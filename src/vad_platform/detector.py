from __future__ import annotations

import base64
import tempfile
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from werkzeug.datastructures import FileStorage

from vad_platform.config import RuntimeConfig, default_config


ProgressCallback = Callable[[str], None]


class ViolenceDetectionService:
    def __init__(self, project_root: Path, config: RuntimeConfig | None = None):
        self.project_root = project_root
        self.config = config or default_config(project_root)
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=self.config.clip_len * self.config.frame_skip)
        self.feature_history: deque[np.ndarray] = deque(maxlen=self.config.live_history)
        self.live_scores: deque[dict[str, Any]] = deque(maxlen=30)
        self.events: deque[dict[str, Any]] = deque(maxlen=100)
        self.live_frame_count = 0
        self.last_live_result: dict[str, Any] | None = None
        self.last_focus_screen = False
        self.live_alert_active = False
        self.last_alert_time = 0.0
        self._runtime: _TorchRuntime | None = None
        self._runtime_error: str | None = None

    def reset(self) -> None:
        self.frame_buffer.clear()
        self.feature_history.clear()
        self.live_scores.clear()
        self.events.clear()
        self.live_frame_count = 0
        self.last_live_result = None
        self.live_alert_active = False
        self.last_alert_time = 0.0

    def health(self) -> dict[str, Any]:
        runtime_ready = self._ensure_runtime()
        return {
            "ready": runtime_ready,
            "checkpoint": str(self.config.checkpoint_path),
            "checkpoint_exists": self.config.checkpoint_path.exists(),
            "device": self._runtime.device_name if self._runtime else None,
            "threshold": self.config.threshold,
            "live_threshold": self.config.live_threshold,
            "live_clear_threshold": self.config.live_clear_threshold,
            "buffered_frames": len(self.frame_buffer),
            "feature_history": len(self.feature_history),
            "error": self._runtime_error,
        }

    def process_live_frame(
        self,
        image_data: str,
        threshold: float | None = None,
        request_focus_screen: bool = False,
    ) -> dict[str, Any]:
        if not self._ensure_runtime():
            return self._not_ready()

        threshold = float(threshold or self.config.live_threshold)
        focus_screen = bool(request_focus_screen)
        if focus_screen != self.last_focus_screen:
            self.frame_buffer.clear()
            self.feature_history.clear()
            self.live_scores.clear()
            self.last_live_result = None
            self.live_frame_count = 0
            self.last_focus_screen = focus_screen
        frame = _decode_data_url(image_data)
        if focus_screen:
            frame = _focus_screen_region(frame)
        self.frame_buffer.append(frame)
        self.live_frame_count += 1

        # Browser frames are already sampled, so live mode uses contiguous frames.
        span = self.config.clip_len
        if len(self.frame_buffer) < span:
            return {
                "ready": False,
                "status": "warming",
                "needed_frames": span - len(self.frame_buffer),
                "events": list(self.events),
            }

        if self.live_frame_count % self.config.live_clip_stride != 0 and self.last_live_result:
            return {
                "ready": True,
                "status": "holding",
                "result": self.last_live_result,
                "feature_count": len(self.feature_history),
                "events": list(self.events),
            }

        clip = list(self.frame_buffer)[-self.config.clip_len :]
        feature = self._runtime.extract_clip_feature(clip)
        self.feature_history.append(feature)
        result = self._score_live_window(threshold)
        result = self._apply_live_hysteresis(result, threshold)
        self.last_live_result = result

        now = time.time()
        if result["prediction"] == "ANOMALY" and now - self.last_alert_time >= self.config.live_cooldown_seconds:
            self.last_alert_time = now
            self.events.appendleft(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "probability": result["prob_anomaly"],
                    "confidence": result["confidence"],
                }
            )

        return {
            "ready": True,
            "status": "scored",
            "result": result,
            "feature_count": len(self.feature_history),
            "events": list(self.events),
        }

    def _score_live_window(self, threshold: float) -> dict[str, Any]:
        features = np.stack(self.feature_history, axis=0)
        history_result = self._runtime.predict(features, threshold)

        recent_count = min(self.config.live_segment_clips, len(features))
        recent_result = self._runtime.predict(features[-recent_count:], threshold)
        operational_score = max(history_result["prob_anomaly"], recent_result["prob_anomaly"])
        result = {
            "prob_anomaly": operational_score,
            "prob_normal": 1.0 - operational_score,
            "prediction": "ANOMALY" if operational_score >= threshold else "NORMAL",
            "confidence": max(operational_score, 1.0 - operational_score),
            "basis": "recent_peak" if recent_result["prob_anomaly"] >= history_result["prob_anomaly"] else "history",
            "recent_prob_anomaly": recent_result["prob_anomaly"],
            "history_prob_anomaly": history_result["prob_anomaly"],
        }
        self.live_scores.append(result)
        return result

    def _apply_live_hysteresis(self, result: dict[str, Any], threshold: float) -> dict[str, Any]:
        score = result["prob_anomaly"]
        clear_threshold = min(self.config.live_clear_threshold, threshold * 0.65)

        if self.live_alert_active:
            if score <= clear_threshold:
                self.live_alert_active = False
        elif score >= threshold:
            self.live_alert_active = True

        result = dict(result)
        result["prediction"] = "ANOMALY" if self.live_alert_active else "NORMAL"
        result["alert_state"] = "active" if self.live_alert_active else "clear"
        result["enter_threshold"] = threshold
        result["clear_threshold"] = clear_threshold
        return result

    def analyze_uploaded_video(
        self,
        video_file: FileStorage,
        threshold: float | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        suffix = Path(video_file.filename or "upload.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_path = Path(tmp.name)
            video_file.save(tmp)

        try:
            return self.analyze_video_path(
                video_path,
                filename=video_file.filename,
                threshold=threshold,
                progress=progress,
            )
        finally:
            video_path.unlink(missing_ok=True)

    def analyze_video_path(
        self,
        video_path: Path,
        filename: str | None = None,
        threshold: float | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        _progress(progress, "$ avt analyze --mode upload")
        _progress(progress, f"[input] file={filename or video_path.name}")
        _progress(progress, f"[input] threshold={threshold if threshold is not None else self.config.threshold:.3f}")
        if not self._ensure_runtime(progress):
            return self._not_ready()

        started_at = time.perf_counter()
        phase_times: dict[str, float] = {}
        read_started = time.perf_counter()
        frames, fps, duration = _read_video(video_path, progress)
        phase_times["read_video_seconds"] = round(time.perf_counter() - read_started, 3)
        _progress(progress, f"[frames] loaded={len(frames)} fps={fps:.2f} duration={duration:.2f}s")
        clips = _build_clips(
            frames,
            clip_len=self.config.clip_len,
            frame_skip=self.config.frame_skip,
            clip_stride=self.config.clip_stride,
        )
        _progress(progress, f"[clips] built={len(clips)} clip_len={self.config.clip_len} frame_skip={self.config.frame_skip}")
        if not clips:
            _progress(progress, "[error] no readable frames found")
            return {"error": "No readable frames found in uploaded video"}

        features = []
        clip_times = []
        extract_started = time.perf_counter()
        for index, (clip_frames, start_idx) in enumerate(clips, start=1):
            _progress(progress, f"[features] extracting feature {index}/{len(clips)} start_frame={start_idx}")
            features.append(self._runtime.extract_clip_feature(clip_frames))
            clip_times.append(start_idx / max(fps, 1.0))
            _progress(progress, f"[features] feature {index} ready shape=(768,) t={clip_times[-1]:.2f}s")
        phase_times["feature_extraction_seconds"] = round(time.perf_counter() - extract_started, 3)

        feature_array = np.stack(features, axis=0)
        active_threshold = threshold if threshold is not None else self.config.threshold
        scoring_started = time.perf_counter()
        _progress(progress, f"[model] scoring full tensor shape={tuple(feature_array.shape)}")
        overall = self._runtime.predict(feature_array, active_threshold, progress=progress, label="overall")
        timeline = self._score_timeline(feature_array, clip_times, fps, duration, active_threshold, progress)
        phase_times["scoring_seconds"] = round(time.perf_counter() - scoring_started, 3)
        _progress(progress, "[metrics] loading timeline metrics")
        anomaly_segments = [s for s in timeline if s["prob_anomaly"] >= active_threshold]
        peak_segment = max(timeline, key=lambda s: s["prob_anomaly"], default=None)
        peak_score = max((s["prob_anomaly"] for s in timeline), default=overall["prob_anomaly"])
        avg_score = float(np.mean([s["prob_anomaly"] for s in timeline])) if timeline else overall["prob_anomaly"]
        anomaly_seconds = _segment_union_seconds(anomaly_segments)
        coverage = anomaly_seconds / max(duration, 0.001)
        frame_samples = _build_frame_samples(frames, fps, timeline, peak_segment, progress=progress)
        operational_score = max(overall["prob_anomaly"], peak_score)
        operational = {
            "prob_anomaly": operational_score,
            "prob_normal": 1.0 - operational_score,
            "prediction": "ANOMALY" if operational_score >= active_threshold else "NORMAL",
            "confidence": max(operational_score, 1.0 - operational_score),
            "basis": "peak_segment" if peak_score >= overall["prob_anomaly"] else "whole_video",
        }
        _progress(progress, f"[done] scoring completed prediction={operational['prediction']} threat={operational_score:.4f}")

        return {
            "filename": filename,
            "duration": duration,
            "fps": fps,
            "clips": len(clips),
            "threshold": active_threshold,
            "overall": overall,
            "operational": operational,
            "timeline": timeline,
            "anomaly_segments": anomaly_segments,
            "peak_score": peak_score,
            "peak_segment": peak_segment,
            "frame_samples": frame_samples,
            "metrics": {
                "frames": len(frames),
                "fps": round(fps, 2),
                "duration_seconds": round(duration, 2),
                "clips": len(clips),
                "features": int(feature_array.shape[0]),
                "feature_dim": int(feature_array.shape[1]),
                "timeline_segments": len(timeline),
                "anomaly_segments": len(anomaly_segments),
                "anomaly_seconds": round(anomaly_seconds, 2),
                "anomaly_coverage": round(coverage, 4),
                "average_score": avg_score,
                "peak_score": peak_score,
                "threshold": active_threshold,
                "processing_seconds": round(time.perf_counter() - started_at, 3),
                "phase_times": phase_times,
            },
        }

    def _score_timeline(
        self,
        features: np.ndarray,
        clip_times: list[float],
        fps: float,
        duration: float,
        threshold: float,
        progress: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        segment_clips = self.config.segment_clips
        step = max(1, segment_clips // 2)
        timeline = []

        for start in range(0, max(1, len(features) - segment_clips + 1), step):
            end = min(start + segment_clips, len(features))
            if start >= end:
                continue
            _progress(progress, f"[timeline] segment {len(timeline) + 1} clips={start}:{end}")
            result = self._runtime.predict(features[start:end], threshold, progress=progress, label=f"segment_{len(timeline) + 1}")
            t_start = clip_times[start]
            t_end = clip_times[end - 1] + (self.config.clip_len * self.config.frame_skip / max(fps, 1.0))
            timeline.append(
                {
                    "start": round(t_start, 2),
                    "end": round(min(t_end, duration), 2),
                    "prob_anomaly": result["prob_anomaly"],
                    "prediction": result["prediction"],
                }
            )

        if not timeline:
            result = self._runtime.predict(features, threshold, progress=progress, label="timeline_fallback")
            timeline.append({"start": 0.0, "end": round(duration, 2), **result})
        return timeline

    def _ensure_runtime(self, progress: ProgressCallback | None = None) -> bool:
        if self._runtime is not None:
            _progress(progress, f"[runtime] cached device={getattr(self._runtime, 'device_name', 'ready')}")
            return True
        try:
            _progress(progress, "[runtime] loading weights and feature extractor")
            self._runtime = _TorchRuntime(self.config, progress=progress)
            self._runtime_error = None
            _progress(progress, "[runtime] ready")
            return True
        except Exception as exc:  # surfaced through /api/health and UI
            self._runtime_error = str(exc)
            _progress(progress, f"[runtime] error={exc}")
            return False

    def _not_ready(self) -> dict[str, Any]:
        return {
            "ready": False,
            "error": self._runtime_error or "Runtime is not ready",
            "install_hint": "Install dependencies with: pip install -r requirements.txt",
        }


class _TorchRuntime:
    def __init__(self, config: RuntimeConfig, progress: ProgressCallback | None = None):
        _progress(progress, "[runtime] importing torch and transformers")
        import torch
        import torch.nn.functional as F
        from transformers.models.auto.image_processing_auto import AutoImageProcessor
        from transformers.models.videomae.modeling_videomae import VideoMAEModel

        from vad_platform.model import AnomalyTransformer

        if not config.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")

        self.torch = torch
        self.F = F
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        _progress(progress, f"[runtime] device={self.device_name}")

        _progress(progress, "[model] building AnomalyTransformer")
        self.model = AnomalyTransformer(
            feat_dim=config.feat_dim,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            max_frames=config.max_frames,
        ).to(self.device)

        _progress(progress, f"[weights] loading {config.checkpoint_path.name}")
        checkpoint = torch.load(config.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state") if isinstance(checkpoint, dict) else None
        if state_dict is None:
            state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        _progress(progress, f"[weights] loaded tensors={len(state_dict)}")

        _progress(progress, f"[videomae] loading processor {config.videomae_name}")
        self.feature_extractor = AutoImageProcessor.from_pretrained(config.videomae_name)
        _progress(progress, f"[videomae] loading model {config.videomae_name}")
        self.feature_model = VideoMAEModel.from_pretrained(config.videomae_name).to(self.device)
        self.feature_model.eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def extract_clip_feature(self, clip_frames: list[np.ndarray]) -> np.ndarray:
        with self.torch.no_grad():
            inputs = self.feature_extractor(clip_frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with self.torch.autocast(
                device_type="cuda",
                enabled=self.device.type == "cuda" and self.config.use_amp,
            ):
                outputs = self.feature_model(pixel_values=pixel_values)
            feature = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()
        return feature.astype(np.float32)

    def predict(
        self,
        feature_array: np.ndarray,
        threshold: float,
        progress: ProgressCallback | None = None,
        label: str = "window",
    ) -> dict[str, Any]:
        feat = self._pad_or_trim(feature_array)
        _progress(progress, f"[calc] {label}: pad_or_trim {feature_array.shape} -> {feat.shape}")
        feat_t = self.torch.from_numpy(feat).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            with self.torch.autocast(
                device_type="cuda",
                enabled=self.device.type == "cuda" and self.config.use_amp,
            ):
                logits = self.model(feat_t)
            probs = self.F.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        prob_normal = float(probs[0])
        prob_anomaly = float(probs[1])
        prediction = "ANOMALY" if prob_anomaly >= threshold else "NORMAL"
        _progress(
            progress,
            f"[calc] {label}: normal={prob_normal:.4f} threat={prob_anomaly:.4f} threshold={threshold:.4f} => {prediction}",
        )
        return {
            "prob_normal": prob_normal,
            "prob_anomaly": prob_anomaly,
            "prediction": prediction,
            "confidence": float(max(probs)),
        }

    def _pad_or_trim(self, feature_array: np.ndarray) -> np.ndarray:
        feature_array = feature_array.astype(np.float32)
        total = feature_array.shape[0]
        if total >= self.config.max_frames:
            start = (total - self.config.max_frames) // 2
            return feature_array[start : start + self.config.max_frames]
        pad = np.zeros((self.config.max_frames - total, feature_array.shape[1]), dtype=np.float32)
        return np.concatenate([feature_array, pad], axis=0)


def _decode_data_url(image_data: str) -> np.ndarray:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    raw = base64.b64decode(image_data)
    image = Image.open(BytesIO(raw)).convert("RGB")
    return np.asarray(image)


def _focus_screen_region(frame: np.ndarray) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    frame_area = width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / max(h, 1)
        if area < frame_area * 0.12 or area > frame_area * 0.92:
            continue
        if aspect < 1.05 or aspect > 2.7:
            continue
        score = area * (1.0 - min(abs(aspect - 1.65), 1.0) * 0.25)
        if best is None or score > best[0]:
            best = (score, x, y, w, h)

    if best is None:
        # Fallback for laptop-screen testing: ignore room edges and focus center.
        crop_w = int(width * 0.82)
        crop_h = int(height * 0.72)
        x = (width - crop_w) // 2
        y = (height - crop_h) // 2
        return frame[y : y + crop_h, x : x + crop_w]

    _, x, y, w, h = best
    pad_x = int(w * 0.03)
    pad_y = int(h * 0.03)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)
    return frame[y0:y1, x0:x1]


def _progress(progress: ProgressCallback | None, message: str) -> None:
    if progress:
        progress(message)


def _read_video(video_path: Path, progress: ProgressCallback | None = None) -> tuple[list[np.ndarray], float, float]:
    import cv2

    _progress(progress, f"[video] opening {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    _progress(progress, f"[video] fps={fps:.2f}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) <= 3 or len(frames) % 50 == 0:
            _progress(progress, f"[frames] extracted frame {len(frames)}")
    cap.release()

    duration = len(frames) / max(fps, 1.0)
    return frames, fps, duration


def _build_clips(
    frames: list[np.ndarray],
    clip_len: int,
    frame_skip: int,
    clip_stride: int,
) -> list[tuple[list[np.ndarray], int]]:
    if not frames:
        return []

    total = len(frames)
    span = clip_len * frame_skip
    clips = []
    start = 0
    while start + span <= total:
        indices = [start + i * frame_skip for i in range(clip_len)]
        clips.append(([frames[i] for i in indices], start))
        start += clip_stride

    if not clips:
        padded = frames + [frames[-1]] * max(0, clip_len - len(frames))
        clips.append((padded[:clip_len], 0))
    return clips


def _segment_union_seconds(segments: list[dict[str, Any]]) -> float:
    if not segments:
        return 0.0

    intervals = sorted((float(seg["start"]), float(seg["end"])) for seg in segments)
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(max(0.0, end - start) for start, end in merged)


def _build_frame_samples(
    frames: list[np.ndarray],
    fps: float,
    timeline: list[dict[str, Any]],
    peak_segment: dict[str, Any] | None,
    max_samples: int = 6,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    if not frames:
        return []

    duration = len(frames) / max(fps, 1.0)
    candidate_times = [0.0, duration * 0.25, duration * 0.5, duration * 0.75]
    if peak_segment:
        candidate_times.insert(0, (float(peak_segment["start"]) + float(peak_segment["end"])) / 2.0)

    top_segments = sorted(timeline, key=lambda seg: seg["prob_anomaly"], reverse=True)[:4]
    candidate_times.extend((float(seg["start"]) + float(seg["end"])) / 2.0 for seg in top_segments)

    samples = []
    used_indices: set[int] = set()
    for sample_time in candidate_times:
        frame_index = min(len(frames) - 1, max(0, int(round(sample_time * max(fps, 1.0)))))
        if frame_index in used_indices:
            continue
        used_indices.add(frame_index)
        scored = _score_at_time(timeline, frame_index / max(fps, 1.0))
        _progress(progress, f"[preview] extracted frame sample {len(samples) + 1} index={frame_index} score={scored['prob_anomaly']:.4f}")
        samples.append(
            {
                "time": round(frame_index / max(fps, 1.0), 2),
                "score": scored["prob_anomaly"],
                "prediction": scored["prediction"],
                "image": _frame_to_data_url(frames[frame_index]),
            }
        )
        if len(samples) >= max_samples:
            break
    return sorted(samples, key=lambda sample: sample["time"])


def _score_at_time(timeline: list[dict[str, Any]], sample_time: float) -> dict[str, Any]:
    if not timeline:
        return {"prob_anomaly": 0.0, "prediction": "NORMAL"}

    for segment in timeline:
        if float(segment["start"]) <= sample_time <= float(segment["end"]):
            return segment
    return min(
        timeline,
        key=lambda seg: min(abs(sample_time - float(seg["start"])), abs(sample_time - float(seg["end"]))),
    )


def _frame_to_data_url(frame: np.ndarray) -> str:
    image = Image.fromarray(frame)
    image.thumbnail((360, 220))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=74, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
