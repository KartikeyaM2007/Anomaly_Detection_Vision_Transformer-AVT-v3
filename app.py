from pathlib import Path
import json
import queue
import sys
import tempfile
import threading
import time
import uuid

from flask import Flask, Response, jsonify, render_template, request, send_from_directory


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vad_platform.detector import ViolenceDetectionService


app = Flask(
    __name__,
    static_folder=str(ROOT / "web" / "static"),
    template_folder=str(ROOT / "web" / "templates"),
)
service = ViolenceDetectionService(project_root=ROOT)
analysis_jobs = {}
analysis_jobs_lock = threading.Lock()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


@app.get("/project-assets/screenshots/<path:filename>")
def project_screenshot(filename):
    return send_from_directory(ROOT / "docs" / "screenshots", filename)


@app.get("/project-assets/training-notebook/<path:filename>")
def training_notebook_screenshot(filename):
    return send_from_directory(ROOT / "docs" / "screenshots" / "training-notebook", filename)


@app.get("/api/health")
def health():
    return jsonify(service.health())


@app.post("/api/reset")
def reset():
    service.reset()
    return jsonify({"ok": True})


@app.post("/api/live-frame")
def live_frame():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "Missing image field"}), 400
    return jsonify(
        service.process_live_frame(
            image_data,
            threshold=payload.get("threshold"),
            request_focus_screen=bool(payload.get("focusScreen")),
        )
    )


@app.post("/api/analyze-video")
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "Upload field must be named video"}), 400

    threshold = request.form.get("threshold", type=float)
    video_file = request.files["video"]
    suffix = Path(video_file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        video_path = Path(tmp.name)
        video_file.save(tmp)

    job = _create_analysis_job(video_path, video_file.filename, threshold)
    return jsonify({"job_id": job["id"]})


@app.get("/api/analyze-video/<job_id>/events")
def analyze_video_events(job_id):
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Analysis job not found"}), 404

    def stream():
        yield _sse({"type": "log", "message": f"$ connected to analysis job {job_id}"})
        while True:
            event = job["events"].get()
            yield _sse(event)
            if event.get("type") in {"complete", "error"}:
                break
        with analysis_jobs_lock:
            analysis_jobs.pop(job_id, None)

    return Response(stream(), mimetype="text/event-stream")


def _create_analysis_job(video_path: Path, filename: str | None, threshold: float | None):
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "events": queue.Queue(),
        "created_at": time.time(),
    }
    with analysis_jobs_lock:
        analysis_jobs[job_id] = job

    def emit(message: str):
        job["events"].put({"type": "log", "message": message, "time": time.time()})

    def run():
        emit("$ job started")
        try:
            result = service.analyze_video_path(video_path, filename=filename, threshold=threshold, progress=emit)
            if "error" in result:
                job["events"].put({"type": "error", "message": result["error"], "time": time.time()})
            else:
                job["events"].put({"type": "complete", "message": "analysis complete", "result": result, "time": time.time()})
        except Exception as exc:
            job["events"].put({"type": "error", "message": str(exc), "time": time.time()})
        finally:
            video_path.unlink(missing_ok=True)

    threading.Thread(target=run, daemon=True).start()
    return job


def _sse(event):
    return f"data: {json.dumps(event)}\n\n"


if __name__ == "__main__":
    import os
    # Hugging Face Spaces use port 7860 by default
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
