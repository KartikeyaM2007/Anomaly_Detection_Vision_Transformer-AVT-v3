const els = {
  runtimeStatus: document.getElementById("runtimeStatus"),
  infoBtn: document.getElementById("infoBtn"),
  closeInfoBtn: document.getElementById("closeInfoBtn"),
  infoModal: document.getElementById("infoModal"),
  cameraState: document.getElementById("cameraState"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  resetBtn: document.getElementById("resetBtn"),
  camera: document.getElementById("camera"),
  capture: document.getElementById("capture"),
  videoShell: document.getElementById("videoShell"),
  verdict: document.getElementById("verdict"),
  threatScore: document.getElementById("threatScore"),
  confidenceScore: document.getElementById("confidenceScore"),
  featureCount: document.getElementById("featureCount"),
  events: document.getElementById("events"),
  device: document.getElementById("device"),
  checkpoint: document.getElementById("checkpoint"),
  threshold: document.getElementById("threshold"),
  clientGpu: document.getElementById("clientGpu"),
  clientCpu: document.getElementById("clientCpu"),
  clientMemory: document.getElementById("clientMemory"),
  thresholdInput: document.getElementById("thresholdInput"),
  thresholdValue: document.getElementById("thresholdValue"),
  liveThresholdInput: document.getElementById("liveThresholdInput"),
  liveThresholdValue: document.getElementById("liveThresholdValue"),
  screenFocusInput: document.getElementById("screenFocusInput"),
  uploadForm: document.getElementById("uploadForm"),
  videoFile: document.getElementById("videoFile"),
  uploadResult: document.getElementById("uploadResult"),
  analysisState: document.getElementById("analysisState"),
  analysisTimer: document.getElementById("analysisTimer"),
  analysisSteps: document.getElementById("analysisSteps"),
  analysisTerminal: document.getElementById("analysisTerminal"),
  clearTerminalBtn: document.getElementById("clearTerminalBtn"),
  visualSummary: document.getElementById("visualSummary"),
  analysisMetrics: document.getElementById("analysisMetrics"),
  frameGallery: document.getElementById("frameGallery"),
  timeline: document.getElementById("timeline"),
};

let stream = null;
let timer = null;
let busy = false;
let analysisStartedAt = null;
let analysisTimerId = null;
let workflowId = null;
let activeStep = "queued";
let terminalStartedAt = null;
let activeEventSource = null;

const stepOrder = ["queued", "uploading", "frames", "features", "scoring", "completed"];
const stepLabels = {
  queued: "File queued",
  uploading: "Uploading video",
  frames: "Preparing frames",
  features: "Extracting features",
  scoring: "Scoring anomaly timeline",
  completed: "Analysis completed",
};

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function pct1(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function fmtTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(mins).padStart(2, "0")}:${secs.toFixed(1).padStart(4, "0")}`;
}

function fmtSeconds(seconds) {
  return `${Number(seconds || 0).toFixed(2)}s`;
}

function terminalTime() {
  if (!terminalStartedAt) return "00:00.000";
  const elapsed = performance.now() - terminalStartedAt;
  const minutes = Math.floor(elapsed / 60000);
  const seconds = Math.floor((elapsed % 60000) / 1000);
  const millis = Math.floor(elapsed % 1000);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function resetTerminal(fileName = "") {
  terminalStartedAt = performance.now();
  els.analysisTerminal.textContent = [
    "Windows PowerShell",
    "Copyright (C) Microsoft Corporation. All rights reserved.",
    "",
    `PS avt> analyze ${fileName || "upload.mp4"}`,
  ].join("\n");
  els.analysisTerminal.scrollTop = els.analysisTerminal.scrollHeight;
}

function appendTerminalLine(message, level = "info") {
  const prefix = level === "error" ? "ERR" : level === "complete" ? "OK " : "   ";
  els.analysisTerminal.textContent += `\n[${terminalTime()}] ${prefix} ${message}`;
  els.analysisTerminal.scrollTop = els.analysisTerminal.scrollHeight;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

async function loadHealth() {
  const health = await api("/api/health");
  els.runtimeStatus.textContent = health.ready ? "Runtime ready" : "Runtime not ready";
  els.runtimeStatus.className = `status ${health.ready ? "ok" : "bad"}`;
  els.device.textContent = health.device || "--";
  els.checkpoint.textContent = health.checkpoint_exists ? "best_model.pt found" : "missing";
  els.threshold.textContent = health.threshold;
  els.thresholdInput.value = health.threshold;
  els.thresholdValue.textContent = pct(health.threshold);
  els.liveThresholdInput.value = health.live_threshold || 0.12;
  els.liveThresholdValue.textContent = pct(health.live_threshold || 0.12);
  if (health.error) {
    els.cameraState.textContent = health.error;
  }
}

function getWebGLRenderer() {
  const canvas = document.createElement("canvas");
  const gl =
    canvas.getContext("webgl2") ||
    canvas.getContext("webgl") ||
    canvas.getContext("experimental-webgl");
  if (!gl) return null;

  const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
  if (debugInfo) {
    return gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
  }
  return gl.getParameter(gl.RENDERER);
}

function loadClientHardware() {
  const renderer = getWebGLRenderer();
  els.clientGpu.textContent = renderer || "Unavailable in this browser";
  els.clientCpu.textContent = navigator.hardwareConcurrency
    ? `${navigator.hardwareConcurrency} logical threads`
    : "Unavailable";
  els.clientMemory.textContent = navigator.deviceMemory
    ? `${navigator.deviceMemory} GB reported`
    : "Unavailable";
}

function openInfoModal() {
  els.infoModal.hidden = false;
  document.body.classList.add("modal-open");
  els.closeInfoBtn.focus();
}

function closeInfoModal() {
  els.infoModal.hidden = true;
  document.body.classList.remove("modal-open");
  els.infoBtn.focus();
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  els.camera.srcObject = stream;
  els.startBtn.disabled = true;
  els.stopBtn.disabled = false;
  els.cameraState.textContent = "Scoring live frames";
  timer = setInterval(sendFrame, 250);
}

function stopCamera() {
  if (timer) clearInterval(timer);
  timer = null;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  els.camera.srcObject = null;
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  els.cameraState.textContent = "Camera idle";
}

async function sendFrame() {
  if (busy || !stream) return;
  busy = true;
  try {
    const ctx = els.capture.getContext("2d");
    ctx.drawImage(els.camera, 0, 0, els.capture.width, els.capture.height);
    const image = els.capture.toDataURL("image/jpeg", 0.8);
    const data = await api("/api/live-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image,
        threshold: Number(els.liveThresholdInput.value),
        focusScreen: els.screenFocusInput.checked,
      }),
    });
    renderLive(data);
  } catch (err) {
    els.cameraState.textContent = err.message;
  } finally {
    busy = false;
  }
}

function renderLive(data) {
  if (!data.ready) {
    els.verdict.textContent = "Warming";
    els.cameraState.textContent = data.needed_frames
      ? `Collecting frames: ${data.needed_frames} more`
      : data.error || "Runtime not ready";
    return;
  }
  const result = data.result;
  const isAlert = result.prediction === "ANOMALY";
  els.videoShell.classList.toggle("alert", isAlert);
  els.verdict.textContent = isAlert ? "Threat" : "Normal";
  els.threatScore.textContent = pct1(result.prob_anomaly);
  els.confidenceScore.textContent = pct(result.confidence);
  els.featureCount.textContent = data.feature_count;
  const basis = result.basis ? `, ${result.basis}` : "";
  const focus = els.screenFocusInput.checked ? ", screen focus" : "";
  const state = result.alert_state ? `, ${result.alert_state}` : "";
  els.cameraState.textContent = `Last score: ${new Date().toLocaleTimeString()}${basis}${state}${focus}`;
  renderEvents(data.events || []);
}

function renderEvents(events) {
  if (!events.length) {
    els.events.className = "events empty";
    els.events.textContent = "No alerts";
    return;
  }
  els.events.className = "events";
  els.events.innerHTML = events
    .map((event) => `<div><strong>${event.time}</strong><span>${pct(event.probability)} threat</span></div>`)
    .join("");
}

function resetWorkflow() {
  stopWorkflowTimer();
  activeStep = "queued";
  els.analysisTimer.textContent = "00:00.0";
  els.analysisState.textContent = "Waiting for upload";
  els.visualSummary.innerHTML = "";
  els.analysisMetrics.innerHTML = "";
  els.frameGallery.innerHTML = "";
  renderWorkflow();
}

function startWorkflow(file) {
  stopWorkflowTimer();
  if (activeEventSource) activeEventSource.close();
  resetTerminal(file.name);
  appendTerminalLine("file queued");
  analysisStartedAt = performance.now();
  activeStep = "queued";
  els.analysisState.textContent = `${file.name} queued`;
  renderWorkflow();
  analysisTimerId = setInterval(() => {
    els.analysisTimer.textContent = fmtTime((performance.now() - analysisStartedAt) / 1000);
  }, 100);
  workflowId = setInterval(advanceWorkflowEstimate, 2600);
}

function stopWorkflowTimer() {
  if (analysisTimerId) clearInterval(analysisTimerId);
  if (workflowId) clearInterval(workflowId);
  analysisTimerId = null;
  workflowId = null;
}

function advanceWorkflowEstimate() {
  const current = stepOrder.indexOf(activeStep);
  if (current >= 1 && current < stepOrder.indexOf("scoring")) {
    setWorkflowStep(stepOrder[current + 1], "running");
  }
}

function setWorkflowStep(step, mode = "running", detail = "") {
  activeStep = step;
  els.analysisState.textContent = detail || stepLabels[step];
  renderWorkflow(mode);
}

function completeWorkflow(data) {
  stopWorkflowTimer();
  if (analysisStartedAt) {
    els.analysisTimer.textContent = fmtTime((performance.now() - analysisStartedAt) / 1000);
  }
  activeStep = "completed";
  els.analysisState.textContent = `${data.filename || "Video"} analysis completed`;
  renderWorkflow("complete");
}

function failWorkflow(message) {
  stopWorkflowTimer();
  els.analysisState.textContent = message;
  renderWorkflow("failed");
}

function renderWorkflow(mode = "running") {
  const activeIndex = stepOrder.indexOf(activeStep);
  els.analysisSteps.querySelectorAll(".step-node").forEach((node) => {
    const index = stepOrder.indexOf(node.dataset.step);
    node.classList.remove("pending", "running", "done", "failed");
    if (mode === "failed" && index === activeIndex) {
      node.classList.add("failed");
    } else if (index < activeIndex || mode === "complete") {
      node.classList.add("done");
    } else if (index === activeIndex) {
      node.classList.add(mode === "complete" ? "done" : "running");
    } else {
      node.classList.add("pending");
    }
  });
}

async function reset() {
  await api("/api/reset", { method: "POST" });
  els.featureCount.textContent = "0";
  els.threatScore.textContent = "--";
  els.confidenceScore.textContent = "--";
  els.verdict.textContent = "Idle";
  els.videoShell.classList.remove("alert");
  renderEvents([]);
}

async function analyzeVideo(event) {
  event.preventDefault();
  const file = els.videoFile.files[0];
  if (!file) return;
  startWorkflow(file);
  els.uploadResult.textContent = `Starting analysis on ${els.device.textContent || "runtime"}...`;
  els.timeline.innerHTML = "";
  els.visualSummary.innerHTML = "";
  els.analysisMetrics.innerHTML = "";
  els.frameGallery.innerHTML = "";

  const form = new FormData();
  form.append("video", file);
  form.append("threshold", els.thresholdInput.value);
  try {
    const job = await uploadForAnalysis(form);
    appendTerminalLine(`server accepted job ${job.job_id}`);
    const data = await streamAnalysisJob(job.job_id);
    setWorkflowStep("completed", "complete");
    completeWorkflow(data);
    renderUpload(data);
  } catch (err) {
    failWorkflow(err.message);
    els.uploadResult.textContent = err.message;
  }
}

function uploadForAnalysis(form) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/analyze-video");
    xhr.upload.addEventListener("loadstart", () => setWorkflowStep("uploading"));
    xhr.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.round((event.loaded / event.total) * 100);
      setWorkflowStep("uploading", "running", `Uploading video: ${percent}%`);
    });
    xhr.upload.addEventListener("load", () => setWorkflowStep("frames", "running", "Upload complete, reading frames"));
    xhr.addEventListener("readystatechange", () => {
      if (xhr.readyState === XMLHttpRequest.HEADERS_RECEIVED) {
        setWorkflowStep("scoring", "running", "Scoring result");
      }
    });
    xhr.addEventListener("load", () => {
      let data = {};
      try {
        data = JSON.parse(xhr.responseText || "{}");
      } catch (err) {
        reject(new Error("Could not parse analysis response"));
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(data);
      } else {
        reject(new Error(data.error || "Request failed"));
      }
    });
    xhr.addEventListener("error", () => reject(new Error("Upload failed")));
    xhr.send(form);
  });
}

function streamAnalysisJob(jobId) {
  return new Promise((resolve, reject) => {
    activeEventSource = new EventSource(`/api/analyze-video/${jobId}/events`);
    activeEventSource.onmessage = (event) => {
      let data = {};
      try {
        data = JSON.parse(event.data || "{}");
      } catch (err) {
        appendTerminalLine("could not parse server event", "error");
        return;
      }

      if (data.type === "log") {
        appendTerminalLine(data.message);
        syncWorkflowFromTerminal(data.message);
      } else if (data.type === "complete") {
        appendTerminalLine(data.message || "analysis complete", "complete");
        activeEventSource.close();
        activeEventSource = null;
        resolve(data.result);
      } else if (data.type === "error") {
        appendTerminalLine(data.message || "analysis failed", "error");
        activeEventSource.close();
        activeEventSource = null;
        reject(new Error(data.message || "Analysis failed"));
      }
    };
    activeEventSource.onerror = () => {
      appendTerminalLine("event stream disconnected", "error");
      if (activeEventSource) activeEventSource.close();
      activeEventSource = null;
      reject(new Error("Analysis event stream disconnected"));
    };
  });
}

function syncWorkflowFromTerminal(message) {
  if (message.startsWith("[video]") || message.startsWith("[frames]")) {
    setWorkflowStep("frames", "running", message.replace(/^\[[^\]]+\]\s*/, ""));
  } else if (message.startsWith("[clips]") || message.startsWith("[features]") || message.startsWith("[videomae]")) {
    setWorkflowStep("features", "running", message.replace(/^\[[^\]]+\]\s*/, ""));
  } else if (message.startsWith("[model] scoring") || message.startsWith("[timeline]") || message.startsWith("[calc]")) {
    setWorkflowStep("scoring", "running", message.replace(/^\[[^\]]+\]\s*/, ""));
  } else if (message.startsWith("[metrics]")) {
    setWorkflowStep("scoring", "running", "Loading metrics");
  } else if (message.startsWith("[done]")) {
    setWorkflowStep("completed", "complete", "Scoring completed");
  }
}

function renderUpload(data) {
  const overall = data.operational || data.overall;
  const raw = data.overall;
  const peak = data.peak_segment;
  const metrics = data.metrics || {};
  els.uploadResult.innerHTML = `
    <strong>${overall.prediction}</strong>
    <span>${pct1(overall.prob_anomaly)} threat probability</span>
    <span>raw model ${pct1(raw.prob_anomaly)}</span>
    <span>peak ${pct1(data.peak_score)}</span>
    <span>${data.clips} clips, ${data.duration.toFixed(1)}s</span>
    ${peak ? `<span>peak at ${peak.start}s-${peak.end}s</span>` : ""}
  `;
  renderVisualSummary(data, overall, raw);
  renderAnalysisMetrics(metrics, overall);
  renderFrameGallery(data.frame_samples || []);
  const duration = Math.max(data.duration, 0.1);
  els.timeline.innerHTML = data.timeline
    .map((seg) => {
      const left = (seg.start / duration) * 100;
      const width = Math.max(((seg.end - seg.start) / duration) * 100, 1);
      const alert = seg.prediction === "ANOMALY";
      return `<div class="${alert ? "danger" : "clear"}" style="left:${left}%;width:${width}%;" title="${seg.start}s-${seg.end}s ${pct(seg.prob_anomaly)}"></div>`;
    })
    .join("");
}

function renderVisualSummary(data, overall, raw) {
  const metrics = data.metrics || {};
  const score = overall.prob_anomaly || 0;
  const normal = raw.prob_normal || 0;
  const threat = raw.prob_anomaly || 0;
  const peak = data.peak_score || score;
  const average = metrics.average_score || 0;
  const coverage = metrics.anomaly_coverage || 0;
  const threshold = metrics.threshold || Number(els.thresholdInput.value);
  const chartBars = (data.timeline || [])
    .slice(0, 80)
    .map((segment) => {
      const height = Math.max(8, Math.round(segment.prob_anomaly * 100));
      const state = segment.prob_anomaly >= threshold ? "danger" : "clear";
      return `<span class="${state}" style="height:${height}%;" title="${segment.start}s-${segment.end}s ${pct1(segment.prob_anomaly)}"></span>`;
    })
    .join("");

  els.visualSummary.innerHTML = `
    <section class="score-gauge" style="--score:${Math.round(score * 100)};">
      <div class="gauge-ring">
        <strong>${pct1(score)}</strong>
        <span>${overall.prediction}</span>
      </div>
      <div class="gauge-copy">
        <span>Operational score</span>
        <strong>${overall.basis === "peak_segment" ? "Peak-driven" : "Whole-video"}</strong>
      </div>
    </section>
    <section class="score-bars">
      <div>
        <span>Normal</span>
        <strong>${pct1(normal)}</strong>
        <i><b style="width:${Math.round(normal * 100)}%;"></b></i>
      </div>
      <div>
        <span>Threat</span>
        <strong>${pct1(threat)}</strong>
        <i><b class="danger" style="width:${Math.round(threat * 100)}%;"></b></i>
      </div>
      <div>
        <span>Peak</span>
        <strong>${pct1(peak)}</strong>
        <i><b class="danger" style="width:${Math.round(peak * 100)}%;"></b></i>
      </div>
    </section>
    <section class="score-strip">
      <div>
        <span>Average score</span>
        <strong>${pct1(average)}</strong>
      </div>
      <div>
        <span>Coverage</span>
        <strong>${pct1(coverage)}</strong>
      </div>
      <div>
        <span>Threshold</span>
        <strong>${pct1(threshold)}</strong>
      </div>
    </section>
    <section class="score-chart" aria-label="Timeline score chart">${chartBars}</section>
  `;
}

function renderAnalysisMetrics(metrics, overall) {
  const phaseTimes = metrics.phase_times || {};
  const rows = [
    ["Duration", fmtSeconds(metrics.duration_seconds)],
    ["Frames", metrics.frames ?? "--"],
    ["FPS", metrics.fps ?? "--"],
    ["Clips", metrics.clips ?? "--"],
    ["Features", metrics.features ? `${metrics.features} x ${metrics.feature_dim}` : "--"],
    ["Timeline Segments", metrics.timeline_segments ?? "--"],
    ["Anomaly Coverage", pct1(metrics.anomaly_coverage || 0)],
    ["Anomaly Time", fmtSeconds(metrics.anomaly_seconds)],
    ["Average Score", pct1(metrics.average_score || 0)],
    ["Peak Score", pct1(metrics.peak_score || overall.prob_anomaly)],
    ["Processing Time", fmtSeconds(metrics.processing_seconds)],
    ["Threshold", pct1(metrics.threshold || Number(els.thresholdInput.value))],
  ];
  const phases = [
    ["Read Video", phaseTimes.read_video_seconds],
    ["Feature Extract", phaseTimes.feature_extraction_seconds],
    ["Model Scoring", phaseTimes.scoring_seconds],
  ];
  els.analysisMetrics.innerHTML = `
    ${rows
      .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
      .join("")}
    <div class="phase-card">
      <span>Step Timing</span>
      ${phases.map(([label, value]) => `<small>${label}: ${fmtSeconds(value)}</small>`).join("")}
    </div>
  `;
}

function renderFrameGallery(samples) {
  if (!samples.length) {
    els.frameGallery.innerHTML = "";
    return;
  }
  els.frameGallery.innerHTML = `
    <div class="gallery-header">
      <h3>Frame Examples</h3>
      <p>Sampled frames with nearest timeline score</p>
    </div>
    <div class="frame-grid">
      ${samples
        .map(
          (sample) => `
            <figure class="${sample.prediction === "ANOMALY" ? "danger" : "clear"}">
              <img src="${sample.image}" alt="Video frame at ${sample.time}s">
              <figcaption>
                <span>${sample.time.toFixed(2)}s</span>
                <strong>${pct1(sample.score)}</strong>
              </figcaption>
            </figure>
          `
        )
        .join("")}
    </div>
  `;
}

els.startBtn.addEventListener("click", startCamera);
els.stopBtn.addEventListener("click", stopCamera);
els.resetBtn.addEventListener("click", reset);
els.clearTerminalBtn.addEventListener("click", () => resetTerminal());
els.infoBtn.addEventListener("click", openInfoModal);
els.closeInfoBtn.addEventListener("click", closeInfoModal);
els.infoModal.addEventListener("click", (event) => {
  if (event.target === els.infoModal) closeInfoModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !els.infoModal.hidden) closeInfoModal();
});
els.uploadForm.addEventListener("submit", analyzeVideo);
els.videoFile.addEventListener("change", resetWorkflow);
els.thresholdInput.addEventListener("input", () => {
  els.thresholdValue.textContent = pct(Number(els.thresholdInput.value));
});
els.liveThresholdInput.addEventListener("input", () => {
  els.liveThresholdValue.textContent = pct(Number(els.liveThresholdInput.value));
});
loadHealth().catch((err) => {
  els.runtimeStatus.textContent = err.message;
  els.runtimeStatus.className = "status bad";
});
loadClientHardware();
resetWorkflow();
