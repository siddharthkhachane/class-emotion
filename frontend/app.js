const MODEL_URL = "https://api-inference.huggingface.co/models/nihar245/Expression-Detection-BEIT-Large";

const EMOTION_TO_STATE = {
  engaged: "engaged",
  neutral: "neutral",
  bored: "bored",
  confused: "confused",
  happy: "engaged",
  sad: "bored",
  fear: "confused",
  surprise: "confused",
  angry: "frustrated",
  disgust: "frustrated"
};

const stateCounts = {
  engaged: 0,
  neutral: 0,
  bored: 0,
  confused: 0,
  frustrated: 0
};

let stream = null;
let timer = null;
let samples = 0;

const tokenEl = document.getElementById("token");
const intervalEl = document.getElementById("interval");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const emotionEl = document.getElementById("emotion");
const stateEl = document.getElementById("state");
const recEl = document.getElementById("recommendation");
const distributionEl = document.getElementById("distributionList");

const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("canvas");
const ctx = canvasEl.getContext("2d");

function normalizeLabel(label) {
  return String(label || "").trim().toLowerCase().replace(/\s+/g, "_");
}

function recommendationFromDistribution(pct) {
  if (pct.bored >= 50) return "Do a quick interactive poll or ask a direct question.";
  if (pct.confused >= 35) return "Re-explain with a concrete example and slow down.";
  if (pct.frustrated >= 25) return "Pause briefly, clarify steps, and address issues.";
  return "Keep going, students look engaged.";
}

function renderDistribution() {
  const pct = {
    engaged: samples ? ((stateCounts.engaged / samples) * 100).toFixed(1) : "0.0",
    neutral: samples ? ((stateCounts.neutral / samples) * 100).toFixed(1) : "0.0",
    bored: samples ? ((stateCounts.bored / samples) * 100).toFixed(1) : "0.0",
    confused: samples ? ((stateCounts.confused / samples) * 100).toFixed(1) : "0.0",
    frustrated: samples ? ((stateCounts.frustrated / samples) * 100).toFixed(1) : "0.0"
  };

  distributionEl.innerHTML = `
    <li>engaged: ${pct.engaged}%</li>
    <li>neutral: ${pct.neutral}%</li>
    <li>bored: ${pct.bored}%</li>
    <li>confused: ${pct.confused}%</li>
    <li>frustrated: ${pct.frustrated}%</li>
  `;

  recEl.textContent = recommendationFromDistribution({
    engaged: Number(pct.engaged),
    neutral: Number(pct.neutral),
    bored: Number(pct.bored),
    confused: Number(pct.confused),
    frustrated: Number(pct.frustrated)
  });
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function blobFromCurrentFrame() {
  canvasEl.width = videoEl.videoWidth;
  canvasEl.height = videoEl.videoHeight;
  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
  return new Promise((resolve) => canvasEl.toBlob(resolve, "image/jpeg", 0.85));
}

async function analyzeCurrentFrame() {
  const token = tokenEl.value.trim();
  if (!token) {
    setStatus("HF token is required.", true);
    return;
  }
  if (!videoEl.videoWidth) {
    return;
  }

  const blob = await blobFromCurrentFrame();
  if (!blob) {
    setStatus("Could not capture webcam frame.", true);
    return;
  }

  try {
    setStatus("Classifying...");
    const response = await fetch(MODEL_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "image/jpeg"
      },
      body: blob
    });

    if (!response.ok) {
      const text = await response.text();
      setStatus(`API error ${response.status}: ${text.slice(0, 120)}`, true);
      return;
    }

    const payload = await response.json();
    if (!Array.isArray(payload) || payload.length === 0) {
      setStatus("Model returned no prediction.", true);
      return;
    }

    const top = payload.reduce((best, item) =>
      (item.score > (best?.score ?? -1) ? item : best), null
    );

    const emotion = normalizeLabel(top?.label);
    if (!emotion || !EMOTION_TO_STATE[emotion]) {
      setStatus("Prediction label not recognized.", true);
      return;
    }

    const state = EMOTION_TO_STATE[emotion];

    emotionEl.textContent = `${emotion} (${Number(top.score).toFixed(2)})`;
    stateEl.textContent = state;
    samples += 1;
    stateCounts[state] += 1;
    renderDistribution();
    setStatus("Live");
  } catch (err) {
    setStatus(`Request failed: ${err.message}`, true);
  }
}

async function startWebcam() {
  const intervalSec = Number(intervalEl.value);
  if (!intervalSec || intervalSec < 0.5) {
    setStatus("Interval must be at least 0.5 seconds.", true);
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoEl.srcObject = stream;
    await videoEl.play();
    setStatus("Camera running. Sampling started.");
    startBtn.disabled = true;
    stopBtn.disabled = false;
    timer = setInterval(analyzeCurrentFrame, intervalSec * 1000);
    analyzeCurrentFrame();
  } catch (err) {
    setStatus(`Camera access failed: ${err.message}`, true);
  }
}

function stopWebcam() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  videoEl.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Stopped.");
}

startBtn.addEventListener("click", startWebcam);
stopBtn.addEventListener("click", stopWebcam);
