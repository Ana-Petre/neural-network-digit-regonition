// Wires drawing → prediction → UI and renders the training-loss chart.

const statusEl = document.getElementById("status");
const predictedDigitEl = document.getElementById("predictedDigit");
const barsEl = document.getElementById("bars");

// Build the 10 probability bar rows once.
for (let d = 0; d < 10; d++) {
  const row = document.createElement("div");
  row.className = "bar-row";
  row.innerHTML = `
    <span class="bar-label">${d}</span>
    <span class="bar-track"><span class="bar-fill" id="bar-${d}"></span></span>
    <span class="bar-pct" id="pct-${d}">0%</span>
  `;
  barsEl.appendChild(row);
}

function updatePrediction() {
  if (!trainedNetwork) return;
  if (!canvasHasDrawing()) {
    predictedDigitEl.textContent = "–";
    for (let d = 0; d < 10; d++) {
      document.getElementById(`bar-${d}`).style.width = "0%";
      document.getElementById(`pct-${d}`).textContent = "0%";
    }
    return;
  }

  const input = getPixelArray();
  const { digit, probabilities } = trainedNetwork.predict(input);
  predictedDigitEl.textContent = digit;

  for (let d = 0; d < 10; d++) {
    const pct = (probabilities[d] * 100).toFixed(1);
    document.getElementById(`bar-${d}`).style.width = `${pct}%`;
    document.getElementById(`pct-${d}`).textContent = `${pct}%`;
  }
}

onStroke(updatePrediction);

loadTrainedNetwork().then((net) => {
  if (net) {
    statusEl.textContent = `Network loaded (${net.hiddenSize} hidden neurons). Start drawing!`;
  } else {
    statusEl.textContent = "Could not load weights.json — run `node train.js` first, then reload this page.";
  }
});



async function renderLossChart() {
  const canvas = document.getElementById("lossCanvas");
  const ctx = canvas.getContext("2d");
  let log;
  try {
    const res = await fetch("training-log.json");
    if (!res.ok) throw new Error("no training-log.json yet");
    log = await res.json();
  } catch (e) {
    ctx.fillStyle = "#666";
    ctx.font = "14px sans-serif";
    ctx.fillText("No training log yet — run `node train.js`.", 12, 24);
    return;
  }

  const losses = log.map((e) => e.avgLoss);
  const maxLoss = Math.max(...losses);
  const w = canvas.width;
  const h = canvas.height;
  const pad = 24;

  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#e0e3e8";
  ctx.beginPath();
  ctx.moveTo(pad, 0);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w, h - pad);
  ctx.stroke();

  ctx.strokeStyle = "#1f3864";
  ctx.lineWidth = 2;
  ctx.beginPath();
  losses.forEach((loss, i) => {
    const x = pad + (i / (losses.length - 1)) * (w - pad - 10);
    const y = h - pad - (loss / maxLoss) * (h - pad - 10);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#666";
  ctx.font = "12px sans-serif";
  ctx.fillText(`loss: ${maxLoss.toFixed(3)}`, pad + 4, 14);
  ctx.fillText("0", pad + 4, h - pad + 14);
  ctx.fillText(`epoch ${losses.length}`, w - 70, h - pad + 14);
}

renderLossChart();
