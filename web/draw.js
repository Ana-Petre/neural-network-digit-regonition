/**
 * draw.js
 * -----------------------------------------------------------------------
 * Handles the 280x280 drawing canvas and converts what's drawn into a
 * 28x28 pixel array (784 values, 0-1) in the SAME format the network was
 * trained on: white strokes on black background, background = 0.
 *
 * This file is complete — no TODOs here, it's UI plumbing, not the ML
 * part of the project.
 * -----------------------------------------------------------------------
 */

const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");

let drawing = false;
let hasDrawn = false;
let onStrokeCallback = null;

function resetCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "white";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  hasDrawn = false;
}
resetCanvas();

function getPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const point = evt.touches ? evt.touches[0] : evt;
  return {
    x: (point.clientX - rect.left) * scaleX,
    y: (point.clientY - rect.top) * scaleY,
  };
}

let lastPos = null;

function startDraw(evt) {
  evt.preventDefault();
  drawing = true;
  hasDrawn = true;
  lastPos = getPos(evt);
}

function moveDraw(evt) {
  if (!drawing) return;
  evt.preventDefault();
  const pos = getPos(evt);
  ctx.beginPath();
  ctx.moveTo(lastPos.x, lastPos.y);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  lastPos = pos;
  if (onStrokeCallback) onStrokeCallback();
}

function endDraw() {
  drawing = false;
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", moveDraw);
window.addEventListener("mouseup", endDraw);

canvas.addEventListener("touchstart", startDraw);
canvas.addEventListener("touchmove", moveDraw);
window.addEventListener("touchend", endDraw);

clearBtn.addEventListener("click", () => {
  resetCanvas();
  if (onStrokeCallback) onStrokeCallback();
});

/**
 * Downsample the 280x280 canvas to a 28x28 grid and return it as a flat
 * 784-length array of values 0-1, matching the MNIST input format
 * (this uses simple block-averaging, not anything fancy).
 */
function getPixelArray() {
  const size = 28;
  const block = canvas.width / size; // 10px per MNIST pixel
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const pixels = new Array(size * size).fill(0);

  for (let by = 0; by < size; by++) {
    for (let bx = 0; bx < size; bx++) {
      let sum = 0;
      for (let y = 0; y < block; y++) {
        for (let x = 0; x < block; x++) {
          const px = bx * block + x;
          const py = by * block + y;
          const idx = (py * canvas.width + px) * 4;
          // Use the red channel (white stroke on black => R=G=B already).
          sum += imgData[idx];
        }
      }
      const avg = sum / (block * block) / 255; // normalize to 0-1
      pixels[by * size + bx] = avg;
    }
  }
  return pixels;
}

/** Register a callback fired every time the drawing changes. */
function onStroke(callback) {
  onStrokeCallback = callback;
}

function canvasHasDrawing() {
  return hasDrawn;
}
