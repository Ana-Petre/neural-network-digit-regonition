/**
 * predict.js
 * -----------------------------------------------------------------------
 * Loads the trained weights (produced by `node train.js`) and exposes a
 * ready-to-use Network instance. Complete — no TODOs, just plumbing that
 * reuses the Network class you implemented in network.js.
 * -----------------------------------------------------------------------
 */

let trainedNetwork = null;
let loadError = null;

async function loadTrainedNetwork() {
  try {
    const res = await fetch("weights.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    trainedNetwork = Network.fromJSON(json);
    return trainedNetwork;
  } catch (err) {
    loadError = err;
    console.error("Could not load weights.json — run `node train.js` first.", err);
    return null;
  }
}
