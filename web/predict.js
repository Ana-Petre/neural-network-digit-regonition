/*
 Loads the trained weights (produced by node train.js)
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
