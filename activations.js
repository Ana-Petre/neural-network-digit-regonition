// activations.js
// relu for the hidden layer, softmax for the output layer, + relu's derivative for backprop

// relu(x) = max(0, x) -> kills negative values, keeps positive ones as is
function relu(x) {
  return Math.max(0, x);
}

// applies relu to every index of the vector, returns a new vec
function reluVector(vec) {
  return vec.map(relu);
}

// slope of relu: 1 where x was positive, 0 where it was cut to zero
// needed in backprop to know how much error passes back through a neuron
function reluDerivative(x) {
  if (x > 0) return 1;
  else return 0;
}

function reluDerivativeVector(vec) {
  return vec.map(reluDerivative);
}

// turns the 10 raw output scores into probabilities that sum to 1
// softmax(x)_i = exp(x_i) / sum(exp(x_j))
function softmax(vec) {
  const m = Math.max(...vec); // ... = spread operator, splits the vec into individual args
  const shifted = vec.map((v) => v - m); // subtract max first so exp() doesn't blow up to Infinity
  const exps = shifted.map((v) => Math.exp(v));
  const sum = exps.reduce((a, b) => a + b, 0); // reduce = collapses all elements into one (the sum)
  return exps.map((v) => v / sum);
}

const ActivationsLib = {
  relu,
  reluVector,
  reluDerivative,
  reluDerivativeVector,
  softmax,
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = ActivationsLib;
} else {
  window.Activations = ActivationsLib;
}