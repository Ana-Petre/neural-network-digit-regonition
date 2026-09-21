/**
 * activations.js
 * -----------------------------------------------------------------------
 * Activation functions for the hidden layer (ReLU) and output layer
 * (softmax), plus the derivative ReLU needs for backpropagation.
 * -----------------------------------------------------------------------
 */

/**
 * TODO (Day 3): ReLU applied to a single number.
 *   relu(x) = max(0, x)
 */
function relu(x) {
  // TODO: implement
}

/**
 * TODO (Day 3): Apply relu() to every element of a vector.
 * Hint: vec.map(...)
 */
function reluVector(vec) {
  // TODO: implement
}

/**
 * TODO (Day 4, backprop): Derivative of ReLU.
 *   reluDerivative(x) = 1 if x > 0, else 0
 */
function reluDerivative(x) {
  // TODO: implement
}

/**
 * TODO (Day 4, backprop): Apply reluDerivative() to every element of a vector.
 */
function reluDerivativeVector(vec) {
  // TODO: implement
}

/**
 * TODO (Day 3): Softmax turns a vector of raw scores into a probability
 * distribution — all values in (0, 1), summing to 1. Used on the output
 * layer so the 10 outputs can be read as "probability it's this digit".
 *
 *   softmax(x)_i = exp(x_i) / sum(exp(x_j) for all j)
 *
 * IMPORTANT numerical-stability trick: subtract the max value in the
 * vector from every element BEFORE calling Math.exp(). This does not
 * change the mathematical result (it cancels out in the division) but
 * prevents Math.exp() from overflowing to Infinity on larger inputs.
 *
 *   const m = Math.max(...x);
 *   const shifted = x.map(v => v - m);
 *   // then exponentiate `shifted`, not `x`
 */
function softmax(vec) {
  // TODO: implement (remember the max-subtraction trick above)
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
