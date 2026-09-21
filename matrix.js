/**
 * matrix.js
 * -----------------------------------------------------------------------
 * Minimal matrix / vector helpers for the neural network.
 * No external libraries — you implement the math yourself.
 *
 * Representation convention used everywhere in this project:
 *   - A "vector" is a plain JS array of numbers:      [x1, x2, x3, ...]
 *   - A "matrix" is an array of row-vectors:           [[..row0..], [..row1..], ...]
 *
 * These are the building blocks for the forward pass (Day 3) and
 * backpropagation (Day 4) from the plan. Fill in every TODO below.
 * Test each function as you go with `node test-matrix.js`.
 * -----------------------------------------------------------------------
 */

/**
 * Create a matrix of the given shape filled with zeros.
 * Already implemented for you — this is just plumbing, not the interesting math.
 */
function zeros(rows, cols) {
  const m = [];
  for (let i = 0; i < rows; i++) {
    m.push(new Array(cols).fill(0));
  }
  return m;
}

/**
 * Create a matrix of the given shape filled with small random values.
 * Already implemented for you.
 *
 * Why small (-0.5..0.5) and not e.g. -100..100? Large initial weights make
 * the weighted sums huge, which saturates activation functions (their
 * gradient becomes ~0) and training stalls before it starts.
 */
function randomMatrix(rows, cols, scale = 1) {
  const m = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      row.push((Math.random() - 0.5) * scale);
    }
    m.push(row);
  }
  return m;
}

/**
 * TODO (Day 3): Dot product of two equal-length vectors.
 *   dot([1,2,3], [4,5,6]) === 1*4 + 2*5 + 3*6 === 32
 *
 * This is THE core operation discussed in the conversation: multiply
 * corresponding elements, then sum the products into a single number.
 */
function dot(vecA, vecB) {
  // TODO: implement
}

/**
 * TODO (Day 3): Multiply a matrix (array of row-vectors, one row of weights
 * per output neuron) by an input vector, producing an output vector.
 *
 *   matVecMul(weights, input)[i] === dot(weights[i], input)
 *
 * Shape: weights is (outputSize x inputSize), input is (inputSize),
 * result is (outputSize). This is how a whole layer computes its
 * weighted sums in one call, instead of looping neuron by neuron
 * everywhere else in the code.
 */
function matVecMul(weights, input) {
  // TODO: implement using dot() above
}

/**
 * TODO (Day 3): Elementwise vector addition. Used to add the bias vector
 * to the weighted sums: weightedSum = matVecMul(weights, input); then
 * addVectors(weightedSum, biases).
 */
function addVectors(vecA, vecB) {
  // TODO: implement
}

/**
 * TODO (Day 4, backprop): Elementwise vector subtraction (a - b).
 * You'll use this to compute the output error: prediction - target.
 */
function subtractVectors(vecA, vecB) {
  // TODO: implement
}

/**
 * TODO (Day 4, backprop): Elementwise (Hadamard) product of two vectors.
 * Needed when combining an error vector with a derivative vector, e.g.
 * delta = elementwiseMul(error, activationDerivative(weightedSum)).
 */
function elementwiseMul(vecA, vecB) {
  // TODO: implement
}

/**
 * TODO (Day 4, backprop): Outer product of two vectors, producing a matrix.
 *   outer([a,b], [x,y,z]) === [[a*x, a*y, a*z],
 *                              [b*x, b*y, b*z]]
 *
 * Why you need it: the gradient for a whole weight MATRIX (not just one
 * weight) is the outer product of "the error at this layer's outputs"
 * and "the activations feeding into this layer". This single function
 * replaces a nested loop over every individual weight.
 */
function outer(vecA, vecB) {
  // TODO: implement
}

/**
 * TODO (Day 4, backprop): Transpose a matrix (swap rows and columns).
 * Needed to propagate the error backward through a layer: the same
 * weight matrix used forward (outputSize x inputSize) must be applied
 * "in reverse" (inputSize x outputSize) to push the error back.
 */
function transpose(matrix) {
  // TODO: implement
}

const MatrixLib = {
  zeros,
  randomMatrix,
  dot,
  matVecMul,
  addVectors,
  subtractVectors,
  elementwiseMul,
  outer,
  transpose,
};

// Works both in Node (require) and in a plain <script> tag in the browser
// (attaches everything to window.Matrix, no bundler needed).
if (typeof module !== "undefined" && module.exports) {
  module.exports = MatrixLib;
} else {
  window.Matrix = MatrixLib;
}
