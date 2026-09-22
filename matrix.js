// matrix.js
// vector = plain array of numbers [x1, x2, ...]
// matrix = array of row-vectors [[row0], [row1], ...]

// already given: empty matrix filled with 0s
function zeros(rows, cols) {
  const m = [];
  for (let i = 0; i < rows; i++) {
    m.push(new Array(cols).fill(0));
  }
  return m;
}

// already given: matrix filled with small random values (-0.5..0.5 * scale)
// small on purpose -> big starting weights saturate the activation and training never gets going
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

// dot([1,2,3],[4,5,6]) = 1*4 + 2*5 + 3*6 = 32
// multiply matching elements, sum everything into one number
function dot(vecA, vecB) {
  let sum = 0;
  for (let i = 0; i < vecA.length; i++) {
    sum += vecA[i] * vecB[i];
  }
  return sum;
}

// weights is a matrix (one row of weights per output neuron), input is a vector
// result[i] = dot(weights[i], input) -> does a whole layer's weighted sums in one call
function matVecMul(weights, input) {
  return weights.map((row) => dot(row, input)); // weights is an array of "rows"
}

// adds bias vector to the weighted sums, elementwise
function addVectors(vecA, vecB) {
  return vecA.map((x, i) => x + vecB[i]);
}

// a - b, elementwise. used for output error = prediction - target
function subtractVectors(vecA, vecB) {
  return vecA.map((x, i) => x - vecB[i]);
}

// elementwise (Hadamard) product, e.g. delta = error * activationDerivative
function elementwiseMul(vecA, vecB) {
  return vecA.map((x, i) => x * vecB[i]);
}

// outer([a,b],[x,y,z]) = [[a*x,a*y,a*z],[b*x,b*y,b*z]]
// gradient for a WHOLE weight matrix = outer(error at this layer, activations feeding in)
// replaces looping over every single weight by hand
function outer(vecA, vecB) {
  let result = [];
  for (let i = 0; i < vecA.length; i++) {
    result[i] = vecB.map((b) => vecA[i] * b);
  }
  return result;
}

// swaps rows and columns. needed to push the error backward through a layer -
// forward pass uses weights as (outputSize x inputSize), backward needs it flipped
function transpose(matrix) {
  let matrixT = [];
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[0].length; j++) {
      if (!matrixT[j]) matrixT[j] = [];
      matrixT[j][i] = matrix[i][j];
    }
  }
  return matrixT;
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

// works in Node (require) and in a plain <script> tag in the browser (window.Matrix)
if (typeof module !== "undefined" && module.exports) {
  module.exports = MatrixLib;
} else {
  window.Matrix = MatrixLib;
}