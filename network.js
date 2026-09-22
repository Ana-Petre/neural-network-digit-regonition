// works both in Node (require) and in a plain <script> tag in the browser,
// where matrix.js and activations.js already attached themselves to
// window.Matrix / window.Activations (see web/index.html)
const Matrix = typeof module !== "undefined" && module.exports ? require("./matrix") : window.Matrix;
const Activations = typeof module !== "undefined" && module.exports ? require("./activations") : window.Activations;

if (typeof module !== "undefined" && module.exports) {
  var { randomMatrix, zeros, matVecMul, addVectors, subtractVectors, elementwiseMul, outer, transpose } = Matrix;
  var { reluVector, reluDerivativeVector, softmax } = Activations;
}

class Network {
  constructor(inputSize, hiddenSize, outputSize) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    // weightsIH: one row per hidden neuron, one column per input pixel
    this.weightsIH = randomMatrix(hiddenSize, inputSize, 1 / Math.sqrt(inputSize));
    this.biasesH = new Array(hiddenSize).fill(0);

    // weightsHO: one row per output neuron, one column per hidden neuron
    this.weightsHO = randomMatrix(outputSize, hiddenSize, 1 / Math.sqrt(hiddenSize));
    this.biasesO = new Array(outputSize).fill(0);
  }

  // push input through both layers, keep everything because backprop needs it
  // 1. hidden weighted sum: weightsIH · input + biasesH
  // 2. relu on that -> hiddenActivation
  // 3. output weighted sum: weightsHO · hiddenActivation + biasesO
  // 4. softmax on that -> probabilities (the actual prediction)
  forward(input) {
    let hiddenWeightedSum = Matrix.addVectors(matVecMul(this.weightsIH, input), this.biasesH);
    let hiddenActivation = Activations.reluVector(hiddenWeightedSum);
    let outputWeightedSum = Matrix.addVectors(matVecMul(this.weightsHO, hiddenActivation), this.biasesO);
    let outputActivation = softmax(outputWeightedSum);

    return {
      input,
      hiddenWeightedSum,
      hiddenActivation,
      outputWeightedSum,
      outputActivation,
    };
  }

  // forward + backprop + gradient descent for one example
  // 1. forward pass
  // 2. outputError = prediction - target (this simplifies cleanly with softmax+cross-entropy)
  // 3. gradients for output layer: outer(outputError, hiddenActivation) for weights, outputError for biases
  // 4. push error back through weightsHO (transpose), mask with relu derivative
  // 5. gradients for hidden layer: same pattern, outer(hiddenError, input)
  // 6. subtract learningRate * gradient from every weight and bias
  // 7. return cross-entropy loss so train() can track progress
  trainOne(input, target, learningRate) {
    const inter = this.forward(input); // getting all the intermediate values
    const outputError = subtractVectors(inter.outputActivation, target);

    const gradWeight = outer(outputError, inter.hiddenActivation);
    const gradBiasesO = outputError;

    const hiddenError = elementwiseMul(
      matVecMul(transpose(this.weightsHO), outputError),
      reluDerivativeVector(inter.hiddenWeightedSum)
    );

    const gradWeightsIH = outer(hiddenError, inter.input);
    const gradBiasesH = hiddenError;

    for (let i = 0; i < this.weightsHO.length; i++) {
      this.biasesO[i] -= learningRate * gradBiasesO[i];
      for (let j = 0; j < this.weightsHO[0].length; j++) {
        this.weightsHO[i][j] -= learningRate * gradWeight[i][j];
      }
    }

    for (let i = 0; i < this.weightsIH.length; i++) {
      this.biasesH[i] -= learningRate * gradBiasesH[i];
      for (let j = 0; j < this.weightsIH[0].length; j++) {
        this.weightsIH[i][j] -= learningRate * gradWeightsIH[i][j];
      }
    }

    return -target.reduce((sum, t, i) => sum + t * Math.log(inter.outputActivation[i] + 1e-12), 0);
  }

  // train over multiple epochs, shuffle each time so order doesn't bias learning
  train(examples, { epochs = 5, learningRate = 0.1, onEpochEnd } = {}) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      const shuffled = [...examples].sort(() => Math.random() - 0.5);
      let totalLoss = 0;
      for (const { input, output } of shuffled) {
        totalLoss += this.trainOne(input, output, learningRate);
      }
      const avgLoss = totalLoss / shuffled.length;
      if (onEpochEnd) onEpochEnd(epoch, avgLoss);
    }
  }

  // run forward, return the digit with highest probability
  predict(input) {
    const { outputActivation } = this.forward(input);
    let best = 0;
    for (let i = 1; i < outputActivation.length; i++) {
      if (outputActivation[i] > outputActivation[best]) best = i;
    }
    return { digit: best, probabilities: outputActivation };
  }

  // % of examples the network gets right
  evaluate(examples) {
    let correct = 0;
    for (const { input, output } of examples) {
      const actualDigit = output.indexOf(1);
      const { digit } = this.predict(input);
      if (digit === actualDigit) correct++;
    }
    return correct / examples.length;
  }

  // serialize to plain object so it can be saved as JSON
  toJSON() {
    return {
      inputSize: this.inputSize,
      hiddenSize: this.hiddenSize,
      outputSize: this.outputSize,
      weightsIH: this.weightsIH,
      biasesH: this.biasesH,
      weightsHO: this.weightsHO,
      biasesO: this.biasesO,
    };
  }

  // rebuild from a saved JSON object
  static fromJSON(obj) {
    const net = new Network(obj.inputSize, obj.hiddenSize, obj.outputSize);
    net.weightsIH = obj.weightsIH;
    net.biasesH = obj.biasesH;
    net.weightsHO = obj.weightsHO;
    net.biasesO = obj.biasesO;
    return net;
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = Network;
} else {
  window.Network = Network;
}