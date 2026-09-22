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

  // forward pass: pushes the input through both layers and returns every
  // intermediate value, because trainOne() needs all of them to compute
  // gradients later. steps:
  //   1. hidden layer weighted sum = weightsIH · input + biasesH
  //   2. hidden layer activation = relu(weighted sum)
  //   3. output layer weighted sum = weightsHO · hiddenActivation + biasesO
  //   4. output layer activation = softmax(weighted sum) -> the 10 probabilities
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

  // one training step (forward + backprop + gradient descent) on a single example.
  // steps:
  //   1. forward pass, keep every intermediate value
  //   2. outputError = prediction - target -> how wrong each output neuron was
  //      (softmax + cross-entropy simplifies to exactly this, known identity)
  //   3. output layer gradients: outer(outputError, hiddenActivation) for the
  //      weights, outputError itself for the biases
  //   4. push the error back into the hidden layer: multiply by weightsHO
  //      transposed, then zero out through neurons that relu had already
  //      cut off (reluDerivativeVector)
  //   5. hidden layer gradients: same idea, outer(hiddenError, input)
  //   6. update every weight/bias: subtract learningRate * gradient
  //   7. return the loss (cross-entropy) so train() can log progress
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

  // trains on a full dataset for a number of epochs. calls trainOne() in a
  // loop, reshuffling each epoch so the network doesn't always see examples
  // in the same order (that can bias learning)
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

  // predicts the digit for one input: run forward(), pick the index with
  // the highest probability
  predict(input) {
    const { outputActivation } = this.forward(input);
    let best = 0;
    for (let i = 1; i < outputActivation.length; i++) {
      if (outputActivation[i] > outputActivation[best]) best = i;
    }
    return { digit: best, probabilities: outputActivation };
  }

  // accuracy over a labeled dataset: percentage of examples predicted correctly
  evaluate(examples) {
    let correct = 0;
    for (const { input, output } of examples) {
      const actualDigit = output.indexOf(1);
      const { digit } = this.predict(input);
      if (digit === actualDigit) correct++;
    }
    return correct / examples.length;
  }

  // dumps weights/biases to a plain object, for saving to JSON
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

  // rebuilds a Network from a plain object (see toJSON above)
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