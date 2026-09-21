/**
 * network.js
 * -----------------------------------------------------------------------
 * A simple feedforward neural network, from scratch:
 *
 *   input (784) --[fully connected + ReLU]--> hidden (H) --[fully connected + softmax]--> output (10)
 *
 * This file is the heart of the project (Day 3 + Day 4 of the plan).
 * The plumbing (constructor, save/load) is done for you; forward() and
 * backward() are TODOs — that's the actual learning.
 *
 * Works unmodified in both Node (training, via train.js) and the browser
 * (prediction, via web/predict.js) as long as the bundler/script setup
 * exposes `Network` — see web/index.html for how it's loaded there.
 * -----------------------------------------------------------------------
 */

// Works both in Node (require) and in a plain <script> tag in the browser,
// where matrix.js and activations.js have already attached themselves to
// window.Matrix / window.Activations (just load those two <script> tags
// before this one — see web/index.html).
const Matrix = typeof module !== "undefined" && module.exports ? require("./matrix") : window.Matrix;
const Activations = typeof module !== "undefined" && module.exports ? require("./activations") : window.Activations;

const { randomMatrix, zeros, matVecMul, addVectors, subtractVectors, elementwiseMul, outer, transpose } = Matrix;
const { reluVector, reluDerivativeVector, softmax } = Activations;

class Network {
  /**
   * @param {number} inputSize  - e.g. 784 (28x28 pixels)
   * @param {number} hiddenSize - e.g. 32 neurons in the hidden layer
   * @param {number} outputSize - e.g. 10 (digits 0-9)
   */
  constructor(inputSize, hiddenSize, outputSize) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    // weightsIH: one row per hidden neuron, one column per input pixel.
    // Small random init (see matrix.js randomMatrix comment for why).
    this.weightsIH = randomMatrix(hiddenSize, inputSize, 1 / Math.sqrt(inputSize));
    this.biasesH = new Array(hiddenSize).fill(0);

    // weightsHO: one row per output neuron, one column per hidden neuron.
    this.weightsHO = randomMatrix(outputSize, hiddenSize, 1 / Math.sqrt(hiddenSize));
    this.biasesO = new Array(outputSize).fill(0);
  }

  /**
   * TODO (Day 3): Forward pass.
   *
   * Given a 784-length input vector, compute and return an object with
   * everything backward() will need later:
   *
   *   {
   *     input,                 // the input vector itself (needed for the weight gradient of layer 1)
   *     hiddenWeightedSum,     // weightsIH · input + biasesH   (before activation)
   *     hiddenActivation,      // relu(hiddenWeightedSum)
   *     outputWeightedSum,     // weightsHO · hiddenActivation + biasesO
   *     outputActivation,      // softmax(outputWeightedSum) — the final prediction (10 probabilities)
   *   }
   *
   * Use matVecMul + addVectors from matrix.js, and reluVector / softmax
   * from activations.js. Don't skip returning the intermediate values —
   * backward() needs every one of them.
   */
  forward(input) {
    // TODO: implement
  }

  /**
   * TODO (Day 4): Backpropagation + gradient descent for ONE training example.
   *
   * @param {number[]} input  - 784-length input vector
   * @param {number[]} target - 10-length one-hot vector, e.g. digit "3" -> [0,0,0,1,0,0,0,0,0,0]
   * @param {number} learningRate - e.g. 0.1
   * @returns {number} the loss for this example (for logging progress) — see note below
   *
   * Steps:
   *  1. Run forward(input) to get all intermediate values.
   *  2. Output layer error:
   *       With softmax output + cross-entropy loss, the gradient of the
   *       loss with respect to the output layer's WEIGHTED SUM (not the
   *       activation!) simplifies beautifully to just:
   *           outputError = outputActivation - target
   *       (This is a well-known identity — you can take it as given here,
   *       but if you want to see WHY it simplifies this way, that's a
   *       great thing to look up and understand for your README / for
   *       explaining this project at an interview.)
   *  3. Gradient for weightsHO: outer(outputError, hiddenActivation)
   *     Gradient for biasesO:   outputError
   *  4. Propagate error back into the hidden layer:
   *       hiddenError = transpose(weightsHO) · outputError,
   *       then multiply elementwise by reluDerivativeVector(hiddenWeightedSum)
   *     (`transpose` is already imported at the top of this file.)
   *  5. Gradient for weightsIH: outer(hiddenError, input)
   *     Gradient for biasesH:   hiddenError
   *  6. Update every weight/bias by subtracting (learningRate * gradient).
   *     Do this in place on this.weightsIH, this.biasesH, this.weightsHO, this.biasesO.
   *  7. Return the loss, e.g. cross-entropy:
   *       -sum(target[i] * Math.log(outputActivation[i] + 1e-12))
   *     (the `+ 1e-12` avoids log(0))
   */
  trainOne(input, target, learningRate) {
    // TODO: implement
  }

  /**
   * Train on a full dataset for a number of epochs. Already implemented —
   * it just calls trainOne() in a loop and logs progress. Shuffle each
   * epoch so the network doesn't see examples in the same fixed order
   * every time (that can bias learning).
   */
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

  /**
   * Predict the digit (0-9) for a single input vector.
   * Already implemented, built on top of your forward().
   */
  predict(input) {
    const { outputActivation } = this.forward(input);
    let best = 0;
    for (let i = 1; i < outputActivation.length; i++) {
      if (outputActivation[i] > outputActivation[best]) best = i;
    }
    return { digit: best, probabilities: outputActivation };
  }

  /**
   * Accuracy over a labeled dataset (each example has .input and .output
   * as a one-hot vector). Already implemented.
   */
  evaluate(examples) {
    let correct = 0;
    for (const { input, output } of examples) {
      const actualDigit = output.indexOf(1);
      const { digit } = this.predict(input);
      if (digit === actualDigit) correct++;
    }
    return correct / examples.length;
  }

  /** Serialize weights/biases to a plain object, for saving to JSON. */
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

  /** Rebuild a Network instance from a plain object (see toJSON above). */
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
