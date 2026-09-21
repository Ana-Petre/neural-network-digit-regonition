/**
 * train.js
 * -----------------------------------------------------------------------
 * Loads MNIST data, trains the network, prints progress + final accuracy,
 * and saves the trained weights to web/weights.json so the browser demo
 * can load them without retraining.
 *
 * Run with:  node train.js
 *
 * This file is already complete — you don't need to edit it. It only
 * becomes useful once forward() and trainOne() in network.js are
 * implemented; until then it will run but predict garbage (~10% accuracy,
 * i.e. random guessing among 10 digits).
 * -----------------------------------------------------------------------
 */

const fs = require("fs");
const path = require("path");
const mnist = require("mnist");
const Network = require("./network");

// Start small on purpose: 2000 training / 500 test examples train in
// seconds and are enough to see whether your math is working at all.
// Once forward/backward are correct, bump these up (e.g. 10000 / 2000)
// for better final accuracy — see the README for guidance.
const TRAIN_SIZE = 2000;
const TEST_SIZE = 500;
const HIDDEN_SIZE = 32;
const EPOCHS = 15;
const LEARNING_RATE = 0.1;

console.log(`Loading MNIST: ${TRAIN_SIZE} train / ${TEST_SIZE} test examples...`);
const set = mnist.set(TRAIN_SIZE, TEST_SIZE);
const trainingData = set.training.map((ex) => ({ input: ex.input, output: ex.output }));
const testData = set.test.map((ex) => ({ input: ex.input, output: ex.output }));

console.log(`Creating network: 784 -> ${HIDDEN_SIZE} -> 10`);
const net = new Network(784, HIDDEN_SIZE, 10);

console.log(`Training for ${EPOCHS} epochs (learning rate ${LEARNING_RATE})...`);
const start = Date.now();
const trainingLog = [];

net.train(trainingData, {
  epochs: EPOCHS,
  learningRate: LEARNING_RATE,
  onEpochEnd: (epoch, avgLoss) => {
    const acc = net.evaluate(testData);
    console.log(
      `  epoch ${String(epoch + 1).padStart(2)}/${EPOCHS}  avg loss: ${avgLoss.toFixed(4)}  test accuracy: ${(acc * 100).toFixed(1)}%`
    );
    trainingLog.push({ epoch: epoch + 1, avgLoss, testAccuracy: acc });
  },
});

const elapsed = ((Date.now() - start) / 1000).toFixed(1);
const finalAccuracy = net.evaluate(testData);
console.log(`\nDone in ${elapsed}s. Final test accuracy: ${(finalAccuracy * 100).toFixed(1)}%`);

const outPath = path.join(__dirname, "web", "weights.json");
fs.writeFileSync(outPath, JSON.stringify(net.toJSON()));
console.log(`Weights saved to ${outPath}`);

const logPath = path.join(__dirname, "web", "training-log.json");
fs.writeFileSync(logPath, JSON.stringify(trainingLog));
console.log(`Training log saved to ${logPath}`);

console.log(`Open web/index.html in a browser (via a local server, not file://) to try the live demo.`);
