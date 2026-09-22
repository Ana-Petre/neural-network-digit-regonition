// Loads MNIST, trains the network, saves weights + loss log to web/.
// Run with: node train.js

const fs = require("fs");
const path = require("path");
const mnist = require("mnist");
const Network = require("./network");

const TRAIN_SIZE = 8000
const TEST_SIZE = 1500;
const HIDDEN_SIZE = 128;
const EPOCHS = 30;
const LEARNING_RATE = 0.05;

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
