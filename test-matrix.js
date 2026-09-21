/**
 * test-matrix.js
 * -----------------------------------------------------------------------
 * Sanity checks for matrix.js, in the same spirit as the checker scripts
 * from your OS assignments (make check). Run with:
 *
 *   node test-matrix.js
 *
 * Implement matrix.js one function at a time and re-run this after each
 * one — don't try to write all of matrix.js before testing anything.
 * -----------------------------------------------------------------------
 */

const {
  dot,
  matVecMul,
  addVectors,
  subtractVectors,
  elementwiseMul,
  outer,
  transpose,
} = require("./matrix");

let passed = 0;
let failed = 0;

function approxEqual(a, b, eps = 1e-9) {
  return Math.abs(a - b) < eps;
}

function arraysEqual(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  return a.every((v, i) => approxEqual(v, b[i]));
}

function matrixEqual(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  return a.every((row, i) => arraysEqual(row, b[i]));
}

function test(name, actual, expected, equalFn = (a, b) => approxEqual(a, b)) {
  const ok = equalFn(actual, expected);
  if (ok) {
    passed++;
    console.log(`  passed  ${name}`);
  } else {
    failed++;
    console.log(`  FAILED  ${name}`);
    console.log(`          expected: ${JSON.stringify(expected)}`);
    console.log(`          actual:   ${JSON.stringify(actual)}`);
  }
}

console.log("\n=== dot() ===");
test("dot basic", dot([1, 2, 3], [4, 5, 6]), 32);
test("dot with zeros", dot([0, 0, 0], [1, 2, 3]), 0);
test("dot single element", dot([7], [3]), 21);

console.log("\n=== matVecMul() ===");
test(
  "matVecMul 2x3 by 3",
  matVecMul(
    [
      [1, 2, 3],
      [4, 5, 6],
    ],
    [1, 1, 1]
  ),
  [6, 15],
  arraysEqual
);

console.log("\n=== addVectors() ===");
test("addVectors basic", addVectors([1, 2, 3], [10, 20, 30]), [11, 22, 33], arraysEqual);

console.log("\n=== subtractVectors() ===");
test("subtractVectors basic", subtractVectors([5, 5, 5], [1, 2, 3]), [4, 3, 2], arraysEqual);

console.log("\n=== elementwiseMul() ===");
test("elementwiseMul basic", elementwiseMul([1, 2, 3], [4, 5, 6]), [4, 10, 18], arraysEqual);

console.log("\n=== outer() ===");
test(
  "outer basic",
  outer([1, 2], [10, 20, 30]),
  [
    [10, 20, 30],
    [20, 40, 60],
  ],
  matrixEqual
);

console.log("\n=== transpose() ===");
test(
  "transpose 2x3",
  transpose([
    [1, 2, 3],
    [4, 5, 6],
  ]),
  [
    [1, 4],
    [2, 5],
    [3, 6],
  ],
  matrixEqual
);

console.log(`\n${passed} passed, ${failed} failed\n`);
if (failed > 0) process.exit(1);
