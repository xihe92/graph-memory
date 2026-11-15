"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
// Run by tests/test_knowledge_ui.py: node layout.test.js <ui/layout.js> <ui/vendor/d3-force-3d.bundle.js>
const { run } = require(process.argv[2]);
// load the vendored layout library as a worker would: its UMD builds attach `d3` to the global
const ctx = { setTimeout, clearTimeout, setInterval, clearInterval, performance };  // what a worker has
vm.runInNewContext(fs.readFileSync(process.argv[3], "utf8"), ctx);
const d3 = ctx.d3;
assert.equal(typeof d3.forceSimulation, "function");

// two clusters joined by one weak link, laid out from nothing
const ids = ["a1", "a2", "a3", "b1", "b2", "b3"];
const job = (dims, x = new Float32Array(6).fill(NaN)) => ({
  dims, ids, x, y: new Float32Array(6).fill(NaN), z: new Float32Array(6).fill(NaN),
  src: Int32Array.from([0, 1, 0, 3, 4, 3, 2]), dst: Int32Array.from([1, 2, 2, 4, 5, 5, 3]),
  about: Uint8Array.from([0, 0, 0, 0, 0, 0, 0]), weight: Float32Array.from([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3]),
  ticks: 200,
});
for (const dims of [2, 3]) {
  const pos = run(d3, job(dims));
  assert.equal(pos.length, 18);
  assert.ok(pos.every(Number.isFinite), "every node placed");
  if (dims === 2) assert.ok([2, 5, 8, 11, 14, 17].every((i) => pos[i] === 0), "2D: no depth");
  const at = (i) => [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
  const dist = (i, j) => Math.hypot(...at(i).map((v, k) => v - at(j)[k]));
  // strongly linked notes end closer than notes in the other cluster
  assert.ok(dist(0, 1) < dist(0, 4), `${dims}D: ${dist(0, 1)} vs ${dist(0, 4)}`);
}
// known starting positions are where it starts from: with few, damped steps a laid-out
// graph barely moves (positions centred on 0, as a layout leaves them: the centring force
// moves the whole graph so its middle is at 0)
const x = Float32Array.from([-60, -50, -40, 40, 50, 60]);
const settled = run(d3, { ...job(2, x), y: new Float32Array(6), ticks: 1, friction: 0.85 });
assert.ok(Math.abs(settled[3 * 3] - 40) < 5, `a placed node stays near its place: ${settled[9]}`);
console.log("ok");
