"use strict";
const assert = require("node:assert/strict");
// Run by tests/test_knowledge_ui.py: node idle.test.js <path to ui/idle.js>
const { quietWhenStill } = require(process.argv[2]);

// A stand-in for the graph: chainable setters, getters with no arguments, and a count of
// how often drawing was paused and resumed.
function fakeGraph() {
  const G = { paused: 0, resumed: 0, drawing: true, calls: [] };
  let onStop = () => G.calls.push("page onEngineStop");
  G.pauseAnimation = () => { G.paused++; G.drawing = false; return G; };
  G.resumeAnimation = () => { G.resumed++; G.drawing = true; return G; };
  G.onEngineStop = (fn) => (fn === undefined ? onStop : ((onStop = fn), G));
  for (const name of ["graphData", "nodeColor", "linkColor", "cameraPosition", "zoomToFit"])
    G[name] = (...args) => (args.length || name === "zoomToFit" ? (G.calls.push(name), G) : "value");
  G.settle = () => G.onEngineStop()();
  return G;
}
function fakeTarget() {
  const handlers = {};
  return {
    addEventListener: (type, fn) => (handlers[type] = fn),
    removeEventListener: (type, fn) => { if (handlers[type] === fn) delete handlers[type]; },
    fire: (type) => handlers[type] && handlers[type](),
    count: () => Object.keys(handlers).length,
  };
}

let t = 0;
const G = fakeGraph();
const page = fakeTarget();
const idle = quietWhenStill(G, { now: () => t, target: page, interval: false, quietMs: 2000 });

// while the layout still moves it keeps drawing, however long nothing else happens
t = 60000; idle.check();
assert.equal(G.drawing, true);

// once settled, it pauses only after the quiet time, and the page's own handler still runs
G.settle();
assert.deepEqual(G.calls, ["page onEngineStop"]);
idle.check();
assert.equal(G.paused, 1);
assert.equal(idle.isPaused(), true);
idle.check();
assert.equal(G.paused, 1); // no repeat pauses

// any input wakes it; it pauses again after the quiet time from that input
t = 70000; page.fire("pointermove");
assert.equal(G.resumed, 1);
t = 71500; idle.check();
assert.equal(G.drawing, true);
t = 72000; idle.check();
assert.equal(G.drawing, false);
for (const type of ["pointerdown", "wheel", "keydown", "touchstart"]) {
  page.fire(type);
  assert.equal(G.drawing, true, type);
  t += 2000; idle.check();
  assert.equal(G.drawing, false, type);
}

// a color change redraws (a click in the side panel), and chaining still works
assert.equal(G.nodeColor(() => "red").linkColor(() => "blue"), G);
assert.equal(G.drawing, true);
t += 2000; idle.check();
assert.equal(G.drawing, false);

// getters don't wake it
assert.equal(G.nodeColor(), "value");
assert.equal(G.cameraPosition(), "value");
assert.equal(G.drawing, false);

// a camera move is drawn to the end before the quiet time starts
G.cameraPosition({ x: 1 }, { x: 0 }, 5000);
t += 6000; idle.check();
assert.equal(G.drawing, true);
t += 1000; idle.check();
assert.equal(G.drawing, false);
G.zoomToFit(600, 60);
t += 2599; idle.check();
assert.equal(G.drawing, true);
t += 1; idle.check();
assert.equal(G.drawing, false);

// new data (a live update): drawing resumes and stays on until the layout settles again
G.graphData({ nodes: [], links: [] });
t += 60000; idle.check();
assert.equal(G.drawing, true);
G.settle(); idle.check();
assert.equal(G.drawing, false);

// the 2D graph has centerAt and zoom instead of cameraPosition: those wake it, the
// missing one is left alone; and a page that swaps graphs stops watching the old one
{
  let now2 = 0;
  const G2 = fakeGraph();
  delete G2.cameraPosition;
  G2.centerAt = (...args) => (args.length ? (G2.calls.push("centerAt"), G2) : "value");
  G2.zoom = (...args) => (args.length ? (G2.calls.push("zoom"), G2) : "value");
  const page2 = fakeTarget();
  const idle2 = quietWhenStill(G2, { now: () => now2, target: page2, interval: false, quietMs: 2000 });
  assert.equal(G2.cameraPosition, undefined);
  G2.settle(); now2 = 5000; idle2.check();
  assert.equal(G2.drawing, false);
  G2.centerAt(1, 2, 800);  // a camera move of 800 ms is drawn to the end
  assert.equal(G2.drawing, true);
  now2 = 7500; idle2.check();
  assert.equal(G2.drawing, true);
  now2 = 7800; idle2.check();
  assert.equal(G2.drawing, false);
  assert.ok(page2.count() > 0);
  idle2.stop();
  assert.equal(page2.count(), 0);
}
console.log("ok");
