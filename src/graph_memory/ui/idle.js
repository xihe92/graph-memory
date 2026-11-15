// Stops a graph (3D or 2D) from redrawing a picture that isn't changing. The graph library draws
// on every screen refresh, so an open page keeps the GPU busy (and a laptop's fans on)
// even when nothing moves. Drawing pauses once the layout has settled and the page has
// had no input for a moment, and resumes on any input or any change to the graph.
// No DOM needed beyond an event target, so the tests run it under Node
// (tests/test_knowledge_ui.py).
(function (root) {
  "use strict";

  const QUIET_MS = 2000; // how long after the last input or change drawing goes on
  const INPUT = ["pointermove", "pointerdown", "wheel", "keydown", "touchstart"];
  // graph methods that change the picture; the number is the argument giving the length
  // of the camera move they start, which must be drawn to the end. cameraPosition is the
  // 3D graph's; centerAt and zoom the 2D graph's. A graph without one is left alone.
  const CHANGES = {
    graphData: -1, nodeColor: -1, linkColor: -1, cameraPosition: 2, zoomToFit: 0, centerAt: 2, zoom: 1,
    width: -1, height: -1, // a resize clears the canvas: draw it again
  };

  function quietWhenStill(G, opts) {
    opts = opts || {};
    const now = opts.now || (() => Date.now());
    const quietMs = opts.quietMs != null ? opts.quietMs : QUIET_MS;
    let settling = true; // the layout is still moving: never pause then
    let paused = false;
    let until = now() + quietMs;

    function wake(ms) {
      until = Math.max(until, now() + quietMs + (ms || 0));
      if (paused) {
        paused = false;
        G.resumeAnimation();
      }
    }

    function check() {
      if (!paused && !settling && now() >= until) {
        paused = true;
        G.pauseAnimation();
      }
    }

    const onStop = G.onEngineStop();
    G.onEngineStop(function () {
      settling = false;
      if (onStop) return onStop.apply(this, arguments);
    });

    for (const name of Object.keys(CHANGES)) {
      const original = G[name];
      if (typeof original !== "function") continue;
      G[name] = function () {
        // called with no arguments these are getters, except zoomToFit, which acts
        if (arguments.length || name === "zoomToFit") {
          if (name === "graphData") settling = true; // new data: the layout moves again
          const at = CHANGES[name];
          wake(at >= 0 ? Number(arguments[at]) || 0 : 0);
        }
        return original.apply(G, arguments);
      };
    }

    const target = opts.target || (typeof document !== "undefined" ? document : null);
    const onInput = () => wake(0);
    if (target) for (const type of INPUT) target.addEventListener(type, onInput, { passive: true });
    const timer = opts.interval !== false ? setInterval(check, 500) : null;

    // a page that swaps graphs (3D <-> 2D) stops watching the old one
    function stop() {
      if (timer) clearInterval(timer);
      if (target) for (const type of INPUT) target.removeEventListener(type, onInput);
    }

    return { wake, check, stop, isPaused: () => paused };
  }

  const GraphIdle = { QUIET_MS, quietWhenStill };
  if (typeof module !== "undefined" && module.exports) module.exports = GraphIdle;
  else root.GraphIdle = GraphIdle;
})(typeof globalThis !== "undefined" ? globalThis : this);
