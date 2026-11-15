// Lays out the Knowledge graph off the page's main thread, in a Web Worker, so the page
// stays responsive while a large graph settles. It runs the same force layout the graph
// libraries run on the page (d3-force-3d, in 2 or 3 dimensions) with the page's settings:
// links pull linked notes together, notes push each other apart, the whole is centred.
// The page sends the nodes (with starting positions where it has them) and the links,
// and gets back one position per node; it then draws them without a layout of its own.
// `run` needs no DOM or worker, so the tests run it under Node (tests/test_knowledge_ui.py).
(function (root) {
  "use strict";

  // job: { dims: 2 | 3, ids, x, y, z (NaN where unknown), src, dst (link ends as node
  //        indexes), about (1: a note-to-entity link), weight, ticks, friction }
  // -> Float32Array of x, y, z per node (z is 0 in 2D)
  function run(d3, job) {
    const n = job.ids.length;
    const nodes = new Array(n);
    for (let i = 0; i < n; i++) {
      const node = { index: i };
      if (!Number.isNaN(job.x[i])) {
        node.x = job.x[i];
        node.y = job.y[i];
        if (job.dims === 3) node.z = Number.isNaN(job.z[i]) ? 0 : job.z[i];
      }
      nodes[i] = node;
    }
    const links = new Array(job.src.length);
    for (let i = 0; i < links.length; i++) {
      links[i] = { source: job.src[i], target: job.dst[i], about: job.about[i] === 1, weight: job.weight[i] };
    }
    const sim = d3
      .forceSimulation(nodes, job.dims)
      .force(
        "link",
        d3
          .forceLink(links)
          .distance((l) => (l.about ? 26 : 40 - l.weight * 20))
          .strength((l) => (l.about ? 0.25 : 0.2 + l.weight * 0.6)),
      )
      .force("charge", d3.forceManyBody().strength(-40))
      .force("center", d3.forceCenter())
      .velocityDecay(job.friction || 0.4)
      .stop();
    for (let t = 0; t < job.ticks; t++) sim.tick();
    const out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      out[3 * i] = nodes[i].x;
      out[3 * i + 1] = nodes[i].y;
      out[3 * i + 2] = job.dims === 3 ? nodes[i].z : 0;
    }
    return out;
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { run };
  } else if (typeof importScripts === "function") {
    // in the worker: the first message names the layout library to load
    root.onmessage = (e) => {
      const m = e.data;
      if (m.deps && !root.d3) importScripts(m.deps);
      if (m.job) {
        const pos = run(root.d3, m.job);
        root.postMessage({ seq: m.seq, pos }, [pos.buffer]);
      }
    };
  }
})(typeof self !== "undefined" ? self : globalThis);
