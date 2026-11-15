// What the knowledge page shows for a set of filters. Pure functions, no DOM, so the
// tests run them under Node (tests/test_knowledge_ui.py).
//
// Filters work the same way in every group: picking chips highlights the items that
// match all groups and fades everything else, so the layout never jumps. Within a group
// the picked chips are alternatives. Items no longer true are left out of the graph
// unless "no longer true" is picked; then each is shown next to the item that replaced it.
(function (root) {
  "use strict";

  const STATUS = ["current", "review", "gone"];
  const AREAS = ["survivor", "old", "dormant"];

  function defaults() {
    return { status: [], area: [], scope: "all", withGlobal: true };
  }

  // A filter read back from storage or a URL: anything unknown is dropped.
  function clean(f) {
    const d = defaults();
    if (!f || typeof f !== "object") return d;
    const pick = (xs, allowed) => (Array.isArray(xs) ? xs.filter((x) => allowed.includes(x)) : []);
    return {
      status: pick(f.status, STATUS),
      area: pick(f.area, AREAS),
      scope: typeof f.scope === "string" && f.scope ? f.scope : "all",
      withGlobal: f.withGlobal !== false,
    };
  }

  function active(f) {
    return f.status.length > 0 || f.area.length > 0 || f.scope !== "all";
  }

  function inScope(item, f) {
    if (f.scope === "all") return true;
    if (f.scope === "global") return item.scope === "global";
    return item.scope === f.scope || (f.withGlobal && item.scope === "global");
  }

  // Needs a person's look: still true as far as memory knows, but disputed by the live
  // judge or past its review date.
  function needsReview(item) {
    return !!item.valid && ((item.disputes || 0) > 0 || !!item.overdue);
  }
  function statuses(item) {
    if (!item.valid) return ["gone"];
    return needsReview(item) ? ["current", "review"] : ["current"];
  }

  function passes(item, f) {
    if (f.status.length && !statuses(item).some((s) => f.status.includes(s))) return false;
    if (f.area.length && !f.area.includes(item.area)) return false;
    return inScope(item, f);
  }

  // -> { items: the items in the graph, lit: ids highlighted (null: no filter, all
  //      normal), pairs: [{ old, by }] items no longer true with what replaced them }
  function view(data, f) {
    const showGone = f.status.includes("gone");
    const items = data.items.filter((i) => i.valid || showGone);
    if (!active(f)) return { items, lit: null, pairs: [] };
    const byId = new Map(data.items.map((i) => [i.id, i]));
    const lit = new Set(items.filter((i) => passes(i, f)).map((i) => i.id));
    const pairs = [];
    if (showGone) {
      for (const i of items) {
        if (i.valid || !lit.has(i.id)) continue;
        const by = i.invalidated_by ? byId.get(i.invalidated_by) || null : null;
        if (by) lit.add(by.id); // the replacement is the other half of the story
        pairs.push({ old: i, by });
      }
      pairs.sort((a, b) => String(b.old.invalid_at || "").localeCompare(String(a.old.invalid_at || "")));
    }
    return { items, lit, pairs };
  }

  // Entities are lit when an item about them is.
  function litEntities(items, lit) {
    const out = new Set();
    if (!lit) return out;
    for (const i of items) if (lit.has(i.id)) for (const e of i.entities) out.add(e);
    return out;
  }

  // How a node looks: area by opacity and colour, no longer true in red, faded when a
  // filter or search leaves it out.
  const LOOK = {
    old: { alpha: 1, grey: false },
    survivor: { alpha: 0.55, grey: false },
    dormant: { alpha: 0.45, grey: true },
  };
  function look(item, lit) {
    const base = LOOK[item.area] || LOOK.survivor;
    const out = { alpha: base.alpha, grey: base.grey, gone: !item.valid };
    if (!item.valid) out.alpha = 0.9;
    if (lit && !lit.has(item.id)) out.alpha = 0.06;
    return out;
  }

  // Counts per chip, over the items the other groups leave in.
  function counts(data, f) {
    const c = { current: 0, review: 0, gone: 0, survivor: 0, old: 0, dormant: 0 };
    for (const i of data.items) {
      if (!inScope(i, f)) continue;
      c[i.valid ? "current" : "gone"] += 1;
      if (needsReview(i)) c.review += 1;
      if (i.valid) c[i.area] = (c[i.area] || 0) + 1;
    }
    return c;
  }

  function toggle(list, value) {
    return list.includes(value) ? list.filter((x) => x !== value) : [...list, value];
  }

  // The items "Select all" picks for deleting: the ones the list shows. In the "no
  // longer true" list those are the items no longer true, not what replaced them.
  function selectable(v, f, on, match) {
    if (f.status.includes("gone") && v.pairs.length) {
      return v.pairs
        .filter((p) => !match || match.has(p.old.id) || (p.by && match.has(p.by.id)))
        .map((p) => p.old.id);
    }
    return v.items.filter((i) => !on || on.has(i.id)).map((i) => i.id);
  }

  // A summary's sections with their citations numbered in reading order; a note keeps
  // its number wherever it is cited. state: "ok", "gone" (no longer true) or "deleted".
  function brief(summary, items) {
    const byId = new Map(items.map((i) => [i.id, i]));
    const num = new Map();
    const sections = (summary.sections || []).map((s) => ({
      title: s.title,
      points: (s.points || []).map((p) => ({
        text: p.text,
        cites: (p.notes || []).map((id) => {
          if (!num.has(id)) num.set(id, num.size + 1);
          const it = byId.get(id);
          return { id, n: num.get(id), statement: it ? it.statement : "", state: !it ? "deleted" : it.valid ? "ok" : "gone" };
        }),
      })),
    }));
    return { sections, cited: num.size };
  }

  // Notes opened by click. A note fetched at memory version v is reused while the page's
  // data is no newer than v (versions only grow, so it is as fresh as the page); the
  // least recently opened go first once `max` are kept.
  function noteCache(max) {
    const notes = new Map();
    return {
      get(id, version) {
        const hit = notes.get(id);
        if (!hit || !(hit.version >= version)) return null;
        notes.delete(id);
        notes.set(id, hit);
        return hit;
      },
      put(id, note) {
        notes.delete(id);
        notes.set(id, note);
        while (notes.size > max) notes.delete(notes.keys().next().value);
      },
      size: () => notes.size,
    };
  }

  // Where a rebuilt graph's node starts: where it was last drawn. Coming from 2D into 3D
  // it gets a small random depth, or the 3D layout would have nothing to spread it along.
  function placed(saved, three, rnd) {
    if (!saved || saved.x === undefined) return {};
    const out = { x: saved.x, y: saved.y };
    if (three) out.z = saved.z !== undefined && saved.z !== null ? saved.z : ((rnd || Math.random)() - 0.5) * 30;
    return out;
  }

  // Where a rebuilt graph's nodes start, from the places each view remembers ("mine": this
  // view's, "other": the other view's), and how much layout it then needs:
  //   restore  every node has a place in this view: drawn as it was, no layout at all
  //   grow     some are new: the known ones are pinned (fx, fy, fz) and each new one starts
  //            next to a placed node it links to, then settles in; the page unpins after
  //   settle   this view was never shown: start from the other view's shape, settle gently
  //   layout   nothing is known: lay out from scratch
  function arrange(nodes, links, mine, other, three, rnd) {
    rnd = rnd || Math.random;
    const jitter = () => (rnd() - 0.5) * 30;
    const known = nodes.filter((n) => mine.has(n.id));
    if (known.length && known.length === nodes.length) {
      for (const n of nodes) Object.assign(n, placed(mine.get(n.id), three, rnd));
      return { mode: "restore", pinned: [] };
    }
    if (known.length) {
      const pinned = [];
      for (const n of known) {
        const at = placed(mine.get(n.id), three, rnd);
        Object.assign(n, at, { fx: at.x, fy: at.y }, three ? { fz: at.z } : {});
        pinned.push(n.id);
      }
      const byId = new Map(nodes.map((n) => [n.id, n]));
      for (const l of links) {
        for (const [a, b] of [[l.source, l.target], [l.target, l.source]]) {
          const n = byId.get(a), m = byId.get(b);
          if (n && m && n.x === undefined && m.x !== undefined) {
            n.x = m.x + jitter(); n.y = m.y + jitter();
            if (three) n.z = (m.z || 0) + jitter();
          }
        }
      }
      return { mode: "grow", pinned };
    }
    let fromOther = 0;
    for (const n of nodes) {
      const at = placed(other.get(n.id), three, rnd);
      if (at.x !== undefined) { Object.assign(n, at); fromOther++; }
    }
    return { mode: nodes.length && fromOther >= 0.8 * nodes.length ? "settle" : "layout", pinned: [] };
  }

  // The rows of one group of the side list: the first `cap` (the list comes sorted, the
  // strongest first) unless the group was opened in full, plus the picked row wherever it
  // is, so a note clicked in the graph always shows. A list of thousands of rows would cost
  // the browser a sixth of a second to lay out on every click.
  // The 3D camera that shows the whole graph from where the page first looks: straight down
  // the z axis, upright. The library's own fit (zoomToFit) keeps whatever angle the camera
  // was turned to; this is its calculation for that one direction. `box` is the graph's
  // bounding box ({x: [min, max], y: …, z: …}); `fov` the camera's vertical field of view in
  // degrees; `pad` the margin in pixels, as zoomToFit takes it.
  function homeCamera(box, fov, aspect, height, pad) {
    const half = ((1 - (2 * pad) / height) * fov) / 2 * (Math.PI / 180);  // vertical
    const wide = Math.atan(Math.tan(half) * aspect);                       // horizontal
    const mid = { x: (box.x[0] + box.x[1]) / 2, y: (box.y[0] + box.y[1]) / 2, z: (box.z[0] + box.z[1]) / 2 };
    // how far back the camera must be for every corner of the box to fit on screen
    let back = 0;
    for (const x of box.x) for (const y of box.y) for (const z of box.z) {
      const dx = x - mid.x, dy = y - mid.y, dz = z - mid.z;
      back = Math.max(back, Math.max(Math.abs(dy) / Math.tan(half), Math.abs(dx) / Math.tan(wide)) + dz);
    }
    return { pos: { x: mid.x, y: mid.y, z: mid.z + back }, target: mid };
  }

  function capped(rows, cap, open, keep) {
    if (open || rows.length <= cap) return { rows, hidden: 0 };
    const head = rows.slice(0, cap);
    const extra = keep ? rows.slice(cap).find((r) => r.id === keep) : null;
    return { rows: extra ? head.concat([extra]) : head, hidden: rows.length - cap - (extra ? 1 : 0) };
  }

  const KView = { STATUS, AREAS, defaults, clean, active, needsReview, passes, view, litEntities, look, counts, toggle, selectable, brief, noteCache, placed, arrange, homeCamera, capped };
  if (typeof module !== "undefined" && module.exports) module.exports = KView;
  else root.KView = KView;
})(typeof globalThis !== "undefined" ? globalThis : this);
