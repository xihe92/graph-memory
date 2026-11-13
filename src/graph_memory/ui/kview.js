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

  const KView = { STATUS, AREAS, defaults, clean, active, needsReview, passes, view, litEntities, look, counts, toggle, selectable, brief };
  if (typeof module !== "undefined" && module.exports) module.exports = KView;
  else root.KView = KView;
})(typeof globalThis !== "undefined" ? globalThis : this);
