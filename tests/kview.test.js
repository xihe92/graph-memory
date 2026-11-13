"use strict";
const assert = require("node:assert/strict");
// Run by tests/test_knowledge_ui.py: node kview.test.js <path to ui/kview.js>
const K = require(process.argv[2]);
const it = (id, o) => ({ id, valid: true, area: "survivor", scope: "global", entities: [], ...o });
const data = { items: [
  it("lock", { valid: false, invalidated_by: "lua", area: "old", scope: "repo:/a", invalid_at: "2026-11-01", entities: ["redis"] }),
  it("lua", { area: "old", scope: "repo:/a", entities: ["redis"] }),
  it("todo", { valid: false, scope: "repo:/a", invalid_at: "2026-11-03", invalid_reason: "by hand" }),
  it("terse", { scope: "global", disputes: 1 }),
  it("old_b", { area: "old", scope: "repo:/b", entities: ["pg"] }),
  it("sleepy", { area: "dormant", scope: "repo:/b", overdue: true }),
] };
const ids = (xs) => xs.map((x) => x.id).sort();
const lit = (v) => (v.lit ? [...v.lit].sort() : null);

// no filter: items no longer true are out of the graph, nothing is highlighted
let v = K.view(data, K.defaults());
assert.deepEqual(ids(v.items), ["lua", "old_b", "sleepy", "terse"]);
assert.equal(v.lit, null);
assert.equal(K.active(K.defaults()), false);

// "no longer true": they come in, lit with what replaced them; the rest fade
v = K.view(data, { ...K.defaults(), status: ["gone"] });
assert.equal(v.items.length, 6);
assert.deepEqual(lit(v), ["lock", "lua", "todo"]);
assert.deepEqual(v.pairs.map((p) => [p.old.id, p.by && p.by.id]), [["todo", null], ["lock", "lua"]]);

// area: old only, current only
v = K.view(data, { ...K.defaults(), area: ["old"] });
assert.deepEqual(lit(v), ["lua", "old_b"]);
v = K.view(data, { ...K.defaults(), status: ["current"] });
assert.deepEqual(lit(v), ["lua", "old_b", "sleepy", "terse"]);
assert.ok(!v.items.some((i) => !i.valid));

// scope: a project with or without global items, global alone
v = K.view(data, { ...K.defaults(), scope: "repo:/b" });
assert.deepEqual(lit(v), ["old_b", "sleepy", "terse"]);
v = K.view(data, { ...K.defaults(), scope: "repo:/b", withGlobal: false });
assert.deepEqual(lit(v), ["old_b", "sleepy"]);
v = K.view(data, { ...K.defaults(), scope: "global" });
assert.deepEqual(lit(v), ["terse"]);

// groups combine: no longer true AND in project b -> nothing; old AND project a
v = K.view(data, { ...K.defaults(), status: ["gone"], scope: "repo:/b", withGlobal: false });
assert.deepEqual(lit(v), []);
assert.deepEqual(v.pairs, []);
v = K.view(data, { status: ["current", "gone"], area: ["old"], scope: "repo:/a", withGlobal: true });
assert.deepEqual(lit(v), ["lock", "lua"]);

// entities light up with their items
assert.deepEqual([...K.litEntities(data.items, new Set(["old_b"]))], ["pg"]);
assert.equal(K.litEntities(data.items, null).size, 0);

// looks: old solid, survivor see-through, dormant grey, no longer true red-ready, faded when out
const byId = Object.fromEntries(data.items.map((i) => [i.id, i]));
assert.equal(K.look(byId.lua, null).alpha, 1);
assert.ok(K.look(byId.terse, null).alpha < 1 && !K.look(byId.terse, null).grey);
assert.ok(K.look(byId.sleepy, null).grey);
assert.ok(K.look(byId.lock, null).gone);
assert.ok(K.look(byId.lua, new Set(["terse"])).alpha < 0.1);

// counts follow the scope, not the chips of their own group
assert.deepEqual(K.counts(data, K.defaults()), { current: 4, review: 2, gone: 2, survivor: 1, old: 2, dormant: 1 });
assert.deepEqual(K.counts(data, { ...K.defaults(), scope: "repo:/b", withGlobal: false }),
  { current: 2, review: 1, gone: 0, survivor: 0, old: 1, dormant: 1 });

// needs review: disputed or overdue, and still true; a retracted item doesn't
v = K.view(data, { ...K.defaults(), status: ["review"] });
assert.deepEqual(lit(v), ["sleepy", "terse"]);
assert.equal(K.needsReview({ valid: false, disputes: 3 }), false);
assert.equal(K.needsReview({ valid: true, disputes: 1 }), true);
v = K.view(data, { ...K.defaults(), status: ["review", "gone"] });
assert.deepEqual(lit(v), ["lock", "lua", "sleepy", "terse", "todo"]);

// "Select all" picks what the list shows: in the no-longer-true list, the items no
// longer true and not their replacements; a search narrows it
v = K.view(data, { ...K.defaults(), status: ["gone"] });
const gone = K.selectable(v, { status: ["gone"] }, v.on, null);
assert.ok(gone.length === v.pairs.length && gone.every((id) => !byId[id].valid));
assert.deepEqual(K.selectable(v, { status: ["gone"] }, v.on, new Set([gone[0]])), [gone[0]]);
v = K.view(data, K.defaults());
assert.deepEqual(K.selectable(v, K.defaults(), new Set(["terse"]), null), ["terse"]);
assert.equal(K.selectable(v, K.defaults(), null, null).length, v.items.length);

// a summary: citations are numbered in reading order, one number per note, and a note
// that is no longer true or was deleted since is marked
const b = K.brief({ sections: [
  { title: "Decisions and why", points: [{ text: "Lua replaced the lock.", notes: ["lua", "lock"] }] },
  { title: "Open items", points: [{ text: "Be terse.", notes: ["terse", "lua", "zapped"] }] },
] }, data.items);
assert.equal(b.cited, 4);
assert.deepEqual(b.sections[0].points[0].cites.map((c) => [c.n, c.state]), [[1, "ok"], [2, "gone"]]);
assert.deepEqual(b.sections[1].points[0].cites.map((c) => [c.id, c.n, c.state]),
  [["terse", 3, "ok"], ["lua", 1, "ok"], ["zapped", 4, "deleted"]]);
assert.deepEqual(K.brief({}, []), { sections: [], cited: 0 });

// stored filters are cleaned; toggling adds and removes
assert.deepEqual(K.clean({ status: ["gone", "bogus"], area: "old", scope: "" }),
  { status: ["gone"], area: [], scope: "all", withGlobal: true });
assert.deepEqual(K.clean(null), K.defaults());
assert.deepEqual(K.toggle(["old"], "old"), []);
assert.deepEqual(K.toggle([], "old"), ["old"]);
console.log("ok");
