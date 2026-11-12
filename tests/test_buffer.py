"""The session buffer: exchanges wait in it until they graduate, then go."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from graph_memory.anchors import SourceRef
from graph_memory.buffer import Exchange, SessionBuffer, Summary

T0 = datetime(2026, 11, 1, 10, tzinfo=UTC)


def _ex(n: int, **kw: object) -> Exchange:
    return Exchange(
        prompt=f"prompt {n}",
        reply=f"reply {n}",
        ts=T0 + timedelta(minutes=n),
        source=SourceRef(first=f"u{n}", last=f"a{n}"),
        **kw,  # type: ignore[arg-type]
    )


def test_append_read_and_survive_a_bad_line(tmp_path: Path) -> None:
    buf = SessionBuffer(tmp_path)
    session = "claude-code:ab/c d"  # any session key works as a folder name
    a, b = buf.append(session, _ex(1)), buf.append(session, _ex(2))
    assert buf.sessions() == [session]
    assert [x.id for x in buf.exchanges(session)] == [a.id, b.id]
    with buf._path(session).open("a") as fh:
        fh.write("{not json\n\n")
    assert len(SessionBuffer(tmp_path).records(session)) == 2  # skipped, not fatal
    assert buf.records("nobody") == [] and buf.last_activity("nobody") is None
    assert buf.last_activity(session) == b.ts
    assert buf.stats()["exchanges"] == 2 and buf.stats()["bytes"] > 0


def test_summarize_replaces_the_covered_exchanges_in_place(tmp_path: Path) -> None:
    buf = SessionBuffer(tmp_path)
    xs = [buf.append("s", _ex(i)) for i in range(4)]
    v = buf.version
    n = buf.summarize(
        "s", [xs[0].id, xs[1].id], Summary(trigger="compact", exchanges=2, text="did 0 and 1")
    )
    assert n == 2 and buf.version > v
    recs = buf.records("s")
    assert [r.kind for r in recs] == ["summary", "exchange", "exchange"]
    assert [r.id for r in recs[1:]] == [xs[2].id, xs[3].id]  # later exchanges stay
    assert buf.summarize("s", ["gone"], Summary(trigger="idle")) == 0
    assert [s.text for s in buf.summaries("s")] == ["did 0 and 1"]


def test_rewind_forget_update_drop_and_delete(tmp_path: Path) -> None:
    buf = SessionBuffer(tmp_path)
    xs = [buf.append("s", _ex(i)) for i in range(4)]
    dropped = buf.drop_after("s", xs[1].id)  # rewound to after the second
    assert [d.id for d in dropped] == [xs[2].id, xs[3].id]
    assert buf.drop_after("s", xs[1].id) == [] and buf.drop_after("s", "nope") == []
    assert buf.remove("s", xs[0].id) == xs[0] and buf.remove("s", "nope") is None
    longer = xs[1].model_copy(update={"reply": "reply 1, and more after the task finished"})
    buf.update("s", longer)
    assert [x.reply for x in buf.exchanges("s")] == [longer.reply]
    assert buf.drop("s", [xs[1].id]) == 1 and buf.exchanges("s") == []
    assert buf.delete("s") and not buf.delete("s") and buf.sessions() == []


def test_vectors_are_stored_compactly_and_old_files_still_load(tmp_path: Path) -> None:
    import json

    from graph_memory.vectors import JsonVectorIndex

    path = tmp_path / "v.json"
    path.write_text(json.dumps({"dim": 3, "vectors": {"a": [1.0, 0.0, 0.0]}, "meta": {"a": {}}}))
    ix = JsonVectorIndex(path)  # the old format: lists of numbers
    assert ix.get("a") == [1.0, 0.0, 0.0]
    ix.upsert("b", [0.3, 0.4, 0.0], {"kind": "item"})
    ix.save()
    data = json.loads(path.read_text())
    assert data["encoding"] == "f32-base64" and isinstance(data["vectors"]["b"], str)
    again = JsonVectorIndex(path)
    b = again.get("b")
    assert b is not None and abs(b[0] - 0.6) < 1e-6 and abs(b[1] - 0.8) < 1e-6
    assert again.query([0.0, 1.0, 0.0], 1, where={"kind": "item"})[0][0] == "b"


def test_vector_queries_are_exact_and_can_be_limited_to_a_set(tmp_path: Path) -> None:
    import math
    import random

    from graph_memory.vectors import JsonVectorIndex

    rnd = random.Random(7)
    ix = JsonVectorIndex(tmp_path / "v.json")
    vecs = {f"n{n}": [rnd.uniform(-1, 1) for _ in range(8)] for n in range(50)}
    for nid, v in vecs.items():
        ix.upsert(nid, v, {"kind": "item" if int(nid[1:]) % 2 else "entity"})
    q = [rnd.uniform(-1, 1) for _ in range(8)]

    def cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))

    want = sorted(((cos(q, v), nid) for nid, v in vecs.items()), reverse=True)
    got = ix.query(q, 5)
    assert [nid for nid, _ in got] == [nid for _, nid in want[:5]]
    assert all(abs(s - w) < 1e-9 for (_, s), (w, _) in zip(got, want, strict=False))
    items = ix.query(q, 50, where={"kind": "item"}, exclude=frozenset({"n1"}))
    assert len(items) == 24 and all(int(n[1:]) % 2 and n != "n1" for n, _ in items)
    assert [n for n, _ in ix.query(q, 5, among={"n3", "n4"})] == [
        nid for _, nid in want if nid in ("n3", "n4")
    ]
    assert ix.query(q, 5, among=set()) == []
    ix.remove(want[0][1])  # a change is seen by the next query
    assert ix.query(q, 1)[0][0] == want[1][1]
