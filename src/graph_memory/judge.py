"""Judging notes live: right after a reply, a model labels each knowledge note recall had
shown for that prompt as useful, neutral, misleading or contradicted, using the reply as
evidence of what was needed. Runs for a sample of turns (`judge_rate`), and always when a
note shown could have gone stale or was never checked (see knowledge.at_risk). The
labels are kept as `knowledge_judge` events; a contradicted note is disputed in the store
(see KnowledgeStore.dispute). Nothing has to be kept to judge later: the hook already
holds the prompt, the reply and the notes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from .knowledge import Item

LABELS = ("useful", "neutral", "misleading", "contradicted")
REPLY_CHARS = 6000

JUDGE_PROMPT = """You judge memory notes shown to an AI assistant before it answered a
prompt. For each note (A1, A2, …) answer two things, using the actual reply as evidence:
- "label", how the note served this prompt: "useful" (the reply uses or depends on it,
  or it clearly prevents a mistake), "neutral" (not needed for this prompt) or
  "misleading" (it pointed this prompt the wrong way)
- "wrong": true when the reply says the opposite of the note, or shows from code, docs or
  tool output that the note is false or out of date; then "why": what in the reply shows
  it. But when the reply itself changes the thing (raises a limit, replaces a tool),
  the note was right until then: not wrong. false when the reply gives no such evidence.

Reply with ONLY a JSON object, one entry per note:
{{"A<n>": {{"label": "<useful|neutral|misleading>", "wrong": <true|false>,
           "why": "<when wrong: what in the reply shows it>"}}}}

<prompt>
{prompt}
</prompt>

<reply>
{reply}
</reply>

<notes>
{notes}
</notes>
"""


def build_judge_prompt(prompt: str, reply: str, items: Sequence[Item]) -> str:
    notes = "\n".join(
        f"A{n + 1} [{i.type.value}, {i.checked.date().isoformat()}] {i.statement}"
        + (f" (why: {i.why})" if i.why else "")
        for n, i in enumerate(items)
    )
    if len(reply) > REPLY_CHARS:
        reply = reply[: REPLY_CHARS // 2] + "\n[…]\n" + reply[-REPLY_CHARS // 2 :]
    return JUDGE_PROMPT.format(prompt=prompt[:3000], reply=reply, notes=notes)


def verdicts(text: str, items: Sequence[Item]) -> dict[str, tuple[str, str]]:
    """Item id -> (label, why) for the notes the model labelled; unknown labels are
    dropped. A verdict is a label, or an object with "label", "wrong" and "why"; a note
    found wrong is "contradicted", whatever its label."""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in the judge's answer: {text[:200]!r}")
    raw = json.loads(s[start : end + 1])
    out: dict[str, tuple[str, str]] = {}
    for key, val in raw.items() if isinstance(raw, dict) else []:
        m = re.fullmatch(r"A?(\d+)", str(key).strip(), re.I)
        if isinstance(val, dict):
            lab, why = str(val.get("label", "")), " ".join(str(val.get("why") or "").split())
            if val.get("wrong") is True or str(val.get("wrong")).lower() == "true":
                lab = "contradicted"
        else:
            lab, why = str(val), ""
        lab = lab.strip().lower()
        if m and 1 <= int(m.group(1)) <= len(items) and lab in LABELS:
            out[items[int(m.group(1)) - 1].item_id] = (lab, why[:300])
    return out


def parse_judgement(text: str, items: Sequence[Item]) -> dict[str, str]:
    """Item id -> label for the notes the model labelled; unknown labels are dropped."""
    return {i: lab for i, (lab, _) in verdicts(text, items).items()}
