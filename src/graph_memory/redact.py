"""Secrets out of memory: text is redacted as it reaches the daemon, before anything
stores it (the session buffer, the event log), embeds it or sends it to a model. Model
answers are redacted too, so a note can never carry a secret it was shown.

Only shapes that are clearly secrets are matched: provider keys with a known prefix,
private key blocks, JWTs, passwords inside URLs and connection strings, and a value
assigned to a name such as `password` or `api_key`. A password written as plain prose
("the password is hunter2") is not caught. A match becomes `[redacted: <kind>]`, so a
note can still say a token was involved; the name a value was assigned to stays.

`redact_patterns` in config.toml adds patterns of your own (Python regular expressions;
the whole match is replaced). `redact = false` turns redaction off.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

from .models import Ask


def _mark(kind: str) -> str:
    return f"[redacted: {kind}]"


_PLACEHOLDERS = frozenset(
    {"password", "passwd", "pass", "pwd", "secret", "p", "xxx", "changeme", "redacted"}
)


def _placeholder(value: str) -> bool:
    """A documented example rather than a secret: `<password>`, `{secret}`, `...`,
    `user:password`. Only these are spared; anything else that matched is redacted."""
    if any(c in value for c in "<>{}$") or not any(c.isalnum() for c in value):
        return True
    word = value.rsplit(":", 1)[-1].strip("*").lower()
    return word in _PLACEHOLDERS or set(word) <= {"x", "*"}


@dataclass(frozen=True)
class _Rule:
    kind: str
    pattern: re.Pattern[str]
    # the group that holds the secret; the text around it stays (a name, `://`, `@`)
    group: int = 0
    spare: Callable[[str], bool] | None = None  # matches it lets through


# A value assigned to one of these names is a secret when it looks like one: see _secretish.
_NAMES = (
    r"(?:password|passwd|pwd|passphrase|secret|client[_-]?secret|api[_-]?key|apikey"
    r"|access[_-]?key|secret[_-]?key|private[_-]?key|auth[_-]?token|access[_-]?token"
    r"|refresh[_-]?token|token)"
)

RULES: tuple[_Rule, ...] = (
    _Rule(
        "private-key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
            re.S,
        ),
    ),
    _Rule("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    _Rule("openai-key", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{32,}")),
    _Rule(
        "github-token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})"),
    ),
    _Rule("aws-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    _Rule("google-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    _Rule("slack-token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    _Rule("stripe-key", re.compile(r"\b[sr]k_(?:live|test)_[0-9A-Za-z]{16,}")),
    _Rule("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    _Rule("bearer-token", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/-]{20,}=*)"), 1),
    # user:password@ in a URL; the user name goes too, it is half the credential
    _Rule(
        "url-credentials",
        re.compile(r"(?<=://)([^\s/:@\[\]]+:[^\s/@\[\]]+)(?=@)"),
        1,
        _placeholder,
    ),
    # Password=...; inside a connection string (the `;` after it says it is one)
    _Rule(
        "password",
        re.compile(r"(?i)(?<![A-Za-z0-9_])(?:password|pwd)=([^;\s\"'\[]+)(?=;)"),
        1,
        _placeholder,
    ),
)

_ASSIGNED = re.compile(
    r"(?i)\b[A-Za-z0-9_]*" + _NAMES + r"\b[\"']?(\s*(?::|=|:=|=>)\s*)([\"']?)([^\s\"'`,;]{8,})"
)


_CODE_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[A-Za-z]\w*_\w*")


def _secretish(value: str) -> bool:
    """A value assigned to a secret-looking name: letters and digits mixed, and not code
    (a call, an index, an attribute, a snake_case name, a variable, a placeholder, a URL)
    or an earlier redaction. Precision over recall: memory is mostly about code."""
    if any(c in value for c in "()[]{}<>$%") or "://" in value:
        return False
    if _CODE_NAME.fullmatch(value):
        return False
    return any(c.isdigit() for c in value) and any(c.isalpha() for c in value)


@dataclass(frozen=True)
class Redactor:
    rules: tuple[_Rule, ...] = RULES
    enabled: bool = True

    def __call__(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        for rule in self.rules:
            text = rule.pattern.sub(partial(_replace, rule=rule), text)
        return _ASSIGNED.sub(_assigned, text)

    def many(self, texts: Iterable[str]) -> list[str]:
        return [self(t) for t in texts]


def _replace(m: re.Match[str], rule: _Rule) -> str:
    if rule.spare is not None and rule.spare(m.group(rule.group)):
        return m.group(0)
    if rule.group == 0:
        return _mark(rule.kind)
    start, end = m.span(rule.group)
    whole = m.group(0)
    s0 = m.start(0)
    return whole[: start - s0] + _mark(rule.kind) + whole[end - s0 :]


def _assigned(m: re.Match[str]) -> str:
    value = m.group(3)
    if not _secretish(value):
        return m.group(0)
    whole, s0 = m.group(0), m.start(0)
    start, end = m.span(3)
    return whole[: start - s0] + _mark("secret") + whole[end - s0 :]


def make_redactor(enabled: bool = True, extra: Iterable[str] = ()) -> Redactor:
    """The built-in rules plus `extra` regular expressions (the whole match is replaced).
    A pattern that does not compile is an error, so a typo in config.toml is not silently
    a secret kept."""
    own = []
    for p in extra:
        try:
            own.append(_Rule("custom", re.compile(p)))
        except re.error as e:
            msg = f"redact_patterns: {p!r} is not a valid regular expression: {e}"
            raise ValueError(msg) from e
    return Redactor(RULES + tuple(own), enabled)


def redacting(ask: Ask | None, redact: Callable[[str], str]) -> Ask | None:
    """A model whose answers are redacted: the backstop, so nothing a model writes into a
    note or a summary can carry a secret."""
    if ask is None:
        return None

    def wrapped(prompt: str) -> str:
        return redact(ask(prompt))

    return wrapped
