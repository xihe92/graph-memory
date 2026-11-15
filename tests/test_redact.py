"""Secrets out of memory: what is redacted, what is left alone, and that nothing the daemon
writes or sends to a model carries a secret."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from graph_memory.client import call
from graph_memory.config import Config, load_config
from graph_memory.redact import make_redactor, redacting

from .test_daemon import _wait
from .test_daemon_buffer import _graduate, _remember, daemon_with
from .test_graduation import Script

# Built at run time, so the repository holds no string shaped like a real key.
GITHUB = "gh" + "p_" + "Z9" * 18
AWS = "AK" + "IA" + "Q7" * 8
ANTHROPIC = "sk-" + "ant-api03-" + "a1B2" * 6
OPENAI = "sk-" + "proj-" + "A1b2" * 10
GOOGLE = "AI" + "za" + "S1" * 17 + "x"
SLACK = "xo" + "xb-" + "1234567890-abcdefABCDEF"
STRIPE = "sk" + "_live_" + "a1" * 12
JWT = "eyJ" + "hbGciOiJIUzI1NiJ9" + ".eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0" + "." + "dozjgNryP4J3jVmN"
PEM = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"

r = make_redactor()


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (f"export GITHUB_TOKEN={GITHUB}", "github-token"),
        ("gh pr create, token github_pat_" + "a1" * 35, "github-token"),
        (f"aws configure set aws_access_key_id {AWS}", "aws-key"),
        (f"ANTHROPIC_API_KEY={ANTHROPIC}", "anthropic-key"),
        (f"client = OpenAI(api_key='{OPENAI}')", "openai-key"),
        (f"maps key {GOOGLE}", "google-key"),
        (f"SLACK_BOT={SLACK}", "slack-token"),
        (f"stripe {STRIPE}", "stripe-key"),
        (f"cookie session={JWT}", "jwt"),
        (f"key file:\n{PEM}\nthen", "private-key"),
    ],
)
def test_known_key_shapes_are_redacted(text: str, kind: str) -> None:
    out = r(text)
    assert f"[redacted: {kind}]" in out
    for secret in (GITHUB, AWS, ANTHROPIC, OPENAI, GOOGLE, SLACK, STRIPE, JWT, "MIIEpAIBAAKCAQEA"):
        assert secret not in out


def test_the_text_around_a_secret_stays() -> None:
    assert r("curl -H 'Authorization: Bearer abc123DEF456ghi789JKL0'") == (
        "curl -H 'Authorization: Bearer [redacted: bearer-token]'"
    )
    assert r("postgres://app:S3cretPw@db:5432/orders") == (
        "postgres://[redacted: url-credentials]@db:5432/orders"
    )
    assert r("Server=db;Password=S3cret!x;Database=orders") == (
        "Server=db;Password=[redacted: password];Database=orders"
    )
    assert r("password=Hunter2Hunter2") == "password=[redacted: secret]"
    assert r('API_KEY = "a1b2c3d4e5f6"') == 'API_KEY = "[redacted: secret]"'
    assert r('{"client_secret": "x9y8z7w6v5"}') == '{"client_secret": "[redacted: secret]"}'
    # a private key cut off mid-way (a clipped reply) still goes, to the end
    assert r("key:\n-----BEGIN PRIVATE KEY-----\nMIIEv") == "key:\n[redacted: private-key]"


@pytest.mark.parametrize(
    "text",
    [
        "commit 3f2a9c1e8b7d6a5f4e3d2c1b0a9f8e7d6c5b4a39 fixed it",
        "item 4b7e19d0a2c3 and id 0e7c6a2e-4b1d-4f3a-9c8e-2a1b3c4d5e6f",
        "max_tokens = 4096",
        "token = tokens(prompt)",
        "token = self.store2",
        "password = getpass()",
        'api_key = os.environ["ANTHROPIC_API_KEY"]',
        "secret_key = SECRET_KEY_V2",
        "the auth_token: see https://example.com/docs2 for how",
        "pip install sk-learn scikit-learn",
        "postgres://user:password@localhost/db",
        "mssql://<user>:<password>@host:1433/db",
        "Server=db;Password={secret};",
        "Server=db;Pwd=...;",
        "password: str = field(repr=False)",
        "a git SHA such as 81fc9ad, the token count was 12345678",
    ],
)
def test_code_hashes_and_placeholders_are_left_alone(text: str) -> None:
    assert r(text) == text


def test_redacting_twice_changes_nothing() -> None:
    text = f"{GITHUB} postgres://app:S3cretPw@db password=Hunter2Hunter2 Pwd=abc12345;"
    once = r(text)
    assert r(once) == once


def test_own_patterns_off_and_a_bad_pattern(short_dir: Path) -> None:
    (short_dir / "config.toml").write_text(
        'redact_patterns = ["ACME-[0-9]{6}"]\n', encoding="utf-8"
    )
    cfg = load_config(short_dir)
    assert cfg.redact_patterns == ("ACME-[0-9]{6}",)
    mine = make_redactor(cfg.redact, cfg.redact_patterns)
    assert mine("badge ACME-123456 and " + GITHUB) == (
        "badge [redacted: custom] and [redacted: github-token]"
    )
    assert make_redactor(False)(GITHUB) == GITHUB
    (short_dir / "config.toml").write_text('redact_patterns = ["ACME-[0-9"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid regular expression"):
        load_config(short_dir)


def test_a_model_answer_is_redacted() -> None:
    ask = redacting(lambda p: f"the key was {AWS}", r)
    assert ask is not None and ask("q") == "the key was [redacted: aws-key]"
    assert redacting(None, r) is None


@pytest.fixture
def cfg(short_dir: Path) -> Config:
    return replace(load_config(short_dir), save_interval=0.05, judge_rate=0.0)


def test_nothing_the_daemon_keeps_or_sends_carries_a_secret(cfg: Config) -> None:
    """A session pastes keys into a prompt, a reply, a command and its compaction summary;
    the model, as if careless, copies one into a note. No file under the memory home, no
    prompt to the model and no note holds any of them."""
    secrets = (GITHUB, AWS, "Hunter2Hunter2")
    careless = {
        "entities": [{"name": "deploy.sh", "kind": "file", "summary": "deploy script"}],
        "items": [
            {
                "type": "fact",
                "statement": f"deploy.sh pushes with the token {GITHUB} from the CI settings",
                "entities": ["deploy.sh"],
                "scope": "project",
                "evidence": ["X1"],
            }
        ],
    }
    ask = Script([careless])
    with daemon_with(cfg, ask) as d:
        prompt = f"push fails with token {GITHUB}, why?"
        call(cfg, "recall", {"session": "s", "prompt": prompt}, autostart=False)
        _remember(
            cfg,
            "s",
            prompt,
            f"Your AWS key {AWS} is fine; set password=Hunter2Hunter2 in deploy.sh.",
            commands=[f"export GITHUB_TOKEN={GITHUB}"],
        )
        (kept,) = d.buffer.exchanges("s")
        assert "[redacted: github-token]" in kept.prompt and "[redacted: aws-key]" in kept.reply
        assert kept.commands == ["export GITHUB_TOKEN=[redacted: github-token]"]
        _graduate(cfg, "s", "compact", f"SUMMARY: deploy uses {AWS}")
        _wait(lambda: d.knowledge.stats()["segments"] == 1)
        notes = [i.statement for i in d.knowledge.valid_items()]
        assert any("[redacted: github-token]" in n for n in notes)
        _graduate(cfg, "s", "session_end")
        _wait(lambda: d.buffer.sessions() == [])
    assert ask.prompts and not any(s in p for p in ask.prompts for s in secrets)
    for f in cfg.home.rglob("*"):
        if f.is_file():
            data = f.read_bytes()
            assert not any(s.encode() in data for s in secrets), f
