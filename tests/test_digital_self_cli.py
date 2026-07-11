import json
import stat
import sys
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from living_brain.identity.evaluation import EvaluationSuite
from living_brain.identity.models import DigitalSelfProfile
from living_brain.main import main


def _write_source(path):
    anchor = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    messages = []
    for index, day in enumerate((0, 31, 59, 90, 120, 151)):
        chat_id = "friend" if index % 2 == 0 else "work"
        timestamp = anchor + timedelta(days=day)
        messages.extend(
            [
                {
                    "source_message_id": f"other-{index}",
                    "source_chat_id": chat_id,
                    "sender_id": f"contact-{chat_id}",
                    "timestamp": timestamp.isoformat(),
                    "from_owner": False,
                    "text": f"SECRET SOURCE PROMPT {index}",
                },
                {
                    "source_message_id": f"owner-{index}",
                    "source_chat_id": chat_id,
                    "sender_id": "owner",
                    "timestamp": (timestamp + timedelta(minutes=1)).isoformat(),
                    "from_owner": True,
                    "text": f"owner response {index}",
                },
            ]
        )
    path.write_text(
        json.dumps(
            {
                "chats": [
                    {
                        "source_chat_id": "friend",
                        "display_name": "Friend\u001b[31m",
                        "kind": "dm",
                    },
                    {"source_chat_id": "work", "display_name": "Work", "kind": "dm"},
                ],
                "messages": messages,
            }
        ),
        encoding="utf-8",
    )


def _answer_interview(path):
    interview = yaml.safe_load(path.read_text(encoding="utf-8"))
    answers = {
        "changes.preferences": "I now prefer focused mornings.",
        "changes.old_self": "I no longer accept every opportunity.",
        "values.core": "Autonomy.",
        "boundaries.qualify": "Anything unsupported by current evidence.",
        "relationships.privacy": "All private details in other people's messages.",
    }
    for section in interview["sections"]:
        for question in section["questions"]:
            if question["id"] in answers:
                question["answer"] = answers[question["id"]]
    interview["completed_at"] = "2026-07-01T12:00:00+00:00"
    path.write_text(yaml.safe_dump(interview, sort_keys=False), encoding="utf-8")


def _run_cli(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["living-brain", *map(str, arguments)])
    main()


def test_self_cli_creates_interview_and_inspects_a_read_only_source(
    tmp_path,
    monkeypatch,
    capsys,
):
    source_path = tmp_path / "messages.json"
    interview_path = tmp_path / "interview.yaml"
    key_path = tmp_path / "pseudonym.key"
    _write_source(source_path)

    _run_cli(
        monkeypatch,
        "self",
        "interview",
        "--owner-name",
        "Amal",
        "--output",
        interview_path,
    )
    _run_cli(
        monkeypatch,
        "self",
        "chats",
        "--source",
        "json",
        "--path",
        source_path,
        "--key-file",
        key_path,
    )
    output = capsys.readouterr().out

    assert yaml.safe_load(interview_path.read_text(encoding="utf-8"))["owner_name"] == "Amal"
    assert "Friend" in output
    assert "Work" in output
    assert "\u001b" not in output
    assert "\\u001b" in output
    assert key_path.exists()
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600


def test_self_cli_builds_validates_and_exports_private_evaluation_artifacts(
    tmp_path,
    monkeypatch,
    capsys,
):
    source_path = tmp_path / "messages.json"
    interview_path = tmp_path / "interview.yaml"
    key_path = tmp_path / "pseudonym.key"
    profile_path = tmp_path / "digital-self.json"
    evaluation_path = tmp_path / "evaluation.json"
    summary_path = tmp_path / "evaluation-summary.json"
    _write_source(source_path)

    _run_cli(
        monkeypatch,
        "self",
        "interview",
        "--owner-name",
        "Amal",
        "--output",
        interview_path,
    )
    _answer_interview(interview_path)
    source_arguments = (
        "--source",
        "json",
        "--path",
        source_path,
        "--chat",
        "friend",
        "--chat",
        "work",
        "--key-file",
        key_path,
        "--owner-name",
        "Amal",
        "--interview",
        interview_path,
    )

    _run_cli(
        monkeypatch,
        "self",
        "build",
        *source_arguments,
        "--output",
        profile_path,
    )
    _run_cli(monkeypatch, "self", "validate", profile_path)
    _run_cli(
        monkeypatch,
        "self",
        "evaluate",
        *source_arguments,
        "--profile",
        profile_path,
        "--output",
        evaluation_path,
        "--summary-output",
        summary_path,
    )
    output = capsys.readouterr().out

    profile = DigitalSelfProfile.load(profile_path)
    evaluation = EvaluationSuite.load(evaluation_path)
    summary_content = summary_path.read_text(encoding="utf-8")

    assert profile.source_summary["chat_count"] == 2
    assert "SECRET SOURCE PROMPT" not in profile.to_json()
    assert "Profile is valid" in output
    assert evaluation.profile_id == profile.profile_id
    assert evaluation.rows
    assert "SECRET SOURCE PROMPT" not in summary_content
    assert "owner response" not in summary_content
    assert stat.S_IMODE(profile_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(evaluation_path.stat().st_mode) == 0o600


def test_root_cli_help_keeps_existing_commands_and_adds_self(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["living-brain", "--help"])

    with pytest.raises(SystemExit, match="0"):
        main()

    output = capsys.readouterr().out
    for command in (
        "parse",
        "train",
        "chat",
        "workbench",
        "ingest",
        "watch",
        "stats",
        "self",
        "research",
        "brain",
    ):
        assert command in output
