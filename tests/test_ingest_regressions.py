import subprocess
import sys
from datetime import datetime

import pytest

from living_brain.ingest.style_analyzer import StyleMetrics
from living_brain.ingest.whatsapp_parser import WhatsAppParser


def test_style_prompt_common_phrases_is_supported_on_all_declared_python_versions():
    metrics = StyleMetrics(common_phrases=[("see you", 3), ("take care", 2)])

    prompt = metrics.to_system_prompt()

    assert '- Common expressions: "see you", "take care"' in prompt


def test_system_events_are_not_appended_to_the_previous_participant_message(tmp_path):
    chat_path = tmp_path / "chat.txt"
    chat_path.write_text(
        "[01/13/24, 12:00:00 AM] Alice: hello\n"
        "[01/13/24, 12:01:00 AM] Bob added Charlie\n"
        "[01/13/24, 12:02:00 AM] Alice: bye\n",
        encoding="utf-8",
    )

    participant_messages = list(WhatsAppParser().parse_file(chat_path))
    all_messages = list(WhatsAppParser().parse_file(chat_path, skip_system=False))

    assert [message.message for message in participant_messages] == ["hello", "bye"]
    assert [message.is_system for message in all_messages] == [False, True, False]
    assert all_messages[1].author == "System"
    assert all_messages[1].message == "Bob added Charlie"


def test_system_events_with_colons_are_not_misclassified_as_participants(tmp_path):
    chat_path = tmp_path / "chat.txt"
    chat_path.write_text(
        "[01/13/24, 12:00:00 AM] Alice: hello\n"
        "[01/13/24, 12:01:00 AM] Bob changed the group description: private\n",
        encoding="utf-8",
    )

    messages = list(WhatsAppParser().parse_file(chat_path, skip_system=False))

    assert len(messages) == 2
    assert messages[1].author == "System"
    assert messages[1].is_system is True
    assert "changed the group description" in messages[1].message


def test_multiline_participant_messages_remain_contiguous(tmp_path):
    chat_path = tmp_path / "chat.txt"
    chat_path.write_text(
        "[01/13/24, 12:00:00 AM] Alice: first line\n"
        "second line\n",
        encoding="utf-8",
    )

    messages = list(WhatsAppParser().parse_file(chat_path))

    assert [message.message for message in messages] == ["first line\nsecond line"]


def test_ambiguous_slash_dates_follow_the_configured_date_order():
    mdy_message = WhatsAppParser(date_order="mdy").parse_line(
        "[01/02/24, 10:00] Alice: hello"
    )
    dmy_message = WhatsAppParser(date_order="dmy").parse_line(
        "[01/02/24, 10:00] Alice: hello"
    )

    assert mdy_message is not None
    assert dmy_message is not None
    assert mdy_message.timestamp == datetime(2024, 1, 2, 10, 0)
    assert dmy_message.timestamp == datetime(2024, 2, 1, 10, 0)


def test_invalid_date_order_is_rejected():
    with pytest.raises(ValueError, match="date_order"):
        WhatsAppParser(date_order="ymd")


def test_parse_cli_accepts_date_order(tmp_path):
    chat_path = tmp_path / "chat.txt"
    output_path = tmp_path / "processed"
    chat_path.write_text(
        "[01/02/24, 10:00] Alice: hello\n"
        "[01/02/24, 10:01] Bob: hi\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "living_brain.main",
            "parse",
            str(chat_path),
            "--your-name",
            "Alice",
            "--date-order",
            "dmy",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        cwd=tmp_path.parent,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.with_suffix(".json").exists()
    assert output_path.with_suffix(".jsonl").exists()
