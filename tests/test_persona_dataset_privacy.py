import json

from living_brain.inference.dataset_ui import DatasetWorkbench
from living_brain.ingest.persona_dataset import PersonaDatasetBuilder


def _write_grouped_chat(path, conversation_count=5):
    lines = []
    for index in range(conversation_count):
        day = index + 1
        lines.extend(
            [
                f"[01/{day:02d}/24, 10:00 AM] Bob: private prompt {index}",
                f"[01/{day:02d}/24, 10:01 AM] Alice: personal reply {index}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_third_party_context_is_excluded_by_default(tmp_path):
    chat_path = tmp_path / "chat.txt"
    _write_grouped_chat(chat_path)

    result = PersonaDatasetBuilder().build_from_file(chat_path, participant="Alice")
    serialized_rows = json.dumps(result.canonical_examples)

    assert result.summary["privacy"]["third_party_context_included"] is False
    assert "private prompt" not in serialized_rows
    assert "Bob" not in serialized_rows
    assert "[third-party message withheld]" in serialized_rows


def test_enabling_third_party_context_requires_separate_confirmation(tmp_path):
    chat_path = tmp_path / "chat.txt"
    _write_grouped_chat(chat_path)

    output = DatasetWorkbench()._build_artifacts(
        upload=chat_path,
        participant="Alice",
        owner_type="self",
        consent_confirmed=True,
        include_third_party_context=True,
        third_party_consent_confirmed=False,
        context_turns=6,
        gap_minutes=60,
    )

    assert "permission" in output[-1].lower()
    assert output[3] is None


def test_shareable_character_artifacts_exclude_source_quotes_and_third_party_names(tmp_path):
    chat_path = tmp_path / "chat.txt"
    _write_grouped_chat(chat_path)

    result = PersonaDatasetBuilder(include_third_party_context=True).build_from_file(
        chat_path,
        participant="Alice",
    )
    shared_artifacts = "\n".join(
        [
            json.dumps(result.canonical_character),
            json.dumps(result.character_card_v2),
            result.persona_markdown,
        ]
    )

    assert "private prompt" not in shared_artifacts
    assert "personal reply" not in shared_artifacts
    assert "Bob" not in shared_artifacts
    assert result.canonical_character["examples"]["dialogue"] == []
    assert result.character_card_v2["data"]["mes_example"] == ""


def test_conversation_splits_and_derived_artifacts_are_isolated(tmp_path):
    chat_path = tmp_path / "chat.txt"
    _write_grouped_chat(chat_path)

    result = PersonaDatasetBuilder().build_from_file(chat_path, participant="Alice")
    rows_by_group = {}
    for row in result.canonical_examples:
        rows_by_group.setdefault(row["split_group_id"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in rows_by_group.values())
    assert {row["split"] for row in result.canonical_examples} == {
        "train",
        "validation",
        "test",
    }
    assert {row["metadata"]["split"] for row in result.sft_alpaca} == {"train"}
    assert {row["metadata"]["split"] for row in result.sft_messages} == {"train"}
    assert {row["metadata"]["split"] for row in result.dpo_trl} == {"train"}
    assert {row["metadata"]["split"] for row in result.dpo_openai} == {"train"}
    assert {row["split"] for row in result.eval_rows} == {"validation", "test"}
    assert result.style_capsule["computed_from_split"] == "train"
    assert result.style_capsule["metrics"]["message_count"] < result.summary[
        "target_message_count"
    ]
