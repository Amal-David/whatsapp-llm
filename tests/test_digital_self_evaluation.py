import hashlib
import json
from datetime import datetime, timedelta, timezone

from living_brain.identity.builder import DigitalSelfBuilder
from living_brain.identity.evaluation import (
    DEFAULT_CONFIGURATIONS,
    REQUIRED_EVALUATION_TAGS,
    EvaluationConfiguration,
    EvaluationSuiteBuilder,
)
from living_brain.identity.sources import NormalizedMessage


def _message(message_id, chat_id, relationship_id, timestamp, text, *, from_owner):
    return NormalizedMessage(
        source="fixture",
        source_message_id=message_id,
        chat_id=chat_id,
        relationship_id=relationship_id,
        sender_id="sender:owner" if from_owner else f"sender:{relationship_id}",
        timestamp=timestamp,
        from_owner=from_owner,
        text=text,
        message_type="text",
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def _messages():
    anchor = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    messages = []
    for index, day in enumerate((0, 31, 59, 90, 120, 151)):
        relationship = "relationship:friend" if index % 2 == 0 else "relationship:work"
        chat = "chat:friend" if index % 2 == 0 else "chat:work"
        timestamp = anchor + timedelta(days=day)
        messages.extend(
            [
                _message(
                    f"other-{index}",
                    chat,
                    relationship,
                    timestamp,
                    f"SECRET THIRD PARTY PROMPT {index}",
                    from_owner=False,
                ),
                _message(
                    f"owner-{index}",
                    chat,
                    relationship,
                    timestamp + timedelta(minutes=1),
                    f"owner reply {index}",
                    from_owner=True,
                ),
            ]
        )
    return messages


def _interview():
    questions = [
        (
            "changes.preferences",
            "What preference changed most recently?",
            "I now prefer focused mornings.",
            ["latest_preference"],
        ),
        (
            "changes.old_self",
            "What old pattern no longer fits?",
            "I no longer say yes to every opportunity.",
            ["supersession", "contradiction"],
        ),
        (
            "values.core",
            "What is non-negotiable?",
            "Autonomy.",
            ["grounding"],
        ),
        (
            "boundaries.qualify",
            "Where should the replica say it is unsure?",
            "Anything not supported by current evidence.",
            ["abstention"],
        ),
        (
            "relationships.privacy",
            "What third-party information must stay private?",
            "All private details from other people's messages.",
            ["relationship_leakage"],
        ),
        (
            "goals.unanswered",
            "What is an unanswered question?",
            None,
            ["grounding"],
        ),
    ]
    return {
        "schema_version": "digital_self_interview.v1",
        "owner_name": "Amal",
        "completed_at": "2026-07-01T12:00:00+00:00",
        "sections": [
            {
                "id": "evaluation",
                "title": "Evaluation",
                "questions": [
                    {
                        "id": question_id,
                        "prompt": prompt,
                        "answer": answer,
                        "evaluation_tags": tags,
                        "valid_from": None,
                        "sensitivity": "private",
                    }
                    for question_id, prompt, answer, tags in questions
                ],
            }
        ],
    }


def _suite(configurations=DEFAULT_CONFIGURATIONS):
    interview = _interview()
    build_result = DigitalSelfBuilder().build(
        _messages(),
        owner_name="Amal",
        interview=interview,
    )
    return EvaluationSuiteBuilder(configurations=configurations).build(
        build_result,
        interview=interview,
    ), build_result


def test_evaluation_rows_use_only_held_out_groups_or_answered_interview_questions():
    suite, build_result = _suite()
    chat_rows = [row for row in suite.rows if row.origin == "held_out_reply"]
    interview_rows = [row for row in suite.rows if row.origin == "interview_retest"]
    train_owner_text = {
        message.text
        for message in build_result.messages_by_split["train"]
        if message.from_owner
    }

    assert chat_rows
    assert all(row.split in {"validation", "test"} for row in chat_rows)
    assert all(
        build_result.split_by_group[row.source_group_id] in {"validation", "test"}
        for row in chat_rows
    )
    assert not train_owner_text.intersection(row.reference_response for row in chat_rows)
    assert len(interview_rows) == 5
    assert all(row.reference_response for row in interview_rows)


def test_evaluation_artifact_reports_required_authenticity_and_safety_coverage():
    suite, _build_result = _suite()

    assert set(suite.coverage()) == set(REQUIRED_EVALUATION_TAGS)
    assert suite.summary()["missing_required_tags"] == []


def test_blind_pairwise_sheet_hides_configuration_labels_and_separates_answer_key():
    suite, _build_result = _suite()
    responses = {
        configuration.id: {
            row.id: f"candidate response {configuration_index} {row_index}"
            for row_index, row in enumerate(suite.rows)
        }
        for configuration_index, configuration in enumerate(suite.configurations)
    }

    sheet, answer_key = suite.build_blind_pairwise_sheet(responses, seed="owner-review")
    public_json = json.dumps(sheet, sort_keys=True)

    assert sheet["comparisons"]
    assert all(
        comparison["more_like_me"] is None and comparison["notes"] is None
        for comparison in sheet["comparisons"]
    )
    assert all(
        configuration.id not in public_json and configuration.label not in public_json
        for configuration in suite.configurations
    )
    assert len(answer_key["assignments"]) == len(sheet["comparisons"])


def test_automatic_summary_never_contains_raw_dialogue():
    suite, _build_result = _suite()

    summary_json = json.dumps(suite.summary(), sort_keys=True)

    assert "SECRET THIRD PARTY PROMPT" not in summary_json
    assert "owner reply" not in summary_json
    assert "All private details" not in summary_json


def test_adding_adapter_configuration_does_not_change_evaluation_row_identity():
    baseline, _build_result = _suite()
    with_adapter, _build_result = _suite(
        DEFAULT_CONFIGURATIONS
        + (
            EvaluationConfiguration(
                id="profile_retrieval_adapter",
                label="Profile + retrieval + adapter",
                uses_profile=True,
                uses_retrieval=True,
                adapter_name="owner-style",
            ),
        )
    )

    assert [row.id for row in baseline.rows] == [row.id for row in with_adapter.rows]
