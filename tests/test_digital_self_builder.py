import hashlib
import json
from datetime import datetime, timedelta, timezone

from living_brain.identity.builder import DigitalSelfBuilder
from living_brain.identity.models import ClaimStatus, ProvenanceType
from living_brain.identity.sources import NormalizedMessage


def _message(
    message_id,
    chat_id,
    relationship_id,
    timestamp,
    text,
    *,
    from_owner,
):
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


def _multi_chat_messages():
    messages = []
    group_specs = [
        ("chat:friend", "relationship:friend", 1, "ok", "question one"),
        ("chat:work", "relationship:work", 2, "ship small", "question two"),
        ("chat:friend", "relationship:friend", 32, "test first", "question three"),
        ("chat:work", "relationship:work", 33, "verify it", "question four"),
        (
            "chat:friend",
            "relationship:friend",
            61,
            "This held-out answer is dramatically longer than the training replies :) :) :)!",
            "private held-out context one",
        ),
        (
            "chat:work",
            "relationship:work",
            62,
            "Another very long held-out answer with expressive punctuation!!!",
            "private held-out context two",
        ),
    ]
    anchor = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    for index, (chat_id, relationship_id, day_offset, owner_text, other_text) in enumerate(
        group_specs
    ):
        timestamp = anchor.replace(day=1) + timedelta(days=day_offset)
        messages.extend(
            [
                _message(
                    f"other-{index}",
                    chat_id,
                    relationship_id,
                    timestamp,
                    other_text,
                    from_owner=False,
                ),
                _message(
                    f"owner-{index}",
                    chat_id,
                    relationship_id,
                    timestamp + timedelta(minutes=1),
                    owner_text,
                    from_owner=True,
                ),
            ]
        )
    return messages


def _interview():
    return {
        "schema_version": "digital_self_interview.v1",
        "owner_name": "Amal",
        "completed_at": "2026-04-01T12:00:00+00:00",
        "sections": [
            {
                "id": "values",
                "title": "Values",
                "questions": [
                    {
                        "id": "values.independence",
                        "prompt": "What matters?",
                        "answer": "I optimize for independence over status.",
                        "valid_from": "2025-01-01T00:00:00+00:00",
                        "sensitivity": "private",
                    },
                    {
                        "id": "communication.message_length",
                        "prompt": "How long are your replies?",
                        "answer": "I vary length based on the stakes.",
                        "valid_from": None,
                        "sensitivity": "private",
                    },
                ],
            }
        ],
    }


def test_builder_combines_multiple_relationships_without_flattening_them():
    result = DigitalSelfBuilder().build(
        _multi_chat_messages(),
        owner_name="Amal",
        interview=_interview(),
    )

    assert result.profile.source_summary["chat_count"] == 2
    assert result.profile.source_summary["owner_message_count"] == 6
    assert {relationship.id for relationship in result.profile.relationships} == {
        "relationship:friend",
        "relationship:work",
    }
    assert set(result.profile.communication_style["relationships"]) == {
        "relationship:friend",
        "relationship:work",
    }


def test_third_party_text_is_excluded_by_default_and_never_identity_evidence():
    messages = _multi_chat_messages()

    private_result = DigitalSelfBuilder().build(messages, owner_name="Amal")
    contextual_result = DigitalSelfBuilder(include_third_party_context=True).build(
        messages,
        owner_name="Amal",
    )

    private_json = private_result.profile.to_json()
    contextual_evidence = [
        evidence
        for evidence in contextual_result.profile.evidence
        if evidence.metadata.get("identity_target") is False
    ]
    claim_evidence_ids = {
        evidence_id
        for claim in contextual_result.profile.claims
        for evidence_id in claim.evidence_ids
    }
    owner_evidence = [
        evidence
        for evidence in private_result.profile.evidence
        if evidence.metadata.get("identity_target") is True
    ]

    assert "private held-out context" not in private_json
    assert owner_evidence
    assert all(evidence.content is None for evidence in owner_evidence)
    assert all(
        evidence.content_hash == evidence.metadata["content_hash"]
        for evidence in owner_evidence
    )
    assert not any(
        evidence.metadata.get("identity_target") is False
        for evidence in private_result.profile.evidence
    )
    assert contextual_evidence
    assert all(evidence.id not in claim_evidence_ids for evidence in contextual_evidence)


def test_style_and_relationship_deltas_use_training_groups_only():
    result = DigitalSelfBuilder().build(_multi_chat_messages(), owner_name="Amal")
    owner_messages = [message for message in _multi_chat_messages() if message.from_owner]
    owner_train_count = len(result.messages_by_split["train"]) // 2

    assert set(result.split_by_group.values()) == {"train", "validation", "test"}
    assert all(
        len({result.split_for(message) for message in group}) == 1
        for group in result.message_groups.values()
    )
    assert result.profile.communication_style["global"]["message_count"] == owner_train_count
    assert owner_train_count < len(owner_messages)
    assert result.profile.communication_style["global"]["avg_message_length"] < 20
    assert all(
        relationship["computed_from_split"] == "train"
        for relationship in result.profile.communication_style["relationships"].values()
    )


def test_interview_claims_are_confirmed_and_override_behavioral_candidates():
    result = DigitalSelfBuilder().build(
        _multi_chat_messages(),
        owner_name="Amal",
        interview=_interview(),
    )

    resolved = result.profile.resolve_dimension(
        "communication.message_length",
        as_of=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    same_dimension = [
        claim
        for claim in result.profile.claims
        if claim.dimension == "communication.message_length"
    ]

    assert resolved is not None
    assert resolved.statement == "I vary length based on the stakes."
    assert resolved.status is ClaimStatus.CONFIRMED
    assert resolved.provenance is ProvenanceType.OWNER_INTERVIEW
    assert {claim.status for claim in same_dimension} == {
        ClaimStatus.CANDIDATE,
        ClaimStatus.CONFIRMED,
    }


def test_equivalent_inputs_produce_stable_profile_and_evidence_ids():
    first = DigitalSelfBuilder().build(
        _multi_chat_messages(),
        owner_name="Amal",
        interview=_interview(),
    )
    second = DigitalSelfBuilder().build(
        list(reversed(_multi_chat_messages())),
        owner_name="Amal",
        interview=json.loads(json.dumps(_interview())),
    )

    assert first.profile.to_json() == second.profile.to_json()
    assert first.split_by_group == second.split_by_group
