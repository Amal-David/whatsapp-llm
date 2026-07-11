import json
from datetime import datetime, timedelta, timezone

from living_brain.brain.coverage import analyze_coverage
from living_brain.brain.inspection import inspect_brain
from living_brain.brain.models import (
    BrainLayer,
    ContextScope,
    DigitalBrain,
    EpistemicStatus,
    Ownership,
    ProvenanceRef,
    ProvenanceRelation,
    Sensitivity,
    StateItem,
    TemporalScope,
)

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _item(
    layer,
    kind,
    summary,
    *,
    observed_at=NOW,
    confidence=0.9,
    epistemic=EpistemicStatus.OWNER_DECLARED,
    ownership=Ownership.OWNER,
    sensitivity=Sensitivity.PRIVATE,
    relationship_id=None,
    payload=None,
):
    return StateItem.create(
        layer=layer,
        kind=kind,
        summary=summary,
        payload=payload or {},
        epistemic_status=epistemic,
        confidence=confidence,
        temporal=TemporalScope(
            observed_at=observed_at,
            recorded_at=observed_at,
            valid_from=observed_at,
        ),
        scope=ContextScope(relationship_id=relationship_id),
        sensitivity=sensitivity,
        ownership=ownership,
        provenance=(
            ProvenanceRef(
                source_id=f"evidence:{kind}",
                source_type="fixture",
                relation=ProvenanceRelation.SUPPORTS,
                observed_at=observed_at,
                content_hash="a" * 64,
                note="Private provenance note.",
            ),
        ),
    )


def _brain(*items):
    return DigitalBrain.create(
        owner_id="owner:test",
        owner_name="Test Owner",
        created_at=min(item.temporal.recorded_at for item in items),
        updated_at=max(item.temporal.recorded_at for item in items),
        items=list(items),
    )


def test_coverage_separates_strong_weak_unknown_stale_and_next_questions():
    strong = _item(
        BrainLayer.SEMANTIC,
        "owner.fact",
        "A confirmed owner fact.",
        observed_at=NOW - timedelta(days=5),
    )
    weak = _item(
        BrainLayer.VALUES_GOALS,
        "value.candidate",
        "A weak inferred value.",
        observed_at=NOW - timedelta(days=10),
        confidence=0.45,
        epistemic=EpistemicStatus.INFERRED,
    )
    stale = _item(
        BrainLayer.COMMUNICATION,
        "tone",
        "An old communication pattern.",
        observed_at=NOW - timedelta(days=400),
        confidence=0.8,
        epistemic=EpistemicStatus.OBSERVED,
    )
    brain = _brain(strong, weak, stale)

    report = analyze_coverage(brain, as_of=NOW, stale_after_days=180)
    data = report.to_dict()

    assert strong.id in data["strong_item_ids"]
    assert weak.id in data["weak_item_ids"]
    assert stale.id in data["stale_item_ids"]
    assert "episode" in data["unknown_layers"]
    assert "social" in data["unknown_layers"]
    assert data["next_questions"]
    assert data["next_questions"][0]["priority"] >= data["next_questions"][-1][
        "priority"
    ]
    assert "payload" not in json.dumps(data)
    assert "Private provenance note" not in json.dumps(data)


def test_inspection_explains_provenance_without_raw_third_party_content_by_default():
    owner = _item(
        BrainLayer.SEMANTIC,
        "owner.fact",
        "The owner likes building things.",
        payload={"raw_text": "private owner text"},
    )
    third_party = _item(
        BrainLayer.EVENT,
        "third_party.message",
        "A contact disclosed a private secret.",
        ownership=Ownership.THIRD_PARTY,
        sensitivity=Sensitivity.RESTRICTED,
        relationship_id="relationship:a",
        payload={"raw_text": "never expose this third-party secret"},
    )
    brain = _brain(owner, third_party)

    global_view = inspect_brain(brain, as_of=NOW)
    relationship_view = inspect_brain(
        brain,
        as_of=NOW,
        relationship_id="relationship:a",
        include_history=True,
    )

    assert len(global_view["items"]) == 1
    assert global_view["items"][0]["summary"] == "The owner likes building things."
    assert "payload" not in global_view["items"][0]
    assert global_view["items"][0]["provenance"][0]["source_id"]
    encoded = json.dumps(relationship_view)
    assert "never expose this third-party secret" not in encoded
    assert "A contact disclosed a private secret" not in encoded
    third_party_view = next(
        item for item in relationship_view["items"] if item["ownership"] == "third_party"
    )
    assert third_party_view["summary"] == "Third-party context (redacted)"
    assert "note" not in third_party_view["provenance"][0]


def test_coverage_excludes_third_party_claims_and_surfaces_owner_conflicts():
    first = _item(
        BrainLayer.SELF_SCHEMA,
        "identity.role",
        "The owner sees themselves as a builder.",
    )
    second = _item(
        BrainLayer.SELF_SCHEMA,
        "identity.role",
        "The owner sees themselves as a teacher.",
        observed_at=NOW - timedelta(days=1),
    )
    third_party = _item(
        BrainLayer.SOCIAL,
        "contact.preference",
        "A contact's preference, not the owner's.",
        ownership=Ownership.THIRD_PARTY,
        relationship_id="relationship:a",
    )

    report = analyze_coverage(_brain(first, second, third_party), as_of=NOW)
    data = report.to_dict()

    assert data["conflict_pairs"] == [sorted((first.id, second.id))]
    assert next(
        layer for layer in data["layers"] if layer["layer"] == "self_schema"
    )["status"] == "conflicted"
    assert "social" in data["unknown_layers"]
    assert third_party.id not in data["strong_item_ids"]
    assert third_party.id not in data["weak_item_ids"]


def test_inspection_isolates_relationships_and_never_reveals_third_party_payloads():
    owner = _item(
        BrainLayer.SEMANTIC,
        "owner.private",
        "A sensitive owner fact.",
        sensitivity=Sensitivity.RESTRICTED,
        payload={"raw_text": "owner-only detail"},
    )
    relation_a = _item(
        BrainLayer.EVENT,
        "message.a",
        "Secret from A.",
        ownership=Ownership.THIRD_PARTY,
        sensitivity=Sensitivity.RESTRICTED,
        relationship_id="relationship:a",
        payload={"raw_text": "third-party A raw text"},
    )
    relation_b = _item(
        BrainLayer.EVENT,
        "message.b",
        "Secret from B.",
        ownership=Ownership.THIRD_PARTY,
        sensitivity=Sensitivity.RESTRICTED,
        relationship_id="relationship:b",
        payload={"raw_text": "third-party B raw text"},
    )
    brain = _brain(owner, relation_a, relation_b)

    default_view = inspect_brain(brain, as_of=NOW)
    relation_view = inspect_brain(
        brain,
        as_of=NOW,
        relationship_id="relationship:a",
        include_payload=True,
        include_sensitive=True,
    )

    assert default_view["items"][0]["summary"] == "Sensitive state (redacted)"
    assert "payload" not in default_view["items"][0]
    encoded = json.dumps(relation_view)
    assert owner.payload["raw_text"] in encoded
    assert "third-party A raw text" not in encoded
    assert "third-party B raw text" not in encoded
    assert relation_a.id in encoded
    assert relation_b.id not in encoded
