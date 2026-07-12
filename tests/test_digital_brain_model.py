from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from living_brain.brain.migration import migrate_v1_profile
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
    StateStatus,
    TemporalScope,
)
from living_brain.identity.models import (
    ClaimStatus,
    DigitalSelfProfile,
    EvidenceRecord,
    IdentityClaim,
    ProvenanceType,
    RelationshipProfile,
)

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _provenance(source_id="evidence:test", *, observed_at=NOW):
    return (
        ProvenanceRef(
            source_id=source_id,
            source_type="test_fixture",
            relation=ProvenanceRelation.SUPPORTS,
            observed_at=observed_at,
            content_hash="a" * 64,
        ),
    )


def _item(
    layer,
    kind,
    summary,
    *,
    valid_from=None,
    valid_to=None,
    relationship_id=None,
    epistemic_status=EpistemicStatus.OWNER_DECLARED,
    ownership=Ownership.OWNER,
    payload=None,
):
    return StateItem.create(
        layer=layer,
        kind=kind,
        summary=summary,
        payload=payload or {},
        epistemic_status=epistemic_status,
        confidence=0.9,
        temporal=TemporalScope(
            observed_at=valid_from or NOW,
            recorded_at=NOW,
            valid_from=valid_from,
            valid_to=valid_to,
        ),
        scope=ContextScope(relationship_id=relationship_id),
        sensitivity=Sensitivity.PRIVATE,
        ownership=ownership,
        provenance=_provenance(f"evidence:{kind}"),
    )


def test_model_distinguishes_every_required_layer_and_round_trips_deterministically():
    items = [
        _item(layer, f"kind-{layer.value}", f"State for {layer.value}")
        for layer in BrainLayer
    ]
    brain = DigitalBrain.create(
        owner_id="owner:test",
        owner_name="Test Owner",
        created_at=NOW,
        items=items,
    )

    brain.validate()
    encoded = brain.to_json()
    restored = DigitalBrain.from_json(encoded)

    assert {item.layer for item in restored.items} == set(BrainLayer)
    assert restored == brain
    assert restored.to_json() == encoded
    assert all(item.provenance for item in restored.items)
    assert all(0.0 <= item.confidence <= 1.0 for item in restored.items)
    assert all(item.temporal.recorded_at.tzinfo for item in restored.items)


def test_temporal_and_relationship_scopes_coexist_without_cross_relationship_leakage():
    earlier = NOW - timedelta(days=30)
    later = NOW + timedelta(days=30)
    global_item = _item(
        BrainLayer.SELF_SCHEMA,
        "role",
        "I am a product builder.",
        valid_from=earlier,
    )
    old_friend_style = _item(
        BrainLayer.COMMUNICATION,
        "tone",
        "More playful with friend A.",
        valid_from=earlier,
        valid_to=NOW,
        relationship_id="relationship:a",
    )
    current_friend_style = _item(
        BrainLayer.COMMUNICATION,
        "tone",
        "More direct with friend A now.",
        valid_from=NOW + timedelta(seconds=1),
        relationship_id="relationship:a",
    )
    work_style = _item(
        BrainLayer.COMMUNICATION,
        "tone",
        "Concise with colleague B.",
        valid_from=earlier,
        relationship_id="relationship:b",
    )
    brain = DigitalBrain.create(
        owner_id="owner:test",
        owner_name="Test Owner",
        created_at=earlier,
        items=[global_item, old_friend_style, current_friend_style, work_style],
    )

    friend_then = brain.query(as_of=NOW, relationship_id="relationship:a")
    friend_later = brain.query(as_of=later, relationship_id="relationship:a")
    global_only = brain.query(as_of=later)

    assert {item.id for item in friend_then} == {global_item.id, old_friend_style.id}
    assert {item.id for item in friend_later} == {
        global_item.id,
        current_friend_style.id,
    }
    assert {item.id for item in global_only} == {global_item.id}
    assert work_style.id not in {item.id for item in friend_later}


def test_owner_correction_supersedes_without_erasing_history():
    inferred = _item(
        BrainLayer.SELF_SCHEMA,
        "communication.preference",
        "Usually prefers brief messages.",
        epistemic_status=EpistemicStatus.INFERRED,
    )
    brain = DigitalBrain.create(
        owner_id="owner:test",
        owner_name="Test Owner",
        created_at=NOW,
        items=[inferred],
    )

    corrected = brain.apply_owner_correction(
        inferred.id,
        summary="Prefers brief messages only for logistics.",
        payload={"scope_note": "logistics only"},
        corrected_at=NOW + timedelta(minutes=5),
        reason="The original inference was too broad.",
    )

    old = brain.item_by_id(inferred.id)
    assert old.status is StateStatus.SUPERSEDED
    assert corrected.supersedes == inferred.id
    assert corrected.epistemic_status is EpistemicStatus.OWNER_DECLARED
    assert corrected.confidence == 1.0
    assert corrected.metadata["correction_reason"] == (
        "The original inference was too broad."
    )
    assert {item.id for item in brain.query(as_of=NOW + timedelta(hours=1))} == {
        corrected.id
    }
    assert len(brain.items) == 2
    assert DigitalBrain.from_json(brain.to_json()) == brain


def test_failed_updates_leave_canonical_state_unchanged():
    original = _item(
        BrainLayer.SEMANTIC,
        "owner.fact",
        "The owner lives in one city.",
    )
    brain = DigitalBrain.create(
        owner_id="owner:test",
        owner_name="Test Owner",
        created_at=NOW,
        items=[original],
    )
    before = brain.to_json()

    with pytest.raises(ValueError, match="kind and summary"):
        brain.apply_owner_correction(
            original.id,
            summary=" ",
            payload={},
            corrected_at=NOW + timedelta(minutes=1),
            reason="Invalid empty correction.",
        )

    invalid_link = StateItem.create(
        layer=BrainLayer.UNCERTAINTY,
        kind="conflict",
        summary="References missing state.",
        payload={},
        epistemic_status=EpistemicStatus.INFERRED,
        confidence=0.5,
        temporal=TemporalScope(observed_at=NOW, recorded_at=NOW),
        scope=ContextScope(),
        sensitivity=Sensitivity.PRIVATE,
        ownership=Ownership.SYSTEM,
        provenance=_provenance("evidence:invalid-link"),
        support_ids=("brain-item:" + "f" * 64,),
    )
    with pytest.raises(ValueError, match="unknown items"):
        brain.add_item(invalid_link, updated_at=NOW + timedelta(minutes=1))

    assert brain.to_json() == before


def test_validation_blocks_protected_inference_and_third_party_self_promotion():
    with pytest.raises(ValueError, match="protected trait"):
        _item(
            BrainLayer.SELF_SCHEMA,
            "religion",
            "The owner has a particular religion.",
            epistemic_status=EpistemicStatus.INFERRED,
        )

    with pytest.raises(ValueError, match="third-party"):
        _item(
            BrainLayer.VALUES_GOALS,
            "value",
            "A contact values status.",
            ownership=Ownership.THIRD_PARTY,
        )


def test_migration_from_v1_preserves_lineage_scope_and_owner_authority():
    owner_message = EvidenceRecord.create(
        source_type="fixture_owner_message",
        source_record_id="message:1",
        observed_at=NOW,
        content_hash="b" * 64,
        metadata={"identity_target": True, "relationship_id": "relationship:a"},
    )
    interview = EvidenceRecord.create(
        source_type="owner_interview",
        source_record_id="interview:values",
        observed_at=NOW + timedelta(minutes=1),
        content="Curiosity matters to me.",
        metadata={"sensitivity": "private"},
    )
    inferred = IdentityClaim.create(
        dimension="communication.message_length",
        statement="Observed messages are usually brief.",
        status=ClaimStatus.CANDIDATE,
        confidence=0.7,
        provenance=ProvenanceType.BEHAVIORAL_INFERENCE,
        created_at=NOW,
        evidence_ids=[owner_message.id],
        relationship_id="relationship:a",
    )
    declared = IdentityClaim.create(
        dimension="values.curiosity",
        statement="Curiosity matters to me.",
        status=ClaimStatus.CONFIRMED,
        confidence=1.0,
        provenance=ProvenanceType.OWNER_INTERVIEW,
        created_at=NOW + timedelta(minutes=1),
        evidence_ids=[interview.id],
    )
    profile = DigitalSelfProfile(
        profile_id="digital-self:test",
        owner_id="owner:test",
        owner_name="Test Owner",
        created_at=NOW,
        updated_at=NOW + timedelta(minutes=1),
        evidence=[owner_message, interview],
        claims=[inferred, declared],
        relationships=[
            RelationshipProfile(
                id="relationship:a",
                label="Friend A",
                evidence_ids=[owner_message.id],
                claim_ids=[inferred.id],
                style_delta={"message_length": "brief"},
            )
        ],
        communication_style={"global": {"message_length": "medium"}},
        source_summary={"message_count": 1},
    )
    profile.validate()
    original = deepcopy(profile)

    brain = migrate_v1_profile(profile)

    assert profile == original
    assert brain.schema_version == "digital_brain.v2"
    assert brain.metadata["migrated_from"] == "digital_self.v1"
    assert {item.layer for item in brain.items} >= {
        BrainLayer.EVENT,
        BrainLayer.COMMUNICATION,
        BrainLayer.VALUES_GOALS,
        BrainLayer.SOCIAL,
    }
    value_item = next(
        item for item in brain.items if item.kind == "values.curiosity"
    )
    inferred_item = next(
        item for item in brain.items if item.kind == "communication.message_length"
    )
    assert value_item.epistemic_status is EpistemicStatus.OWNER_DECLARED
    assert inferred_item.epistemic_status is EpistemicStatus.INFERRED
    assert inferred_item.scope.relationship_id == "relationship:a"
    assert DigitalBrain.from_json(brain.to_json()) == brain
    assert migrate_v1_profile(profile).to_json() == brain.to_json()
