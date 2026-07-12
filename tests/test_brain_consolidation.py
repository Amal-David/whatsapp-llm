from datetime import datetime, timedelta, timezone

import pytest

from living_brain.brain.consolidation import (
    ConsolidationEngine,
    ConsolidationProposal,
    StructuredEventPolicy,
    UpdateAction,
    rank_for_retrieval,
)
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

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _source_event(
    event_id,
    *,
    observed_at=NOW,
    ownership=Ownership.OWNER,
    payload=None,
):
    return StateItem.create(
        layer=BrainLayer.EVENT,
        kind="source.message",
        summary=f"Source event {event_id}",
        payload=payload or {},
        epistemic_status=EpistemicStatus.OBSERVED,
        confidence=1.0,
        temporal=TemporalScope(
            observed_at=observed_at,
            recorded_at=observed_at,
            valid_from=observed_at,
        ),
        scope=ContextScope(),
        sensitivity=Sensitivity.PRIVATE,
        ownership=ownership,
        provenance=(
            ProvenanceRef(
                source_id=event_id,
                source_type="fixture",
                relation=ProvenanceRelation.OBSERVES,
                observed_at=observed_at,
                content_hash="a" * 64,
            ),
        ),
    )


def _state(layer, kind, summary, *, observed_at=NOW, relationship_id=None):
    return StateItem.create(
        layer=layer,
        kind=kind,
        summary=summary,
        payload={},
        epistemic_status=EpistemicStatus.INFERRED,
        confidence=0.6,
        temporal=TemporalScope(
            observed_at=observed_at,
            recorded_at=observed_at,
            valid_from=observed_at,
        ),
        scope=ContextScope(relationship_id=relationship_id),
        sensitivity=Sensitivity.PRIVATE,
        ownership=Ownership.OWNER,
        provenance=(
            ProvenanceRef(
                source_id=f"fixture:{kind}",
                source_type="fixture",
                relation=ProvenanceRelation.SUPPORTS,
                observed_at=observed_at,
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


def test_structured_policy_creates_episode_semantic_state_and_auditable_reflections():
    event = _source_event(
        "message:structured",
        payload={
            "consolidation_proposals": [
                {
                    "action": "create",
                    "layer": "episode",
                    "kind": "episode.project_launch",
                    "summary": "The owner launched a project.",
                    "payload": {"project": "fixture"},
                    "confidence": 0.85,
                    "rationale": "The owner explicitly described the launch.",
                    "uncertainty": "The exact launch time is approximate.",
                    "deliberately_not_inferred": ["success", "motive"],
                },
                {
                    "action": "create",
                    "layer": "semantic",
                    "kind": "owner.project",
                    "summary": "The owner has worked on the fixture project.",
                    "payload": {"project": "fixture"},
                    "confidence": 0.8,
                    "rationale": "The source directly names the project.",
                    "uncertainty": "Current involvement is unknown.",
                    "deliberately_not_inferred": ["current commitment"],
                },
            ]
        },
    )
    brain = _brain(event)

    result = ConsolidationEngine(StructuredEventPolicy()).run(
        brain,
        source_item_ids=[event.id],
        as_of=NOW + timedelta(minutes=1),
    )

    assert len(result.added_item_ids) == 2
    assert len(result.reflection_item_ids) == 2
    assert {brain.item_by_id(item_id).layer for item_id in result.added_item_ids} == {
        BrainLayer.EPISODE,
        BrainLayer.SEMANTIC,
    }
    for reflection_id in result.reflection_item_ids:
        reflection = brain.item_by_id(reflection_id)
        assert reflection.layer is BrainLayer.REFLECTION
        assert reflection.payload["source_item_ids"] == [event.id]
        assert reflection.payload["rationale"]
        assert reflection.payload["uncertainty"]
        assert reflection.payload["deliberately_not_inferred"]
        assert reflection.epistemic_status is EpistemicStatus.GENERATED_PROPOSAL


class _ActionPolicy:
    policy_id = "fixture-actions.v1"

    def __init__(self, proposals):
        self.proposals = proposals

    def propose(self, brain, source_items, as_of):
        return list(reversed(self.proposals))


def test_updates_strengthen_contextualize_supersede_and_conflict_without_erasure():
    event = _source_event("message:updates")
    strength_target = _state(
        BrainLayer.SEMANTIC,
        "owner.project",
        "The owner works on project A.",
    )
    context_target = _state(
        BrainLayer.COMMUNICATION,
        "tone",
        "The owner is playful.",
    )
    supersede_target = _state(
        BrainLayer.VALUES_GOALS,
        "goal.active",
        "The owner plans to finish project B.",
    )
    conflict_target = _state(
        BrainLayer.SELF_SCHEMA,
        "role",
        "The owner identifies as a manager.",
    )
    brain = _brain(
        event,
        strength_target,
        context_target,
        supersede_target,
        conflict_target,
    )
    proposals = [
        ConsolidationProposal.create(
            action=UpdateAction.STRENGTHEN,
            layer=BrainLayer.SEMANTIC,
            kind=strength_target.kind,
            summary=strength_target.summary,
            payload={"corroborated": True},
            confidence=0.9,
            source_item_ids=(event.id,),
            target_item_id=strength_target.id,
            rationale="A second direct source corroborates the project.",
            uncertainty="The current time allocation is unknown.",
            deliberately_not_inferred=("priority",),
        ),
        ConsolidationProposal.create(
            action=UpdateAction.CONTEXTUALIZE,
            layer=BrainLayer.COMMUNICATION,
            kind=context_target.kind,
            summary="The owner is playful with friend A.",
            payload={"relationship_id": "relationship:a"},
            confidence=0.8,
            source_item_ids=(event.id,),
            target_item_id=context_target.id,
            scope=ContextScope(relationship_id="relationship:a"),
            rationale="The new source limits the pattern to one relationship.",
            uncertainty="Other relationships remain unknown.",
            deliberately_not_inferred=("global trait",),
        ),
        ConsolidationProposal.create(
            action=UpdateAction.SUPERSEDE,
            layer=BrainLayer.VALUES_GOALS,
            kind=supersede_target.kind,
            summary="The owner paused project B.",
            payload={"status": "paused"},
            confidence=0.95,
            source_item_ids=(event.id,),
            target_item_id=supersede_target.id,
            rationale="The owner explicitly paused the goal.",
            uncertainty="A restart date is unknown.",
            deliberately_not_inferred=("abandoned forever",),
        ),
        ConsolidationProposal.create(
            action=UpdateAction.CONFLICT,
            layer=BrainLayer.SELF_SCHEMA,
            kind=conflict_target.kind,
            summary="The owner rejects being described as a manager.",
            payload={"owner_rejection": True},
            confidence=0.75,
            source_item_ids=(event.id,),
            target_item_id=conflict_target.id,
            rationale="A new source conflicts with the existing role claim.",
            uncertainty="The role may be context-specific.",
            deliberately_not_inferred=("which statement is globally true",),
        ),
    ]

    result = ConsolidationEngine(_ActionPolicy(proposals)).run(
        brain,
        source_item_ids=[event.id],
        as_of=NOW + timedelta(hours=1),
    )

    assert len(result.added_item_ids) == 4
    assert len(result.reflection_item_ids) == 4
    assert brain.item_by_id(strength_target.id).status is StateStatus.SUPERSEDED
    assert brain.item_by_id(context_target.id).status is StateStatus.SUPERSEDED
    assert brain.item_by_id(supersede_target.id).status is StateStatus.SUPERSEDED
    assert brain.item_by_id(conflict_target.id).status is StateStatus.ACTIVE
    changed = [brain.item_by_id(item_id) for item_id in result.added_item_ids]
    assert any(item.scope.relationship_id == "relationship:a" for item in changed)
    conflict = next(item for item in changed if item.kind == conflict_target.kind)
    assert conflict.contradiction_ids == (conflict_target.id,)
    assert conflict_target.id in {
        item.id for item in brain.query(as_of=NOW + timedelta(hours=2))
    }
    assert conflict.id in {
        item.id for item in brain.query(as_of=NOW + timedelta(hours=2))
    }


def test_protected_and_third_party_promotions_fail_transactionally():
    third_party_event = _source_event(
        "message:third-party",
        ownership=Ownership.THIRD_PARTY,
    )
    brain = _brain(third_party_event)
    before = brain.to_json()
    third_party_proposal = ConsolidationProposal.create(
        action=UpdateAction.CREATE,
        layer=BrainLayer.SELF_SCHEMA,
        kind="role",
        summary="A contact has a stable role.",
        payload={},
        confidence=0.8,
        source_item_ids=(third_party_event.id,),
        rationale="Unsafe fixture proposal.",
        uncertainty="The contact did not consent.",
        deliberately_not_inferred=("owner identity",),
    )

    with pytest.raises(ValueError, match="third-party"):
        ConsolidationEngine(_ActionPolicy([third_party_proposal])).run(
            brain,
            source_item_ids=[third_party_event.id],
            as_of=NOW + timedelta(minutes=1),
        )
    assert brain.to_json() == before

    misowned_social = ConsolidationProposal.create(
        action=UpdateAction.CREATE,
        layer=BrainLayer.SOCIAL,
        kind="relationship.context",
        summary="Context supplied by another person.",
        payload={},
        confidence=0.8,
        source_item_ids=(third_party_event.id,),
        ownership=Ownership.OWNER,
        rationale="The context may be useful but is not owner identity.",
        uncertainty="Third-party consent is unknown.",
        deliberately_not_inferred=("contact traits",),
    )
    with pytest.raises(ValueError, match="shared or third-party"):
        ConsolidationEngine(_ActionPolicy([misowned_social])).run(
            brain,
            source_item_ids=[third_party_event.id],
            as_of=NOW + timedelta(minutes=1),
        )
    assert brain.to_json() == before

    owner_event = _source_event("message:protected")
    owner_brain = _brain(owner_event)
    owner_before = owner_brain.to_json()
    protected_proposal = ConsolidationProposal.create(
        action=UpdateAction.CREATE,
        layer=BrainLayer.SELF_SCHEMA,
        kind="religion",
        summary="The owner has a particular religion.",
        payload={},
        confidence=0.8,
        source_item_ids=(owner_event.id,),
        rationale="Unsafe protected inference.",
        uncertainty="No owner declaration exists.",
        deliberately_not_inferred=(),
    )
    with pytest.raises(ValueError, match="protected trait"):
        ConsolidationEngine(_ActionPolicy([protected_proposal])).run(
            owner_brain,
            source_item_ids=[owner_event.id],
            as_of=NOW + timedelta(minutes=1),
        )
    assert owner_brain.to_json() == owner_before


def test_decay_changes_retrieval_priority_without_mutating_or_erasing_audit_state():
    old = _state(
        BrainLayer.SEMANTIC,
        "owner.interest",
        "An old interest.",
        observed_at=NOW - timedelta(days=120),
    )
    recent = _state(
        BrainLayer.SEMANTIC,
        "owner.interest",
        "A recent interest.",
        observed_at=NOW - timedelta(days=2),
    )
    event = _source_event(
        "message:audit",
        observed_at=NOW - timedelta(days=365),
    )
    brain = _brain(old, recent, event)
    before = brain.to_json()

    ranked = rank_for_retrieval(brain.items, as_of=NOW, half_life_days=30)

    by_id = {item.item_id: item for item in ranked}
    assert by_id[recent.id].score > by_id[old.id].score
    assert by_id[event.id].protected_audit is True
    assert brain.item_by_id(event.id) == event
    assert brain.to_json() == before


def test_provider_policy_replay_is_deterministic_and_idempotent():
    event = _source_event("message:replay")
    proposal = ConsolidationProposal.create(
        action=UpdateAction.CREATE,
        layer=BrainLayer.EPISODE,
        kind="episode.replay",
        summary="A deterministic episode.",
        payload={"sequence": 1},
        confidence=0.8,
        source_item_ids=(event.id,),
        rationale="Fixture provider proposal.",
        uncertainty="Synthetic fixture.",
        deliberately_not_inferred=("motive",),
    )
    engine = ConsolidationEngine(_ActionPolicy([proposal]))
    first = _brain(event)
    second = DigitalBrain.from_json(first.to_json())

    first_result = engine.run(
        first,
        source_item_ids=[event.id],
        as_of=NOW + timedelta(minutes=1),
    )
    second_result = engine.run(
        second,
        source_item_ids=[event.id],
        as_of=NOW + timedelta(minutes=1),
    )

    assert first.to_json() == second.to_json()
    assert first_result == second_result
    before_replay = first.to_json()
    replay = engine.run(
        first,
        source_item_ids=[event.id],
        as_of=NOW + timedelta(minutes=1),
    )
    assert replay.added_item_ids == ()
    assert replay.reflection_item_ids == ()
    assert first.to_json() == before_replay
