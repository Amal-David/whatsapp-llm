import json
from datetime import datetime, timedelta, timezone

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
from living_brain.brain.simulation import (
    CandidateResponse,
    LLMDeliberationProvider,
    RequestedAuthority,
    SimulationEngine,
    Situation,
    Stakes,
)

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _item(
    layer,
    kind,
    summary,
    *,
    relationship_id=None,
    confidence=0.9,
    sensitivity=Sensitivity.PRIVATE,
    ownership=Ownership.OWNER,
    contradiction_ids=(),
    evidence_source_id=None,
):
    return StateItem.create(
        layer=layer,
        kind=kind,
        summary=summary,
        payload={},
        epistemic_status=EpistemicStatus.OWNER_DECLARED,
        confidence=confidence,
        temporal=TemporalScope(
            observed_at=NOW - timedelta(days=1),
            recorded_at=NOW - timedelta(days=1),
            valid_from=NOW - timedelta(days=1),
        ),
        scope=ContextScope(relationship_id=relationship_id),
        sensitivity=sensitivity,
        ownership=ownership,
        provenance=(
            ProvenanceRef(
                source_id=(
                    evidence_source_id
                    or f"evidence:{kind}:{relationship_id or 'global'}"
                ),
                source_type="fixture",
                relation=ProvenanceRelation.SUPPORTS,
                observed_at=NOW - timedelta(days=1),
            ),
        ),
        contradiction_ids=contradiction_ids,
    )


def _brain(*items):
    return DigitalBrain.create(
        owner_id="owner:test",
        owner_name="Test Owner",
        created_at=NOW - timedelta(days=2),
        updated_at=NOW - timedelta(days=1),
        items=list(items),
    )


def _situation(
    *,
    relationship_id="relationship:a",
    stakes=Stakes.LOW,
    authority=RequestedAuthority.PRIVATE_DRAFT,
):
    return Situation.create(
        request="Draft a warm reply about the project.",
        audience="Friend A",
        relationship_id=relationship_id,
        role="friend",
        intent="reply warmly without making a commitment",
        stakes=stakes,
        target_time=NOW,
        channel="whatsapp",
        authority=authority,
        assumptions=("The owner has not replied yet.",),
    )


class _CapturingProvider:
    provider_id = "fixture-provider.v1"

    def __init__(self, *, unknown_state=False, requires_authority=False):
        self.calls = 0
        self.context = None
        self.unknown_state = unknown_state
        self.requires_authority = requires_authority

    def deliberate(self, situation, context):
        self.calls += 1
        self.context = context
        state_ids = context.state_item_ids
        used = ("brain-item:" + "f" * 64,) if self.unknown_state else state_ids[:2]
        return [
            CandidateResponse.create(
                text="That sounds exciting. Tell me how it goes.",
                rationale="Warm, interested, and avoids a commitment.",
                used_state_ids=used,
                assumptions=("Friend A welcomes an informal tone.",),
                confidence=0.85,
                requires_live_authority=self.requires_authority,
            ),
            CandidateResponse.create(
                text="Nice. Keep me posted.",
                rationale="A shorter relationship-consistent alternative.",
                used_state_ids=state_ids[:1],
                assumptions=(),
                confidence=0.7,
            ),
        ]


def test_simulation_builds_explicit_situation_and_isolates_relationship_context():
    episode = _item(
        BrainLayer.EPISODE,
        "episode.project",
        "Friend A recently asked about the project.",
        relationship_id="relationship:a",
    )
    self_knowledge = _item(
        BrainLayer.SEMANTIC,
        "owner.preference",
        "The owner avoids overpromising.",
    )
    values = _item(
        BrainLayer.VALUES_GOALS,
        "value.reliability",
        "Reliability matters to the owner.",
    )
    relationship_a = _item(
        BrainLayer.SOCIAL,
        "relationship.profile",
        "Friend A appreciates warmth.",
        relationship_id="relationship:a",
    )
    relationship_b = _item(
        BrainLayer.SOCIAL,
        "relationship.profile",
        "Colleague B expects formal detail.",
        relationship_id="relationship:b",
        sensitivity=Sensitivity.RESTRICTED,
    )
    affect = _item(
        BrainLayer.AFFECT,
        "affect.tendency",
        "The owner is upbeat about this project.",
    )
    communication = _item(
        BrainLayer.COMMUNICATION,
        "tone",
        "The owner is warm and concise with Friend A.",
        relationship_id="relationship:a",
    )
    provider = _CapturingProvider()
    brain = _brain(
        episode,
        self_knowledge,
        values,
        relationship_a,
        relationship_b,
        affect,
        communication,
    )

    result = SimulationEngine(provider, minimum_grounded_items=2).simulate(
        brain,
        _situation(),
    )

    assert result.selected_response == "That sounds exciting. Tell me how it goes."
    assert len(result.alternatives) == 2
    assert result.situation.audience == "Friend A"
    assert result.situation.relationship_id == "relationship:a"
    assert result.situation.intent
    assert result.situation.channel == "whatsapp"
    assert result.situation.authority is RequestedAuthority.PRIVATE_DRAFT
    assert result.context.episodes
    assert result.context.self_knowledge
    assert result.context.values_goals
    assert result.context.relationship_state
    assert result.context.affect
    assert result.context.communication
    assert relationship_b.id not in result.context.state_item_ids
    assert relationship_a.id in result.context.state_item_ids
    assert result.evidence_ids
    assert result.assumptions
    assert result.abstention_reasons == ()
    assert result.authority_granted is False
    assert result.synthetic_disclosure


def test_third_party_social_state_never_reaches_deliberation_context():
    owner_fact = _item(
        BrainLayer.SEMANTIC,
        "owner.fact",
        "The owner prefers a careful response.",
    )
    owner_value = _item(
        BrainLayer.VALUES_GOALS,
        "owner.value",
        "The owner values privacy.",
    )
    third_party = _item(
        BrainLayer.SOCIAL,
        "contact.secret",
        "PRIVATE THIRD PARTY SECRET",
        relationship_id="relationship:a",
        sensitivity=Sensitivity.PRIVATE,
        ownership=Ownership.THIRD_PARTY,
    )
    sensitive_owner_state = _item(
        BrainLayer.SEMANTIC,
        "owner.sensitive",
        "SENSITIVE OWNER SECRET",
        sensitivity=Sensitivity.SENSITIVE,
    )
    shared_state = _item(
        BrainLayer.SOCIAL,
        "relationship.shared",
        "SHARED THIRD PARTY SECRET",
        relationship_id="relationship:a",
        ownership=Ownership.SHARED,
    )
    provider = _CapturingProvider()

    result = SimulationEngine(provider).simulate(
        _brain(
            owner_fact,
            owner_value,
            third_party,
            sensitive_owner_state,
            shared_state,
        ),
        _situation(),
    )

    assert provider.context is not None
    assert third_party.id not in provider.context.state_item_ids
    assert sensitive_owner_state.id not in provider.context.state_item_ids
    assert shared_state.id not in provider.context.state_item_ids
    assert third_party.id not in result.state_item_ids
    assert "PRIVATE THIRD PARTY SECRET" not in result.to_json()
    assert "SENSITIVE OWNER SECRET" not in result.to_json()
    assert "SHARED THIRD PARTY SECRET" not in result.to_json()


def test_high_stakes_and_live_authority_requests_stop_before_generation():
    provider = _CapturingProvider()
    brain = _brain(
        _item(BrainLayer.SEMANTIC, "owner.fact", "The owner has relevant context."),
        _item(BrainLayer.VALUES_GOALS, "value", "The owner values care."),
    )

    high_stakes = SimulationEngine(provider).simulate(
        brain,
        _situation(stakes=Stakes.HIGH),
    )
    live_action = SimulationEngine(provider).simulate(
        brain,
        _situation(authority=RequestedAuthority.SEND_MESSAGE),
    )

    assert provider.calls == 0
    assert high_stakes.selected_response is None
    assert "high_stakes_owner_required" in high_stakes.abstention_reasons
    assert high_stakes.owner_question
    assert live_action.selected_response is None
    assert "live_authority_not_permitted" in live_action.abstention_reasons
    assert high_stakes.authority_granted is False
    assert live_action.authority_granted is False


def test_under_supported_or_ungrounded_generation_abstains():
    sparse_provider = _CapturingProvider()
    sparse_brain = _brain(
        _item(BrainLayer.SEMANTIC, "owner.fact", "Only one relevant fact."),
    )
    sparse = SimulationEngine(
        sparse_provider,
        minimum_grounded_items=2,
    ).simulate(sparse_brain, _situation())

    assert sparse_provider.calls == 0
    assert sparse.selected_response is None
    assert "insufficient_grounding" in sparse.abstention_reasons
    assert sparse.owner_question

    ungrounded_provider = _CapturingProvider(unknown_state=True)
    grounded_brain = _brain(
        _item(BrainLayer.SEMANTIC, "owner.fact", "One supported fact."),
        _item(BrainLayer.VALUES_GOALS, "value", "One supported value."),
    )
    ungrounded = SimulationEngine(
        ungrounded_provider,
        minimum_grounded_items=2,
    ).simulate(grounded_brain, _situation())

    assert ungrounded.selected_response == "Nice. Keep me posted."
    assert "ungrounded_candidate_rejected" in ungrounded.abstention_reasons
    assert len(ungrounded.alternatives) == 1


def test_conflicts_are_exposed_and_live_authority_candidates_are_rejected():
    first = _item(
        BrainLayer.SELF_SCHEMA,
        "role",
        "The owner identifies as a manager.",
    )
    second = _item(
        BrainLayer.SELF_SCHEMA,
        "role",
        "The owner rejects the manager label.",
        contradiction_ids=(first.id,),
    )
    value = _item(
        BrainLayer.VALUES_GOALS,
        "value",
        "The owner values clarity.",
    )
    provider = _CapturingProvider(requires_authority=True)

    result = SimulationEngine(provider, minimum_grounded_items=2).simulate(
        _brain(first, second, value),
        _situation(relationship_id=None),
    )

    assert set(result.conflict_item_ids) >= {first.id, second.id}
    assert "live_authority_candidate_rejected" in result.abstention_reasons
    assert result.selected_response == "Nice. Keep me posted."
    assert all(not candidate.requires_live_authority for candidate in result.alternatives)


def test_supporting_evidence_links_do_not_create_false_conflicts():
    supported = _item(
        BrainLayer.SEMANTIC,
        "owner.fact",
        "The owner has a project review tomorrow.",
    )
    value = _item(
        BrainLayer.VALUES_GOALS,
        "value",
        "The owner values preparation.",
        evidence_source_id=supported.id,
    )

    result = SimulationEngine(
        _CapturingProvider(),
        minimum_grounded_items=2,
    ).simulate(_brain(supported, value), _situation(relationship_id=None))

    assert result.conflict_item_ids == ()


class _CompletionProvider:
    provider_id = "fixture-completion.v1"

    def __init__(self, state_id):
        self.state_id = state_id
        self.request = None

    def complete(self, request):
        self.request = json.loads(request)
        return json.dumps(
            {
                "alternatives": [
                    {
                        "text": "A grounded optional-provider reply.",
                        "rationale": "Uses one supplied state item.",
                        "used_state_ids": [self.state_id],
                        "assumptions": [],
                        "confidence": 0.8,
                        "requires_live_authority": False,
                    }
                ]
            }
        )


def test_optional_llm_provider_uses_a_structured_json_contract():
    fact = _item(
        BrainLayer.SEMANTIC,
        "owner.fact",
        "The owner prefers a concise reply.",
    )
    value = _item(
        BrainLayer.VALUES_GOALS,
        "value",
        "The owner values honesty.",
    )
    completion = _CompletionProvider(fact.id)
    provider = LLMDeliberationProvider(completion)

    result = SimulationEngine(provider, minimum_grounded_items=2).simulate(
        _brain(fact, value),
        _situation(relationship_id=None),
    )

    assert result.selected_response == "A grounded optional-provider reply."
    assert completion.request["schema_version"] == "digital_brain_deliberation.v1"
    assert completion.request["situation"]["authority"] == "private_draft"
    assert set(completion.request["context"]["state_item_ids"]) == {fact.id, value.id}
