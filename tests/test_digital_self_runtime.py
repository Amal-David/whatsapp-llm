from datetime import datetime, timezone

from living_brain.core.config import Config, FactConfig, MemoryConfig, TrainingConfig
from living_brain.identity.models import (
    ClaimStatus,
    DigitalSelfProfile,
    IdentityClaim,
    ProvenanceType,
)
from living_brain.identity.prompt import DigitalSelfPromptBuilder
from living_brain.inference.orchestrator import Orchestrator
from living_brain.memory.retriever import MemoryRetriever, RetrievalResult
from living_brain.memory.vector_store import VectorStore


def _at(month, day=1):
    return datetime(2026, month, day, 12, 0, tzinfo=timezone.utc)


def _claim(
    dimension,
    statement,
    *,
    status=ClaimStatus.CONFIRMED,
    provenance=ProvenanceType.OWNER_INTERVIEW,
    created_at=None,
    valid_from=None,
    valid_to=None,
    supersedes=None,
    relationship_id=None,
    sensitivity="public",
):
    return IdentityClaim.create(
        dimension=dimension,
        statement=statement,
        status=status,
        confidence=1.0 if status is ClaimStatus.CONFIRMED else 0.65,
        provenance=provenance,
        created_at=created_at or _at(1),
        valid_from=valid_from,
        valid_to=valid_to,
        supersedes=supersedes,
        relationship_id=relationship_id,
        sensitivity=sensitivity,
    )


def _profile():
    old_location = _claim(
        "preferences.location",
        "Preferred Bengaluru.",
        valid_from=_at(1),
        valid_to=_at(2),
    )
    current_location = _claim(
        "preferences.location",
        "Prefers Kochi.",
        created_at=_at(3),
        valid_from=_at(3),
        supersedes=old_location.id,
    )
    return DigitalSelfProfile(
        profile_id="digital-self:amal",
        owner_id="owner:amal",
        owner_name="Amal",
        created_at=_at(1),
        updated_at=_at(3),
        claims=[
            old_location,
            current_location,
            _claim(
                "communication.message_length",
                "Observed messages are usually brief.",
                status=ClaimStatus.CANDIDATE,
                provenance=ProvenanceType.BEHAVIORAL_INFERENCE,
            ),
            _claim("values.autonomy", "Values autonomy."),
            _claim("decisions.risk", "Tests risky ideas cheaply first."),
            _claim(
                "decisions.risk",
                "Sometimes accepts a large bet when timing matters.",
                created_at=_at(2),
            ),
            _claim(
                "communication.support",
                "Uses more reassurance with close friends.",
                relationship_id="relationship:friend",
            ),
        ],
    )


def test_prompt_distinguishes_confirmed_inferred_conflicting_and_unknown_information():
    prompt = DigitalSelfPromptBuilder(_profile()).render(
        as_of=_at(5),
        relationship_id="relationship:friend",
    )

    assert "<confirmed_profile>" in prompt
    assert "Values autonomy" in prompt
    assert "<inferred_candidates>" in prompt
    assert "Observed messages are usually brief" in prompt
    assert "<conflicts>" in prompt
    assert "decisions.risk" in prompt
    assert "<unknowns>" in prompt
    assert "Uses more reassurance with close friends" in prompt


def test_confirmed_self_report_does_not_become_an_equal_conflict_with_inference():
    profile = DigitalSelfProfile(
        profile_id="digital-self:amal",
        owner_id="owner:amal",
        owner_name="Amal",
        claims=[
            _claim(
                "communication.message_length",
                "Observed messages are usually brief.",
                status=ClaimStatus.CANDIDATE,
                provenance=ProvenanceType.BEHAVIORAL_INFERENCE,
            ),
            _claim(
                "communication.message_length",
                "I vary reply length based on the stakes.",
            ),
        ],
    )

    prompt = DigitalSelfPromptBuilder(profile).render(as_of=_at(5))
    confirmed = prompt.split("<confirmed_profile>", 1)[1].split(
        "</confirmed_profile>", 1
    )[0]
    inferred = prompt.split("<inferred_candidates>", 1)[1].split(
        "</inferred_candidates>", 1
    )[0]
    conflicts = prompt.split("<conflicts>", 1)[1].split("</conflicts>", 1)[0]

    assert "I vary reply length" in confirmed
    assert "Observed messages are usually brief" in inferred
    assert "communication.message_length" not in conflicts


def test_prompt_excludes_expired_and_superseded_claims_but_profile_keeps_them():
    profile = _profile()

    prompt = DigitalSelfPromptBuilder(profile).render(as_of=_at(5))

    assert "Prefers Kochi" in prompt
    assert "Preferred Bengaluru" not in prompt
    assert any(claim.statement == "Preferred Bengaluru." for claim in profile.claims)


def test_relationship_scoped_prompt_excludes_private_claims():
    profile = DigitalSelfProfile(
        profile_id="digital-self:amal",
        owner_id="owner:amal",
        owner_name="Amal",
        claims=[
            _claim("preferences.drink", "Likes tea."),
            _claim(
                "health.private",
                "PRIVATE OWNER HEALTH NOTE",
                sensitivity="private",
            ),
        ],
    )

    relationship_prompt = DigitalSelfPromptBuilder(profile).render(
        as_of=_at(5),
        relationship_id="relationship:friend",
    )
    private_prompt = DigitalSelfPromptBuilder(profile).render(as_of=_at(5))

    assert "Likes tea" in relationship_prompt
    assert "PRIVATE OWNER HEALTH NOTE" not in relationship_prompt
    assert "PRIVATE OWNER HEALTH NOTE" in private_prompt


def test_prompt_discloses_simulation_and_denies_live_authority():
    prompt = DigitalSelfPromptBuilder(_profile()).render(as_of=_at(5))

    assert "simulation of Amal" in prompt
    assert "not Amal" in prompt
    assert "Never send messages" in prompt
    assert "uncertain" in prompt.lower()


def test_orchestrator_loads_profile_without_a_style_adapter(tmp_path):
    profile_path = tmp_path / "digital-self.json"
    _profile().save(profile_path)
    config = Config(
        data_dir=str(tmp_path / "data"),
        memory=MemoryConfig(chroma_persist_dir=str(tmp_path / "chroma")),
        facts=FactConfig(facts_path=str(tmp_path / "facts.json")),
        training=TrainingConfig(output_dir=str(tmp_path / "models" / "adapter")),
    )

    orchestrator = Orchestrator(config=config, profile_path=profile_path)

    assert orchestrator.adapter_name is None
    assert orchestrator.digital_self_profile is not None
    assert "simulation of Amal" in orchestrator._get_system_prompt(as_of=_at(5))


def test_orchestrator_defaults_profile_retrieval_to_the_current_time(tmp_path):
    profile_path = tmp_path / "digital-self.json"
    _profile().save(profile_path)
    config = Config(
        data_dir=str(tmp_path / "data"),
        memory=MemoryConfig(chroma_persist_dir=str(tmp_path / "chroma")),
        facts=FactConfig(facts_path=str(tmp_path / "facts.json")),
        training=TrainingConfig(output_dir=str(tmp_path / "models" / "adapter")),
    )

    class FakeRetriever:
        def __init__(self):
            self.kwargs = None

        def retrieve(self, **kwargs):
            self.kwargs = kwargs
            return RetrievalResult(memories=[], facts=[], context_text="")

    retriever = FakeRetriever()
    orchestrator = Orchestrator(config=config, profile_path=profile_path)
    orchestrator._get_retriever = lambda: retriever
    orchestrator._generate_transformers = lambda *_args: ("draft", 1)

    before = datetime.now(timezone.utc)
    orchestrator.generate("What would I choose?", use_facts=False)
    after = datetime.now(timezone.utc)

    assert retriever.kwargs is not None
    assert before <= retriever.kwargs["as_of"] <= after


def test_memory_retriever_forwards_identity_and_time_filters():
    class FakeVectorStore:
        def __init__(self):
            self.search_kwargs = None

        def search(self, **kwargs):
            self.search_kwargs = kwargs
            return []

    class FakeFactStore:
        def get_all_active(self):
            return []

    vector_store = FakeVectorStore()
    retriever = MemoryRetriever(vector_store=vector_store, fact_store=FakeFactStore())

    retriever.retrieve(
        "What would I choose?",
        include_facts=False,
        owner_id="owner:amal",
        source="whatsapp_mac",
        relationship_id="relationship:friend",
        as_of=_at(5),
        after=_at(2),
    )

    assert vector_store.search_kwargs["filter_metadata"] == {
        "owner_id": "owner:amal",
        "source": "whatsapp_mac",
        "relationship_id": "relationship:friend",
    }
    assert vector_store.search_kwargs["before"] == _at(5)
    assert vector_store.search_kwargs["after"] == _at(2)


def test_vector_store_applies_metadata_and_temporal_filters_after_search():
    class FakeCollection:
        def __init__(self):
            self.query_kwargs = None

        def query(self, **kwargs):
            self.query_kwargs = kwargs
            return {
                "ids": [["old", "current", "future"]],
                "documents": [["old memory", "current memory", "future memory"]],
                "metadatas": [[
                    {"timestamp": _at(1).isoformat(), "owner_id": "owner:amal"},
                    {"timestamp": _at(3).isoformat(), "owner_id": "owner:amal"},
                    {"timestamp": _at(6).isoformat(), "owner_id": "owner:amal"},
                ]],
                "distances": [[0.1, 0.1, 0.1]],
            }

    store = object.__new__(VectorStore)
    store._collection = FakeCollection()
    store.embed = lambda texts: [[0.1, 0.2]]

    results = store.search(
        "choice",
        top_k=2,
        filter_metadata={"owner_id": "owner:amal"},
        after=_at(2),
        before=_at(5),
    )

    assert [entry.id for entry, _score in results] == ["current"]
    assert store._collection.query_kwargs["where"] == {"owner_id": "owner:amal"}
