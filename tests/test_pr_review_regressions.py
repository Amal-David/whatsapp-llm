from datetime import datetime, timezone

from living_brain.core.config import Config, FactConfig, MemoryConfig, TrainingConfig
from living_brain.inference.orchestrator import Orchestrator
from living_brain.memory.fact_store import FactStore
from living_brain.memory.vector_store import VectorStore


def test_fact_extraction_prefers_specific_patterns_and_skips_overlaps(tmp_path):
    store = FactStore(str(tmp_path / "facts.json"))

    cases = {
        "I'm from NYC.": ["I am from nyc"],
        "I am 25 years old.": ["I am years old 25"],
        "My name is Amal.": ["I am named amal"],
        "I'm a developer.": ["I am developer"],
    }

    for text, expected in cases.items():
        got = [fact.to_natural() for fact in store.extract_facts(text)]
        assert got == expected


def test_generic_am_fact_does_not_supersede_specific_am_predicates(tmp_path):
    store = FactStore(str(tmp_path / "facts.json"))

    store.add("I", "am named", "Amal")
    store.add("I", "am from", "NYC")
    store.add("I", "am years old", "25")
    store.add("I", "am", "happy")

    active = sorted(fact.to_natural() for fact in store.get_all_active())
    assert active == [
        "I am from NYC",
        "I am happy",
        "I am named Amal",
        "I am years old 25",
    ]

    store.add("I", "am from", "Kochi")

    active = sorted(fact.to_natural() for fact in store.get_all_active())
    assert active == [
        "I am from Kochi",
        "I am happy",
        "I am named Amal",
        "I am years old 25",
    ]


def test_vector_add_batch_pairs_new_ids_with_new_documents_when_duplicates_appear_first():
    store = object.__new__(VectorStore)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    duplicate_id = store._generate_id("duplicate", timestamp)

    class FakeCollection:
        def __init__(self, existing_ids):
            self.existing_ids = set(existing_ids)
            self.added = None

        def get(self, ids):
            return {"ids": [entry_id for entry_id in ids if entry_id in self.existing_ids]}

        def add(self, ids, embeddings, documents, metadatas):
            self.added = {
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
            }

    fake_collection = FakeCollection({duplicate_id})
    store._collection = fake_collection
    store.embed = lambda contents: [[float(index)] for index, _ in enumerate(contents)]

    ids = store.add_batch([
        ("duplicate", timestamp, {"kind": "existing"}),
        ("new one", timestamp, {"kind": "new"}),
        ("new two", timestamp, {"kind": "new"}),
    ])

    expected_new_ids = [
        store._generate_id("new one", timestamp),
        store._generate_id("new two", timestamp),
    ]

    assert ids == [duplicate_id, *expected_new_ids]
    assert fake_collection.added["ids"] == expected_new_ids
    assert fake_collection.added["documents"] == ["new one", "new two"]


def test_vector_store_default_write_timestamps_are_utc_aware():
    store = object.__new__(VectorStore)

    class FakeCollection:
        def __init__(self):
            self.added = None

        def get(self, ids):
            return {"ids": []}

        def add(self, ids, embeddings, documents, metadatas):
            self.added = metadatas

    collection = FakeCollection()
    store._collection = collection
    store.embed = lambda contents: [[0.0] for _ in contents]

    store.add("default timestamp")

    added_at = datetime.fromisoformat(collection.added[0]["timestamp"])
    assert added_at.utcoffset() == timezone.utc.utcoffset(added_at)

    captured_entries = []

    def capture_batch(entries):
        captured_entries.extend(entries)
        return ["chunk"] * len(entries)

    store.add_batch = capture_batch
    store.add_chunked("one two three", chunk_size=2, chunk_overlap=0)

    assert captured_entries
    assert all(timestamp.utcoffset() == timezone.utc.utcoffset(timestamp) for _, timestamp, _ in captured_entries)


def test_orchestrator_defers_memory_store_initialization(tmp_path):
    config = Config(
        data_dir=str(tmp_path / "data"),
        memory=MemoryConfig(chroma_persist_dir=str(tmp_path / "chroma")),
        facts=FactConfig(facts_path=str(tmp_path / "facts.json")),
        training=TrainingConfig(output_dir=str(tmp_path / "models" / "lora_adapter")),
    )

    orchestrator = Orchestrator(config=config, use_gguf=True)

    assert orchestrator.vector_store is None
    assert orchestrator.fact_store is None
    assert orchestrator.retriever is None
