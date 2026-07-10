"""
Vector store for episodic memory using ChromaDB.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class VectorStore:
    """
    Vector store for episodic memories using ChromaDB.

    Stores conversation episodes as embeddings for retrieval-augmented generation.
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "conversations",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required for VectorStore. "
                "Install with: pip install chromadb"
            )

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize embedding model
        self._embedding_model = None
        self._embedding_model_name = embedding_model

        logger.info(
            f"VectorStore initialized: {self._collection.count()} memories loaded"
        )

    def _get_embedding_model(self) -> "SentenceTransformer":
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is required for embeddings. "
                    "Install with: pip install sentence-transformers"
                )
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _generate_id(self, content: str, timestamp: datetime) -> str:
        """Generate a unique ID for a memory entry."""
        unique_str = f"{content}_{timestamp.isoformat()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]

    @staticmethod
    def chunk_text(content: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
        """Split text into overlapping word chunks."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

        words = content.split()
        if not words:
            return []

        step = chunk_size - chunk_overlap
        chunks = []
        start = 0
        while start < len(words):
            chunks.append(" ".join(words[start : start + chunk_size]))
            if start + chunk_size >= len(words):
                break
            start += step
        return chunks

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        model = self._get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return cast(list[list[float]], embeddings.tolist())

    def add(
        self,
        content: str,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Add a memory entry to the store.

        Args:
            content: The text content to store
            timestamp: When this memory was created
            metadata: Additional metadata (participants, topic, etc.)

        Returns:
            The ID of the added entry
        """
        timestamp = timestamp or datetime.now()
        metadata = metadata or {}

        entry_id = self._generate_id(content, timestamp)

        # Check if already exists
        existing = self._collection.get(ids=[entry_id])
        if existing["ids"]:
            logger.debug(f"Memory {entry_id} already exists, skipping")
            return entry_id

        # Generate embedding
        embedding = self.embed([content])[0]

        # Prepare metadata for ChromaDB (must be flat)
        chroma_metadata = {
            "timestamp": timestamp.isoformat(),
            "content_preview": content[:200],
            **{k: str(v) for k, v in metadata.items() if v is not None},
        }

        self._collection.add(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[chroma_metadata],
        )

        logger.debug(f"Added memory {entry_id}: {content[:50]}...")
        return entry_id

    def add_batch(
        self,
        entries: list[tuple[str, datetime, dict]],
    ) -> list[str]:
        """
        Add multiple memory entries efficiently.

        Args:
            entries: List of (content, timestamp, metadata) tuples

        Returns:
            List of entry IDs
        """
        if not entries:
            return []

        ids = []
        new_ids = []
        contents = []
        metadatas = []
        timestamps = []

        for content, timestamp, metadata in entries:
            entry_id = self._generate_id(content, timestamp)

            # Skip duplicates
            existing = self._collection.get(ids=[entry_id])
            if existing["ids"]:
                ids.append(entry_id)
                continue

            ids.append(entry_id)
            new_ids.append(entry_id)
            contents.append(content)
            timestamps.append(timestamp)
            metadatas.append({
                "timestamp": timestamp.isoformat(),
                "content_preview": content[:200],
                **{k: str(v) for k, v in metadata.items() if v is not None},
            })

        if contents:
            embeddings = self.embed(contents)

            self._collection.add(
                ids=new_ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )

            logger.info(f"Added {len(contents)} new memories (batch)")

        return ids

    def add_chunked(
        self,
        content: str,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[str]:
        """Chunk content and add each chunk with stable source metadata."""
        timestamp = timestamp or datetime.now()
        metadata = metadata or {}
        chunks = self.chunk_text(content, chunk_size, chunk_overlap)
        entries = [
            (
                chunk,
                timestamp,
                {
                    **metadata,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                },
            )
            for index, chunk in enumerate(chunks)
        ]
        return self.add_batch(entries)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict | None = None,
        before: datetime | None = None,
        after: datetime | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        Search for relevant memories.

        Args:
            query: The search query
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            filter_metadata: Optional metadata filter
            before: Exclude memories newer than this time
            after: Exclude memories older than this time

        Returns:
            List of (MemoryEntry, score) tuples sorted by relevance
        """
        query_embedding = self.embed([query])[0]

        # Build where filter if provided
        where = None
        if filter_metadata:
            where = {k: str(v) for k, v in filter_metadata.items()}

        candidate_count = top_k * 4 if before or after else top_k
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, candidate_count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        entries = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB returns L2 distance by default, convert to similarity
            # For cosine distance: similarity = 1 - distance
            distance = results["distances"][0][i]
            score = 1 - distance  # Convert distance to similarity

            if score < min_score:
                continue

            metadata = dict(results["metadatas"][0][i])
            timestamp = datetime.fromisoformat(metadata.pop("timestamp"))
            metadata.pop("content_preview", None)

            comparable_timestamp = self._as_utc(timestamp)
            if after and comparable_timestamp < self._as_utc(after):
                continue
            if before and comparable_timestamp > self._as_utc(before):
                continue

            entry = MemoryEntry(
                id=doc_id,
                content=results["documents"][0][i],
                timestamp=timestamp,
                metadata=metadata,
            )
            entries.append((entry, score))
            if len(entries) >= top_k:
                break

        return entries

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        """Get the most recent memories."""
        # Get all memories and sort by timestamp
        all_data = self._collection.get(include=["documents", "metadatas"])

        entries = []
        for i, doc_id in enumerate(all_data["ids"]):
            metadata = all_data["metadatas"][i]
            timestamp = datetime.fromisoformat(metadata.pop("timestamp"))
            metadata.pop("content_preview", None)

            entry = MemoryEntry(
                id=doc_id,
                content=all_data["documents"][i],
                timestamp=timestamp,
                metadata=metadata,
            )
            entries.append(entry)

        # Sort by timestamp descending
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        try:
            self._collection.delete(ids=[entry_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {entry_id}: {e}")
            return False

    def clear(self) -> None:
        """Clear all memories from the store."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared")

    def count(self) -> int:
        """Get the number of memories in the store."""
        return int(self._collection.count())

    def export(self, filepath: str | Path) -> int:
        """Export all memories to a JSON file."""
        all_data = self._collection.get(include=["documents", "metadatas"])

        entries = []
        for i, doc_id in enumerate(all_data["ids"]):
            entries.append({
                "id": doc_id,
                "content": all_data["documents"][i],
                "metadata": all_data["metadatas"][i],
            })

        with open(filepath, 'w') as f:
            json.dump(entries, f, indent=2)

        return len(entries)

    def import_from_file(self, filepath: str | Path) -> int:
        """Import memories from a JSON file."""
        with open(filepath) as f:
            entries = json.load(f)

        count = 0
        for entry in entries:
            timestamp = datetime.fromisoformat(
                entry["metadata"].get("timestamp", datetime.now().isoformat())
            )
            self.add(
                content=entry["content"],
                timestamp=timestamp,
                metadata={k: v for k, v in entry["metadata"].items() if k != "timestamp"},
            )
            count += 1

        return count
