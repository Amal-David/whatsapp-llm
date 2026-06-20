"""
Memory retriever combining vector store and fact store for RAG.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .vector_store import VectorStore, MemoryEntry
from .fact_store import FactStore, Fact

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Combined retrieval result from all memory sources."""
    memories: list[tuple[MemoryEntry, float]]
    facts: list[Fact]
    context_text: str


class MemoryRetriever:
    """
    Unified memory retriever combining episodic (vector) and semantic (fact) memory.

    Used for RAG at inference time.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        fact_store: FactStore,
        top_k_memories: int = 5,
        min_memory_score: float = 0.3,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: The vector store for episodic memories
            fact_store: The fact store for semantic memory
            top_k_memories: Number of memories to retrieve
            min_memory_score: Minimum similarity score for memories
        """
        self.vector_store = vector_store
        self.fact_store = fact_store
        self.top_k_memories = top_k_memories
        self.min_memory_score = min_memory_score

    def retrieve(
        self,
        query: str,
        include_facts: bool = True,
        include_memories: bool = True,
        fact_query: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant context for a query.

        Args:
            query: The user's query/message
            include_facts: Whether to include facts in the result
            include_memories: Whether to include episodic memories
            fact_query: Optional separate query for fact search

        Returns:
            RetrievalResult with memories, facts, and formatted context
        """
        memories = []
        facts = []

        # Retrieve episodic memories
        if include_memories:
            memories = self.vector_store.search(
                query=query,
                top_k=self.top_k_memories,
                min_score=self.min_memory_score,
            )
            logger.debug(f"Retrieved {len(memories)} memories for query: {query[:50]}...")

        # Retrieve facts
        if include_facts:
            # Get all active facts (knowledge base)
            facts = self.fact_store.get_all_active()

            # Also search for query-specific facts
            if fact_query:
                query_facts = self.fact_store.search(fact_query)
                # Merge without duplicates
                fact_ids = {f.id for f in facts}
                for f in query_facts:
                    if f.id not in fact_ids:
                        facts.append(f)

            logger.debug(f"Retrieved {len(facts)} facts")

        # Format context
        context_text = self._format_context(memories, facts)

        return RetrievalResult(
            memories=memories,
            facts=facts,
            context_text=context_text,
        )

    def _format_context(
        self,
        memories: list[tuple[MemoryEntry, float]],
        facts: list[Fact],
    ) -> str:
        """
        Format memories and facts into a context string for the LLM.

        Args:
            memories: List of (memory, score) tuples
            facts: List of facts

        Returns:
            Formatted context string
        """
        sections = []

        # Facts section
        if facts:
            fact_lines = ["<facts>"]
            for fact in facts:
                fact_lines.append(f"- {fact.to_natural()}")
            fact_lines.append("</facts>")
            sections.append("\n".join(fact_lines))

        # Memories section
        if memories:
            memory_lines = ["<memories>"]
            for memory, score in memories:
                # Format timestamp nicely
                time_str = memory.timestamp.strftime("%Y-%m-%d")
                memory_lines.append(f"[{time_str}] {memory.content}")
            memory_lines.append("</memories>")
            sections.append("\n".join(memory_lines))

        return "\n\n".join(sections)

    def format_prompt_with_context(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        system_prompt: str = "",
    ) -> str:
        """
        Format a complete prompt with retrieved context.

        Args:
            query: The user's query
            retrieval_result: The retrieval result
            system_prompt: Base system prompt

        Returns:
            Complete prompt with context injected
        """
        parts = []

        if system_prompt:
            parts.append(system_prompt)

        if retrieval_result.context_text:
            parts.append(
                "Here is some relevant context from my memories and knowledge:"
            )
            parts.append(retrieval_result.context_text)

        parts.append(f"Current message: {query}")

        return "\n\n".join(parts)

    def add_conversation_to_memory(
        self,
        conversation_text: str,
        timestamp=None,
        metadata: Optional[dict] = None,
        extract_facts: bool = True,
    ) -> tuple[str, list[Fact]]:
        """
        Add a conversation to memory and optionally extract facts.

        Args:
            conversation_text: The conversation text to add
            timestamp: When this conversation occurred
            metadata: Additional metadata
            extract_facts: Whether to extract facts from the conversation

        Returns:
            Tuple of (memory_id, extracted_facts)
        """
        from datetime import datetime

        timestamp = timestamp or datetime.now()
        metadata = metadata or {}

        # Add to vector store
        memory_id = self.vector_store.add(
            content=conversation_text,
            timestamp=timestamp,
            metadata=metadata,
        )

        # Extract and store facts
        extracted_facts = []
        if extract_facts:
            extracted_facts = self.fact_store.ingest_and_extract(
                conversation_text,
                auto_add=True,
            )
            logger.info(f"Extracted {len(extracted_facts)} facts from conversation")

        return memory_id, extracted_facts

    def stats(self) -> dict:
        """Get statistics about the memory system."""
        return {
            "total_memories": self.vector_store.count(),
            "total_facts": self.fact_store.count(),
            "recent_memories": [
                m.to_dict() for m in self.vector_store.get_recent(5)
            ],
            "sample_facts": [
                f.to_dict() for f in self.fact_store.get_all_active()[:5]
            ],
        }
