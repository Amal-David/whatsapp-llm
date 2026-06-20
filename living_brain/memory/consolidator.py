"""
Memory consolidator for compressing old memories.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from .vector_store import VectorStore, MemoryEntry

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """
    Consolidates old memories by summarizing and compressing them.

    This helps keep the memory system efficient while preserving
    important information from older conversations.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        consolidation_age_days: int = 30,
        min_memories_to_consolidate: int = 10,
    ):
        """
        Initialize the consolidator.

        Args:
            vector_store: The vector store containing memories
            consolidation_age_days: Memories older than this will be consolidated
            min_memories_to_consolidate: Minimum memories needed to trigger consolidation
        """
        self.vector_store = vector_store
        self.consolidation_age_days = consolidation_age_days
        self.min_memories_to_consolidate = min_memories_to_consolidate

    def get_old_memories(self) -> list[MemoryEntry]:
        """Get memories older than the consolidation threshold."""
        cutoff = datetime.now() - timedelta(days=self.consolidation_age_days)

        all_memories = self.vector_store.get_recent(limit=10000)  # Get all
        old_memories = [m for m in all_memories if m.timestamp < cutoff]

        return old_memories

    def summarize_memories(
        self,
        memories: list[MemoryEntry],
        use_llm: bool = False,
        llm_client=None,
    ) -> str:
        """
        Summarize a group of memories.

        Args:
            memories: Memories to summarize
            use_llm: Whether to use an LLM for summarization
            llm_client: Optional LLM client for summarization

        Returns:
            Summary text
        """
        if not memories:
            return ""

        if use_llm and llm_client:
            return self._summarize_with_llm(memories, llm_client)
        else:
            return self._summarize_simple(memories)

    def _summarize_simple(self, memories: list[MemoryEntry]) -> str:
        """Simple rule-based summarization."""
        # Group by date
        by_date: dict[str, list[MemoryEntry]] = {}
        for m in memories:
            date_key = m.timestamp.strftime("%Y-%m-%d")
            if date_key not in by_date:
                by_date[date_key] = []
            by_date[date_key].append(m)

        # Create summary
        lines = []
        for date_key in sorted(by_date.keys()):
            date_memories = by_date[date_key]
            lines.append(f"On {date_key}:")

            # Take first and last message from each day
            if len(date_memories) == 1:
                lines.append(f"  - {date_memories[0].content[:200]}...")
            else:
                lines.append(f"  - Started: {date_memories[0].content[:100]}...")
                lines.append(f"  - Ended: {date_memories[-1].content[:100]}...")
                lines.append(f"  - ({len(date_memories)} conversations total)")

        return "\n".join(lines)

    def _summarize_with_llm(
        self,
        memories: list[MemoryEntry],
        llm_client,
    ) -> str:
        """Use an LLM to create a summary."""
        # Prepare the content
        content = "\n\n---\n\n".join([
            f"[{m.timestamp.strftime('%Y-%m-%d')}]\n{m.content}"
            for m in memories[:50]  # Limit to avoid token limits
        ])

        prompt = f"""Summarize the following conversation memories into a concise summary
that preserves key facts, topics discussed, and important details.

MEMORIES:
{content}

SUMMARY:"""

        try:
            response = llm_client.generate(prompt, max_tokens=500)
            return response
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return self._summarize_simple(memories)

    def consolidate(
        self,
        use_llm: bool = False,
        llm_client=None,
        dry_run: bool = False,
    ) -> dict:
        """
        Perform memory consolidation.

        Args:
            use_llm: Whether to use an LLM for summarization
            llm_client: Optional LLM client
            dry_run: If True, don't actually delete/modify memories

        Returns:
            Dictionary with consolidation statistics
        """
        old_memories = self.get_old_memories()

        if len(old_memories) < self.min_memories_to_consolidate:
            logger.info(
                f"Only {len(old_memories)} old memories, "
                f"skipping consolidation (min: {self.min_memories_to_consolidate})"
            )
            return {
                "consolidated": False,
                "reason": "not_enough_memories",
                "old_memories_count": len(old_memories),
            }

        # Create summary
        summary = self.summarize_memories(old_memories, use_llm, llm_client)

        if dry_run:
            return {
                "consolidated": False,
                "dry_run": True,
                "memories_to_consolidate": len(old_memories),
                "summary_preview": summary[:500],
            }

        # Store the consolidated summary as a new memory
        oldest_timestamp = min(m.timestamp for m in old_memories)
        newest_timestamp = max(m.timestamp for m in old_memories)

        self.vector_store.add(
            content=f"[CONSOLIDATED MEMORIES from {oldest_timestamp.date()} to {newest_timestamp.date()}]\n\n{summary}",
            timestamp=newest_timestamp,
            metadata={
                "type": "consolidated",
                "original_count": len(old_memories),
                "date_range_start": oldest_timestamp.isoformat(),
                "date_range_end": newest_timestamp.isoformat(),
            },
        )

        # Delete old memories
        deleted_count = 0
        for memory in old_memories:
            if self.vector_store.delete(memory.id):
                deleted_count += 1

        logger.info(
            f"Consolidated {len(old_memories)} memories into 1 summary, "
            f"deleted {deleted_count} old memories"
        )

        return {
            "consolidated": True,
            "memories_consolidated": len(old_memories),
            "memories_deleted": deleted_count,
            "date_range": {
                "start": oldest_timestamp.isoformat(),
                "end": newest_timestamp.isoformat(),
            },
        }

    def get_consolidation_status(self) -> dict:
        """Get the current status of memories for consolidation."""
        old_memories = self.get_old_memories()
        total_memories = self.vector_store.count()

        return {
            "total_memories": total_memories,
            "old_memories": len(old_memories),
            "threshold_days": self.consolidation_age_days,
            "needs_consolidation": len(old_memories) >= self.min_memories_to_consolidate,
        }
