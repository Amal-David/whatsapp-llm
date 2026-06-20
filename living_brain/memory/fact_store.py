"""
Fact store for semantic memory (knowledge graph lite).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """A single fact about the persona."""
    subject: str  # e.g., "I", "my dog", "my job"
    predicate: str  # e.g., "live in", "is named", "is"
    object: str  # e.g., "NYC", "Max", "software engineer"
    confidence: float = 1.0  # 0-1 confidence score
    source: str = "extracted"  # How this fact was created
    timestamp: datetime = field(default_factory=datetime.now)
    superseded_by: str | None = None  # ID of newer fact that replaces this

    @property
    def id(self) -> str:
        """Generate a unique ID for this fact."""
        return f"{self.subject}:{self.predicate}:{self.object}".lower().replace(" ", "_")

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Fact":
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "imported"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            superseded_by=data.get("superseded_by"),
        )

    def to_natural(self) -> str:
        """Convert to natural language."""
        return f"{self.subject} {self.predicate} {self.object}"


class FactStore:
    """
    Simple fact store for semantic memory.

    Stores facts as subject-predicate-object triples with timestamps
    for conflict resolution.
    """

    # Regex patterns for extracting facts from text
    EXTRACTION_PATTERNS = [
        # Name patterns
        (r"(?:my name is|i'm called|call me)\s+(.+?)(?:\.|$)", "I", "am named"),
        # Age patterns
        (r"i(?:'m|\s+am)\s+(\d+)\s*(?:years?\s*old)?", "I", "am years old"),
        # Location patterns
        (r"i(?:'m|\s+am)\s+(?:from|in)\s+(.+?)(?:\.|$)", "I", "am from"),
        # Job patterns
        (r"i\s+work\s+(?:as\s+)?(?:a\s+)?(.+?)(?:\.|$)", "I", "work as"),
        # I live/work patterns
        (r"i\s+(?:live|work|stay)\s+(?:in|at)\s+(.+?)(?:\.|$)", "I", "live in"),
        # My X is Y patterns
        (r"my\s+(\w+)\s+(?:is|was)\s+(?:called\s+|named\s+)?(.+?)(?:\.|$)", "my {0}", "is"),
        # I have patterns
        (r"i\s+have\s+(?:a\s+)?(.+?)(?:\.|$)", "I", "have"),
        # Hobby patterns
        (r"i\s+(?:love|like|enjoy)\s+(.+?)(?:\.|$)", "I", "enjoy"),
        # I am/I'm patterns. Keep this generic pattern after specific predicates.
        (r"(?:i am|i'm|im)\s+(?:a\s+)?(.+?)(?:\.|$)", "I", "am"),
    ]

    def __init__(self, facts_path: str = "./data/facts.json"):
        """
        Initialize the fact store.

        Args:
            facts_path: Path to persist facts
        """
        self.facts_path = Path(facts_path)
        self.facts_path.parent.mkdir(parents=True, exist_ok=True)

        self._facts: dict[str, Fact] = {}
        self._load()

    def _load(self) -> None:
        """Load facts from disk."""
        if self.facts_path.exists():
            try:
                with open(self.facts_path) as f:
                    data = json.load(f)
                for fact_data in data:
                    fact = Fact.from_dict(fact_data)
                    self._facts[fact.id] = fact
                logger.info(f"Loaded {len(self._facts)} facts")
            except Exception as e:
                logger.error(f"Failed to load facts: {e}")

    def _save(self) -> None:
        """Save facts to disk."""
        try:
            with open(self.facts_path, 'w') as f:
                json.dump([f.to_dict() for f in self._facts.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save facts: {e}")

    def add(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "manual",
    ) -> Fact:
        """
        Add a new fact to the store.

        If a conflicting fact exists, the newer one takes precedence.

        Args:
            subject: Subject of the fact
            predicate: Predicate/relationship
            obj: Object of the fact
            confidence: Confidence score (0-1)
            source: Source of the fact (manual, extracted, llm)

        Returns:
            The created Fact
        """
        fact = Fact(
            subject=subject.strip(),
            predicate=predicate.strip(),
            object=obj.strip(),
            confidence=confidence,
            source=source,
        )

        # Check for existing fact with same subject+predicate
        existing_key = f"{fact.subject}:{fact.predicate}".lower().replace(" ", "_")
        for key, existing in list(self._facts.items()):
            if key.startswith(existing_key + ":") and key != fact.id:
                # Mark old fact as superseded
                existing.superseded_by = fact.id
                logger.info(f"Fact superseded: {existing.to_natural()} -> {fact.to_natural()}")

        self._facts[fact.id] = fact
        self._save()
        return fact

    def add_batch(self, facts: list[tuple[str, str, str]]) -> list[Fact]:
        """Add multiple facts at once."""
        results = []
        for subject, predicate, obj in facts:
            fact = self.add(subject, predicate, obj)
            results.append(fact)
        return results

    def get(self, fact_id: str) -> Fact | None:
        """Get a fact by ID."""
        return self._facts.get(fact_id)

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        include_superseded: bool = False,
    ) -> list[Fact]:
        """
        Query facts by subject and/or predicate.

        Args:
            subject: Filter by subject (case-insensitive partial match)
            predicate: Filter by predicate (case-insensitive partial match)
            include_superseded: Include facts that have been superseded

        Returns:
            List of matching facts
        """
        results = []
        for fact in self._facts.values():
            if not include_superseded and fact.superseded_by:
                continue
            if subject and subject.lower() not in fact.subject.lower():
                continue
            if predicate and predicate.lower() not in fact.predicate.lower():
                continue
            results.append(fact)

        # Sort by timestamp descending
        results.sort(key=lambda f: f.timestamp, reverse=True)
        return results

    def search(self, query: str) -> list[Fact]:
        """
        Search facts by any field.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching facts
        """
        query = query.lower()
        results = []

        for fact in self._facts.values():
            if fact.superseded_by:
                continue
            if (query in fact.subject.lower() or
                query in fact.predicate.lower() or
                query in fact.object.lower()):
                results.append(fact)

        return results

    def extract_facts(self, text: str) -> list[Fact]:
        """
        Extract facts from natural language text.

        Args:
            text: Text to extract facts from

        Returns:
            List of extracted facts (not yet added to store)
        """
        facts = []
        seen_ids = set()
        occupied_spans: list[tuple[int, int]] = []
        text_lower = text.lower()

        for pattern, subject_template, predicate in self.EXTRACTION_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if any(match.start() < end and start < match.end() for start, end in occupied_spans):
                    continue

                groups = match.groups()

                if len(groups) == 1:
                    obj = groups[0].strip()
                    subject = subject_template
                elif len(groups) == 2:
                    # Pattern like "my X is Y"
                    subject = subject_template.format(groups[0].strip())
                    obj = groups[1].strip()
                else:
                    continue

                if obj and len(obj) > 1:
                    fact = Fact(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=0.7,  # Lower confidence for extracted facts
                        source="extracted",
                    )
                    if fact.id in seen_ids:
                        continue
                    facts.append(fact)
                    seen_ids.add(fact.id)
                    occupied_spans.append(match.span())

        return facts

    def ingest_and_extract(self, text: str, auto_add: bool = True) -> list[Fact]:
        """
        Extract facts from text and optionally add them to the store.

        Args:
            text: Text to process
            auto_add: Whether to automatically add extracted facts

        Returns:
            List of extracted facts
        """
        facts = self.extract_facts(text)

        if auto_add:
            for fact in facts:
                self.add(
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    confidence=fact.confidence,
                    source=fact.source,
                )

        return facts

    def get_all_active(self) -> list[Fact]:
        """Get all non-superseded facts."""
        return [f for f in self._facts.values() if not f.superseded_by]

    def to_context(self) -> str:
        """
        Generate a context string from all active facts.

        Returns:
            A formatted string of facts for injection into prompts
        """
        facts = self.get_all_active()
        if not facts:
            return ""

        lines = ["Here are some facts about me:"]
        for fact in facts:
            lines.append(f"- {fact.to_natural()}")

        return "\n".join(lines)

    def delete(self, fact_id: str) -> bool:
        """Delete a fact."""
        if fact_id in self._facts:
            del self._facts[fact_id]
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Clear all facts."""
        self._facts.clear()
        self._save()

    def count(self) -> int:
        """Get number of active facts."""
        return len(self.get_all_active())

    def export(self, filepath: str | Path) -> int:
        """Export facts to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump([f.to_dict() for f in self._facts.values()], f, indent=2)
        return len(self._facts)

    def import_from_file(self, filepath: str | Path) -> int:
        """Import facts from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        count = 0
        for fact_data in data:
            fact = Fact.from_dict(fact_data)
            self._facts[fact.id] = fact
            count += 1

        self._save()
        return count
