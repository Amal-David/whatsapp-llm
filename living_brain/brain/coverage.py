"""Privacy-preserving coverage analysis and owner elicitation priorities."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .models import (
    BrainLayer,
    DigitalBrain,
    EpistemicStatus,
    Ownership,
    StateItem,
    StateStatus,
)

_PERSISTENT_LAYERS = {
    BrainLayer.SEMANTIC,
    BrainLayer.PROCEDURAL,
    BrainLayer.SELF_SCHEMA,
    BrainLayer.VALUES_GOALS,
    BrainLayer.AFFECT,
    BrainLayer.SOCIAL,
    BrainLayer.NARRATIVE,
    BrainLayer.COMMUNICATION,
    BrainLayer.UNCERTAINTY,
}

_LAYER_IMPORTANCE = {
    BrainLayer.SELF_SCHEMA: 20,
    BrainLayer.VALUES_GOALS: 19,
    BrainLayer.SOCIAL: 18,
    BrainLayer.NARRATIVE: 17,
    BrainLayer.COMMUNICATION: 16,
    BrainLayer.SEMANTIC: 15,
    BrainLayer.PROCEDURAL: 14,
    BrainLayer.AFFECT: 13,
    BrainLayer.EPISODE: 12,
    BrainLayer.UNCERTAINTY: 11,
    BrainLayer.EVENT: 0,
    BrainLayer.REFLECTION: 0,
}

_QUESTION_PROMPTS = {
    BrainLayer.EPISODE: "Which recent experience best explains how you act today?",
    BrainLayer.SEMANTIC: "What enduring fact about your life should this model know?",
    BrainLayer.PROCEDURAL: "How do you usually approach an unfamiliar hard problem?",
    BrainLayer.SELF_SCHEMA: "Which description of who you are feels most central to you?",
    BrainLayer.VALUES_GOALS: "Which value or goal should win when your priorities conflict?",
    BrainLayer.AFFECT: "What emotional pattern most changes how you decide or communicate?",
    BrainLayer.SOCIAL: "How does your behavior change across important relationships?",
    BrainLayer.NARRATIVE: "Which life chapter most shapes the story you tell about yourself?",
    BrainLayer.COMMUNICATION: "What makes a response sound unmistakably like you?",
    BrainLayer.UNCERTAINTY: "What about you should the model remain unsure about?",
}


@dataclass(frozen=True)
class OwnerQuestion:
    """A content-free prompt for resolving one coverage deficit."""

    question_id: str
    layer: BrainLayer
    priority: int
    prompt: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "layer": self.layer.value,
            "priority": self.priority,
            "prompt": self.prompt,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LayerCoverage:
    layer: BrainLayer
    status: str
    strong_count: int
    weak_count: int
    stale_count: int
    conflict_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer.value,
            "status": self.status,
            "strong_count": self.strong_count,
            "weak_count": self.weak_count,
            "stale_count": self.stale_count,
            "conflict_count": self.conflict_count,
        }


@dataclass(frozen=True)
class CoverageReport:
    brain_id: str
    brain_version: int
    as_of: datetime
    strong_item_ids: tuple[str, ...]
    weak_item_ids: tuple[str, ...]
    stale_item_ids: tuple[str, ...]
    unknown_layers: tuple[BrainLayer, ...]
    conflict_pairs: tuple[tuple[str, str], ...]
    layers: tuple[LayerCoverage, ...]
    next_questions: tuple[OwnerQuestion, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "brain_id": self.brain_id,
            "brain_version": self.brain_version,
            "as_of": self.as_of.isoformat(),
            "strong_item_ids": list(self.strong_item_ids),
            "weak_item_ids": list(self.weak_item_ids),
            "stale_item_ids": list(self.stale_item_ids),
            "unknown_layers": [layer.value for layer in self.unknown_layers],
            "conflict_pairs": [list(pair) for pair in self.conflict_pairs],
            "layers": [layer.to_dict() for layer in self.layers],
            "next_questions": [question.to_dict() for question in self.next_questions],
        }


def analyze_coverage(
    brain: DigitalBrain,
    *,
    as_of: datetime,
    stale_after_days: int = 180,
    max_questions: int = 5,
) -> CoverageReport:
    """Classify current owner state without copying summaries or source notes."""

    _require_aware(as_of)
    if stale_after_days < 1:
        raise ValueError("stale_after_days must be positive")
    if max_questions < 0:
        raise ValueError("max_questions cannot be negative")
    brain.validate()

    current = [
        item
        for item in _current_items(brain, as_of)
        if item.ownership is not Ownership.THIRD_PARTY
    ]
    cutoff = as_of - timedelta(days=stale_after_days)
    strong = tuple(sorted(item.id for item in current if _is_strong(item)))
    stale = tuple(
        sorted(
            item.id
            for item in current
            if item.layer in _PERSISTENT_LAYERS and _freshness_time(item) < cutoff
        )
    )
    strong_set = set(strong)
    weak = tuple(sorted(item.id for item in current if item.id not in strong_set))
    conflicts = _conflict_pairs(current)

    by_layer: dict[BrainLayer, list[StateItem]] = defaultdict(list)
    for item in current:
        by_layer[item.layer].append(item)
    stale_set = set(stale)
    conflict_counts: dict[BrainLayer, int] = defaultdict(int)
    by_id = {item.id: item for item in current}
    for left, _right in conflicts:
        conflict_counts[by_id[left].layer] += 1

    unknown = tuple(layer for layer in BrainLayer if not by_layer[layer])
    layer_reports = tuple(
        _layer_coverage(
            layer,
            by_layer[layer],
            strong_set=strong_set,
            stale_set=stale_set,
            conflict_count=conflict_counts[layer],
        )
        for layer in BrainLayer
    )
    questions = _next_questions(layer_reports, max_questions=max_questions)

    return CoverageReport(
        brain_id=brain.brain_id,
        brain_version=brain.version,
        as_of=as_of,
        strong_item_ids=strong,
        weak_item_ids=weak,
        stale_item_ids=stale,
        unknown_layers=unknown,
        conflict_pairs=conflicts,
        layers=layer_reports,
        next_questions=questions,
    )


def _current_items(brain: DigitalBrain, as_of: datetime) -> list[StateItem]:
    superseded_ids = {
        item.supersedes
        for item in brain.items
        if item.supersedes
        and item.status not in {StateStatus.REJECTED, StateStatus.DELETED}
        and item.temporal.contains(as_of)
    }
    return sorted(
        (
            item
            for item in brain.items
            if item.id not in superseded_ids
            and item.status in {StateStatus.ACTIVE, StateStatus.PROPOSED}
            and item.temporal.contains(as_of)
        ),
        key=lambda item: (item.layer.value, item.kind, item.id),
    )


def _is_strong(item: StateItem) -> bool:
    return (
        item.status is StateStatus.ACTIVE
        and item.ownership is Ownership.OWNER
        and item.confidence >= 0.8
        and item.epistemic_status
        in {EpistemicStatus.OWNER_DECLARED, EpistemicStatus.OBSERVED}
    )


def _freshness_time(item: StateItem) -> datetime:
    return item.temporal.last_confirmed_at or item.temporal.observed_at


def _conflict_pairs(items: list[StateItem]) -> tuple[tuple[str, str], ...]:
    known_ids = {item.id for item in items}
    pairs: set[tuple[str, str]] = set()
    for item in items:
        for other_id in item.contradiction_ids:
            if other_id in known_ids:
                pairs.add(_ordered_pair(item.id, other_id))

    grouped: dict[str, list[StateItem]] = defaultdict(list)
    for item in items:
        key = json.dumps(
            {
                "layer": item.layer.value,
                "kind": item.kind,
                "scope": item.scope.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        grouped[key].append(item)
    for group in grouped.values():
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                if left.summary != right.summary:
                    pairs.add(_ordered_pair(left.id, right.id))
    return tuple(sorted(pairs))


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _layer_coverage(
    layer: BrainLayer,
    items: list[StateItem],
    *,
    strong_set: set[str],
    stale_set: set[str],
    conflict_count: int,
) -> LayerCoverage:
    strong_count = sum(item.id in strong_set for item in items)
    weak_count = len(items) - strong_count
    stale_count = sum(item.id in stale_set for item in items)
    if not items:
        status = "unknown"
    elif conflict_count:
        status = "conflicted"
    elif stale_count:
        status = "stale"
    elif weak_count:
        status = "weak"
    else:
        status = "strong"
    return LayerCoverage(
        layer=layer,
        status=status,
        strong_count=strong_count,
        weak_count=weak_count,
        stale_count=stale_count,
        conflict_count=conflict_count,
    )


def _next_questions(
    layer_reports: tuple[LayerCoverage, ...],
    *,
    max_questions: int,
) -> tuple[OwnerQuestion, ...]:
    candidates: list[OwnerQuestion] = []
    base_priority = {
        "unknown": 100,
        "conflicted": 90,
        "stale": 75,
        "weak": 65,
    }
    reasons = {
        "unknown": "No owner-grounded state exists for this dimension.",
        "conflicted": "Current state contains unresolved contradictory evidence.",
        "stale": "The most relevant state has not been confirmed recently.",
        "weak": "Current state is inferred, proposed, shared, or low confidence.",
    }
    for report in layer_reports:
        prompt = _QUESTION_PROMPTS.get(report.layer)
        if prompt is None or report.status == "strong":
            continue
        priority = base_priority[report.status] + _LAYER_IMPORTANCE[report.layer]
        reason = reasons[report.status]
        identity = json.dumps(
            {"layer": report.layer.value, "reason": report.status, "prompt": prompt},
            sort_keys=True,
            separators=(",", ":"),
        )
        question_id = "owner-question:" + hashlib.sha256(identity.encode()).hexdigest()
        candidates.append(
            OwnerQuestion(
                question_id=question_id,
                layer=report.layer,
                priority=priority,
                prompt=prompt,
                reason=reason,
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda question: (
                -question.priority,
                question.layer.value,
                question.question_id,
            ),
        )[:max_questions]
    )


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("coverage as_of must be timezone-aware")
