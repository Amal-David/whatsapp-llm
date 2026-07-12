"""Editable guided interview for owner-confirmed digital-self evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

INTERVIEW_SECTIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "life_story",
        "title": "Life story and current roles",
        "questions": (
            ("life_story.summary", "What parts of your life story most shape who you are now?"),
            ("roles.current", "Which roles or responsibilities matter most to you today?"),
            ("identity.self_description", "How do you describe yourself when nobody else is defining you?"),
        ),
    },
    {
        "id": "values",
        "title": "Values and trade-offs",
        "questions": (
            ("values.core", "Which values are non-negotiable for you?"),
            ("values.tradeoffs", "When two good values conflict, how do you decide between them?"),
            ("values.regret", "What kind of choice are you most likely to regret?"),
        ),
    },
    {
        "id": "goals",
        "title": "Goals and active priorities",
        "questions": (
            ("goals.current", "What are you trying to accomplish in the next year?"),
            ("goals.long_term", "What would a deeply satisfying longer-term life look like?"),
            ("goals.avoid", "Which outcomes are you actively trying to avoid?"),
        ),
    },
    {
        "id": "decision_rules",
        "title": "Decisions and recurring heuristics",
        "questions": (
            ("decisions.uncertainty", "How do you make decisions when evidence is incomplete?"),
            ("decisions.speed", "When do you move quickly, and when do you slow down?"),
            ("decisions.advice", "Whose advice do you trust, and what makes it persuasive?"),
            ("decisions.examples", "Describe two recent choices that felt distinctly like you."),
        ),
    },
    {
        "id": "preferences",
        "title": "Preferences and aversions",
        "questions": (
            ("preferences.energy", "Which activities give you energy, and which reliably drain it?"),
            ("preferences.work", "What conditions help you do your best work?"),
            ("preferences.social", "What kinds of social situations do you seek or avoid?"),
        ),
    },
    {
        "id": "communication",
        "title": "Communication and conflict",
        "questions": (
            ("communication.default", "How do you usually explain something important?"),
            ("communication.conflict", "How do you behave when you disagree with someone you care about?"),
            ("communication.support", "How do you prefer to give and receive emotional support?"),
        ),
    },
    {
        "id": "relationships",
        "title": "Relationship patterns",
        "questions": (
            ("relationships.roles", "Which relationship roles bring out noticeably different sides of you?"),
            ("relationships.trust", "What causes you to trust someone more or less?"),
            ("relationships.privacy", "What third-party information must the replica never repeat?"),
        ),
    },
    {
        "id": "recent_changes",
        "title": "Recent changes",
        "questions": (
            ("changes.preferences", "Which preferences or beliefs have changed recently?"),
            ("changes.context", "What changed in your circumstances, and when?"),
            ("changes.old_self", "Which older patterns no longer describe you?"),
        ),
    },
    {
        "id": "boundaries",
        "title": "Boundaries and uncertainty",
        "questions": (
            ("boundaries.refuse", "Which topics should the replica refuse to answer as you?"),
            ("boundaries.qualify", "Where should it explicitly say it is unsure?"),
            ("boundaries.authority", "Which decisions must always remain yours alone?"),
            ("boundaries.protected", "Which personal traits must never be inferred from behavior?"),
        ),
    },
)


def create_interview_template(owner_name: str) -> dict[str, Any]:
    """Create a deterministic YAML-friendly interview document."""
    if not owner_name.strip():
        raise ValueError("owner_name cannot be empty")

    return {
        "schema_version": "digital_self_interview.v1",
        "owner_name": owner_name.strip(),
        "instructions": [
            "Answer in your own words; short answers are fine.",
            "Leave an answer null when you do not want the replica to learn it.",
            "Add valid_from when an answer describes a recent change.",
            "Interview answers are owner-confirmed evidence, not public profile copy.",
        ],
        "sections": [
            {
                "id": section["id"],
                "title": section["title"],
                "questions": [
                    {
                        "id": question_id,
                        "prompt": prompt,
                        "answer": None,
                        "valid_from": None,
                        "sensitivity": "private",
                    }
                    for question_id, prompt in section["questions"]
                ],
            }
            for section in INTERVIEW_SECTIONS
        ],
    }


def write_interview_template(path: str | Path, owner_name: str) -> Path:
    """Write an editable interview template without sending data anywhere."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            create_interview_template(owner_name),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_path.chmod(0o600)
    return output_path


def load_interview(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an interview document."""
    content = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("interview must be a YAML object")
    if content.get("schema_version") != "digital_self_interview.v1":
        raise ValueError("unsupported interview schema")
    if not isinstance(content.get("sections"), list):
        raise ValueError("interview sections must be a list")
    return content


__all__ = [
    "INTERVIEW_SECTIONS",
    "create_interview_template",
    "load_interview",
    "write_interview_template",
]
