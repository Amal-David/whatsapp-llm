"""
Sample-text recommendation engine for authentic persona drafting.

The engine is deliberately heuristic and model-free. It converts a pasted sample
into concrete guidance for method choice, code-switching, paralinguistic tags,
low-data augmentation, and evaluation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .persona_dataset import FILLER_PARTICLES, SCRIPT_RANGES, stable_hash
from .style_analyzer import StyleAnalyzer
from .whatsapp_parser import ChatMessage

LAUGHTER_PATTERNS = (
    r"\b(?:ha){2,}\b",
    r"\b(?:he){2,}\b",
    r"\b(?:ja){2,}\b",
    r"\blol+\b",
    r"\blmao+\b",
    r"\brofl+\b",
)


@dataclass
class SampleRecommendation:
    """Structured recommendation returned to the UI and export code."""

    summary: dict[str, Any]
    method_recommendations: list[dict[str, Any]]
    what_works: list[str]
    authenticity_gaps: list[str]
    code_switching: dict[str, Any]
    paralinguistic_tags: list[dict[str, Any]]
    augmentation_plan: list[dict[str, Any]]
    generation_constraints: list[str]
    evaluation_plan: list[dict[str, Any]]
    style_card: dict[str, Any]
    prompt_snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "method_recommendations": self.method_recommendations,
            "what_works": self.what_works,
            "authenticity_gaps": self.authenticity_gaps,
            "code_switching": self.code_switching,
            "paralinguistic_tags": self.paralinguistic_tags,
            "augmentation_plan": self.augmentation_plan,
            "generation_constraints": self.generation_constraints,
            "evaluation_plan": self.evaluation_plan,
            "style_card": self.style_card,
            "prompt_snippet": self.prompt_snippet,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    def to_markdown(self) -> str:
        lines = [
            "# Sample Text Recommendation",
            "",
            f"Default method: `{self.summary['default_method']}`",
            f"Data tier: `{self.summary['data_tier']}`",
            "",
            "## What Works",
        ]
        lines.extend(f"- {item}" for item in self.what_works)
        lines.append("")
        lines.append("## Authenticity Gaps")
        lines.extend(f"- {item}" for item in self.authenticity_gaps)
        lines.append("")
        lines.append("## Method Ranking")
        for item in self.method_recommendations:
            lines.append(f"- `{item['method']}`: {item['fit']} - {item['reason']}")
        lines.append("")
        lines.append("## Code-Switching")
        lines.append(self.code_switching["guidance"])
        lines.append("")
        lines.append("## Paralinguistic Tags")
        for tag in self.paralinguistic_tags:
            lines.append(f"- `{tag['tag']}`: {tag['guidance']}")
        lines.append("")
        lines.append("## Low-Data Augmentation")
        for recipe in self.augmentation_plan:
            lines.append(f"- `{recipe['recipe']}`: {recipe['how_to_use']}")
        lines.append("")
        lines.append("## Prompt Snippet")
        lines.append("")
        lines.append("```text")
        lines.append(self.prompt_snippet)
        lines.append("```")
        return "\n".join(lines) + "\n"


class SampleTextRecommender:
    """Analyze pasted sample text and recommend a training/drafting path."""

    def recommend(
        self,
        sample_text: str,
        target_message_count: int = 0,
        desired_outputs: int = 100,
        persona_name: str = "Target",
        context: str = "",
        unary_feedback_labels: int = 0,
    ) -> SampleRecommendation:
        messages = self._extract_messages(sample_text)
        texts = [message for message in messages if message]
        metrics = self._metrics(texts)
        code_switching = self._code_switching(texts)
        tags = self._paralinguistic_tags(texts, metrics)
        data_tier = self._data_tier(target_message_count, len(texts))
        default_method = self._default_method(data_tier)
        methods = self._method_recommendations(
            data_tier,
            target_message_count,
            len(texts),
            unary_feedback_labels,
        )
        what_works = self._what_works(metrics, code_switching, tags)
        gaps = self._authenticity_gaps(metrics, code_switching, tags, len(texts))
        augmentation_plan = self._augmentation_plan(
            data_tier=data_tier,
            desired_outputs=desired_outputs,
            tags=tags,
            code_switching=code_switching,
        )
        constraints = self._generation_constraints(metrics, code_switching, tags)
        evaluation = self._evaluation_plan(metrics, code_switching, tags)
        style_card = self._style_card(
            persona_name=persona_name,
            metrics=metrics,
            code_switching=code_switching,
            tags=tags,
            data_tier=data_tier,
            context=context,
        )
        prompt = self._prompt_snippet(style_card, constraints)
        summary = {
            "persona_name": persona_name,
            "sample_id": "sha256:" + stable_hash(sample_text, context)[:24],
            "sample_message_count": len(texts),
            "target_message_count": target_message_count,
            "desired_outputs": desired_outputs,
            "unary_feedback_labels": unary_feedback_labels,
            "data_tier": data_tier,
            "default_method": default_method,
            "authenticity_score": self._authenticity_score(metrics, code_switching, tags, len(texts)),
            "metrics": metrics,
        }
        return SampleRecommendation(
            summary=summary,
            method_recommendations=methods,
            what_works=what_works,
            authenticity_gaps=gaps,
            code_switching=code_switching,
            paralinguistic_tags=tags,
            augmentation_plan=augmentation_plan,
            generation_constraints=constraints,
            evaluation_plan=evaluation,
            style_card=style_card,
            prompt_snippet=prompt,
        )

    def _extract_messages(self, sample_text: str) -> list[str]:
        messages: list[str] = []
        for raw_line in sample_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\[[^\]]+\]\s*", "", line)
            line = re.sub(r"^[^:]{1,40}:\s+", "", line)
            messages.append(line)
        if not messages and sample_text.strip():
            messages = [sample_text.strip()]
        return messages

    def _metrics(self, texts: list[str]) -> dict[str, Any]:
        count = len(texts) or 1
        lengths = [len(text) for text in texts]
        words = [re.findall(r"\b[\w']+\b", text.lower()) for text in texts]
        word_counts = [len(item) for item in words]
        emoji_counter: Counter[str] = Counter()
        particle_counter: Counter[str] = Counter()
        opener_counter: Counter[str] = Counter()
        closer_counter: Counter[str] = Counter()
        laughter_count = 0
        ellipsis_count = 0
        question_count = 0
        exclaim_count = 0
        repeated_punctuation = 0
        elongation_count = 0
        lowercase_starts = 0
        all_caps = 0
        one_word = 0

        for text, word_list in zip(texts, words):
            emoji_counter.update(StyleAnalyzer.EMOJI_PATTERN.findall(text))
            particle_counter.update(word for word in word_list if word in FILLER_PARTICLES)
            if word_list:
                opener_counter[word_list[0]] += 1
                closer_counter[word_list[-1]] += 1
            if len(word_list) == 1:
                one_word += 1
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in LAUGHTER_PATTERNS):
                laughter_count += 1
            if "..." in text or "\u2026" in text:
                ellipsis_count += 1
            if text.rstrip().endswith("?"):
                question_count += 1
            if text.rstrip().endswith("!"):
                exclaim_count += 1
            if re.search(r"([!?])\1{1,}|\.{3,}|\u2026", text):
                repeated_punctuation += 1
            if re.search(r"([A-Za-z])\1{2,}", text):
                elongation_count += 1
            first_alpha = next((char for char in text if char.isalpha()), "")
            if first_alpha and first_alpha.islower():
                lowercase_starts += 1
            letters = [char for char in text if char.isalpha()]
            if letters and sum(1 for char in letters if char.isupper()) / len(letters) > 0.75:
                all_caps += 1

        return {
            "message_count": len(texts),
            "avg_chars": sum(lengths) / count if lengths else 0,
            "median_chars": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "avg_words": sum(word_counts) / count if word_counts else 0,
            "one_word_rate": one_word / count,
            "emoji_rate": sum(1 for text in texts if StyleAnalyzer.EMOJI_PATTERN.search(text)) / count,
            "top_emojis": emoji_counter.most_common(10),
            "particle_rate": sum(1 for word_list in words if any(w in FILLER_PARTICLES for w in word_list))
            / count,
            "top_particles": particle_counter.most_common(12),
            "top_openers": opener_counter.most_common(8),
            "top_closers": closer_counter.most_common(8),
            "laughter_rate": laughter_count / count,
            "ellipsis_rate": ellipsis_count / count,
            "question_rate": question_count / count,
            "exclaim_rate": exclaim_count / count,
            "repeated_punctuation_rate": repeated_punctuation / count,
            "elongation_rate": elongation_count / count,
            "lowercase_start_rate": lowercase_starts / count,
            "all_caps_rate": all_caps / count,
        }

    def _code_switching(self, texts: list[str]) -> dict[str, Any]:
        script_counter: Counter[str] = Counter()
        mixed_messages = 0
        romanized_markers = Counter()
        romanized_terms = {
            "acha",
            "achha",
            "arre",
            "bas",
            "haan",
            "hai",
            "kya",
            "matlab",
            "nahi",
            "na",
            "theek",
            "yaar",
        }

        for text in texts:
            scripts = self._script_counts(text)
            script_counter.update(scripts)
            active = [name for name, count in scripts.items() if count > 0]
            if len(active) > 1:
                mixed_messages += 1
            words = re.findall(r"\b[\w']+\b", text.lower())
            romanized_markers.update(word for word in words if word in romanized_terms)

        total = len(texts) or 1
        primary = script_counter.most_common(1)[0][0] if script_counter else "unknown"
        secondary = [
            name for name, count in script_counter.items() if name != primary and count > 0
        ]
        marker_rate = sum(romanized_markers.values()) / total
        mixed_rate = mixed_messages / total
        level = "frequent" if mixed_rate > 0.2 or marker_rate > 0.5 else "light" if mixed_rate > 0 or marker_rate > 0 else "none"

        if level == "none":
            guidance = "No strong code-switching signal. Keep generation in the sample's dominant script unless the prompt itself switches."
        elif level == "light":
            guidance = "Use code-switching as a contextual accent: preserve common particles and switch only around emphasis, jokes, softeners, or relationship markers."
        else:
            guidance = "Model code-switching as a distribution: preserve matrix language, switch points, and romanized markers instead of translating whole messages."

        return {
            "level": level,
            "primary_script": primary,
            "secondary_scripts": secondary,
            "mixed_script_message_rate": mixed_rate,
            "romanized_marker_rate": marker_rate,
            "romanized_markers": romanized_markers.most_common(12),
            "guidance": guidance,
            "generation_rule": (
                "Switch only where the sample shows switching: particles, emphasis, quoted speech, affect, or short discourse markers."
            ),
        }

    def _paralinguistic_tags(
        self,
        texts: list[str],
        metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        tags: list[dict[str, Any]] = []
        candidates = [
            ("LAUGHTER", metrics["laughter_rate"], "Preserve laughter tokens as affect, not literal content."),
            ("ELLIPSIS", metrics["ellipsis_rate"], "Use trailing pauses for uncertainty, softening, or unfinished thoughts."),
            ("EMOJI", metrics["emoji_rate"], "Match emoji density and placement; avoid adding emoji to every message."),
            (
                "REPEATED_PUNCTUATION",
                metrics["repeated_punctuation_rate"],
                "Use repeated punctuation only for emphasis bursts.",
            ),
            ("ELONGATION", metrics["elongation_rate"], "Use repeated letters for warmth, teasing, or emphasis."),
            ("LOWERCASE_START", metrics["lowercase_start_rate"], "Keep casual lowercase openings when the sample uses them."),
            ("ALL_CAPS", metrics["all_caps_rate"], "Reserve all-caps for rare high-affect turns."),
            ("DISCOURSE_PARTICLE", metrics["particle_rate"], "Reuse observed particles as turn openers, softeners, and backchannels."),
        ]
        for tag, rate, guidance in candidates:
            if rate > 0:
                tags.append({"tag": tag, "rate": rate, "guidance": guidance})
        if not tags:
            tags.append(
                {
                    "tag": "PLAIN_TEXT_BASELINE",
                    "rate": 1.0,
                    "guidance": "The sample has few overt paralinguistic cues. Keep generation clean and do not force emoji, caps, or laughter.",
                }
            )
        return tags

    def _data_tier(self, target_count: int, sample_count: int) -> str:
        effective = max(target_count, sample_count)
        if effective < 20:
            return "tiny"
        if effective < 100:
            return "small"
        if effective < 500:
            return "medium"
        if effective < 2000:
            return "sft_candidate"
        return "sft_ready"

    def _default_method(self, data_tier: str) -> str:
        return {
            "tiny": "prompt_card_plus_manual_examples",
            "small": "prompt_card_plus_retrieved_exemplars",
            "medium": "rag_persona_memory",
            "sft_candidate": "qlora_sft_with_strict_holdout",
            "sft_ready": "qlora_sft_then_preference_tuning",
        }[data_tier]

    def _method_recommendations(
        self,
        data_tier: str,
        target_count: int,
        sample_count: int,
        unary_feedback_labels: int,
    ) -> list[dict[str, Any]]:
        enough_for_sft = data_tier in {"sft_candidate", "sft_ready"}
        enough_for_rag = data_tier in {"medium", "sft_candidate", "sft_ready"}
        preference_fit = "high" if unary_feedback_labels >= 500 else "medium" if unary_feedback_labels >= 100 else "low"
        return [
            {
                "method": "prompt_card",
                "fit": "high",
                "reason": "Reversible and useful even from a small sample.",
                "minimum_data": "5-20 representative messages",
            },
            {
                "method": "rag_persona_memory",
                "fit": "high" if enough_for_rag else "medium",
                "reason": "Best when there are enough examples to retrieve relationship-specific turns.",
                "minimum_data": "100+ target messages preferred",
            },
            {
                "method": "qlora_sft",
                "fit": "high" if enough_for_sft else "low",
                "reason": (
                    "Train only after enough clean target replies exist; otherwise it will overfit and memorize."
                ),
                "minimum_data": "500+ target replies; 2,000+ better",
            },
            {
                "method": "dpo_or_kto",
                "fit": preference_fit,
                "reason": (
                    "DPO needs chosen/rejected pairs; KTO needs unary desirable/undesirable labels. "
                    f"Current unary label count: {unary_feedback_labels}."
                ),
                "minimum_data": "200+ preference labels or generated variants reviewed by a human",
            },
            {
                "method": "synthetic_augmentation",
                "fit": "supporting",
                "reason": f"Use to reach {sample_count} -> richer coverage, not to replace real held-out data.",
                "minimum_data": "Works from small data only with strict provenance labels",
            },
        ]

    def _what_works(
        self,
        metrics: dict[str, Any],
        code_switching: dict[str, Any],
        tags: list[dict[str, Any]],
    ) -> list[str]:
        wins = []
        if metrics["message_count"] >= 10:
            wins.append("The sample has enough turns to estimate basic length and punctuation distributions.")
        if metrics["top_openers"]:
            wins.append("Repeated openings/closings can become style-card constraints.")
        if code_switching["level"] != "none":
            wins.append("Code-switching markers are visible enough to preserve as explicit generation rules.")
        if any(tag["tag"] != "PLAIN_TEXT_BASELINE" for tag in tags):
            wins.append("Paralinguistic cues can be represented as tags and distribution targets.")
        if metrics["one_word_rate"] > 0:
            wins.append("Short-response behavior is visible; preserve one-word/brief reply rates.")
        if not wins:
            wins.append("The sample can still seed a prompt card, but it needs more real messages.")
        return wins

    def _authenticity_gaps(
        self,
        metrics: dict[str, Any],
        code_switching: dict[str, Any],
        tags: list[dict[str, Any]],
        sample_count: int,
    ) -> list[str]:
        gaps = []
        if sample_count < 20:
            gaps.append("Too few messages to estimate stable distributions; collect more real target replies.")
        if metrics["question_rate"] == 0:
            gaps.append("No question behavior observed; generated conversations may miss initiative and repair turns.")
        if metrics["laughter_rate"] == 0 and metrics["emoji_rate"] == 0 and metrics["particle_rate"] == 0:
            gaps.append("Few affect/backchannel cues; do not synthesize them aggressively without evidence.")
        if code_switching["level"] == "none":
            gaps.append("No code-switching evidence; only switch languages when the live context does.")
        if len(tags) == 1 and tags[0]["tag"] == "PLAIN_TEXT_BASELINE":
            gaps.append("Paralinguistic tag inventory is sparse; treat tags as optional constraints.")
        return gaps

    def _augmentation_plan(
        self,
        data_tier: str,
        desired_outputs: int,
        tags: list[dict[str, Any]],
        code_switching: dict[str, Any],
    ) -> list[dict[str, Any]]:
        plan = [
            {
                "recipe": "intent_preserving_paraphrase",
                "how_to_use": "Create 2-3 variants per real reply that preserve intent, length band, and style tags.",
                "use_for": "SFT only after human review; otherwise keep as prompt exemplars.",
                "synthetic_label": True,
            },
            {
                "recipe": "context_style_pairing",
                "how_to_use": "Pair real contexts with style-preserving replies, but never invent private facts.",
                "use_for": "Prompt/RAG rehearsal and supervised examples with provenance labels.",
                "synthetic_label": True,
            },
            {
                "recipe": "negative_formality_mutation",
                "how_to_use": "Generate too-formal, too-long, or context-ignoring variants as rejected responses.",
                "use_for": "DPO/KTO rejected rows, not SFT targets.",
                "synthetic_label": True,
            },
            {
                "recipe": "kto_labeling_queue",
                "how_to_use": "Collect thumbs-up/down labels on drafted replies before recommending KTO training.",
                "use_for": "KTO desirable/undesirable records after enough human labels exist.",
                "synthetic_label": False,
            },
            {
                "recipe": "tag_controlled_variants",
                "how_to_use": "Vary only observed tags such as laughter, ellipsis, emoji, or particles within measured rates.",
                "use_for": "Teach controllability and preserve paralinguistic distribution.",
                "synthetic_label": True,
            },
        ]
        if code_switching["level"] != "none":
            plan.append(
                {
                    "recipe": "code_switch_position_variants",
                    "how_to_use": "Move observed switch markers across softeners, emphasis, or closings without changing facts.",
                    "use_for": "Preference comparisons and style-card examples.",
                    "synthetic_label": True,
                }
            )
        if data_tier in {"tiny", "small"}:
            plan.append(
                {
                    "recipe": "active_learning_queue",
                    "how_to_use": f"Ask the owner to write or approve {min(desired_outputs, 50)} missing reply types before training.",
                    "use_for": "Best way to escape low-data overfit.",
                    "synthetic_label": False,
                }
            )
        return plan

    def _generation_constraints(
        self,
        metrics: dict[str, Any],
        code_switching: dict[str, Any],
        tags: list[dict[str, Any]],
    ) -> list[str]:
        constraints = [
            f"Target median message length around {metrics['median_chars']:.0f} characters.",
            f"Keep one-word replies near {metrics['one_word_rate']:.0%} unless context demands more.",
            "Preserve current-context facts; do not retrieve or invent private details from training text.",
            code_switching["generation_rule"],
        ]
        for tag in tags[:5]:
            constraints.append(f"{tag['tag']}: {tag['guidance']}")
        return constraints

    def _evaluation_plan(
        self,
        metrics: dict[str, Any],
        code_switching: dict[str, Any],
        tags: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        checks = [
            {
                "check": "length_distribution",
                "metric": "KS or Wasserstein distance",
                "target": f"median near {metrics['median_chars']:.0f} chars",
            },
            {
                "check": "paralinguistic_distribution",
                "metric": "Jensen-Shannon distance",
                "target": "emoji, laughter, ellipsis, punctuation, and particle rates close to held-out samples",
            },
            {
                "check": "code_switching_distribution",
                "metric": "script mix and marker rate",
                "target": code_switching["level"],
            },
            {
                "check": "human_review",
                "metric": "style fit, context fit, privacy, no memorization",
                "target": "human accepts draft as useful, not deceptive",
            },
            {
                "check": "memorization_guard",
                "metric": "exact/near duplicate search",
                "target": "no output copies private source turns unless explicitly supplied in prompt",
            },
        ]
        if tags:
            checks.append(
                {
                    "check": "tag_precision",
                    "metric": "manual audit of tag use",
                    "target": "tags appear only where they support conversation function",
                }
            )
        return checks

    def _style_card(
        self,
        persona_name: str,
        metrics: dict[str, Any],
        code_switching: dict[str, Any],
        tags: list[dict[str, Any]],
        data_tier: str,
        context: str,
    ) -> dict[str, Any]:
        return {
            "schema": "whatsapp.sample_recommendation.style_card",
            "schema_version": "1.0",
            "persona_name": persona_name,
            "data_tier": data_tier,
            "context_note": context,
            "length": {
                "median_chars": metrics["median_chars"],
                "avg_chars": metrics["avg_chars"],
                "one_word_rate": metrics["one_word_rate"],
            },
            "code_switching": code_switching,
            "paralinguistic_tags": tags,
            "rules": self._generation_constraints(metrics, code_switching, tags),
        }

    def _prompt_snippet(self, style_card: dict[str, Any], constraints: list[str]) -> str:
        lines = [
            f"Draft as {style_card['persona_name']} using an authorized style profile.",
            "Match the observed text-message mechanics, not private facts.",
            "Constraints:",
        ]
        lines.extend(f"- {item}" for item in constraints)
        lines.append("If the prompt lacks context, ask a short clarifying question instead of inventing details.")
        return "\n".join(lines)

    def _authenticity_score(
        self,
        metrics: dict[str, Any],
        code_switching: dict[str, Any],
        tags: list[dict[str, Any]],
        sample_count: int,
    ) -> float:
        score = 35.0
        score += min(sample_count, 50) * 0.6
        score += min(len(tags), 6) * 4
        if metrics["top_openers"]:
            score += 5
        if code_switching["level"] != "none":
            score += 6
        if metrics["question_rate"] > 0:
            score += 5
        return min(100.0, round(score, 1))

    def _script_counts(self, text: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for char in text:
            code = ord(char)
            for name, (start, end) in SCRIPT_RANGES:
                if start <= code <= end and char.isalpha():
                    counts[name] += 1
                    break
        return counts


def recommend_from_messages(
    messages: list[ChatMessage],
    persona_name: str,
    target_message_count: int = 0,
    desired_outputs: int = 100,
) -> SampleRecommendation:
    """Build a recommendation from parsed chat messages."""
    sample = "\n".join(message.message for message in messages)
    return SampleTextRecommender().recommend(
        sample,
        target_message_count=target_message_count or len(messages),
        desired_outputs=desired_outputs,
        persona_name=persona_name,
    )


def example_recommendation() -> SampleRecommendation:
    """Small deterministic example used by smoke tests and demos."""
    sample = "\n".join(
        [
            "haan wait lol",
            "I can come after 7...",
            "arre send the location na",
            "perfecttt, see you",
        ]
    )
    return SampleTextRecommender().recommend(
        sample,
        target_message_count=42,
        desired_outputs=120,
        persona_name="Sample",
    )
