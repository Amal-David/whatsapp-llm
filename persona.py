import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable

import pandas as pd
import yaml

from privacy import PIIRedactor


@dataclass
class PersonaProfile:
    persona: Dict[str, object]

    def summary(self) -> str:
        return self.persona.get('persona_summary', '')


class PersonaBuilder:
    """Build a structured persona document from chat analytics."""

    def __init__(
        self,
        your_name: str,
        contact_name: str,
        redactor: Optional[PIIRedactor] = None,
        redact: bool = False
    ) -> None:
        self.your_name = your_name
        self.contact_name = contact_name
        self.redactor = redactor or PIIRedactor()
        self.redact = redact

    def _tone_descriptors(self, tones: Iterable[str]) -> List[str]:
        filtered = [tone for tone in tones if tone]
        counter = Counter(filtered)
        if not counter:
            return ['balanced']
        ordered = counter.most_common(3)
        descriptors = [tone for tone, _ in ordered if tone != 'general']
        return descriptors or ['balanced']

    def _intent_distribution(self, intents: Iterable[str]) -> Dict[str, int]:
        filtered = [intent for intent in intents if intent]
        counter = Counter(filtered)
        return dict(counter)

    def _topic_overview(self, topics: Iterable[str]) -> List[Dict[str, object]]:
        counter = Counter(topic for topic in topics if topic and topic != 'general')
        overview = []
        for topic, count in counter.most_common(5):
            overview.append({'topic': topic, 'frequency': count})
        return overview

    def _conversation_role(self, df: pd.DataFrame) -> str:
        counts = df['Author'].value_counts()
        contact_msgs = counts.get(self.contact_name, 0)
        your_msgs = counts.get(self.your_name, 0)
        if contact_msgs > your_msgs * 1.2:
            return 'primary_initiator'
        if your_msgs > contact_msgs * 1.2:
            return 'responsive_partner'
        return 'balanced_partner'

    def _notable_quirks(self, style_metrics: Dict[str, object]) -> List[str]:
        quirks = []
        emoji_rate = style_metrics.get('emoji_usage_rate', 0)
        slang_rate = style_metrics.get('slang_usage_rate', 0)
        capitalization_rate = style_metrics.get('capitalization_rate', 0)

        if emoji_rate > 0.4:
            quirks.append('Heavy emoji usage in responses')
        elif emoji_rate > 0.2:
            quirks.append('Frequent emoji accents')

        if slang_rate > 0.3:
            quirks.append('Uses casual slang regularly')
        elif slang_rate > 0.15:
            quirks.append('Sprinkles casual abbreviations')

        if capitalization_rate < 0.3:
            quirks.append('Rarely capitalizes sentence openings')
        elif capitalization_rate > 0.8:
            quirks.append('Consistently formal capitalization')

        common_phrases = style_metrics.get('common_phrases', {})
        for phrase, count in list(common_phrases.items())[:3]:
            quirks.append(f"Signature phrase '{phrase}' used {count}×")

        return quirks

    def _memory_slots(self, df: pd.DataFrame) -> List[Dict[str, object]]:
        facts = {}
        contact_rows = df[df['Author'] == self.contact_name]
        patterns = [
            (r"\bI am ([^.?!]+)", 'identity'),
            (r"\bI'm ([^.?!]+)", 'identity'),
            (r"\bI have ([^.?!]+)", 'possession'),
            (r"\bMy ([^.?!]+?) is ([^.?!]+)", 'personal_detail'),
            (r"\bI (?:like|love) ([^.?!]+)", 'preference'),
            (r"\bI live in ([^.?!]+)", 'location')
        ]

        for _, row in contact_rows.iterrows():
            message = str(row.get('Message', ''))
            timestamp = row.get('Timestamp')
            for pattern, slot_type in patterns:
                matches = re.findall(pattern, message, flags=re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        statement = ' '.join(match).strip()
                    else:
                        statement = str(match).strip()
                    key = statement.lower()
                    facts[key] = {
                        'type': slot_type,
                        'value': statement,
                        'source_message': message,
                        'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else timestamp,
                        'confidence': 0.9
                    }

        slots = list(facts.values())
        if self.redact:
            for slot in slots:
                slot['value'] = self.redactor.redact_text(slot['value'])
                slot['source_message'] = self.redactor.redact_text(slot['source_message'])
        return slots

    def _prompt_snippets(self, tone_descriptors: List[str], topics: List[Dict[str, object]]) -> List[str]:
        topic_examples = ', '.join(topic['topic'] for topic in topics[:3]) if topics else 'their favourite subjects'
        tone_label = tone_descriptors[0] if tone_descriptors else 'balanced'
        return [
            (
                f"Stay {tone_label} and reference {topic_examples}. "
                f"Mirror {self.contact_name}'s messaging cadence and quirks."
            ),
            (
                f"Respond as {self.contact_name} would, prioritizing {tone_label} energy and "
                "consistent persona facts."
            )
        ]

    def _compose_summary(
        self,
        tone_descriptors: List[str],
        topics: List[Dict[str, object]],
        intents: Dict[str, int]
    ) -> str:
        tone_text = ', '.join(tone_descriptors) if tone_descriptors else 'balanced'
        if topics:
            topic_text = ', '.join(topic['topic'] for topic in topics[:3])
        else:
            topic_text = 'varied day-to-day topics'
        dominant_intent = max(intents.items(), key=lambda item: item[1])[0] if intents else 'general conversation'
        return (
            f"{self.contact_name} typically communicates with a {tone_text} tone, "
            f"leaning on {topic_text}. Their primary intent trend is {dominant_intent}."
        )

    def build_profile(
        self,
        chat_records: List[Dict[str, object]],
        style_metrics: Dict[str, object],
        episode_metadata: Optional[List[Dict[str, object]]] = None
    ) -> PersonaProfile:
        df = pd.DataFrame(chat_records)
        if df.empty:
            df = pd.DataFrame(columns=['Author', 'Message', 'Timestamp', 'Tone', 'Intent', 'Topic', 'Episode'])

        tone_descriptors = self._tone_descriptors(df.get('Tone', []))
        intent_distribution = self._intent_distribution(df.get('Intent', []))
        topic_overview = self._topic_overview(df.get('Topic', []))
        conversation_role = self._conversation_role(df)
        memory_slots = self._memory_slots(df)
        prompt_snippets = self._prompt_snippets(tone_descriptors, topic_overview)
        summary = self._compose_summary(tone_descriptors, topic_overview, intent_distribution)

        persona_payload = {
            'contact_name': self.redactor.redact_text(self.contact_name) if self.redact else self.contact_name,
            'conversation_role': conversation_role,
            'tone_descriptors': tone_descriptors,
            'sentiment_distribution': dict(Counter(tone for tone in df.get('Tone', []) if tone)),
            'intent_distribution': intent_distribution,
            'dominant_topics': topic_overview,
            'notable_quirks': self._notable_quirks(style_metrics),
            'memory_slots': memory_slots,
            'prompt_snippets': prompt_snippets,
            'episodes': episode_metadata or [],
            'style_metrics': style_metrics,
            'persona_summary': summary
        }

        if self.redact:
            persona_payload['prompt_snippets'] = [self.redactor.redact_text(snippet) for snippet in prompt_snippets]

        return PersonaProfile(persona=persona_payload)


def save_persona_profile(profile: PersonaProfile, path: str) -> None:
    if path.endswith(('.yaml', '.yml')):
        with open(path, 'w') as f:
            yaml.safe_dump(profile.persona, f, sort_keys=False)
    else:
        with open(path, 'w') as f:
            json.dump(profile.persona, f, indent=2)


def load_persona_profile(path: str) -> PersonaProfile:
    with open(path, 'r') as f:
        if path.endswith(('.yaml', '.yml')):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return PersonaProfile(persona=data)


def persona_summary_from_file(path: str) -> str:
    profile = load_persona_profile(path)
    return profile.summary()
