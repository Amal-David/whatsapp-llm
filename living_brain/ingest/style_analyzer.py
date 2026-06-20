"""
Style analyzer for extracting personal writing style metrics.
"""

import re
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .whatsapp_parser import ChatMessage


@dataclass
class StyleMetrics:
    """Container for style analysis results."""
    avg_message_length: float = 0.0
    median_message_length: float = 0.0
    emoji_usage_rate: float = 0.0
    top_emojis: list[tuple[str, int]] = field(default_factory=list)
    slang_usage_rate: float = 0.0
    top_slang: list[tuple[str, int]] = field(default_factory=list)
    capitalization_rate: float = 0.0
    punctuation_patterns: dict[str, float] = field(default_factory=dict)
    common_phrases: list[tuple[str, int]] = field(default_factory=list)
    response_time_avg_minutes: Optional[float] = None
    message_count: int = 0
    vocabulary_size: int = 0
    avg_words_per_message: float = 0.0

    def to_dict(self) -> dict:
        return {
            "avg_message_length": self.avg_message_length,
            "median_message_length": self.median_message_length,
            "emoji_usage_rate": self.emoji_usage_rate,
            "top_emojis": self.top_emojis,
            "slang_usage_rate": self.slang_usage_rate,
            "top_slang": self.top_slang,
            "capitalization_rate": self.capitalization_rate,
            "punctuation_patterns": self.punctuation_patterns,
            "common_phrases": self.common_phrases,
            "response_time_avg_minutes": self.response_time_avg_minutes,
            "message_count": self.message_count,
            "vocabulary_size": self.vocabulary_size,
            "avg_words_per_message": self.avg_words_per_message,
        }

    def save(self, path: str | Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "StyleMetrics":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_system_prompt(self) -> str:
        """Generate a system prompt describing this style."""
        lines = ["You are mimicking a person with the following conversation style:"]

        # Message length
        if self.avg_message_length < 50:
            lines.append("- Writes SHORT, concise messages (usually under 50 characters)")
        elif self.avg_message_length < 150:
            lines.append("- Writes MEDIUM-length messages")
        else:
            lines.append("- Writes LONGER, detailed messages")

        # Emoji usage
        if self.emoji_usage_rate > 0.5:
            emoji_str = ", ".join([e for e, _ in self.top_emojis[:5]])
            lines.append(f"- Uses emojis FREQUENTLY, favorites: {emoji_str}")
        elif self.emoji_usage_rate > 0.2:
            lines.append("- Uses emojis OCCASIONALLY")
        else:
            lines.append("- RARELY uses emojis")

        # Slang
        if self.slang_usage_rate > 0.15:
            slang_str = ", ".join([s for s, _ in self.top_slang[:5]])
            lines.append(f"- Uses casual slang frequently: {slang_str}")
        elif self.slang_usage_rate > 0.05:
            lines.append("- Occasionally uses casual language")
        else:
            lines.append("- Uses formal language")

        # Capitalization
        if self.capitalization_rate > 0.8:
            lines.append("- ALWAYS capitalizes properly")
        elif self.capitalization_rate > 0.4:
            lines.append("- Sometimes capitalizes, sometimes doesn't")
        else:
            lines.append("- Rarely capitalizes (lowercase style)")

        # Punctuation
        if self.punctuation_patterns:
            if self.punctuation_patterns.get('!', 0) > 0.2:
                lines.append("- Uses exclamation marks frequently!")
            if self.punctuation_patterns.get('...', 0) > 0.1:
                lines.append("- Trails off with ellipsis...")
            if self.punctuation_patterns.get('?', 0) > 0.3:
                lines.append("- Asks questions often")

        # Common phrases
        if self.common_phrases:
            phrases = [p for p, _ in self.common_phrases[:3]]
            lines.append(f"- Common expressions: \"{'\", \"'.join(phrases)}\"")

        lines.append("\nMimic this style while maintaining natural conversation flow.")
        return "\n".join(lines)


class StyleAnalyzer:
    """Analyzes chat messages to extract personal style metrics."""

    # Extended emoji pattern covering most Unicode emoji ranges
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols Extended-A
        "\U00002600-\U000026FF"  # Misc Symbols
        "\U00002300-\U000023FF"  # Misc Technical
        "]+"
    )

    # Common internet slang/abbreviations
    SLANG_TERMS = {
        'lol', 'lmao', 'lmfao', 'rofl', 'omg', 'idk', 'tbh', 'imo', 'imho',
        'fyi', 'brb', 'btw', 'afk', 'nvm', 'aka', 'asap', 'fomo', 'fwiw',
        'iirc', 'irl', 'jk', 'lmk', 'nbd', 'np', 'nsfw', 'rn', 'tbf',
        'tfw', 'tldr', 'tl;dr', 'tysm', 'ty', 'thx', 'pls', 'plz',
        'w/', 'w/o', 'ya', 'ymmv', 'yolo', 'smh', 'icymi', 'ftw',
        'goat', 'gg', 'ez', 'ngl', 'fr', 'ong', 'istg', 'wbu', 'hbu',
        'wya', 'hmu', 'dm', 'pm', 'ofc', 'obvi', 'obvs', 'srsly',
        'whatevs', 'totes', 'prob', 'probs', 'def', 'deffo', 'gonna',
        'wanna', 'gotta', 'kinda', 'sorta', 'dunno', 'lemme', 'gimme',
        'cuz', 'coz', 'bc', 'b4', 'gr8', 'l8r', 'u', 'ur', 'r', 'y',
        'k', 'ok', 'okay', 'kk', 'yea', 'yeah', 'yep', 'yup', 'nah',
        'nope', 'mhm', 'hmm', 'hmmm', 'umm', 'ummm', 'ahh', 'ohh',
        'aww', 'omw', 'otw', 'eta', 'bff', 'bf', 'gf', 'bae',
    }

    def __init__(self):
        self._slang_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(s) for s in self.SLANG_TERMS) + r')\b',
            re.IGNORECASE
        )

    def analyze(self, messages: list[ChatMessage]) -> StyleMetrics:
        """
        Analyze a list of messages and extract style metrics.

        Args:
            messages: List of ChatMessage objects to analyze

        Returns:
            StyleMetrics containing the analysis results
        """
        if not messages:
            return StyleMetrics()

        texts = [m.message for m in messages if m.message]
        total = len(texts)

        if total == 0:
            return StyleMetrics()

        # Message lengths
        lengths = [len(t) for t in texts]
        avg_length = sum(lengths) / total
        sorted_lengths = sorted(lengths)
        median_length = sorted_lengths[total // 2]

        # Word counts
        word_counts = [len(t.split()) for t in texts]
        avg_words = sum(word_counts) / total

        # Vocabulary
        all_words = set()
        for t in texts:
            all_words.update(w.lower() for w in re.findall(r'\b\w+\b', t))
        vocab_size = len(all_words)

        # Emoji analysis
        emoji_counter = Counter()
        emoji_msg_count = 0
        for t in texts:
            emojis = self.EMOJI_PATTERN.findall(t)
            if emojis:
                emoji_msg_count += 1
                emoji_counter.update(emojis)
        emoji_rate = emoji_msg_count / total
        top_emojis = emoji_counter.most_common(10)

        # Slang analysis
        slang_counter = Counter()
        slang_msg_count = 0
        for t in texts:
            matches = self._slang_pattern.findall(t.lower())
            if matches:
                slang_msg_count += 1
                slang_counter.update(m.lower() for m in matches)
        slang_rate = slang_msg_count / total
        top_slang = slang_counter.most_common(10)

        # Capitalization
        cap_count = sum(1 for t in texts if t and t[0].isupper())
        cap_rate = cap_count / total

        # Punctuation patterns
        punct_counts = defaultdict(int)
        for t in texts:
            if t.endswith('!'):
                punct_counts['!'] += 1
            if t.endswith('?'):
                punct_counts['?'] += 1
            if t.endswith('.'):
                punct_counts['.'] += 1
            if '...' in t or '…' in t:
                punct_counts['...'] += 1
            if '!!' in t:
                punct_counts['!!'] += 1
            if '??' in t:
                punct_counts['??'] += 1
        punct_patterns = {k: v / total for k, v in punct_counts.items()}

        # Common phrases (2-grams and 3-grams)
        ngram_counter = Counter()
        for t in texts:
            words = t.lower().split()
            # 2-grams
            for i in range(len(words) - 1):
                ngram = ' '.join(words[i:i+2])
                if len(ngram) > 4:  # Skip very short phrases
                    ngram_counter[ngram] += 1
            # 3-grams
            for i in range(len(words) - 2):
                ngram = ' '.join(words[i:i+3])
                if len(ngram) > 6:
                    ngram_counter[ngram] += 1

        # Filter to phrases that appear multiple times
        common_phrases = [
            (phrase, count) for phrase, count in ngram_counter.most_common(20)
            if count >= 3
        ][:10]

        # Response time (if timestamps available)
        response_times = []
        for i in range(1, len(messages)):
            if messages[i].author != messages[i-1].author:
                delta = (messages[i].timestamp - messages[i-1].timestamp).total_seconds()
                if 0 < delta < 86400:  # Within 24 hours
                    response_times.append(delta / 60)  # Convert to minutes

        avg_response = sum(response_times) / len(response_times) if response_times else None

        return StyleMetrics(
            avg_message_length=avg_length,
            median_message_length=median_length,
            emoji_usage_rate=emoji_rate,
            top_emojis=top_emojis,
            slang_usage_rate=slang_rate,
            top_slang=top_slang,
            capitalization_rate=cap_rate,
            punctuation_patterns=punct_patterns,
            common_phrases=common_phrases,
            response_time_avg_minutes=avg_response,
            message_count=total,
            vocabulary_size=vocab_size,
            avg_words_per_message=avg_words,
        )

    def analyze_file(
        self,
        filepath: str | Path,
        your_name: str,
    ) -> StyleMetrics:
        """
        Analyze a WhatsApp export file for a specific person's style.

        Args:
            filepath: Path to the chat export file
            your_name: Name of the person to analyze

        Returns:
            StyleMetrics for that person
        """
        from .whatsapp_parser import WhatsAppParser

        parser = WhatsAppParser(your_name=your_name)
        messages = parser.get_your_messages(filepath)
        return self.analyze(messages)
