"""
WhatsApp chat parser with support for multiple regional formats.

Supports date formats from various regions:
- US: MM/DD/YY, MM/DD/YYYY
- EU: DD/MM/YY, DD/MM/YYYY
- ISO: YYYY-MM-DD
- Various bracket styles: [], none
- Time formats: 12h (AM/PM), 24h
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
import json

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single chat message."""
    author: str
    message: str
    timestamp: datetime
    is_system: bool = False

    def to_dict(self) -> dict:
        return {
            "author": self.author,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "is_system": self.is_system,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(
            author=data["author"],
            message=data["message"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            is_system=data.get("is_system", False),
        )


@dataclass
class Conversation:
    """A conversation (episode) containing multiple messages."""
    messages: list[ChatMessage] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def add_message(self, msg: ChatMessage) -> None:
        self.messages.append(msg)
        if self.start_time is None or msg.timestamp < self.start_time:
            self.start_time = msg.timestamp
        if self.end_time is None or msg.timestamp > self.end_time:
            self.end_time = msg.timestamp

    def to_text(self, include_timestamps: bool = False) -> str:
        """Convert conversation to text format."""
        lines = []
        for msg in self.messages:
            if include_timestamps:
                lines.append(f"[{msg.timestamp}] {msg.author}: {msg.message}")
            else:
                lines.append(f"{msg.author}: {msg.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class WhatsAppParser:
    """
    Parser for WhatsApp chat exports with support for multiple regional formats.
    """

    # Comprehensive regex patterns for different WhatsApp formats
    PATTERNS = [
        # Bracketed format with seconds: [MM/DD/YY, HH:MM:SS AM/PM]
        r"\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}:\d{2}\s*[APap][Mm])\]\s*([^:]+):\s*(.+)",
        # Bracketed format without seconds: [MM/DD/YY, HH:MM AM/PM]
        r"\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}\s*[APap][Mm])\]\s*([^:]+):\s*(.+)",
        # Bracketed 24h format: [DD/MM/YY, HH:MM:SS]
        r"\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}:\d{2})\]\s*([^:]+):\s*(.+)",
        # Bracketed 24h without seconds: [DD/MM/YY, HH:MM]
        r"\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2})\]\s*([^:]+):\s*(.+)",
        # No brackets with dash: MM/DD/YY, HH:MM:SS AM/PM -
        r"(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}:\d{2}\s*[APap][Mm])\s*-\s*([^:]+):\s*(.+)",
        # No brackets without seconds: MM/DD/YY, HH:MM AM/PM -
        r"(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}\s*[APap][Mm])\s*-\s*([^:]+):\s*(.+)",
        # 24h format with dash: DD/MM/YY, HH:MM:SS -
        r"(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}:\d{2})\s*-\s*([^:]+):\s*(.+)",
        # 24h format without seconds with dash: DD/MM/YY, HH:MM -
        r"(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2})\s*-\s*([^:]+):\s*(.+)",
        # ISO format: YYYY-MM-DD, HH:MM:SS
        r"(\d{4}-\d{2}-\d{2}),?\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\s*-\s*([^:]+):\s*(.+)",
        # Dot-separated date: DD.MM.YY, HH:MM
        r"(\d{1,2}\.\d{1,2}\.\d{2,4}),?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s*(.+)",
    ]

    # Date format candidates for parsing
    DATE_FORMATS = [
        "%m/%d/%y",      # US short
        "%m/%d/%Y",      # US long
        "%d/%m/%y",      # EU short
        "%d/%m/%Y",      # EU long
        "%Y-%m-%d",      # ISO
        "%d.%m.%y",      # Dot short
        "%d.%m.%Y",      # Dot long
    ]

    # Time format candidates
    TIME_FORMATS = [
        "%I:%M:%S %p",   # 12h with seconds
        "%I:%M %p",      # 12h without seconds
        "%H:%M:%S",      # 24h with seconds
        "%H:%M",         # 24h without seconds
    ]

    # System message patterns
    SYSTEM_PATTERNS = [
        r"<media omitted>",
        r"<attached:",
        r"image omitted",
        r"video omitted",
        r"audio omitted",
        r"document omitted",
        r"sticker omitted",
        r"GIF omitted",
        r"Contact card omitted",
        r"location:",
        r"Live location shared",
        r"Messages and calls are end-to-end encrypted",
        r"created group",
        r"added you",
        r"left$",
        r"removed",
        r"changed the subject",
        r"changed this group's icon",
        r"changed the group description",
        r"deleted this message",
        r"This message was deleted",
        r"You deleted this message",
        r"missed .* call",
        r"Missed voice call",
        r"Missed video call",
    ]

    def __init__(self, your_name: Optional[str] = None):
        """
        Initialize the parser.

        Args:
            your_name: Your name as it appears in the chat export.
                      Used to identify your messages vs others.
        """
        self.your_name = your_name
        self._compiled_patterns = [re.compile(p) for p in self.PATTERNS]
        self._system_patterns = [re.compile(p, re.IGNORECASE) for p in self.SYSTEM_PATTERNS]

    def _parse_timestamp(self, date_str: str, time_str: str) -> Optional[datetime]:
        """Parse date and time strings into a datetime object."""
        time_str = time_str.strip().upper()

        for date_fmt in self.DATE_FORMATS:
            for time_fmt in self.TIME_FORMATS:
                try:
                    combined = f"{date_str} {time_str}"
                    fmt = f"{date_fmt} {time_fmt}"
                    return datetime.strptime(combined, fmt)
                except ValueError:
                    continue

        # Fallback: try parsing time without AM/PM marker for 24h
        for date_fmt in self.DATE_FORMATS:
            try:
                # Remove any trailing AM/PM that didn't match
                clean_time = re.sub(r'\s*[APap][Mm]$', '', time_str)
                if ':' in clean_time:
                    parts = clean_time.split(':')
                    if len(parts) == 2:
                        combined = f"{date_str} {clean_time}"
                        return datetime.strptime(combined, f"{date_fmt} %H:%M")
                    elif len(parts) == 3:
                        combined = f"{date_str} {clean_time}"
                        return datetime.strptime(combined, f"{date_fmt} %H:%M:%S")
            except ValueError:
                continue

        return None

    def _is_system_message(self, message: str) -> bool:
        """Check if a message is a system message."""
        for pattern in self._system_patterns:
            if pattern.search(message):
                return True
        return False

    def _clean_message(self, message: str) -> str:
        """Clean and normalize message text."""
        # Strip whitespace
        message = message.strip()
        # Normalize whitespace
        message = ' '.join(message.split())
        # Replace URLs with placeholder (optional)
        message = re.sub(r'https?://\S+', '[URL]', message)
        return message

    def parse_line(self, line: str, prev_message: Optional[ChatMessage] = None) -> Optional[ChatMessage]:
        """
        Parse a single line of chat.

        Args:
            line: The line to parse
            prev_message: Previous message (for multi-line message handling)

        Returns:
            ChatMessage if parsed successfully, None otherwise
        """
        line = line.strip()
        if not line:
            return None

        # Try each pattern
        for pattern in self._compiled_patterns:
            match = pattern.match(line)
            if match:
                date_str, time_str, author, message = match.groups()

                timestamp = self._parse_timestamp(date_str, time_str)
                if timestamp is None:
                    logger.debug(f"Could not parse timestamp: {date_str} {time_str}")
                    continue

                author = author.strip()
                message = self._clean_message(message)
                is_system = self._is_system_message(message)

                return ChatMessage(
                    author=author,
                    message=message,
                    timestamp=timestamp,
                    is_system=is_system,
                )

        return None

    def parse_file(self, filepath: str | Path, skip_system: bool = True) -> Iterator[ChatMessage]:
        """
        Parse a WhatsApp chat export file.

        Args:
            filepath: Path to the chat export file
            skip_system: Whether to skip system messages

        Yields:
            ChatMessage objects
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Chat file not found: {filepath}")

        current_message: Optional[ChatMessage] = None

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parsed = self.parse_line(line, current_message)

                if parsed:
                    # Yield previous message if exists
                    if current_message:
                        if not (skip_system and current_message.is_system):
                            yield current_message
                    current_message = parsed
                elif current_message and line.strip():
                    # Multi-line message continuation
                    current_message.message += "\n" + line.strip()

        # Yield last message
        if current_message and not (skip_system and current_message.is_system):
            yield current_message

    def parse_to_conversations(
        self,
        filepath: str | Path,
        gap_minutes: int = 60,
        skip_system: bool = True,
    ) -> list[Conversation]:
        """
        Parse chat file and split into conversations based on time gaps.

        Args:
            filepath: Path to the chat export file
            gap_minutes: Minimum gap in minutes to start new conversation
            skip_system: Whether to skip system messages

        Returns:
            List of Conversation objects
        """
        conversations = []
        current_conv = Conversation()
        last_timestamp = None

        for msg in self.parse_file(filepath, skip_system):
            # Check if we should start a new conversation
            if last_timestamp:
                gap = (msg.timestamp - last_timestamp).total_seconds() / 60
                if gap > gap_minutes:
                    if current_conv.messages:
                        conversations.append(current_conv)
                    current_conv = Conversation()

            current_conv.add_message(msg)
            last_timestamp = msg.timestamp

        # Add final conversation
        if current_conv.messages:
            conversations.append(current_conv)

        return conversations

    def get_your_messages(self, filepath: str | Path) -> list[ChatMessage]:
        """Get only your messages from the chat."""
        if not self.your_name:
            raise ValueError("your_name must be set to filter messages")

        return [
            msg for msg in self.parse_file(filepath)
            if msg.author == self.your_name
        ]

    def get_participants(self, filepath: str | Path) -> set[str]:
        """Get all unique participants in the chat."""
        return {msg.author for msg in self.parse_file(filepath, skip_system=True)}

    def export_jsonl(self, filepath: str | Path, output_path: str | Path) -> int:
        """
        Export parsed messages to JSONL format.

        Returns:
            Number of messages exported
        """
        output_path = Path(output_path)
        count = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for msg in self.parse_file(filepath):
                f.write(json.dumps(msg.to_dict()) + '\n')
                count += 1

        return count
