"""
Data formatter for creating training datasets from conversations.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..ingest.whatsapp_parser import ChatMessage, Conversation
from ..ingest.style_analyzer import StyleMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example."""
    instruction: str
    input_text: str
    output: str
    system: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output,
        }
        if self.system:
            d["system"] = self.system
        return d

    def to_chatml(self) -> str:
        """Format as ChatML (used by many models)."""
        parts = []
        if self.system:
            parts.append(f"<|im_start|>system\n{self.system}<|im_end|>")
        parts.append(f"<|im_start|>user\n{self.instruction}")
        if self.input_text:
            parts.append(f"\n{self.input_text}")
        parts.append(f"<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{self.output}<|im_end|>")
        return "\n".join(parts)

    def to_llama(self) -> str:
        """Format as Llama 3 chat format."""
        parts = ["<|begin_of_text|>"]
        if self.system:
            parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{self.system}<|eot_id|>")
        parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{self.instruction}")
        if self.input_text:
            parts.append(f"\n{self.input_text}")
        parts.append(f"<|eot_id|>")
        parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{self.output}<|eot_id|>")
        return "".join(parts)


class DataFormatter:
    """
    Formats conversation data into training examples.

    Supports multiple output formats for different models.
    """

    def __init__(
        self,
        your_name: str,
        context_turns: int = 3,
        style_metrics: Optional[StyleMetrics] = None,
    ):
        """
        Initialize the formatter.

        Args:
            your_name: Name of the person to train as
            context_turns: Number of previous turns to include as context
            style_metrics: Style metrics for generating system prompts
        """
        self.your_name = your_name
        self.context_turns = context_turns
        self.style_metrics = style_metrics

    def _get_system_prompt(self) -> str:
        """Generate system prompt based on style metrics."""
        if self.style_metrics:
            return self.style_metrics.to_system_prompt()

        return (
            f"You are {self.your_name}. Respond naturally in your personal style. "
            "Keep responses conversational and authentic."
        )

    def format_conversation(
        self,
        messages: list[ChatMessage],
        include_system: bool = True,
    ) -> list[TrainingExample]:
        """
        Format a conversation into training examples.

        Creates examples where your messages are the target outputs
        and previous messages are the context/input.

        Args:
            messages: List of chat messages
            include_system: Whether to include system prompt

        Returns:
            List of TrainingExample objects
        """
        examples = []
        system = self._get_system_prompt() if include_system else None

        for i, msg in enumerate(messages):
            # Only create examples for your messages
            if msg.author != self.your_name:
                continue

            # Skip if no context (need at least one previous message)
            if i == 0:
                continue

            # Build context from previous messages
            start_idx = max(0, i - self.context_turns)
            context_messages = messages[start_idx:i]

            # Format context
            context_lines = []
            for ctx_msg in context_messages:
                context_lines.append(f"{ctx_msg.author}: {ctx_msg.message}")
            context = "\n".join(context_lines)

            # Create instruction
            instruction = "Continue this conversation as yourself."

            example = TrainingExample(
                instruction=instruction,
                input_text=context,
                output=msg.message,
                system=system,
            )
            examples.append(example)

        return examples

    def format_conversations(
        self,
        conversations: list[Conversation],
        include_system: bool = True,
    ) -> list[TrainingExample]:
        """
        Format multiple conversations into training examples.

        Args:
            conversations: List of Conversation objects
            include_system: Whether to include system prompt

        Returns:
            List of TrainingExample objects
        """
        all_examples = []
        for conv in conversations:
            examples = self.format_conversation(
                conv.messages,
                include_system=include_system,
            )
            all_examples.extend(examples)
        return all_examples

    def export_jsonl(
        self,
        examples: list[TrainingExample],
        output_path: str | Path,
        format_type: str = "alpaca",
    ) -> int:
        """
        Export training examples to JSONL file.

        Args:
            examples: List of TrainingExample objects
            output_path: Output file path
            format_type: Output format ("alpaca", "chatml", "llama")

        Returns:
            Number of examples exported
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                if format_type == "alpaca":
                    # Alpaca format (instruction, input, output)
                    data = example.to_dict()
                elif format_type == "chatml":
                    # ChatML format (text field)
                    data = {"text": example.to_chatml()}
                elif format_type == "llama":
                    # Llama 3 format (text field)
                    data = {"text": example.to_llama()}
                else:
                    raise ValueError(f"Unknown format type: {format_type}")

                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        logger.info(f"Exported {len(examples)} examples to {output_path}")
        return len(examples)

    def create_dataset_from_file(
        self,
        chat_file: str | Path,
        output_path: str | Path,
        format_type: str = "alpaca",
        gap_minutes: int = 60,
    ) -> int:
        """
        Create a training dataset from a WhatsApp export file.

        Args:
            chat_file: Path to WhatsApp export file
            output_path: Output JSONL file path
            format_type: Output format type
            gap_minutes: Gap to split conversations

        Returns:
            Number of examples created
        """
        from ..ingest.whatsapp_parser import WhatsAppParser
        from ..ingest.style_analyzer import StyleAnalyzer

        # Parse conversations
        parser = WhatsAppParser(your_name=self.your_name)
        conversations = parser.parse_to_conversations(
            chat_file,
            gap_minutes=gap_minutes,
        )

        # Analyze style if not provided
        if self.style_metrics is None:
            analyzer = StyleAnalyzer()
            your_messages = parser.get_your_messages(chat_file)
            self.style_metrics = analyzer.analyze(your_messages)

        # Format and export
        examples = self.format_conversations(conversations)
        return self.export_jsonl(examples, output_path, format_type)
