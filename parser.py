import pandas as pd
from collections import defaultdict
import re
import argparse
import datetime
import json
import logging
from typing import List, Dict, Tuple, Optional
import dataclasses # For asdict
from dataclasses import dataclass
from enum import Enum
import os

from .elizaos_character import ElizaOsCharacter, MessageTurn, MessageContent, Style, KnowledgeItem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMFormat(Enum):
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    FALCON = "falcon"
    GPT = "gpt"
    QWEN = "qwen"
    ELIZAOS = "elizaos"

    @classmethod
    def from_string(cls, s: str) -> 'LLMFormat':
        try:
            return cls[s.upper()]
        except KeyError:
            try:
                return next(m for m in cls if m.value.lower() == s.lower())
            except StopIteration:
                raise ValueError(f"Unknown LLM format: {s}")

@dataclass
class ChatMessage:
    author: str
    message: str
    timestamp: datetime.datetime
    
class PersonalStyleExtractor:
    def __init__(self):
        self.emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
        self.slang_abbreviations = {
            'lol', 'omg', 'idk', 'tbh', 'imo', 'fyi', 'brb', 'btw', 'afk',
            'nvm', 'aka', 'asap', 'fomo', 'fwiw', 'iirc', 'imho', 'irl',
            'jk', 'lmk', 'nbd', 'np', 'nsfw', 'rn', 'tbf', 'tfw', 'tl;dr',
            'tysm', 'w/', 'w/o', 'ya', 'ymmv', 'yolo'
        }
        
    def analyze_style(self, messages: List[str]) -> Dict:
        """Analyze personal conversation style metrics."""
        style_metrics = {
            'avg_message_length': 0,
            'emoji_usage_rate': 0,
            'slang_usage_rate': 0,
            'punctuation_patterns': defaultdict(int),
            'capitalization_rate': 0,
            'common_phrases': defaultdict(int),
            'response_patterns': []
        }
        
        total_messages = len(messages)
        if total_messages == 0:
            return style_metrics
            
        # Analyze messages
        total_length = 0
        emoji_count = 0
        slang_count = 0
        capitalized_count = 0
        
        for msg in messages:
            # Message length
            total_length += len(msg)
            
            # Emoji usage
            emoji_count += len(self.emoji_pattern.findall(msg))
            
            # Slang usage
            words = msg.lower().split()
            slang_count += sum(1 for word in words if word in self.slang_abbreviations)
            
            # Capitalization
            if msg and msg[0].isupper():
                capitalized_count += 1
            
            # Punctuation patterns
            for char in '.,!?...':
                if char in msg:
                    style_metrics['punctuation_patterns'][char] += 1
            
            # Common phrases (3-grams)
            words = msg.split()
            for i in range(len(words)-2):
                phrase = ' '.join(words[i:i+3])
                style_metrics['common_phrases'][phrase] += 1
        
        # Calculate averages
        style_metrics['avg_message_length'] = total_length / total_messages
        style_metrics['emoji_usage_rate'] = emoji_count / total_messages
        style_metrics['slang_usage_rate'] = slang_count / total_messages
        style_metrics['capitalization_rate'] = capitalized_count / total_messages
        
        # Keep only most common phrases
        style_metrics['common_phrases'] = dict(
            sorted(style_metrics['common_phrases'].items(), 
                  key=lambda x: x[1], 
                  reverse=True)[:10]
        )
        
        return style_metrics

class DataCleaner:
    @staticmethod
    def clean_message(message: str) -> str:
        """Clean and normalize message text while preserving personal style."""
        if not message:
            return ""
        
        # Basic cleaning while preserving emojis and style
        message = message.strip()
        message = ' '.join(message.split())
        
        # Remove URLs but keep other elements
        message = re.sub(r'http\S+|www.\S+', '[URL]', message)
        
        return message

    @staticmethod
    def validate_message(message: str) -> bool:
        """Validate if a message should be included in the dataset."""
        if not message or len(message.strip()) < 2:
            return False
            
        # Skip media and system messages
        skip_patterns = [
            '<media omitted>',
            '[redacted]',
            'message deleted',
            'image omitted',
            'video omitted',
            'audio omitted',
            'document omitted'
        ]
        
        return not any(pattern.lower() in message.lower() for pattern in skip_patterns)

class ChatParser:
    def __init__(self):
        self.date_patterns = [
            # Standard WhatsApp format (12/31/23, 11:59:59 PM)
            r"(\d{1,2}/\d{1,2}/\d{2}),\s*(\d{1,2}:\d{2}:\d{2}\s*[APM]{2})\s*-\s*([^:]+):\s*(.+)",
            # Without seconds (12/31/23, 11:59 PM)
            r"(\d{1,2}/\d{1,2}/\d{2}),\s*(\d{1,2}:\d{2}\s*[APM]{2})\s*-\s*([^:]+):\s*(.+)",
            # Bracketed format [12/31/23, 11:59:59 PM]
            r"\[(\d{1,2}/\d{1,2}/\d{2}),\s*(\d{1,2}:\d{2}:\d{2}\s*[APM]{2})\]\s*([^:]+):\s*(.+)",
            # International format (31/12/23)
            r"(\d{1,2}/\d{1,2}/\d{2}),\s*(\d{1,2}:\d{2}(?::\d{2})?\s*[APM]{2})\s*-\s*([^:]+):\s*(.+)",
            # ISO-like format (2023-12-31)
            r"(\d{4}-\d{2}-\d{2}),\s*(\d{1,2}:\d{2}(?::\d{2})?\s*[APM]{2})\s*-\s*([^:]+):\s*(.+)"
        ]

    def parse_line(self, line: str) -> Optional[ChatMessage]:
        """Parse a single line of chat."""
        line = line.strip()
        if not line:
            return None
            
        for pattern in self.date_patterns:
            match = re.search(pattern, line)
            if match:
                date, time, author, message = match.groups()
                try:
                    # Try multiple date formats
                    for date_format in ["%m/%d/%y", "%d/%m/%y", "%Y-%m-%d"]:
                        try:
                            if "-" in date:
                                date_format = "%Y-%m-%d"
                            timestamp = datetime.datetime.strptime(f"{date} {time}", f"{date_format} %I:%M:%S %p")
                            break
                        except ValueError:
                            try:
                                timestamp = datetime.datetime.strptime(f"{date} {time}", f"{date_format} %I:%M %p")
                                break
                            except ValueError:
                                continue
                    return ChatMessage(
                        author=author.strip(),
                        message=message.strip(),
                        timestamp=timestamp
                    )
                except ValueError as e:
                    logger.warning(f"Error parsing timestamp for date string '{date}' and time string '{time}': {e}. Line: {line}")
                    continue
        return None

class ConversationManager:
    def __init__(self, context_length: int = 3):
        self.context_length = context_length
        self.current_conversation: List[ChatMessage] = []
        self.conversation_id = 0
        self.style_extractor = PersonalStyleExtractor()

    def format_context(self, messages: List[ChatMessage]) -> str:
        """Format conversation context."""
        return "\n".join([f"{msg.author}: {msg.message}" for msg in messages])

    def add_message(self, message: ChatMessage) -> Optional[Dict]:
        """Add a message to the current conversation and return formatted data if appropriate."""
        self.current_conversation.append(message)
        
        # Trim conversation if it exceeds twice the context length
        if len(self.current_conversation) > self.context_length * 2:
            self.current_conversation = self.current_conversation[-self.context_length:]
            
        return None

def create_system_prompt(style_metrics: Dict) -> str:
    """Create a personalized system prompt based on style metrics."""
    prompt = "You are now mimicking a person with the following conversation style:\n"
    prompt += f"- Average message length: {style_metrics['avg_message_length']:.1f} characters\n"
    prompt += f"- Emoji usage: {'High' if style_metrics['emoji_usage_rate'] > 0.3 else 'Moderate' if style_metrics['emoji_usage_rate'] > 0.1 else 'Low'}\n"
    prompt += f"- Slang usage: {'High' if style_metrics['slang_usage_rate'] > 0.2 else 'Moderate' if style_metrics['slang_usage_rate'] > 0.1 else 'Low'}\n"
    prompt += f"- Capitalization: {'Usually' if style_metrics['capitalization_rate'] > 0.7 else 'Sometimes' if style_metrics['capitalization_rate'] > 0.3 else 'Rarely'} starts sentences with capital letters\n"
    
    if style_metrics['common_phrases']:
        prompt += "- Frequently used phrases:\n"
        for phrase, count in list(style_metrics['common_phrases'].items())[:5]:
            prompt += f"  * {phrase}\n"
    
    prompt += "\nMimic this conversational style while maintaining context and natural flow."
    return prompt

class LLMFormatter:
    @staticmethod
    def format_llama2(context: str, response: str, system_prompt: str) -> str:
        """Format data for Llama-2 style models."""
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{context} [/INST] {response} </s>"

    @staticmethod
    def format_llama3(context: str, response: str, system_prompt: str) -> str:
        """Format data for Llama-3 style models."""
        # Llama3 uses a slightly different format with tiktoken-based tokenizer
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{context} [/INST] {response} </s>"

    @staticmethod
    def format_mistral(context: str, response: str, system_prompt: str) -> str:
        """Format data for Mistral style models."""
        return f"<s>[INST] {system_prompt}\n\n{context} [/INST] {response} </s>"

    @staticmethod
    def format_falcon(context: str, response: str, system_prompt: str) -> str:
        """Format data for Falcon style models."""
        return f"System: {system_prompt}\nUser: {context}\nAssistant: {response}"

    @staticmethod
    def format_gpt(context: str, response: str, system_prompt: str) -> str:
        """Format data for GPT style models."""
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
                {"role": "assistant", "content": response}
            ]
        }
        
    @staticmethod
    def format_qwen(context: str, response: str, system_prompt: str) -> str:
        """Format data for Qwen style models."""
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

def jsonl_data(df: pd.DataFrame, your_name: str, system_prompt: str, llm_format: LLMFormat, 
               context_length: int = 3) -> List[Dict]:
    """Formats chat data for fine-tuning LLMs (excluding ElizaOS)."""
    conversation_manager = ConversationManager(context_length)
    formatted_data = []
    formatter = LLMFormatter()
    
    # This function should not be called if llm_format is ELIZAOS by the new logic in converter_with_debug
    if llm_format == LLMFormat.ELIZAOS:
        # Should not happen if converter_with_debug is correctly implemented
        logger.warning("jsonl_data called with ELIZAOS format, returning empty list.")
        return []
    
    for _, row in df.iterrows():
        message = ChatMessage(
            author=row['Author'],
            message=row['Message'],
            timestamp=row['Timestamp']  # Use the actual timestamp from the DataFrame
        )
        
        if not DataCleaner.validate_message(message.message):
            continue
            
        message.message = DataCleaner.clean_message(message.message)
        
        if message.author == your_name and len(conversation_manager.current_conversation) > 0:
            context = conversation_manager.format_context(
                conversation_manager.current_conversation[-context_length:]
            )
            
            # Format based on selected LLM type
            if llm_format == LLMFormat.LLAMA2:
                formatted_text = formatter.format_llama2(context, message.message, system_prompt)
            elif llm_format == LLMFormat.LLAMA3:
                formatted_text = formatter.format_llama3(context, message.message, system_prompt)
            elif llm_format == LLMFormat.MISTRAL:
                formatted_text = formatter.format_mistral(context, message.message, system_prompt)
            elif llm_format == LLMFormat.FALCON:
                formatted_text = formatter.format_falcon(context, message.message, system_prompt)
            elif llm_format == LLMFormat.QWEN:
                formatted_text = formatter.format_qwen(context, message.message, system_prompt)
            else:  # GPT format
                formatted_text = formatter.format_gpt(context, message.message, system_prompt)
            
            entry = {
                "text": formatted_text,
                "conversation_id": len(formatted_data),
                "turn_number": len(conversation_manager.current_conversation),
                "timestamp": message.timestamp.isoformat() if message.timestamp else None
            }
            formatted_data.append(entry)
        
        conversation_manager.add_message(message)
    
    return formatted_data

def converter_with_debug(filepath: str, prompter: str, responder: str, your_name: str,
                        llm_format: LLMFormat = LLMFormat.LLAMA2, 
                        context_length: int = 3) -> Tuple[pd.DataFrame, Optional[List[Dict]], Optional[ElizaOsCharacter], Dict]:
    """
    Enhanced converter function with style analysis and ElizaOS character generation.
    The context_length parameter is now passed from main.
    """
    chat_parser = ChatParser()

    if not isinstance(filepath, str):
        raise TypeError(f"Filepath must be a string, got {type(filepath)}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Chat file not found: {filepath}")
        
    if os.path.getsize(filepath) == 0:
        raise ValueError("Chat file is empty")
    
    df_finetune = pd.DataFrame(columns=['Author', 'Message', 'Timestamp'])
    result_dict = defaultdict(dict)
    count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as fp:
            for line_num, line in enumerate(fp, 1):
                try:
                    chat_message = chat_parser.parse_line(line)
                    if chat_message:
                        df_finetune.loc[len(df_finetune)] = {
                            'Author': chat_message.author,
                            'Message': chat_message.message,
                            'Timestamp': chat_message.timestamp
                        }
                        
                        if chat_message.author == prompter:
                            result_dict[count]['prompt'] = chat_message.message
                        elif chat_message.author == responder and count in result_dict:
                            result_dict[count]['completion'] = chat_message.message
                            count += 1
                    elif line.strip():  # Log only non-empty unmatched lines
                        logger.warning(f"Line {line_num} did not match any pattern: {line.strip()}")
                except Exception as e:
                    logger.error(f"Error processing line {line_num} in file {filepath}: {e}. Content: {line.strip()}")
                    # Optionally, re-raise or continue based on desired behavior for line-level errors
                    continue # Continue to the next line if one line fails
    
    except IOError as e:
        logger.error(f"IOError occurred while reading file {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during main processing loop of file {filepath}: {e}")
        raise
    
    if count == 0:
        logger.warning("No valid conversation pairs found in the chat file")
    else:
        logger.info(f"Processed {count} conversation pairs")
    
    # Convert result_dict to DataFrame
    df = pd.DataFrame.from_dict(result_dict, orient='index')

    # Calculate style metrics and system prompt first, as they are common
    your_messages = df_finetune[df_finetune['Author'] == your_name]['Message'].tolist()
    style_metrics = PersonalStyleExtractor().analyze_style(your_messages)
    system_prompt = create_system_prompt(style_metrics)

    if llm_format == LLMFormat.ELIZAOS:
        elizaos_char_data = format_elizaos_character(
            df_finetune, 
            your_name, 
            style_metrics, 
            system_prompt, 
            context_length  # Pass context_length here
        )
        return df, None, elizaos_char_data, style_metrics
    else:
        # For other LLM formats, generate the JSONL data
        # Note: jsonl_data now takes system_prompt as an argument
        formatted_llm_data = jsonl_data(
            df_finetune, 
            your_name, 
            system_prompt, 
            llm_format, 
            context_length
        )
        return df, formatted_llm_data, None, style_metrics

# Helper functions for ElizaOS character object population

def _get_elizaos_bio(system_prompt: str) -> List[str]:
    """
    Extracts a bio from the system prompt.
    Splits by newlines and filters empty strings.
    """
    bio_lines = [line.strip() for line in system_prompt.split('\n') if line.strip()]
    if not bio_lines:
        return ["A character bio."]
    return bio_lines

def _get_elizaos_lore() -> List[str]:
    """
    Returns placeholder lore.
    """
    return ["Character lore to be added."]

def _get_elizaos_adjectives(style_metrics: Dict) -> List[str]:
    """
    Derives descriptive adjectives from style_metrics.
    """
    adjectives = []
    if style_metrics.get('emoji_usage_rate', 0) > 0.3:
        adjectives.append("expressive")
    elif style_metrics.get('emoji_usage_rate', 0) > 0.1:
        adjectives.append("uses emojis moderately")
    
    if style_metrics.get('slang_usage_rate', 0) > 0.2:
        adjectives.append("informal")
    elif style_metrics.get('slang_usage_rate', 0) > 0.05:
        adjectives.append("uses some slang")

    capitalization_rate = style_metrics.get('capitalization_rate', 0.5) # Default to mid-range if not present
    if capitalization_rate < 0.3:
        adjectives.append("casual typist")
    elif capitalization_rate > 0.7:
        adjectives.append("formal typist")
    
    avg_len = style_metrics.get('avg_message_length', 100) # Default to mid-range
    if avg_len < 50:
        adjectives.append("concise")
    elif avg_len > 150:
        adjectives.append("verbose")
        
    if not adjectives:
        return ["descriptive adjective to be added"]
    return adjectives

def _get_elizaos_topics(style_metrics: Dict) -> List[str]:
    """
    Extracts top common phrases as topics from style_metrics.
    """
    common_phrases_dict = style_metrics.get('common_phrases', {})
    # common_phrases in PersonalStyleExtractor is already a dict of phrase: count
    # and sorted by count, taking top 10. We can take top 3-5 from this.
    topics = list(common_phrases_dict.keys())[:5] # Take up to the first 5
    
    if not topics:
        return ["general topics"]
    return topics

def _get_elizaos_post_examples(your_messages: List[str]) -> List[str]:
    """
    Selects the longest 1-3 messages from your_messages as post examples.
    Filters out very short messages.
    """
    if not your_messages:
        return ["A sample post by the character."]
    
    # Filter out potentially empty or very short messages, and non-string messages
    valid_messages = [msg for msg in your_messages if isinstance(msg, str) and len(msg) > 10]

    if not valid_messages:
        return ["A sample post by the character."]
        
    # Sort by length in descending order
    sorted_messages = sorted(valid_messages, key=len, reverse=True)
    
    # Return top 1-3 messages
    return sorted_messages[:3]

def _get_elizaos_message_examples(all_messages: List[ChatMessage], your_name: str, 
                                 context_length: int = 3, max_examples: int = 5) -> List[List[MessageTurn]]:
    """
    Generates example conversation snippets based on chat messages.
    """
    example_snippets: List[List[MessageTurn]] = []

    if not all_messages:
        # Ensure MessageContent is a dict as it's a TypedDict
        return [[MessageTurn(user="System", content={'text': "No message examples available.", 'action': None})]]

    for i, current_message in enumerate(all_messages):
        if current_message.author == your_name:
            current_snippet_turns: List[MessageTurn] = []
            
            # Gather context (messages before current_message, from other authors)
            context_start_index = max(0, i - context_length)
            for j in range(context_start_index, i):
                context_message = all_messages[j]
                # Ensure context messages are not from 'your_name' for a natural flow,
                # though typically context is just the preceding messages.
                # For this implementation, we'll include all preceding messages in the window.
                current_snippet_turns.append(
                    MessageTurn(
                        user=context_message.author,
                        content={'text': context_message.message, 'action': None}
                    )
                )
            
            # Add the character's response
            current_snippet_turns.append(
                MessageTurn(
                    user=current_message.author, # This is your_name
                    content={'text': current_message.message, 'action': "CONTINUE"} # Or None
                )
            )
            
            # Add to example_snippets if it contains at least one context message and the response,
            # or simply if it's a valid snippet. The schema requires minItems: 1 for turns.
            # We only add if there was at least one character message that initiated this.
            if len(current_snippet_turns) > 0: # Ensures the character's message itself is added.
                                               # The problem asks for context + response.
                                               # A more strict check could be `len(current_snippet_turns) > 1` 
                                               # if context is strictly required.
                                               # Given the schema, even a single turn is a valid list of turns.
                example_snippets.append(current_snippet_turns)

            if len(example_snippets) >= max_examples:
                break
                
    if not example_snippets:
        return [[MessageTurn(user="System", content={'text': "No message examples available.", 'action': None})]]
        
    return example_snippets

def _get_elizaos_style(system_prompt: str) -> Dict[str, List[str]]:
    """
    Creates the style dictionary from the system prompt.
    """
    style_lines = [line.strip() for line in system_prompt.split('\n') if line.strip()]
    if not style_lines:
        style_lines = ["General character style."]
    
    return {
        "all": style_lines,
        "chat": style_lines, # For now, reuse the same list
        "post": style_lines  # For now, reuse the same list
    }

def format_elizaos_character(
    df_finetune: pd.DataFrame, 
    your_name: str, 
    style_metrics: Dict, 
    system_prompt: str, 
    context_length: int = 3, 
    max_examples: int = 5
) -> ElizaOsCharacter:
    """
    Formats the processed chat data and metrics into an ElizaOsCharacter object.
    """
    
    # Convert DataFrame to List[ChatMessage]
    all_messages_list: List[ChatMessage] = []
    for row in df_finetune.itertuples():
        # Assuming df_finetune has columns 'Author', 'Message', 'Timestamp'
        # and 'Timestamp' is already a datetime.datetime object.
        all_messages_list.append(
            ChatMessage(
                author=getattr(row, 'Author'),
                message=getattr(row, 'Message'),
                timestamp=getattr(row, 'Timestamp')
            )
        )

    name_val = your_name
    bio_val = _get_elizaos_bio(system_prompt)
    lore_val = _get_elizaos_lore()
    
    message_examples_val = _get_elizaos_message_examples(
        all_messages_list, your_name, context_length, max_examples
    )
    
    your_messages_str_list = [msg.message for msg in all_messages_list if msg.author == your_name]
    post_examples_val = _get_elizaos_post_examples(your_messages_str_list)
    
    adjectives_val = _get_elizaos_adjectives(style_metrics)
    topics_val = _get_elizaos_topics(style_metrics)
    
    style_data_dict = _get_elizaos_style(system_prompt)
    # Style is a dataclass, so instantiate it
    style_obj = Style(
        all=style_data_dict['all'], 
        chat=style_data_dict['chat'], 
        post=style_data_dict['post']
    )
    
    # Initialize knowledge as an empty list of KnowledgeItem
    knowledge_val: List[KnowledgeItem] = []

    character = ElizaOsCharacter(
        name=name_val,
        bio=bio_val,
        lore=lore_val,
        messageExamples=message_examples_val,
        postExamples=post_examples_val,
        adjectives=adjectives_val,
        topics=topics_val,
        style=style_obj,
        knowledge=knowledge_val
    )
    
    return character

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process WhatsApp chat data for personal style LLM fine-tuning.')
    parser.add_argument('path', type=str, help='Path to chat file')
    parser.add_argument('prompter', type=str, help='Name of Prompter')
    parser.add_argument('responder', type=str, help='Name of Responder')
    parser.add_argument('your_name', type=str, help='Your Name')
    parser.add_argument('--llm_format', type=str, 
                        choices=[f.value for f in LLMFormat], # Use enum values for choices
                        default=LLMFormat.LLAMA2.value, help='Target LLM format or elizaos')
    parser.add_argument('--context_length', type=int, default=3,
                        help='Number of previous messages to include as context for LLM fine-tuning and ElizaOS examples')

    args = parser.parse_args()
    
    try:
        selected_llm_format = LLMFormat.from_string(args.llm_format)

        # Convert and parse chat data with style analysis
        # Pass context_length from args to converter_with_debug
        parsed_data, fine_tune_output, elizaos_char_output, style_metrics_output = converter_with_debug(
            args.path,
            args.prompter,
            args.responder,
            args.your_name,
            selected_llm_format,
            args.context_length 
        )
        
        # Save the original parsed data (e.g. prompter/responder pairs)
        output_file = f'parsed_chat_data_{args.prompter}_{args.responder}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv'
        parsed_data.to_csv(output_file)
        logger.info(f"Saved parsed data to {output_file}")

        if elizaos_char_output:
            # Save ElizaOS character data
            elizaos_file = f'elizaos_character_{args.your_name}.json'
            # Convert dataclass to dict for JSON serialization
            elizaos_dict = dataclasses.asdict(elizaos_char_output)
            with open(elizaos_file, 'w', encoding='utf-8') as f:
                json.dump(elizaos_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved ElizaOS character data to {elizaos_file}")
        elif fine_tune_output:
            # Save the formatted data in JSONL format for LLMs
            jsonl_file = f'formatted_llm_data_{args.your_name}_{args.llm_format}.jsonl'
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for item in fine_tune_output:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Saved formatted LLM data to {jsonl_file}")
        else:
            logger.info("No specific output data generated (ElizaOS or LLM fine-tune).")

        # Save style metrics (always produced)
        style_file = f'style_metrics_{args.your_name}.json'
        with open(style_file, 'w', encoding='utf-8') as f:
            json.dump(style_metrics_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved style metrics to {style_file}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
