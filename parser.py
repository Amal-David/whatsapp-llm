import pandas as pd
from collections import defaultdict
import re
import argparse
import datetime
import json
import logging
from typing import List, Dict, Tuple, Optional, Iterable
from dataclasses import dataclass
from enum import Enum
import os
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

class ConversationEpisodeDetector:
    """Detect conversation episodes based on temporal gaps."""

    def __init__(self, gap_hours: float = 6.0):
        self.gap = datetime.timedelta(hours=gap_hours)
        self._metadata: List[Dict[str, object]] = []

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'Timestamp' not in df.columns:
            self._metadata = []
            df = df.copy()
            df['Episode'] = []
            return df

        df = df.sort_values('Timestamp').reset_index(drop=True)
        episodes: List[int] = []
        metadata: List[Dict[str, object]] = []
        current_episode = 1
        episode_start_idx = 0
        last_timestamp = df.at[0, 'Timestamp']

        for idx, row in df.iterrows():
            timestamp = row['Timestamp']
            if idx != 0 and timestamp - last_timestamp > self.gap:
                metadata.append({
                    'episode_id': current_episode,
                    'start': df.at[episode_start_idx, 'Timestamp'].isoformat(),
                    'end': df.at[idx - 1, 'Timestamp'].isoformat(),
                    'message_count': idx - episode_start_idx
                })
                current_episode += 1
                episode_start_idx = idx
            episodes.append(current_episode)
            last_timestamp = timestamp

        metadata.append({
            'episode_id': current_episode,
            'start': df.at[episode_start_idx, 'Timestamp'].isoformat(),
            'end': df.at[len(df) - 1, 'Timestamp'].isoformat(),
            'message_count': len(df) - episode_start_idx
        })

        df['Episode'] = episodes
        self._metadata = metadata
        return df

    @property
    def metadata(self) -> List[Dict[str, object]]:
        return self._metadata

class PersonaTagger:
    """Apply lightweight tone, topic, and intent tagging to chat messages."""

    def __init__(self, stop_words: Optional[Iterable[str]] = None):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stop_words) if stop_words is not None else set(ENGLISH_STOP_WORDS)
        self.intent_keywords: Dict[str, Iterable[str]] = {
            'planning': ['schedule', 'plan', 'meet', 'arrive', 'time', 'calendar', 'book'],
            'informational': ['info', 'details', 'link', 'send', 'update', 'share', 'because'],
            'emotional_support': ['sorry', 'feel', 'proud', 'congrats', 'miss', 'love', 'tough'],
            'logistics': ['drive', 'flight', 'ticket', 'order', 'deliver', 'pickup', 'drop']
        }

    def _classify_tone(self, message: str) -> str:
        if not message:
            return 'neutral'
        scores = self.sentiment_analyzer.polarity_scores(message)
        compound = scores['compound']
        if compound >= 0.35:
            return 'positive'
        if compound <= -0.35:
            return 'negative'
        return 'neutral'

    def _classify_intent(self, message: str) -> str:
        text = message.lower()
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in text for keyword in keywords):
                return intent
        if '?' in message:
            return 'question'
        if text.startswith(('let', "let's", 'shall')):
            return 'planning'
        return 'general'

    def _top_terms(self, messages: List[str]) -> List[str]:
        if not messages:
            return []
        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            max_features=25,
            ngram_range=(1, 2)
        )
        try:
            matrix = vectorizer.fit_transform(messages)
        except ValueError:
            return []

        sums = matrix.sum(axis=0)
        terms = vectorizer.get_feature_names_out()
        ranking = [terms[i] for i in sums.A1.argsort()[::-1]]
        return ranking

    def tag_dataframe(self, df: pd.DataFrame, contact_name: str, your_name: str) -> pd.DataFrame:
        if df.empty:
            df = df.copy()
            df['Tone'] = []
            df['Intent'] = []
            df['Topic'] = []
            return df

        df = df.copy()
        messages = df['Message'].astype(str).tolist()
        top_terms = self._top_terms(messages)

        def determine_topic(message: str) -> str:
            lower_msg = message.lower()
            for term in top_terms:
                term_lower = term.lower()
                if len(term_lower) < 3:
                    continue
                if term_lower in lower_msg:
                    return term_lower
            if contact_name.lower() in lower_msg:
                return 'relationship'
            if your_name.lower() in lower_msg:
                return 'self-referential'
            return 'general'

        df['Tone'] = df['Message'].apply(self._classify_tone)
        df['Intent'] = df['Message'].apply(self._classify_intent)
        df['Topic'] = df['Message'].apply(determine_topic)
        return df

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
                timestamp = None
                # Try multiple date formats
                for date_format in ["%m/%d/%y", "%d/%m/%y", "%Y-%m-%d"]:
                    try:
                        fmt = "%Y-%m-%d" if "-" in date else date_format
                        timestamp = datetime.datetime.strptime(f"{date} {time}", f"{fmt} %I:%M:%S %p")
                        break
                    except ValueError:
                        try:
                            timestamp = datetime.datetime.strptime(f"{date} {time}", f"{fmt} %I:%M %p")
                            break
                        except ValueError:
                            timestamp = None
                            continue
                if timestamp is None:
                    logger.warning(f"Error parsing timestamp for line: {line}")
                    continue
                return ChatMessage(
                    author=author.strip(),
                    message=message.strip(),
                    timestamp=timestamp
                )
        return None

def parse_chat_file(filepath: str) -> pd.DataFrame:
    chat_parser = ChatParser()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Chat file not found: {filepath}")

    if os.path.getsize(filepath) == 0:
        raise ValueError("Chat file is empty")

    records = []

    with open(filepath, 'r', encoding='utf-8') as fp:
        for line_num, line in enumerate(fp, 1):
            chat_message = chat_parser.parse_line(line)
            if chat_message:
                records.append({
                    'Author': chat_message.author,
                    'Message': chat_message.message,
                    'Timestamp': chat_message.timestamp
                })
            elif line.strip():
                logger.warning(f"Line {line_num} did not match any pattern: {line.strip()}")

    df = pd.DataFrame(records, columns=['Author', 'Message', 'Timestamp'])
    return df

def filter_contact_chat(
    df: pd.DataFrame,
    contact_name: str,
    your_name: Optional[str] = None,
    include_self: bool = True
) -> pd.DataFrame:
    if df.empty:
        return df

    normalized_contact = contact_name.lower()
    allowed_authors = {normalized_contact}
    if include_self and your_name:
        allowed_authors.add(your_name.lower())

    mask = df['Author'].str.lower().isin(allowed_authors)
    filtered = df[mask].reset_index(drop=True)

    unexpected_authors = set(filtered['Author'].str.lower()) - allowed_authors
    if unexpected_authors:
        logger.warning(
            "Detected unexpected authors after contact filtering: %s",
            ', '.join(sorted(unexpected_authors))
        )

    media_placeholders = filtered['Message'].str.contains(
        r'<media omitted>|omitted', case=False, na=False
    )
    if media_placeholders.any():
        logger.warning(
            "Media placeholders remain in filtered chat. Consider removing %d entries.",
            media_placeholders.sum()
        )

    return filtered

def extract_contact_chat(
    chat_path: str,
    contact_name: str,
    your_name: Optional[str] = None,
    include_self: bool = True
) -> pd.DataFrame:
    df = parse_chat_file(chat_path)
    return filter_contact_chat(df, contact_name, your_name, include_self)

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

def jsonl_data(
    df: pd.DataFrame,
    your_name: str,
    llm_format: LLMFormat = LLMFormat.LLAMA2,
    context_length: int = 3,
    contact_name: Optional[str] = None,
    episode_gap_hours: float = 6.0
) -> Tuple[List[Dict], Dict, Dict[str, object]]:
    """Enhanced function to format chat data for fine-tuning with style and persona analysis."""
    conversation_manager = ConversationManager(context_length)
    formatted_data: List[Dict] = []

    if not df.empty:
        df = df.sort_values('Timestamp').reset_index(drop=True)

    detector = ConversationEpisodeDetector(gap_hours=episode_gap_hours)
    if df.empty:
        df_with_episodes = pd.DataFrame(columns=['Author', 'Message', 'Timestamp', 'Episode'])
    else:
        df_with_episodes = detector.annotate(df)

    persona_tagger = PersonaTagger()
    target_contact = contact_name or your_name
    tagged_df = persona_tagger.tag_dataframe(df_with_episodes, target_contact, your_name)

    style_extractor = PersonalStyleExtractor()
    author_style_metrics: Dict[str, Dict] = {}
    if not tagged_df.empty and 'Author' in tagged_df.columns:
        for author in tagged_df['Author'].dropna().unique():
            author_messages = tagged_df[tagged_df['Author'] == author]['Message'].tolist()
            author_style_metrics[str(author)] = style_extractor.analyze_style(author_messages)

    # Extract personal style for training examples that model your responses.
    your_messages = tagged_df[tagged_df['Author'] == your_name]['Message'].tolist()
    style_metrics = style_extractor.analyze_style(your_messages)
    system_prompt = create_system_prompt(style_metrics)

    formatter = LLMFormatter()

    for _, row in tagged_df.iterrows():
        message = ChatMessage(
            author=row['Author'],
            message=row['Message'],
            timestamp=row['Timestamp']
        )

        if not DataCleaner.validate_message(message.message):
            continue

        message.message = DataCleaner.clean_message(message.message)

        if message.author == your_name and len(conversation_manager.current_conversation) > 0:
            context = conversation_manager.format_context(
                conversation_manager.current_conversation[-context_length:]
            )

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
            else:
                formatted_text = formatter.format_gpt(context, message.message, system_prompt)

            entry = {
                "text": formatted_text,
                "conversation_id": len(formatted_data),
                "turn_number": len(conversation_manager.current_conversation),
                "timestamp": message.timestamp.isoformat() if message.timestamp else None,
                "persona_tags": {
                    "tone": row.get('Tone', 'neutral'),
                    "intent": row.get('Intent', 'general'),
                    "topic": row.get('Topic', 'general'),
                    "episode_id": row.get('Episode')
                }
            }
            formatted_data.append(entry)

        conversation_manager.add_message(message)

    persona_context = {
        "episode_metadata": detector.metadata if not df.empty else [],
        "tagged_messages": tagged_df.to_dict('records'),
        "author_style_metrics": author_style_metrics
    }

    return formatted_data, style_metrics, persona_context

def converter_with_debug(
    filepath: str,
    prompter: str,
    responder: str,
    your_name: str,
    llm_format: LLMFormat = LLMFormat.LLAMA2,
    context_length: int = 3,
    contact_name: Optional[str] = None,
    include_self: bool = True,
    episode_gap_hours: float = 6.0
) -> Tuple[pd.DataFrame, List[Dict], Dict, Dict[str, object]]:
    """Enhanced converter function with style analysis."""
    df_finetune = parse_chat_file(filepath)

    if contact_name:
        df_finetune = filter_contact_chat(
            df_finetune,
            contact_name=contact_name,
            your_name=your_name,
            include_self=include_self
        )

    result_dict: Dict[int, Dict[str, str]] = defaultdict(dict)
    pair_index = 0

    for _, row in df_finetune.iterrows():
        author = row['Author']
        if author == prompter:
            result_dict[pair_index]['prompt'] = row['Message']
        elif author == responder and pair_index in result_dict and 'prompt' in result_dict[pair_index]:
            result_dict[pair_index]['completion'] = row['Message']
            pair_index += 1

    completed_pairs = sum(1 for record in result_dict.values() if 'prompt' in record and 'completion' in record)
    if completed_pairs == 0:
        logger.warning("No valid conversation pairs found in the chat file")
    else:
        logger.info("Processed %d conversation pairs", completed_pairs)

    df_pairs = pd.DataFrame.from_dict(result_dict, orient='index')

    formatted_data, style_metrics, persona_context = jsonl_data(
        df_finetune,
        your_name,
        llm_format,
        context_length,
        contact_name=contact_name,
        episode_gap_hours=episode_gap_hours
    )

    return df_pairs, formatted_data, style_metrics, persona_context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process WhatsApp chat data for personal style LLM fine-tuning.')
    parser.add_argument('path', type=str, help='Path to chat file')
    parser.add_argument('prompter', type=str, help='Name of Prompter')
    parser.add_argument('responder', type=str, help='Name of Responder')
    parser.add_argument('your_name', type=str, help='Your Name')
    parser.add_argument('--llm_format', type=str, choices=['llama2', 'llama3', 'mistral', 'falcon', 'gpt', 'qwen'],
                        default='llama2', help='Target LLM format')
    parser.add_argument('--context_length', type=int, default=3,
                        help='Number of previous messages to include as context')
    parser.add_argument('--contact', type=str, default=None,
                        help='Focus on a specific contact when exporting the chat')
    parser.add_argument('--include_self', action='store_true',
                        help='Include your own messages when filtering by contact')
    parser.add_argument('--episode_gap_hours', type=float, default=6.0,
                        help='Gap (in hours) that splits conversation episodes')
    parser.add_argument('--persona_metadata', type=str, default=None,
                        help='Optional path to store persona tagging metadata (JSON)')

    args = parser.parse_args()

    try:
        # Convert and parse chat data with style analysis
        parsed_data, finetune_data, style_metrics, persona_context = converter_with_debug(
            args.path,
            args.prompter,
            args.responder,
            args.your_name,
            LLMFormat[args.llm_format.upper()],
            args.context_length,
            contact_name=args.contact,
            include_self=args.include_self,
            episode_gap_hours=args.episode_gap_hours
        )

        # Save the original parsed data
        output_file = f'output_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv'
        parsed_data.to_csv(output_file)
        logger.info(f"Saved parsed data to {output_file}")

        # Save the formatted data in JSONL format
        jsonl_file = f'formatted_{args.prompter}.jsonl'
        with open(jsonl_file, 'w') as f:
            for item in finetune_data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved formatted data to {jsonl_file}")

        # Save style metrics
        style_file = f'style_metrics_{args.your_name}.json'
        with open(style_file, 'w') as f:
            json.dump(style_metrics, f, indent=2)
        logger.info(f"Saved style metrics to {style_file}")

        if args.persona_metadata:
            with open(args.persona_metadata, 'w') as f:
                json.dump(persona_context, f, indent=2, default=str)
            logger.info("Saved persona metadata to %s", args.persona_metadata)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
