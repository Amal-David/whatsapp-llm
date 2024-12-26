import pandas as pd
from collections import defaultdict
import re
import argparse
import datetime
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMFormat(Enum):
    LLAMA2 = "llama2"
    MISTRAL = "mistral"
    FALCON = "falcon"
    GPT = "gpt"

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
            r"(\d{1,2}/\d{1,2}/\d{2}),\s*(\d{1,2}:\d{2}:\d{2}\s*[APM]{2})\s*-\s*([^:]+):\s*(.+)",
            r"(\d{1,2}/\d{1,2}/\d{2}),\s*(\d{1,2}:\d{2}\s*[APM]{2})\s*-\s*([^:]+):\s*(.+)",
            r"\[(\d{1,2}/\d{1,2}/\d{2}),\s*(\d{1,2}:\d{2}:\d{2}\s*[APM]{2})\]\s*([^:]+):\s*(.+)"
        ]

    def parse_line(self, line: str) -> Optional[ChatMessage]:
        """Parse a single line of chat."""
        for pattern in self.date_patterns:
            match = re.search(pattern, line)
            if match:
                date, time, author, message = match.groups()
                try:
                    timestamp = datetime.datetime.strptime(f"{date} {time}", "%m/%d/%y %I:%M:%S %p")
                    return ChatMessage(
                        author=author.strip(),
                        message=message.strip(),
                        timestamp=timestamp
                    )
                except ValueError as e:
                    logger.warning(f"Error parsing timestamp: {e}")
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

def jsonl_data(df: pd.DataFrame, your_name: str, llm_format: LLMFormat = LLMFormat.LLAMA2, 
               context_length: int = 3) -> Tuple[List[Dict], Dict]:
    """Enhanced function to format chat data for fine-tuning with style analysis."""
    conversation_manager = ConversationManager(context_length)
    formatted_data = []
    
    # Extract personal style
    your_messages = df[df['Author'] == your_name]['Message'].tolist()
    style_metrics = PersonalStyleExtractor().analyze_style(your_messages)
    system_prompt = create_system_prompt(style_metrics)
    
    formatter = LLMFormatter()
    
    for _, row in df.iterrows():
        message = ChatMessage(
            author=row['Author'],
            message=row['Message'],
            timestamp=datetime.datetime.now()
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
            elif llm_format == LLMFormat.MISTRAL:
                formatted_text = formatter.format_mistral(context, message.message, system_prompt)
            elif llm_format == LLMFormat.FALCON:
                formatted_text = formatter.format_falcon(context, message.message, system_prompt)
            else:  # GPT format
                formatted_text = formatter.format_gpt(context, message.message, system_prompt)
            
            entry = {
                "text": formatted_text,
                "conversation_id": len(formatted_data),
                "turn_number": len(conversation_manager.current_conversation),
                "timestamp": message.timestamp.isoformat()
            }
            formatted_data.append(entry)
        
        conversation_manager.add_message(message)
    
    return formatted_data, style_metrics

def converter_with_debug(filepath: str, prompter: str, responder: str, your_name: str,
                        llm_format: LLMFormat = LLMFormat.LLAMA2) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """Enhanced converter function with style analysis."""
    chat_parser = ChatParser()
    
    df_finetune = pd.DataFrame(columns=['Author', 'Message', 'Timestamp'])
    result_dict = defaultdict(dict)
    count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as fp:
            for line in fp:
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
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise
    
    logger.info(f"Processed {count} conversation pairs")
    
    # Convert result_dict to DataFrame
    df = pd.DataFrame.from_dict(result_dict, orient='index')
    
    # Format data for fine-tuning with style analysis
    formatted_data, style_metrics = jsonl_data(df_finetune, your_name, llm_format)
    
    return df, formatted_data, style_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process WhatsApp chat data for personal style LLM fine-tuning.')
    parser.add_argument('path', type=str, help='Path to chat file')
    parser.add_argument('prompter', type=str, help='Name of Prompter')
    parser.add_argument('responder', type=str, help='Name of Responder')
    parser.add_argument('your_name', type=str, help='Your Name')
    parser.add_argument('--llm_format', type=str, choices=['llama2', 'mistral', 'falcon', 'gpt'],
                        default='llama2', help='Target LLM format')
    parser.add_argument('--context_length', type=int, default=3,
                        help='Number of previous messages to include as context')

    args = parser.parse_args()
    
    try:
        # Convert and parse chat data with style analysis
        parsed_data, finetune_data, style_metrics = converter_with_debug(
            args.path,
            args.prompter,
            args.responder,
            args.your_name,
            LLMFormat[args.llm_format.upper()]
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
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
