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
    
class DataCleaner:
    @staticmethod
    def clean_message(message: str) -> str:
        """Clean and normalize message text."""
        if not message:
            return ""
        
        # Basic cleaning
        message = message.strip()
        message = ' '.join(message.split())
        
        # Remove special characters but keep essential punctuation
        message = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', message)
        
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
                    # Parse timestamp
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

class SensitiveInfoRemover:
    def __init__(self):
        self.patterns = {
            'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'password': r'(?i)\b(password|passwd|pass|pwd)[ :]+[^\s]+\b',
            'otp': r'\b\d{4,6}\b',
            'pin': r'\b\d{4}\b'
        }

    def remove_sensitive_info(self, text: str) -> str:
        """Remove various types of sensitive information from text."""
        for info_type, pattern in self.patterns.items():
            text = re.sub(pattern, f'[REDACTED_{info_type.upper()}]', text)
        return text

class LLMFormatter:
    @staticmethod
    def format_llama2(context: str, response: str, system_prompt: str) -> str:
        """Format data for Llama-2 style models."""
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

class ConversationManager:
    def __init__(self, context_length: int = 3):
        self.context_length = context_length
        self.current_conversation: List[ChatMessage] = []
        self.conversation_id = 0

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

def jsonl_data(df: pd.DataFrame, your_name: str, system_prompt: str, 
               llm_format: LLMFormat = LLMFormat.LLAMA2, context_length: int = 3) -> List[Dict]:
    """
    Enhanced function to format chat data for fine-tuning various LLM models.
    """
    formatter = LLMFormatter()
    conversation_manager = ConversationManager(context_length)
    formatted_data = []
    
    for _, row in df.iterrows():
        message = ChatMessage(
            author=row['Author'],
            message=row['Message'],
            timestamp=datetime.datetime.now()  # Using current time as default
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
    
    return formatted_data

def converter_with_debug(filepath: str, prompter: str, responder: str, your_name: str,
                        llm_format: LLMFormat = LLMFormat.LLAMA2) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Enhanced converter function with better error handling and debugging.
    """
    chat_parser = ChatParser()
    sensitive_info_remover = SensitiveInfoRemover()
    
    df_finetune = pd.DataFrame(columns=['Author', 'Message', 'Timestamp'])
    result_dict = defaultdict(dict)
    count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as fp:
            for line in fp:
                # Remove sensitive information
                line = sensitive_info_remover.remove_sensitive_info(line)
                
                # Parse the chat message
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
    
    # Format data for fine-tuning
    system_prompt = "Mimic the conversational style of the user, considering the context of the conversation."
    formatted_data = jsonl_data(df_finetune, your_name, system_prompt, llm_format)
    
    return df, formatted_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process WhatsApp chat data for LLM fine-tuning.')
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
        # Convert and parse chat data
        parsed_data, finetune_data = converter_with_debug(
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
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
