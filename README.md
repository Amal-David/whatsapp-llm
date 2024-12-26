# WhatsApp Chat Parser for LLM Fine-tuning

A Python tool to parse WhatsApp chat exports and format them for fine-tuning various Large Language Models (LLMs).

## Features

- Support for multiple LLM formats:
  - Llama-2
  - Mistral
  - Falcon
  - GPT
- Intelligent conversation context management
- Comprehensive sensitive information removal
- Advanced data cleaning and validation
- Proper timestamp handling
- Detailed logging and error handling

## Requirements

```bash
pip install pandas
```

## Usage

### Basic Usage

```bash
python parser.py chat.txt "Prompter" "Responder" "YourName"
```

### Advanced Usage

1. Using different LLM formats:
```bash
python parser.py chat.txt "Prompter" "Responder" "YourName" --llm_format mistral
```

2. Adjusting conversation context length:
```bash
python parser.py chat.txt "Prompter" "Responder" "YourName" --context_length 5
```

### Arguments

- `path`: Path to the WhatsApp chat export file
- `prompter`: Name of the person asking questions
- `responder`: Name of the person responding
- `your_name`: Your name in the chat
- `--llm_format`: Target LLM format (choices: llama2, mistral, falcon, gpt)
- `--context_length`: Number of previous messages to include as context (default: 3)

## Output Files

The script generates two output files:

1. `output_YYYYMMDD_HHMMSS.csv`: Original parsed conversation pairs
2. `formatted_[prompter].jsonl`: Formatted data ready for LLM fine-tuning

## Data Processing Features

### Sensitive Information Removal
- Credit card numbers
- Phone numbers
- Email addresses
- IP addresses
- Passwords
- OTPs and PINs

### Message Validation
- Removes empty or too short messages
- Filters out media messages
- Removes system messages
- Cleans special characters while preserving essential punctuation

### Conversation Context
- Maintains conversation flow
- Configurable context length
- Proper message attribution
- Timestamp preservation

## LLM Format Examples

### Llama-2
```
<s>[INST] <<SYS>>
System prompt
<</SYS>>

Context [/INST] Response </s>
```

### Mistral
```
<s>[INST] System prompt

Context [/INST] Response </s>
```

### Falcon
```
System: System prompt
User: Context
Assistant: Response
```

### GPT
```json
{
    "messages": [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Context"},
        {"role": "assistant", "content": "Response"}
    ]
}
```

## Error Handling

The script includes comprehensive error handling and logging:
- Input file validation
- Timestamp parsing errors
- Data processing issues
- Output file writing errors

## Contributing

Feel free to submit issues and enhancement requests!
