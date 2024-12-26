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
- Built-in fine-tuning support using HuggingFace

## Requirements

```bash
pip install pandas transformers torch datasets tensorboard
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

### Fine-tuning

After generating the formatted data, you can fine-tune a model using the provided `finetune.py` script:

```bash
python finetune.py \
    --data_path formatted_prompter.jsonl \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --output_dir "./fine_tuned_model" \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_epochs 3
```

#### Fine-tuning Arguments

- `--data_path`: Path to the JSONL file containing the training data
- `--model_name`: Name or path of the base model to fine-tune
- `--output_dir`: Directory to save the fine-tuned model
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--num_epochs`: Number of training epochs (default: 3)
- `--max_length`: Maximum sequence length (default: 2048)
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 4)
- `--tokenizer_name`: Name or path of the tokenizer if different from model

#### Recommended Models for Fine-tuning

1. For Llama2 format:
   - meta-llama/Llama-2-7b-chat-hf
   - meta-llama/Llama-2-13b-chat-hf

2. For Mistral format:
   - mistralai/Mistral-7B-v0.1
   - mistralai/Mistral-7B-Instruct-v0.1

3. For Falcon format:
   - tiiuae/falcon-7b
   - tiiuae/falcon-40b

4. For GPT format:
   - gpt2
   - gpt2-medium
   - gpt2-large

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

## Fine-tuning Tips

1. **Hardware Requirements**:
   - 7B models: At least 16GB GPU VRAM
   - 13B models: At least 24GB GPU VRAM
   - 40B+ models: Multiple GPUs recommended

2. **Optimization Tips**:
   - Use gradient accumulation for larger effective batch sizes
   - Enable mixed precision training (fp16)
   - Start with a small learning rate (2e-5 to 5e-5)
   - Monitor training with TensorBoard

3. **Best Practices**:
   - Clean and validate your data thoroughly
   - Use appropriate context length for your use case
   - Save checkpoints regularly
   - Test the model periodically during training

## Contributing

Feel free to submit issues and enhancement requests!
