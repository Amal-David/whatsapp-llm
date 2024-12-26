# Personal Chat Style Cloning Tool

A Python toolkit for creating a personalized AI chatbot that mimics your WhatsApp conversation style. This tool analyzes your chat patterns, writing style, and personality traits to create a fine-tuned language model that can interact just like you.

## Features

- **Advanced Style Analysis**:
  - Message length patterns
  - Emoji usage frequency
  - Slang and abbreviation patterns
  - Capitalization habits
  - Punctuation patterns
  - Common phrases and expressions
  - Response patterns

- **Multiple LLM Support**:
  - Llama-2
  - Mistral
  - Falcon
  - GPT

- **Optimization Features**:
  - Parameter-Efficient Fine-Tuning (LoRA)
  - 8-bit quantization support
  - Gradient accumulation
  - Mixed precision training

## Requirements

```bash
pip install pandas transformers torch datasets tensorboard peft
```

## How Style Analysis Works

The tool performs a comprehensive analysis of your chat style through multiple layers:

### 1. Message Structure Analysis

#### Length Patterns
- Calculates average message length
- Tracks message length distribution
- Identifies typical message structures
```python
avg_message_length = total_length / total_messages
```

#### Emoji Usage
- Uses Unicode pattern matching for emoji detection
- Calculates emoji frequency per message
- Identifies favorite/most used emojis
```python
emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
emoji_count = len(emoji_pattern.findall(message))
emoji_rate = emoji_count / total_messages
```

#### Slang and Abbreviations
- Maintains a comprehensive slang dictionary
- Tracks usage frequency of common internet abbreviations
- Examples: 'lol', 'omg', 'idk', 'tbh', etc.
```python
slang_words = {'lol', 'omg', 'idk', 'tbh', ...}
slang_rate = slang_count / total_messages
```

### 2. Writing Style Metrics

#### Capitalization Patterns
- Analyzes sentence start capitalization
- Tracks ALL CAPS usage
- Identifies stylistic capitalization choices
```python
capitalization_rate = capitalized_messages / total_messages
```

#### Punctuation Analysis
- Tracks usage of different punctuation marks
- Identifies multiple punctuation patterns (e.g., "!!!")
- Analyzes sentence ending styles
```python
punctuation_patterns = {
    '.': frequency,
    '!': frequency,
    '?': frequency,
    '...': frequency
}
```

#### Common Phrases
- Extracts and ranks frequently used phrases (3-grams)
- Identifies personal catchphrases
- Tracks conversation starters/enders
```python
for i in range(len(words)-2):
    phrase = ' '.join(words[i:i+3])
    common_phrases[phrase] += 1
```

### 3. Style Integration

#### System Prompt Generation
The analysis generates a detailed system prompt:
```
You are now mimicking a person with the following conversation style:
- Average message length: X characters
- Emoji usage: [High/Moderate/Low]
- Slang usage: [High/Moderate/Low]
- Capitalization: [Usually/Sometimes/Rarely] starts sentences with capital letters
- Frequently used phrases:
  * [phrase 1]
  * [phrase 2]
  ...
```

#### Training Optimization
Style metrics influence training parameters:
- Adjusts learning rate based on style complexity
- Modifies batch size for longer messages
- Adds common phrases as special tokens

### 4. Style Preservation

#### Token Management
- Adds common phrases as special tokens
- Preserves emoji and special characters
- Maintains punctuation patterns

#### Data Cleaning
- Removes URLs while preserving style
- Keeps emojis and formatting
- Preserves personal abbreviations

## Creating Your Digital Twin

### Step 1: Process Your Chat Data

Export your WhatsApp chat and process it:

```bash
python parser.py chat.txt "YourName" "OtherPerson" "YourName"
```

This will generate:
- `formatted_YourName.jsonl`: Training data
- `style_metrics_YourName.json`: Your conversation style analysis

### Step 2: Fine-tune the Model

Choose a base model and fine-tune it with your style:

```bash
python finetune.py \
    --data_path formatted_YourName.jsonl \
    --model_name "mistralai/Mistral-7B-v0.1" \
    --output_dir "./my_chatbot" \
    --style_metrics_path style_metrics_YourName.json \
    --use_peft \
    --use_8bit
```

### Advanced Usage

1. Different LLM formats:
```bash
python parser.py chat.txt "YourName" "OtherPerson" "YourName" --llm_format mistral
```

2. Adjust context length:
```bash
python parser.py chat.txt "YourName" "OtherPerson" "YourName" --context_length 5
```

3. Memory-efficient training:
```bash
python finetune.py \
    --data_path formatted_YourName.jsonl \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --use_8bit \
    --use_peft \
    --batch_size 1 \
    --gradient_accumulation_steps 8
```

## Style Analysis Features

The tool analyzes various aspects of your chat style:

### Message Patterns
- Average message length
- Response timing patterns
- Message structure preferences

### Language Usage
- Emoji frequency and patterns
- Slang and abbreviation usage
- Capitalization habits
- Punctuation style
- Common phrases and expressions

### Conversation Flow
- Response patterns
- Context utilization
- Topic transition style

## Fine-tuning Options

### Model Selection Guide

1. For Personal Use (Lower Resources):
   - Mistral 7B
   - Llama-2 7B
   - GPT-2 Medium

2. For Better Quality (Higher Resources):
   - Llama-2 13B
   - Falcon 40B

### Optimization Techniques

1. Memory Optimization:
   - 8-bit quantization
   - LoRA fine-tuning
   - Gradient accumulation

2. Training Optimization:
   - Learning rate adaptation
   - Batch size adjustment
   - Warmup steps
   - Mixed precision training

## Output Files

The process generates several files:

1. `output_YYYYMMDD_HHMMSS.csv`: Original conversation pairs
2. `formatted_[YourName].jsonl`: Training data
3. `style_metrics_[YourName].json`: Style analysis
4. Fine-tuned model in `output_dir`:
   - Model weights
   - Tokenizer
   - Style configuration
   - Training logs

## Best Practices

1. **Data Quality**:
   - Use at least 1000 messages for good results
   - Include diverse conversations
   - Clean out irrelevant messages

2. **Model Selection**:
   - Start with smaller models (7B)
   - Use 8-bit quantization for larger models
   - Enable LoRA for efficient training

3. **Training Tips**:
   - Monitor training with TensorBoard
   - Start with default hyperparameters
   - Adjust based on style metrics
   - Use gradient accumulation for stability

4. **Hardware Requirements**:
   - 7B models: 16GB+ GPU VRAM
   - 13B models: 24GB+ GPU VRAM
   - 40B models: Multiple GPUs

## Troubleshooting

1. **Memory Issues**:
   - Enable 8-bit quantization
   - Reduce batch size
   - Increase gradient accumulation
   - Use LoRA fine-tuning

2. **Quality Issues**:
   - Increase training data
   - Adjust context length
   - Try different base models
   - Fine-tune hyperparameters

## Contributing

Feel free to submit issues and enhancement requests!
