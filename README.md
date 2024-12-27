# Personal Clone with WhatsApp Messages

A Python toolkit for creating a personalized AI clone of yourself using WhatsApp chat data. This tool analyzes your chat patterns, writing style, and personality traits to create a fine-tuned language model that can interact just like you.

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
   - Llama3-8B
   - Qwen-7B
   - Mistral 7B
   - Llama-2 7B
   - GPT-2 Medium

2. For Better Quality (Higher Resources):
   - Llama3-70B
   - Qwen-14B
   - Llama-2 13B
   - Falcon 40B

### Llama3 Model Support

The tool now includes special support for Meta's Llama3 models with optimizations:

#### Llama3-Specific Features
- BF16 precision training (recommended by Meta)
- Flash Attention 2 support
- Optimized LoRA configuration
- Tiktoken-based tokenizer support
- Proper padding token handling

#### Using Llama3 Models

1. Basic Usage:
```bash
python parser.py chat.txt "YourName" "OtherPerson" "YourName" --llm_format llama3
```

2. Fine-tuning with Llama3:
```bash
python finetune.py \
    --data_path formatted_YourName.jsonl \
    --model_name "meta-llama/Llama-3-8b" \
    --output_dir "./my_chatbot" \
    --style_metrics_path style_metrics_YourName.json \
    --use_peft \
    --use_8bit \
    --use_flash_attention
```

#### Llama3 Model Options
- Llama3-8B: Excellent balance of performance and resource usage
- Llama3-70B: State-of-the-art performance, requires significant resources
- Both models support context lengths up to 8192 tokens

#### Llama3-Specific Training Tips
1. **Memory Optimization**:
   - Use BF16 precision (default for Llama3)
   - Enable Flash Attention 2 for better performance
   - Use 8-bit quantization for larger models

2. **Training Settings**:
   - Optimized LoRA configuration (r=32)
   - Target all attention modules and MLP layers
   - Balanced learning rates for stability

3. **Best Practices**:
   - Always use BF16 precision
   - Enable Flash Attention when possible
   - Keep batch size moderate (2-4)
   - Use gradient accumulation for stability

4. **Important Notes**:
   - Requires Meta's model access approval
   - Uses tiktoken-based tokenizer (different from Llama2)
   - Needs proper padding token handling

### Qwen Model Support

The tool now includes special support for Qwen models with optimizations:

#### Qwen-Specific Features
- BF16 precision training (better performance than FP16)
- Flash Attention support
- Optimized LoRA configuration
- Special token handling
- Custom chat template

#### Using Qwen Models

1. Basic Usage:
```bash
python parser.py chat.txt "YourName" "OtherPerson" "YourName" --llm_format qwen
```

2. Fine-tuning with Qwen:
```bash
python finetune.py \
    --data_path formatted_YourName.jsonl \
    --model_name "Qwen/Qwen-7B" \
    --output_dir "./my_chatbot" \
    --style_metrics_path style_metrics_YourName.json \
    --use_peft \
    --use_8bit \
    --use_flash_attention
```

#### Qwen Model Options
- Qwen-7B: Good balance of performance and resource usage
- Qwen-14B: Better performance, requires more resources
- Both models support context lengths up to 8192 tokens

#### Qwen-Specific Training Tips
1. **Memory Optimization**:
   - Use BF16 precision (default for Qwen)
   - Enable Flash Attention for better performance
   - Use 8-bit quantization for larger models

2. **Training Settings**:
   - Higher LoRA rank (r=64) for better fine-tuning
   - Custom attention module targeting
   - Optimized learning rates

3. **Best Practices**:
   - Use Flash Attention when possible
   - Keep batch size moderate (2-4)
   - Use gradient accumulation for stability

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

## Known Issues and Limitations

1. **Data Format Sensitivity**:
   - WhatsApp export format changes may break parsing
   - Some regional date formats might not be recognized
   - Media messages are skipped and may affect context

2. **Memory Requirements**:
   - 7B models need at least 16GB VRAM
   - Loading times can be long on slower systems
   - System may freeze if RAM is insufficient

3. **Training Limitations**:
   - Short conversations may not provide enough context
   - Mixed language chats may reduce quality
   - Group chat dynamics can confuse the model
   - Style analysis may be inaccurate with limited data

4. **Model Access**:
   - Llama models require Meta's approval
   - Some models may have usage restrictions
   - Commercial use may require special licenses

5. **Performance Issues**:
   - Flash Attention may not work on older GPUs
   - 8-bit training can be unstable on some systems
   - CPU training is extremely slow and not recommended

6. **Common Error Cases**:
   - "CUDA out of memory": Reduce batch size or use 8-bit
   - "Token length exceeded": Adjust max_length parameter
   - "Invalid date format": Check chat export format
   - "Tokenizer errors": Update transformers library

## Chat Format Requirements

Your WhatsApp chat export should follow this format:

```
[MM/DD/YY, HH:MM:SS AM/PM] Author: Message text here
```

Example:
```
[01/13/24, 12:24:48 AM] Alex: Have you finished that project for work?
[01/13/24, 12:52:48 AM] Jamie: I love those! Send me the title later.
[01/13/24, 01:03:48 AM] Alex: Have you finished that project for work?
```

### Supported Date Formats

The parser supports multiple date formats:
1. Standard WhatsApp format: `[MM/DD/YY, HH:MM:SS AM/PM]`
2. Without seconds: `[MM/DD/YY, HH:MM AM/PM]`
3. International format: `[DD/MM/YY, HH:MM:SS AM/PM]`
4. ISO-like format: `[YYYY-MM-DD, HH:MM:SS AM/PM]`

### Format Guidelines

1. **Message Structure**:
   - Each message must be on a new line
   - Date and time in brackets
   - Author name followed by colon
   - Message text after the colon

2. **What to Include**:
   - Regular text messages
   - Emoji messages
   - URLs (they will be cleaned automatically)
   - Normal conversation text

3. **What to Exclude**:
   - Media messages (images, videos, documents)
   - System messages
   - Group settings changes
   - Contact cards
   - Location shares

4. **Export Instructions**:
   1. Open WhatsApp chat
   2. Tap ⋮ (three dots) > More > Export chat
   3. Choose 'Without media'
   4. Save the .txt file
   5. Use this file with the parser

### Common Issues

1. **Invalid Format**:
   ```
   # Wrong format:
   13/01/24 12:24 - Alex: Message
   
   # Correct format:
   [01/13/24, 12:24:48 AM] Alex: Message
   ```

2. **Missing Components**:
   ```
   # Missing brackets:
   01/13/24, 12:24:48 AM Alex: Message
   
   # Missing colon:
   [01/13/24, 12:24:48 AM] Alex Message
   ```

3. **System Messages**:
   ```
   # These will be automatically filtered:
   [01/13/24, 12:24:48 AM] Messages and calls are end-to-end encrypted
   [01/13/24, 12:24:48 AM] Alex changed the group description
   ```
