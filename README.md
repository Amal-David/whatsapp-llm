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
  - Llama-3
  - Mistral
  - Falcon
  - GPT
  - Qwen

- **Optimization Features**:
  - Parameter-Efficient Fine-Tuning (LoRA)
  - 8-bit quantization support
  - Gradient accumulation
  - Mixed precision training

## Requirements

```bash
pip install pandas transformers torch datasets tensorboard peft
```

## Getting Started

### 1. Chat Export and Parsing

#### Chat Format Requirements

Your WhatsApp chat export should follow this format:
```
[MM/DD/YY, HH:MM:SS AM/PM] Author: Message text here
```

Example:
```
[01/13/24, 12:24:48 AM] Alex: Have you finished that project for work?
[01/13/24, 12:52:48 AM] Jamie: I love those! Send me the title later.
```

#### Supported Date Formats
1. Standard WhatsApp format: `[MM/DD/YY, HH:MM:SS AM/PM]`
2. Without seconds: `[MM/DD/YY, HH:MM AM/PM]`
3. International format: `[DD/MM/YY, HH:MM:SS AM/PM]`
4. ISO-like format: `[YYYY-MM-DD, HH:MM:SS AM/PM]`

#### Export Instructions
1. Open WhatsApp chat
2. Tap ⋮ (three dots) > More > Export chat
3. Choose 'Without media'
4. Save the .txt file

#### What to Include/Exclude
- ✅ Include:
  - Regular text messages
  - Emoji messages
  - URLs (they will be cleaned automatically)
  - Normal conversation text

- ❌ Exclude:
  - Media messages (images, videos, documents)
  - System messages
  - Group settings changes
  - Contact cards
  - Location shares

#### Parsing Your Chat

Basic usage:
```bash
python parser.py chat.txt "YourName" "OtherPerson" "YourName"
```

Advanced usage:
```bash
python parser.py chat.txt "YourName" "OtherPerson" "YourName" \
    --llm_format mistral \
    --context_length 5
```

This generates:
- `output_YYYYMMDD_HHMMSS.csv`: Original conversation pairs
- `formatted_YourName.jsonl`: Training data
- `style_metrics_YourName.json`: Style analysis

### 2. Fine-tuning Process

After parsing your chat data, you can fine-tune a model:

```bash
python finetune.py \
    --data_path formatted_YourName.jsonl \
    --model_name "mistralai/Mistral-7B-v0.1" \
    --output_dir "./my_chatbot" \
    --style_metrics_path style_metrics_YourName.json \
    --use_peft \
    --use_8bit
```

#### Model Selection Guide

1. For Personal Use (Lower Resources):
   - Llama3-8B (16GB VRAM)
   - Qwen-7B (16GB VRAM)
   - Mistral 7B (16GB VRAM)
   - Llama-2 7B (16GB VRAM)

2. For Better Quality (Higher Resources):
   - Llama3-70B (80GB VRAM)
   - Qwen-14B (28GB VRAM)
   - Llama-2 13B (24GB VRAM)
   - Falcon 40B (Multiple GPUs)

## How Style Analysis Works

The tool performs a comprehensive analysis of your chat style through multiple layers:

### Message Structure Analysis

- **Length Patterns**
  - Average message length
  - Message length distribution
  - Typical response lengths

- **Emoji Usage**
  - Detection and frequency analysis
  - Favorite emoji patterns
  - Contextual emoji usage

- **Slang and Abbreviations**
  - Common internet slang (e.g., 'lol', 'omg', 'idk')
  - Personal abbreviations
  - Informal language patterns

### Writing Style Metrics

- **Capitalization**
  - Sentence start patterns
  - Stylistic caps usage
  - Name/proper noun capitalization

- **Punctuation**
  - End-of-sentence patterns
  - Multiple punctuation usage (!!!, ???)
  - Informal punctuation style

- **Common Phrases**
  - Frequent expressions
  - Conversation starters/enders
  - Personal catchphrases

## Advanced Training Options

### Memory Optimization

1. **8-bit Quantization**
   ```bash
   python finetune.py \
       --data_path formatted_YourName.jsonl \
       --use_8bit \
       --batch_size 2
   ```

2. **Gradient Accumulation**
   ```bash
   python finetune.py \
       --gradient_accumulation_steps 8 \
       --batch_size 1
   ```

### Model-Specific Features

#### Llama3 Support
- BF16 precision training
- Flash Attention 2
- 8192 token context
- Example:
  ```bash
  python finetune.py \
      --model_name "meta-llama/Llama-3-8b" \
      --use_flash_attention \
      --bf16
  ```

#### Qwen Support
- Custom chat template
- Flash Attention
- Example:
  ```bash
  python finetune.py \
      --model_name "Qwen/Qwen-7B" \
      --use_flash_attention
  ```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size
   - Enable 8-bit training
   - Use gradient accumulation
   ```bash
   python finetune.py \
       --batch_size 1 \
       --use_8bit \
       --gradient_accumulation_steps 8
   ```

2. **Training Issues**
   - Adjust learning rate
   - Increase training data
   - Try different models
   ```bash
   python finetune.py \
       --learning_rate 1e-5 \
       --num_epochs 5
   ```

3. **Parsing Issues**
   - Check date format
   - Verify chat export
   - Clean input data

### Hardware Requirements

1. **GPU Memory Requirements**
   - 7B models: 16GB VRAM
   - 13B models: 24GB VRAM
   - 70B models: 80GB VRAM
   - Multiple GPUs: Falcon 40B

2. **Recommended Setup**
   - NVIDIA GPU with CUDA support
   - 32GB+ System RAM
   - SSD for faster data loading

## Best Practices

### Data Preparation
1. Use at least 1000 messages
2. Include diverse conversations
3. Clean irrelevant messages
4. Maintain conversation context

### Training Configuration
1. Start with smaller models
2. Use default hyperparameters
3. Monitor with TensorBoard
4. Save checkpoints regularly

### Style Preservation
1. Keep emoji patterns
2. Maintain punctuation style
3. Preserve message length patterns
4. Retain personal phrases

## Known Limitations

1. **Data Format**
   - WhatsApp format changes
   - Regional date formats
   - Media message handling

2. **Resource Requirements**
   - High VRAM usage
   - Long training times
   - System RAM needs

3. **Model Access**
   - Llama model approval
   - Usage restrictions
   - License requirements

4. **Quality Factors**
   - Short conversation impact
   - Mixed language handling
   - Group chat dynamics

## Contributing

Feel free to:
- Submit issues
- Propose features
- Share improvements
- Report bugs

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- HuggingFace Transformers
- Meta AI (Llama models)
- WhatsApp chat format documentation
