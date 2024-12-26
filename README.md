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
