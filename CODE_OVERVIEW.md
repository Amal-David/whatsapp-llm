# WhatsApp LLM Toolkit – Code Overview

This document summarizes the main modules in the repository and how they work together to convert WhatsApp chats into a personal fine-tuning dataset and train a conversational model.

## `parser.py`

The parser module turns raw WhatsApp exports into structured conversations while extracting personal style statistics.

### Key building blocks

- **`LLMFormat` enum** – enumerates the supported target model families (LLaMA-2/3, Mistral, Falcon, GPT, Qwen) and provides a helper to parse string inputs.
- **`ChatMessage` dataclass** – normalized representation for a single chat line with author, message text, and timestamp fields.
- **`PersonalStyleExtractor`** – analyzes message lists to compute personal metrics (average length, emoji/slang usage, capitalization, frequent phrases, punctuation counts, and response patterns).
- **`DataCleaner`** – filters out media/system placeholders and normalizes message text while preserving emojis and replacing links with `[URL]`.
- **`ChatParser`** – recognizes multiple WhatsApp timestamp formats via regex patterns and converts matching lines to `ChatMessage` objects.
- **`ConversationManager`** – tracks rolling context windows so that when the target speaker replies, the preceding turns become the prompt portion of the training pair.
- **`LLMFormatter`** – renders prompt/response pairs in the instruction format expected by each model family (e.g., LLaMA `[INST]` blocks, GPT chat message dicts, Qwen `<|im_start|>` tags).

### Processing flow

1. **`converter_with_debug`** streams the chat file, uses `ChatParser` to build a DataFrame, and pairs prompts/responses for inspection.
2. **`jsonl_data`** filters and cleans messages, keeps sliding windows of context, and when "your" character replies it formats an instruction example with the chosen `LLMFormat`.
3. While formatting, the function also computes personal style metrics and generates a system prompt (via `create_system_prompt`) describing those traits.
4. CLI execution writes three artifacts: a CSV of parsed pairs, a JSONL fine-tuning file with formatted conversations, and a JSON style summary.

## `finetune.py`

This script fine-tunes a causal language model using the formatted dataset and the style metrics.

### Data preparation

- **`load_and_process_data`** loads the JSONL file with 🤗 Datasets, validates it, then tokenizes text examples with optional length truncation.

### Model setup

- **`setup_model_and_tokenizer`** configures tokenizers/models with special handling for LLaMA 3 and Qwen (pad tokens, bf16 usage, optional 8-bit loading). For other models it can append special tokens seeded from common phrases to help preserve style.
- When LoRA is requested, **`setup_peft_config`** selects sensible rank, alpha, and target modules tailored to LLaMA 3, Qwen, or default architectures.

### Training loop

- **`train`** constructs a 🤗 `Trainer`, optionally adjusts hyperparameters based on style metrics (e.g., reducing learning rate for longer average messages), and saves both the fine-tuned weights and the metrics JSON alongside them.
- The CLI (`main`) wires everything together: parses arguments, loads style metrics, sets up 8-bit/PEFT/Flash Attention options, prepares data and a `DataCollatorForLanguageModeling`, and launches training.

## End-to-end usage

1. Run `parser.py` on a WhatsApp export to create `formatted_<name>.jsonl` and style metrics.
2. Provide those files to `finetune.py`, choose a base model and options (LoRA, 8-bit, etc.), and fine-tune a personalized assistant that mirrors the original chat style.

