from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch
import argparse
import logging
import os
import json
from typing import Dict, List, Optional
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_data(
    data_path: str,
    tokenizer,
    max_length: int = 2048
) -> Dict:
    """Load and process the JSONL dataset."""
    # Load the dataset
    dataset = load_dataset('json', data_files=data_path)
    
    def tokenize_function(examples):
        """Tokenize the texts."""
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset['train'].column_names,
        batched=True
    )
    
    return tokenized_dataset

def setup_model_and_tokenizer(
    model_name: str,
    tokenizer_name: str = None,
    device_map: str = "auto",
    load_in_8bit: bool = False,
    style_metrics: Optional[Dict] = None
) -> tuple:
    """Setup the model and tokenizer with optimizations for personal style training."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or model_name,
        trust_remote_code=True
    )
    
    # Add special tokens if they don't exist
    special_tokens = {"pad_token": "[PAD]"}
    if style_metrics:
        # Add common phrases as special tokens for better preservation
        for phrase in style_metrics.get('common_phrases', {}).keys():
            special_tokens[f"phrase_{len(special_tokens)}"] = phrase
    
    tokenizer.add_special_tokens(special_tokens)
    
    # Load the model with optimizations
    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            load_in_8bit=True,
            device_map=device_map,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.float16
        )
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def setup_peft_config(model_name: str) -> LoraConfig:
    """Setup Parameter-Efficient Fine-Tuning configuration."""
    return LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Alpha scaling
        target_modules=["q_proj", "v_proj"],  # Target attention modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def train(
    model,
    tokenized_dataset,
    training_args: TrainingArguments,
    data_collator,
    style_metrics: Optional[Dict] = None
) -> None:
    """Train the model with style-aware optimizations."""
    # If we have style metrics, adjust training parameters
    if style_metrics:
        # Adjust learning rate based on dataset size and style complexity
        msg_length = style_metrics.get('avg_message_length', 0)
        if msg_length > 100:
            training_args.learning_rate *= 0.8  # Reduce learning rate for complex styles
        
        # Adjust batch size based on message length
        if msg_length > 200:
            training_args.per_device_train_batch_size = max(1, training_args.per_device_train_batch_size - 2)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Save the model and style metrics
    trainer.save_model()
    if style_metrics:
        with open(os.path.join(training_args.output_dir, "style_metrics.json"), "w") as f:
            json.dump(style_metrics, f, indent=2)
    
    logger.info(f"Model and style metrics saved to {training_args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a language model to mimic personal chat style.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the JSONL file containing the training data')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name or path of the base model to fine-tune')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--tokenizer_name', type=str, default=None,
                        help='Name or path of the tokenizer (if different from model)')
    parser.add_argument('--style_metrics_path', type=str, default=None,
                        help='Path to the style metrics JSON file')
    parser.add_argument('--use_8bit', action='store_true',
                        help='Use 8-bit quantization for training')
    parser.add_argument('--use_peft', action='store_true',
                        help='Use Parameter-Efficient Fine-Tuning (LoRA)')
    
    args = parser.parse_args()
    
    # Load style metrics if available
    style_metrics = None
    if args.style_metrics_path and os.path.exists(args.style_metrics_path):
        with open(args.style_metrics_path, 'r') as f:
            style_metrics = json.load(f)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        args.tokenizer_name,
        load_in_8bit=args.use_8bit,
        style_metrics=style_metrics
    )
    
    # Apply LoRA if requested
    if args.use_peft:
        logger.info("Applying LoRA for parameter-efficient fine-tuning")
        peft_config = setup_peft_config(args.model_name)
        model = get_peft_model(model, peft_config)
    
    # Load and process data
    tokenized_dataset = load_and_process_data(
        args.data_path,
        tokenizer,
        args.max_length
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        # Add warmup steps for better style adaptation
        warmup_steps=100,
        # Add evaluation steps
        evaluation_strategy="steps",
        eval_steps=100,
    )
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Train the model
    train(model, tokenized_dataset, training_args, data_collator, style_metrics)

if __name__ == "__main__":
    main() 