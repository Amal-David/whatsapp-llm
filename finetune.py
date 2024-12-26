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
from typing import Dict, List

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
    device_map: str = "auto"
) -> tuple:
    """Setup the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or model_name,
        trust_remote_code=True
    )
    
    # Add special tokens if they don't exist
    special_tokens = {"pad_token": "[PAD]"}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16
    )
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def train(
    model,
    tokenized_dataset,
    training_args: TrainingArguments,
    data_collator
) -> None:
    """Train the model."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Save the model
    trainer.save_model()
    logger.info(f"Model saved to {training_args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a language model on WhatsApp chat data.')
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
    
    args = parser.parse_args()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        args.tokenizer_name
    )
    
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
    )
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Train the model
    train(model, tokenized_dataset, training_args, data_collator)

if __name__ == "__main__":
    main() 