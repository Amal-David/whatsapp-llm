"""
Style trainer using Unsloth for efficient LoRA fine-tuning.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import training dependencies
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

try:
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./models/lora_adapter"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    save_steps: int = 100
    logging_steps: int = 10
    max_seq_length: int = 2048
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05


class StyleTrainer:
    """
    Trainer for fine-tuning LLMs on personal style using Unsloth.

    Uses QLoRA for memory-efficient training.
    """

    # Supported base models
    SUPPORTED_MODELS = {
        "llama-3.2-1b": "unsloth/Llama-3.2-1B-Instruct",
        "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
        "llama-3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "qwen-2.5-3b": "unsloth/Qwen2.5-3B-Instruct",
        "qwen-2.5-7b": "unsloth/Qwen2.5-7B-Instruct",
        "mistral-7b": "unsloth/Mistral-7B-Instruct-v0.3",
        "phi-3-mini": "unsloth/Phi-3-mini-4k-instruct",
    }

    def __init__(
        self,
        model_name: str = "llama-3.2-3b",
        config: TrainingConfig | None = None,
        load_in_4bit: bool = True,
    ):
        """
        Initialize the trainer.

        Args:
            model_name: Model name (short name or full HuggingFace path)
            config: Training configuration
            load_in_4bit: Whether to load model in 4-bit quantization
        """
        if not UNSLOTH_AVAILABLE:
            raise ImportError(
                "unsloth is required for StyleTrainer. "
                "Install with: pip install unsloth"
            )

        if not TRL_AVAILABLE:
            raise ImportError(
                "trl and datasets are required for StyleTrainer. "
                "Install with: pip install trl datasets"
            )

        self.config = config or TrainingConfig()
        self.load_in_4bit = load_in_4bit

        # Resolve model name
        if model_name in self.SUPPORTED_MODELS:
            self.model_path = self.SUPPORTED_MODELS[model_name]
        else:
            self.model_path = model_name

        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def _require_loaded_state(self) -> tuple[Any, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model and tokenizer did not initialize")
        return self.model, self.tokenizer

    def load_model(self) -> None:
        """Load the base model with LoRA adapters."""
        logger.info(f"Loading model: {self.model_path}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=self.load_in_4bit,
        )

        # Add LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        logger.info("Model loaded with LoRA adapters")

    def _get_formatting_func(self):
        """Get the formatting function based on model type."""
        # Detect model type from path
        model_lower = self.model_path.lower()

        if "llama" in model_lower:
            def format_llama(examples):
                texts = []
                for i in range(len(examples["instruction"])):
                    system = examples.get("system", [""] * len(examples["instruction"]))[i]
                    instruction = examples["instruction"][i]
                    input_text = examples["input"][i]
                    output = examples["output"][i]

                    text = "<|begin_of_text|>"
                    if system:
                        text += f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                    text += f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}"
                    if input_text:
                        text += f"\n{input_text}"
                    text += "<|eot_id|>"
                    text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
                    texts.append(text)
                return {"text": texts}
            return format_llama

        elif "qwen" in model_lower:
            def format_qwen(examples):
                texts = []
                for i in range(len(examples["instruction"])):
                    system = examples.get("system", [""] * len(examples["instruction"]))[i]
                    instruction = examples["instruction"][i]
                    input_text = examples["input"][i]
                    output = examples["output"][i]

                    text = ""
                    if system:
                        text += f"<|im_start|>system\n{system}<|im_end|>\n"
                    text += f"<|im_start|>user\n{instruction}"
                    if input_text:
                        text += f"\n{input_text}"
                    text += "<|im_end|>\n"
                    text += f"<|im_start|>assistant\n{output}<|im_end|>"
                    texts.append(text)
                return {"text": texts}
            return format_qwen

        else:
            # Generic ChatML format
            def format_chatml(examples):
                texts = []
                for i in range(len(examples["instruction"])):
                    system = examples.get("system", [""] * len(examples["instruction"]))[i]
                    instruction = examples["instruction"][i]
                    input_text = examples["input"][i]
                    output = examples["output"][i]

                    text = ""
                    if system:
                        text += f"<|im_start|>system\n{system}<|im_end|>\n"
                    text += f"<|im_start|>user\n{instruction}"
                    if input_text:
                        text += f"\n{input_text}"
                    text += "<|im_end|>\n"
                    text += f"<|im_start|>assistant\n{output}<|im_end|>"
                    texts.append(text)
                return {"text": texts}
            return format_chatml

    def train(self, data_path: str | Path) -> str:
        """
        Train the model on a dataset.

        Args:
            data_path: Path to JSONL training data (Alpaca format)

        Returns:
            Path to the saved adapter
        """
        if self.model is None:
            self.load_model()
        model, tokenizer = self._require_loaded_state()

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        dataset = load_dataset("json", data_files=str(data_path), split="train")

        # Format dataset
        formatting_func = self._get_formatting_func()
        dataset = dataset.map(formatting_func, batched=True)

        # Training arguments
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            fp16=not self.load_in_4bit,
            bf16=self.load_in_4bit,
            optim="adamw_8bit",
            seed=42,
            report_to="none",
        )

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=training_args,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save adapter
        adapter_path = output_dir / "adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        logger.info(f"Adapter saved to {adapter_path}")
        return str(adapter_path)

    def export_gguf(
        self,
        adapter_path: str | Path,
        output_path: str | Path,
        quantization: str = "q4_k_m",
    ) -> str:
        """
        Export the model to GGUF format for llama.cpp.

        Args:
            adapter_path: Path to the trained adapter
            output_path: Output path for GGUF file
            quantization: Quantization method (q4_k_m, q5_k_m, q8_0, f16)

        Returns:
            Path to the exported GGUF file
        """
        if self.model is None:
            self.load_model()
        model, tokenizer = self._require_loaded_state()

        adapter_path = Path(adapter_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load the adapter
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(model, str(adapter_path))

        logger.info(f"Exporting to GGUF with {quantization} quantization...")

        # Use Unsloth's GGUF export
        self.model.save_pretrained_gguf(
            str(output_path.parent),
            tokenizer,
            quantization_method=quantization,
        )

        # Find the generated file
        gguf_files = list(output_path.parent.glob("*.gguf"))
        if gguf_files:
            final_path = gguf_files[0]
            logger.info(f"GGUF exported to {final_path}")
            return str(final_path)

        raise RuntimeError("GGUF export failed - no output file generated")

    def push_to_hub(
        self,
        adapter_path: str | Path,
        repo_name: str,
        private: bool = True,
    ) -> str:
        """
        Push the adapter to HuggingFace Hub.

        Args:
            adapter_path: Path to the trained adapter
            repo_name: Repository name on HuggingFace
            private: Whether to make the repo private

        Returns:
            URL of the uploaded model
        """
        if self.model is None:
            self.load_model()
        model, tokenizer = self._require_loaded_state()

        from peft import PeftModel
        self.model = PeftModel.from_pretrained(model, str(adapter_path))

        logger.info(f"Pushing to HuggingFace Hub: {repo_name}")

        self.model.push_to_hub(
            repo_name,
            private=private,
            token=True,  # Use cached token
        )
        tokenizer.push_to_hub(
            repo_name,
            private=private,
            token=True,
        )

        url = f"https://huggingface.co/{repo_name}"
        logger.info(f"Model pushed to {url}")
        return url
