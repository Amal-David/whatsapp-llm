"""Configuration management for Living Brain."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    base_model: str = "unsloth/Llama-3.2-3B-Instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "conversations"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    max_context_chars: int = 12000
    max_facts: int = 50


@dataclass
class FactConfig:
    """Fact store configuration."""
    facts_path: str = "./data/facts.json"
    auto_extract: bool = True
    llm_extraction: bool = False  # Use LLM for fact extraction (more accurate but slower)


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


@dataclass
class InferenceConfig:
    """Inference configuration."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1
    use_gguf: bool = False
    gguf_path: str | None = None


@dataclass
class Config:
    """Main configuration container."""
    persona_name: str = "Assistant"
    data_dir: str = "./data"
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    facts: FactConfig = field(default_factory=FactConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self):
        """Ensure directories exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.memory.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.output_dir).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | None = None) -> Config:
    """Load configuration from YAML file or return defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Build config from nested dict
        model_cfg = ModelConfig(**data.get('model', {}))
        memory_cfg = MemoryConfig(**data.get('memory', {}))
        facts_cfg = FactConfig(**data.get('facts', {}))
        training_cfg = TrainingConfig(**data.get('training', {}))
        inference_cfg = InferenceConfig(**data.get('inference', {}))

        return Config(
            persona_name=data.get('persona_name', 'Assistant'),
            data_dir=data.get('data_dir', './data'),
            model=model_cfg,
            memory=memory_cfg,
            facts=facts_cfg,
            training=training_cfg,
            inference=inference_cfg,
        )

    return Config()


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    data = {
        'persona_name': config.persona_name,
        'data_dir': config.data_dir,
        'model': {
            'base_model': config.model.base_model,
            'embedding_model': config.model.embedding_model,
            'max_seq_length': config.model.max_seq_length,
            'load_in_4bit': config.model.load_in_4bit,
            'lora_r': config.model.lora_r,
            'lora_alpha': config.model.lora_alpha,
            'lora_dropout': config.model.lora_dropout,
        },
        'memory': {
            'chroma_persist_dir': config.memory.chroma_persist_dir,
            'collection_name': config.memory.collection_name,
            'chunk_size': config.memory.chunk_size,
            'chunk_overlap': config.memory.chunk_overlap,
            'top_k_retrieval': config.memory.top_k_retrieval,
            'max_context_chars': config.memory.max_context_chars,
            'max_facts': config.memory.max_facts,
        },
        'facts': {
            'facts_path': config.facts.facts_path,
            'auto_extract': config.facts.auto_extract,
            'llm_extraction': config.facts.llm_extraction,
        },
        'training': {
            'output_dir': config.training.output_dir,
            'num_epochs': config.training.num_epochs,
            'batch_size': config.training.batch_size,
            'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
            'learning_rate': config.training.learning_rate,
            'warmup_ratio': config.training.warmup_ratio,
            'max_grad_norm': config.training.max_grad_norm,
            'save_steps': config.training.save_steps,
            'logging_steps': config.training.logging_steps,
        },
        'inference': {
            'temperature': config.inference.temperature,
            'top_p': config.inference.top_p,
            'top_k': config.inference.top_k,
            'max_new_tokens': config.inference.max_new_tokens,
            'repetition_penalty': config.inference.repetition_penalty,
            'use_gguf': config.inference.use_gguf,
            'gguf_path': config.inference.gguf_path,
        },
    }

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
