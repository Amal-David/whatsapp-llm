"""
Adapter manager for loading and managing LoRA adapters.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AdapterInfo:
    """Information about a trained adapter."""
    name: str
    path: str
    base_model: str
    created_at: datetime
    training_examples: int = 0
    style_metrics_path: Optional[str] = None
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "base_model": self.base_model,
            "created_at": self.created_at.isoformat(),
            "training_examples": self.training_examples,
            "style_metrics_path": self.style_metrics_path,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdapterInfo":
        return cls(
            name=data["name"],
            path=data["path"],
            base_model=data["base_model"],
            created_at=datetime.fromisoformat(data["created_at"]),
            training_examples=data.get("training_examples", 0),
            style_metrics_path=data.get("style_metrics_path"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )


class AdapterManager:
    """
    Manages LoRA adapters - loading, listing, and switching between them.
    """

    def __init__(self, adapters_dir: str = "./models/adapters"):
        """
        Initialize the adapter manager.

        Args:
            adapters_dir: Directory to store adapters
        """
        self.adapters_dir = Path(adapters_dir)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)

        self._registry_path = self.adapters_dir / "registry.json"
        self._registry: dict[str, AdapterInfo] = {}
        self._load_registry()

        self._current_adapter: Optional[str] = None

    def _load_registry(self) -> None:
        """Load adapter registry from disk."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r') as f:
                    data = json.load(f)
                for name, info in data.items():
                    self._registry[name] = AdapterInfo.from_dict(info)
                logger.info(f"Loaded {len(self._registry)} adapters from registry")
            except Exception as e:
                logger.error(f"Failed to load adapter registry: {e}")

    def _save_registry(self) -> None:
        """Save adapter registry to disk."""
        try:
            with open(self._registry_path, 'w') as f:
                json.dump(
                    {name: info.to_dict() for name, info in self._registry.items()},
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save adapter registry: {e}")

    def register(
        self,
        name: str,
        adapter_path: str | Path,
        base_model: str,
        training_examples: int = 0,
        style_metrics_path: Optional[str] = None,
        description: str = "",
        tags: Optional[list[str]] = None,
        copy_files: bool = True,
    ) -> AdapterInfo:
        """
        Register a new adapter.

        Args:
            name: Unique name for the adapter
            adapter_path: Path to the adapter files
            base_model: Base model the adapter was trained on
            training_examples: Number of examples used in training
            style_metrics_path: Path to associated style metrics
            description: Description of the adapter
            tags: Tags for categorization
            copy_files: Whether to copy adapter files to the adapters directory

        Returns:
            AdapterInfo for the registered adapter
        """
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        # Copy to adapters directory if requested
        if copy_files:
            dest_path = self.adapters_dir / name
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(adapter_path, dest_path)
            adapter_path = dest_path

        info = AdapterInfo(
            name=name,
            path=str(adapter_path),
            base_model=base_model,
            created_at=datetime.now(),
            training_examples=training_examples,
            style_metrics_path=style_metrics_path,
            description=description,
            tags=tags or [],
        )

        self._registry[name] = info
        self._save_registry()

        logger.info(f"Registered adapter: {name}")
        return info

    def get(self, name: str) -> Optional[AdapterInfo]:
        """Get adapter info by name."""
        return self._registry.get(name)

    def list(self, tag: Optional[str] = None) -> list[AdapterInfo]:
        """
        List all registered adapters.

        Args:
            tag: Optional tag to filter by

        Returns:
            List of AdapterInfo objects
        """
        adapters = list(self._registry.values())

        if tag:
            adapters = [a for a in adapters if tag in a.tags]

        # Sort by creation date (newest first)
        adapters.sort(key=lambda a: a.created_at, reverse=True)
        return adapters

    def delete(self, name: str, delete_files: bool = True) -> bool:
        """
        Delete an adapter.

        Args:
            name: Adapter name
            delete_files: Whether to delete adapter files

        Returns:
            True if deleted, False if not found
        """
        if name not in self._registry:
            return False

        info = self._registry[name]

        if delete_files:
            adapter_path = Path(info.path)
            if adapter_path.exists():
                shutil.rmtree(adapter_path)

        del self._registry[name]
        self._save_registry()

        logger.info(f"Deleted adapter: {name}")
        return True

    def load(self, name: str, model=None, tokenizer=None):
        """
        Load an adapter onto a model.

        Args:
            name: Adapter name
            model: Base model (will be loaded if not provided)
            tokenizer: Tokenizer (will be loaded if not provided)

        Returns:
            Tuple of (model, tokenizer) with adapter loaded
        """
        info = self.get(name)
        if info is None:
            raise ValueError(f"Adapter not found: {name}")

        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft is required to load adapters")

        if model is None:
            # Load base model
            try:
                from unsloth import FastLanguageModel
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=info.base_model,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                )
            except ImportError:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(info.base_model)
                tokenizer = AutoTokenizer.from_pretrained(info.base_model)

        # Load adapter
        model = PeftModel.from_pretrained(model, info.path)
        self._current_adapter = name

        logger.info(f"Loaded adapter: {name}")
        return model, tokenizer

    def get_current(self) -> Optional[str]:
        """Get the currently loaded adapter name."""
        return self._current_adapter

    def get_latest(self) -> Optional[AdapterInfo]:
        """Get the most recently created adapter."""
        adapters = self.list()
        return adapters[0] if adapters else None

    def export_to_gguf(
        self,
        name: str,
        output_path: str | Path,
        quantization: str = "q4_k_m",
    ) -> str:
        """
        Export an adapter to GGUF format.

        Args:
            name: Adapter name
            output_path: Output path for GGUF file
            quantization: Quantization method

        Returns:
            Path to the exported GGUF file
        """
        info = self.get(name)
        if info is None:
            raise ValueError(f"Adapter not found: {name}")

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("unsloth is required for GGUF export")

        # Load model with adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=info.base_model,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        from peft import PeftModel
        model = PeftModel.from_pretrained(model, info.path)

        # Export to GGUF
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.save_pretrained_gguf(
            str(output_path.parent),
            tokenizer,
            quantization_method=quantization,
        )

        # Find the generated file
        gguf_files = list(output_path.parent.glob("*.gguf"))
        if gguf_files:
            return str(gguf_files[0])

        raise RuntimeError("GGUF export failed")

    def merge_adapter(self, name: str, output_path: str | Path) -> str:
        """
        Merge an adapter with its base model.

        Args:
            name: Adapter name
            output_path: Output path for merged model

        Returns:
            Path to the merged model
        """
        info = self.get(name)
        if info is None:
            raise ValueError(f"Adapter not found: {name}")

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("unsloth is required for merging")

        # Load model with adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=info.base_model,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,  # Need full precision for merging
        )

        from peft import PeftModel
        model = PeftModel.from_pretrained(model, info.path)

        # Merge adapter
        model = model.merge_and_unload()

        # Save merged model
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        logger.info(f"Merged model saved to {output_path}")
        return str(output_path)
