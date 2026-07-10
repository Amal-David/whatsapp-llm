"""
Orchestrator that combines style, memory, and facts for inference.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.config import Config
from ..ingest.style_analyzer import StyleMetrics
from ..memory.fact_store import FactStore
from ..memory.retriever import MemoryRetriever, RetrievalResult
from ..memory.vector_store import VectorStore
from ..style.adapter_manager import AdapterManager

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a generation."""
    response: str
    retrieval: RetrievalResult | None = None
    tokens_generated: int = 0
    generation_time_ms: float = 0.0


class Orchestrator:
    """
    Orchestrates inference by combining:
    - Style (LoRA adapter)
    - Memory (vector store for episodic memories)
    - Facts (knowledge graph for semantic memory)
    """

    def __init__(
        self,
        config: Config | None = None,
        adapter_name: str | None = None,
        use_gguf: bool = False,
        gguf_path: str | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Configuration object
            adapter_name: Name of the LoRA adapter to use
            use_gguf: Whether to use a GGUF model via llama.cpp
            gguf_path: Path to GGUF model file
        """
        self.config = config or Config()
        self.adapter_name = adapter_name
        self.use_gguf = use_gguf
        self.gguf_path = gguf_path

        # Memory is initialized lazily so GGUF or prompt-only flows can run
        # without the optional ChromaDB dependency.
        self.vector_store: VectorStore | None = None
        self.fact_store: FactStore | None = None
        self.retriever: MemoryRetriever | None = None

        # Initialize adapter manager
        self.adapter_manager = AdapterManager(
            adapters_dir=str(Path(self.config.training.output_dir).parent / "adapters")
        )

        # Style metrics (loaded from adapter if available)
        self.style_metrics: StyleMetrics | None = None

        # Model (lazy loaded)
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._llama_model: Any | None = None  # For GGUF

        # Conversation history
        self._history: list[dict] = []

    def _get_fact_store(self) -> FactStore:
        """Load the lightweight fact store without requiring ChromaDB."""
        if self.fact_store is None:
            self.fact_store = FactStore(
                facts_path=self.config.facts.facts_path,
            )
        return self.fact_store

    def _get_retriever(self) -> MemoryRetriever:
        """Lazy load memory stores only when memory or facts are requested."""
        if self.retriever is None:
            self.vector_store = VectorStore(
                persist_directory=self.config.memory.chroma_persist_dir,
                collection_name=self.config.memory.collection_name,
                embedding_model=self.config.model.embedding_model,
            )
            self.retriever = MemoryRetriever(
                vector_store=self.vector_store,
                fact_store=self._get_fact_store(),
                top_k_memories=self.config.memory.top_k_retrieval,
                max_context_chars=self.config.memory.max_context_chars,
                max_facts=self.config.memory.max_facts,
            )
        return self.retriever

    def _load_style_metrics(self) -> None:
        """Load style metrics from the adapter."""
        if self.adapter_name:
            adapter_info = self.adapter_manager.get(self.adapter_name)
            if adapter_info and adapter_info.style_metrics_path:
                try:
                    self.style_metrics = StyleMetrics.load(adapter_info.style_metrics_path)
                    logger.info("Loaded style metrics from adapter")
                except Exception as e:
                    logger.warning(f"Could not load style metrics: {e}")

    def _get_system_prompt(self) -> str:
        """Generate the system prompt."""
        parts = [f"You are {self.config.persona_name}."]

        if self.style_metrics:
            parts.append(self.style_metrics.to_system_prompt())
        else:
            parts.append(
                "Respond naturally in a conversational style. "
                "Keep responses authentic and personal."
            )

        return "\n\n".join(parts)

    def load_model(self) -> None:
        """Load the model for inference."""
        if self.use_gguf:
            self._load_gguf_model()
        else:
            self._load_transformers_model()

        self._load_style_metrics()

    def _load_gguf_model(self) -> None:
        """Load a GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF inference. "
                "Install with: pip install llama-cpp-python"
            )

        if not self.gguf_path:
            raise ValueError("gguf_path must be provided for GGUF inference")

        gguf_path = Path(self.gguf_path)
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        logger.info(f"Loading GGUF model: {gguf_path}")

        self._llama_model = Llama(
            model_path=str(gguf_path),
            n_ctx=self.config.model.max_seq_length,
            n_gpu_layers=-1,  # Use all GPU layers
            verbose=False,
        )

        logger.info("GGUF model loaded")

    def _load_transformers_model(self) -> None:
        """Load a transformers model with optional LoRA adapter."""
        try:
            from unsloth import FastLanguageModel
            use_unsloth = True
        except ImportError:
            use_unsloth = False

        if use_unsloth:
            logger.info(f"Loading model with Unsloth: {self.config.model.base_model}")
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model.base_model,
                max_seq_length=self.config.model.max_seq_length,
                dtype=None,
                load_in_4bit=self.config.model.load_in_4bit,
            )

            # Enable inference mode
            FastLanguageModel.for_inference(self._model)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading model: {self.config.model.base_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.base_model)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        # Load adapter if specified
        if self.adapter_name:
            self._model, self._tokenizer = self.adapter_manager.load(
                self.adapter_name,
                model=self._model,
                tokenizer=self._tokenizer,
            )

        logger.info("Model loaded")

    def _format_prompt(
        self,
        message: str,
        retrieval: RetrievalResult | None = None,
    ) -> str:
        """Format the prompt with context and history."""
        system = self._get_system_prompt()

        # Add memory context if available
        if retrieval and retrieval.context_text:
            system += "\n\n" + retrieval.context_text

        # Format based on model type
        model_name = self.config.model.base_model.lower()

        if "llama" in model_name:
            return self._format_llama(system, message)
        elif "qwen" in model_name:
            return self._format_chatml(system, message)
        else:
            return self._format_chatml(system, message)

    def _format_llama(self, system: str, message: str) -> str:
        """Format for Llama 3 models."""
        parts = ["<|begin_of_text|>"]
        parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>")

        # Add history
        for turn in self._history[-5:]:  # Last 5 turns
            parts.append(
                f"<|start_header_id|>user<|end_header_id|>\n\n{turn['user']}<|eot_id|>"
            )
            parts.append(
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{turn['assistant']}<|eot_id|>"
            )

        # Add current message
        parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(parts)

    def _format_chatml(self, system: str, message: str) -> str:
        """Format in ChatML format."""
        parts = [f"<|im_start|>system\n{system}<|im_end|>"]

        # Add history
        for turn in self._history[-5:]:
            parts.append(f"<|im_start|>user\n{turn['user']}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>")

        # Add current message
        parts.append(f"<|im_start|>user\n{message}<|im_end|>")
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def generate(
        self,
        message: str,
        use_memory: bool = True,
        use_facts: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """
        Generate a response to a message.

        Args:
            message: The user's message
            use_memory: Whether to retrieve relevant memories
            use_facts: Whether to include facts
            temperature: Override temperature
            max_tokens: Override max tokens

        Returns:
            GenerationResult with the response and metadata
        """
        import time

        start_time = time.time()

        # Retrieve context
        retrieval = None
        if use_memory or use_facts:
            retrieval = self._get_retriever().retrieve(
                query=message,
                include_facts=use_facts,
                include_memories=use_memory,
            )

        # Format prompt
        prompt = self._format_prompt(message, retrieval)

        # Generate
        inference_config = self.config.inference
        temp = temperature if temperature is not None else inference_config.temperature
        max_new = max_tokens if max_tokens is not None else inference_config.max_new_tokens

        if self.use_gguf:
            response, tokens = self._generate_gguf(prompt, temp, max_new)
        else:
            response, tokens = self._generate_transformers(prompt, temp, max_new)

        generation_time = (time.time() - start_time) * 1000

        # Update history
        self._history.append({
            "user": message,
            "assistant": response,
            "timestamp": datetime.now().isoformat(),
        })

        return GenerationResult(
            response=response,
            retrieval=retrieval,
            tokens_generated=tokens,
            generation_time_ms=generation_time,
        )

    def _generate_gguf(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Generate using llama.cpp."""
        if self._llama_model is None:
            self.load_model()
        if self._llama_model is None:
            raise RuntimeError("GGUF model did not initialize")

        output = self._llama_model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.inference.top_p,
            top_k=self.config.inference.top_k,
            repeat_penalty=self.config.inference.repetition_penalty,
            stop=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>"],
        )

        response = output["choices"][0]["text"].strip()
        tokens = output["usage"]["completion_tokens"]

        return response, tokens

    def _generate_transformers(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Generate using transformers."""
        if self._model is None:
            self.load_model()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Transformers model did not initialize")

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_length = inputs.input_ids.shape[1]

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.inference.top_p,
            top_k=self.config.inference.top_k,
            repetition_penalty=self.config.inference.repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        response = self._tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        ).strip()

        tokens = len(outputs[0]) - input_length

        return response, tokens

    def chat(self, message: str, **kwargs) -> str:
        """Simple chat interface that returns just the response text."""
        result = self.generate(message, **kwargs)
        return result.response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    def add_to_memory(
        self,
        conversation: str,
        timestamp: datetime | None = None,
    ) -> tuple[str, list]:
        """Add the current conversation to memory."""
        return self._get_retriever().add_conversation_to_memory(
            conversation_text=conversation,
            timestamp=timestamp,
            extract_facts=self.config.facts.auto_extract,
        )

    def add_fact(self, subject: str, predicate: str, obj: str):
        """Add a fact without initializing vector memory."""
        return self._get_fact_store().add(
            subject=subject,
            predicate=predicate,
            obj=obj,
            source="manual",
        )

    def get_stats(self) -> dict:
        """Get statistics about the system."""
        base = {
            "model": self.config.model.base_model,
            "adapter": self.adapter_name,
            "use_gguf": self.use_gguf,
            "history_length": len(self._history),
        }
        if self.retriever is None:
            return {**base, "memory_loaded": False}
        return {**base, "memory_loaded": True, **self.retriever.stats()}
