# Living Brain: Continuous Personal AI Clone

A modular system for creating a **living AI clone** that learns and evolves with new conversations over time. Unlike one-shot fine-tuning, Living Brain maintains persistent memory, growing knowledge, and consistent personality.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LIVING BRAIN                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   STYLE      │    │   MEMORY     │    │   FACTS      │  │
│  │   (LoRA)     │    │   (Vector DB)│    │   (Knowledge │  │
│  │              │    │              │    │    Graph)    │  │
│  │ Fine-tuned   │    │ Episodic     │    │ "I live in   │  │
│  │ personality  │    │ memories,    │    │  NYC", etc   │  │
│  │ & writing    │    │ conversations│    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │          │
│         └───────────────────┼───────────────────┘          │
│                             │                              │
│                    ┌────────▼────────┐                     │
│                    │   ORCHESTRATOR  │                     │
│                    │   (Combines all │                     │
│                    │    at inference)│                     │
│                    └─────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Style Layer (LoRA Adapter)**: Fine-tuned on your writing style, tone, emoji usage. Captures "how you talk" not "what you know".

2. **Memory Layer (ChromaDB)**: Stores conversation episodes as embeddings. Continuously updated with new conversations. Enables RAG retrieval at inference.

3. **Facts Layer (Knowledge Graph)**: Extracted facts like "I work at X", "My dog is named Y". Timestamp-based conflict resolution. Can be manually edited.

4. **Orchestrator**: Combines all layers at inference time for coherent responses.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/living-brain
cd living-brain

# Install base package
pip install -e .

# Install with memory support (ChromaDB)
pip install -e ".[core]"

# Install with training support (Unsloth - fastest)
pip install -e ".[train]"

# Install with chat interface
pip install -e ".[chat]"

# Install everything
pip install -e ".[all]"
```

### GPU Requirements

| Model | VRAM Required |
|-------|---------------|
| Llama 3.2 1B | 4GB |
| Llama 3.2 3B | 8GB |
| Qwen 2.5 3B | 8GB |
| Qwen 2.5 7B | 16GB |
| Llama 3.1 8B | 16GB |

## Quick Start

### 1. Parse Your WhatsApp Export

Export your WhatsApp chat (Settings > Export Chat > Without Media).

```bash
living-brain parse chat.txt --your-name "Your Name" --output processed
```

This generates:
- `processed.json` - Style metrics
- `processed.jsonl` - Parsed messages

### Dataset Workbench

Launch the local UI for per-participant persona datasets and character files:

```bash
living-brain workbench --port 7861
```

The workbench parses a WhatsApp `.txt` export, lets you choose a participant, and downloads a ZIP with canonical examples, SFT JSONL, DPO-style preference JSONL, eval rows, a style capsule, ElizaOS JSON, Character Card v2 JSON, and Markdown persona files. It is for messages you wrote or profiles with explicit permission.

The same workbench also has a **Sample Text** tab. Paste a small set of messages to get a research-backed recommendation for prompt cards, RAG, QLoRA SFT, DPO/KTO, low-data augmentation, code-switching behavior, and paralinguistic tags such as laughter, ellipsis, emoji, repeated punctuation, elongation, lowercase starts, all-caps, and discourse particles. The underlying study is in `docs/persona-methods-study.md`, with a machine-checkable 120-source corpus in `docs/persona-authenticity-source-corpus.jsonl`.

### 2. Train Your Style Adapter

```bash
living-brain train chat.txt \
    --your-name "Your Name" \
    --model llama-3.2-3b \
    --output ./models/my_style
```

Supported models:
- `llama-3.2-1b`, `llama-3.2-3b` (recommended)
- `qwen-2.5-3b`, `qwen-2.5-7b`
- `llama-3.1-8b`, `mistral-7b`

### 3. Ingest Conversations into Memory

```bash
living-brain ingest chat.txt \
    --your-name "Your Name" \
    --extract-facts
```

### 4. Chat with Your Clone

```bash
living-brain chat --adapter my_style
```

Or with a GGUF model for faster CPU inference:

```bash
living-brain chat --gguf ./models/my_style.gguf
```

## Configuration

Create a `config.yaml` from the example:

```bash
cp config.example.yaml config.yaml
```

Key settings:

```yaml
persona_name: "Your Name"

model:
  base_model: "unsloth/Llama-3.2-3B-Instruct"
  load_in_4bit: true

memory:
  top_k_retrieval: 5

inference:
  temperature: 0.7
```

## Continuous Learning

### Watch for New Exports

Automatically process new WhatsApp exports:

```bash
living-brain watch ./exports --your-name "Your Name" --extract-facts
```

Drop new chat exports into `./exports` and they'll be automatically ingested.

### Memory Consolidation

Old memories are automatically consolidated to save space while preserving key information.

## Project Structure

```
living_brain/
├── ingest/
│   ├── whatsapp_parser.py     # Parse WhatsApp exports
│   ├── style_analyzer.py      # Analyze writing style
│   └── watcher.py             # Watch for new exports
├── memory/
│   ├── vector_store.py        # ChromaDB wrapper
│   ├── fact_store.py          # Knowledge graph
│   ├── retriever.py           # RAG retrieval
│   └── consolidator.py        # Memory compression
├── style/
│   ├── trainer.py             # Unsloth/LoRA training
│   ├── data_formatter.py      # Training data formatting
│   └── adapter_manager.py     # Adapter management
├── inference/
│   ├── orchestrator.py        # Combines all layers
│   └── chat.py                # Gradio interface
├── core/
│   └── config.py              # Configuration
└── main.py                    # CLI entry point
```

## API Usage

```python
from living_brain.core.config import Config
from living_brain.inference.orchestrator import Orchestrator

# Initialize
config = Config(persona_name="Alice")
orchestrator = Orchestrator(config=config, adapter_name="alice_style")

# Load model
orchestrator.load_model()

# Chat
response = orchestrator.chat("Hey, what's up?")
print(response)

# Add new memory
orchestrator.add_to_memory("We talked about the weather today.")
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base Models | Llama 3.2, Qwen 2.5, Mistral |
| Fine-tuning | Unsloth + QLoRA (2x faster) |
| Vector DB | ChromaDB (local, persistent) |
| Embeddings | sentence-transformers |
| Inference | llama.cpp (GGUF) or transformers |
| UI | Gradio |

## Supported Date Formats

The parser supports many WhatsApp export formats:
- US: `[MM/DD/YY, HH:MM:SS AM/PM]`
- EU: `[DD/MM/YY, HH:MM:SS]`
- ISO: `YYYY-MM-DD, HH:MM:SS`
- With/without brackets, with/without seconds

## Privacy

All data stays local. No external APIs required for core functionality.

## License

MIT License - see [LICENSE](LICENSE) for details.
