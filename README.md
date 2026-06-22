# WhatsApp LLM

Local tools for turning authorized WhatsApp conversations into persona datasets,
style cards, character files, fine-tuning inputs, and memory-backed chat.

The package is still named `living-brain`, and the CLI command is
`living-brain`.

## What It Does

- Parse WhatsApp `.txt` exports across common date formats.
- Analyze per-person writing style, emoji use, punctuation, and message length.
- Build consented persona datasets for one selected participant.
- Export SFT, DPO, eval, style capsule, and character-card artifacts.
- Recommend prompt, RAG, QLoRA SFT, DPO, KTO, ORPO, and DRPO paths.
- Analyze small pasted samples when a full chat export is unavailable.
- Detect code-switching signals and paralinguistic cues.
- Suggest low-data augmentation recipes with synthetic provenance labels.
- Store episodic memories in ChromaDB and facts in a local knowledge graph.
- Run a local chat UI with adapters, GGUF models, RAG, and facts.

The project is designed for authorized drafting and research workflows. It is
not designed for deceptive impersonation or detector evasion.

## Feature Map

| Area | Features |
| --- | --- |
| WhatsApp parsing | Participant discovery, conversation splitting, system/media filtering, JSONL export |
| Style analysis | Length, emoji, punctuation, capitalization, common phrases, response behavior |
| Dataset workbench | Gradio UI, participant picker, consent gate, ZIP artifact download |
| Training data | Canonical rows, Alpaca SFT, chat-message SFT, TRL DPO, OpenAI DPO, eval JSONL |
| Recommendations | Prompt card, RAG memory, QLoRA SFT, DPO/KTO, synthetic augmentation, holdout checks |
| Sample text | Method ranking, what works, gaps, style card JSON, prompt snippet, export JSON |
| Authenticity cues | Laughter, ellipsis, emoji, repeated punctuation, elongation, lowercase starts, all-caps, particles |
| Code-switching | Script mix, romanized markers, switch rate, generation rules |
| Character files | Canonical character JSON, ElizaOS character JSON, Character Card v2, persona Markdown |
| Memory | ChromaDB vector store, fact extraction, recency retrieval, memory consolidation |
| Inference | Lazy orchestrator, transformers or llama.cpp GGUF, Gradio chat |
| Safety | Consent checks, PII redaction, third-party context controls, no raw shared quotes by default |

## Installation

```bash
git clone https://github.com/Amal-David/whatsapp-llm
cd whatsapp-llm
pip install -e .
```

Install optional feature groups as needed:

```bash
pip install -e ".[chat]"
pip install -e ".[core]"
pip install -e ".[train-compat]"
pip install -e ".[inference-gguf]"
pip install -e ".[watch]"
pip install -e ".[all]"
```

Use `.[train]` only when you want the Unsloth-backed training stack.

## Quick Start

Export a WhatsApp chat as text:

1. Open a WhatsApp chat.
2. Choose **Export Chat**.
3. Export **Without Media**.
4. Save the `.txt` file locally.

Parse and inspect the chat:

```bash
living-brain parse chat.txt --your-name "Your Name" --output processed
```

This writes:

- `processed.json` with style metrics.
- `processed.jsonl` with parsed messages.

Launch the persona dataset workbench:

```bash
living-brain workbench --port 7861
```

Open the local URL, upload the `.txt` export, choose a participant, confirm
consent, and download the generated ZIP.

## Dataset Workbench

The workbench has two tabs.

### WhatsApp Export

Use this when you have a full chat export.

It produces:

- `summary.json`
- `canonical_examples.jsonl`
- `sft_alpaca.jsonl`
- `sft_messages.jsonl`
- `dpo_trl.jsonl`
- `dpo_openai.jsonl`
- `eval.jsonl`
- `style_capsule.json`
- `canonical_character.json`
- `elizaos.character.json`
- `character_card_v2.json`
- `persona.md`
- `recommendation.md`

The canonical rows preserve source metadata, context, target replies, privacy
redactions, quality labels, splits, and synthetic flags. SFT rows use observed
target replies. DPO rows pair the observed reply with a labeled synthetic
negative, such as a too-formal variant.

### Sample Text

Use this when you only have pasted messages or a small sample.

It returns:

- A ranked method recommendation.
- What works in the sample.
- Authenticity gaps.
- Code-switching guidance.
- Paralinguistic tag inventory.
- Low-data augmentation recipes.
- Generation constraints.
- Evaluation checks.
- A style card JSON object.
- A prompt snippet for immediate drafting.

The recommender is deliberately conservative:

| Data available | Default path |
| --- | --- |
| Under 20 messages | Prompt card plus manual examples |
| 20-99 messages | Prompt card plus retrieved exemplars |
| 100-499 messages | RAG persona memory |
| 500-1,999 messages | QLoRA SFT candidate with strict holdout |
| 2,000+ messages | QLoRA SFT, then preference tuning if labels exist |

## Fine-Tuning Paths

Use the generated artifacts according to your data volume and labels.

| Method | Input | Use When |
| --- | --- | --- |
| Prompt card | Style card and examples | Any small import or first prototype |
| RAG persona memory | Retrieved examples and context | You have 100+ target messages |
| SFT / QLoRA | Target replies with context | You have 500+ clean target replies |
| DPO / ORPO | Chosen and rejected pairs | You have pairwise preference examples |
| KTO | Desirable and undesirable labels | You have unary thumbs-up/down feedback |
| DRPO-style robust tuning | Preference rows with subgroup metadata | You have large shifted datasets |
| GRPO / RL | Rollouts and verifiable rewards | Avoid for normal conversational style |

The research notes and ranking are in
[`docs/persona-methods-study.md`](docs/persona-methods-study.md). The source
corpus is in
[`docs/persona-authenticity-source-corpus.jsonl`](docs/persona-authenticity-source-corpus.jsonl).

## Train A Style Adapter

Create training rows from a chat and train a LoRA adapter:

```bash
living-brain train chat.txt \
  --your-name "Your Name" \
  --model llama-3.2-3b \
  --output ./models/my_style
```

Supported model aliases include:

- `llama-3.2-1b`
- `llama-3.2-3b`
- `qwen-2.5-3b`
- `qwen-2.5-7b`
- `llama-3.1-8b`
- `mistral-7b`

Export a GGUF model when needed:

```bash
living-brain train chat.txt \
  --your-name "Your Name" \
  --model llama-3.2-3b \
  --output ./models/my_style \
  --export-gguf
```

## Memory And Chat

Create a config file:

```bash
cp config.example.yaml config.yaml
```

Ingest conversations into memory:

```bash
living-brain ingest chat.txt --your-name "Your Name" --extract-facts
```

Launch chat with an adapter:

```bash
living-brain chat --adapter my_style
```

Launch chat with a GGUF model:

```bash
living-brain chat --gguf ./models/my_style.gguf
```

Watch a directory for new exports:

```bash
living-brain watch ./exports --your-name "Your Name" --extract-facts
```

Show local memory, facts, and adapter counts:

```bash
living-brain stats
```

The orchestrator loads memory stores lazily, so GGUF-only workflows can start
without ChromaDB until memory features are used.

## Python API

```python
from living_brain.core.config import Config
from living_brain.inference.orchestrator import Orchestrator

config = Config(persona_name="Alice")
orchestrator = Orchestrator(config=config, adapter_name="alice_style")

orchestrator.load_model()
print(orchestrator.chat("hey, are we still on for later?"))

orchestrator.add_to_memory("We planned to meet at 7 near the cafe.")
```

Build dataset artifacts directly:

```python
from living_brain.ingest.persona_dataset import PersonaDatasetBuilder

builder = PersonaDatasetBuilder(context_turns=6)
result = builder.build_from_file("chat.txt", participant="Alice", owner_type="self")
builder.write_artifacts(result, "out/alice")
```

Analyze pasted sample text:

```python
from living_brain.ingest.sample_recommender import SampleTextRecommender

recommendation = SampleTextRecommender().recommend(
    sample_text="haan wait lol\nI can come after 7...",
    target_message_count=42,
    persona_name="Alice",
)

print(recommendation.to_markdown())
```

## Project Structure

```text
living_brain/
  core/          configuration
  ingest/        WhatsApp parsing, style analysis, dataset builders
  inference/     Gradio chat, dataset UI, orchestrator
  memory/        ChromaDB store, facts, retrieval, consolidation
  style/         dataset formatting, adapter training, adapter management
  main.py        CLI entry point
docs/
  persona-methods-study.md
  persona-authenticity-source-corpus.jsonl
tests/
  test_pr_review_regressions.py
```

## Privacy And Safety

- Use the workbench only for your own messages, an organization voice you manage,
  or someone who gave explicit consent.
- Keep raw WhatsApp exports local.
- Review every generated dataset before training.
- Treat synthetic rows as synthetic.
- Do not train SFT on unreviewed synthetic replies as if they were real messages.
- Keep held-out real replies for distribution and memorization checks.
- Do not invent private facts.
- Label AI-assisted drafts when they leave private testing.

Core parsing and dataset generation do not require external APIs.

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run the regression tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_pr_review_regressions.py -q -o addopts=''
```

Run lint:

```bash
uvx ruff check tests/test_pr_review_regressions.py \
  living_brain/memory/fact_store.py \
  living_brain/memory/vector_store.py \
  living_brain/inference/orchestrator.py
```

## License

MIT License. See [`LICENSE`](LICENSE).
