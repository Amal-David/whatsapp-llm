# WhatsApp LLM

Local tools for building an evidence-grounded digital self from authorized
WhatsApp conversations, owner interviews, relationship-specific behavior,
time-aware memory, and optional style tuning.

The package is still named `living-brain`, and the CLI command is
`living-brain`.

## What It Does

- Parse WhatsApp `.txt` exports across common date formats.
- Read selected chats directly from WhatsApp for Mac or a local `wacli` mirror.
- Build a versioned self model with provenance, confidence, validity windows,
  contradictions, and owner confirmation.
- Reproduce a ten-seat, 187-paper research council for human representation.
- Exercise a typed, versioned digital-brain PoC with coverage, correction,
  deliberation, and hard-gated evaluation.
- Keep global identity separate from relationship-specific communication style.
- Evaluate profile-only and retrieval-backed behavior on held-out chats and
  explicit interview retests.
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
| Digital self | Guided owner interview, read-only multi-chat sources, versioned claims, temporal validity |
| Research council | Ten bounded seats, independent cross-review, 187 unique papers, evidence map, synthesis |
| Digital brain PoC | Twelve typed state layers, deterministic migration, consolidation, simulation, correction |
| Relationships | Pseudonymous per-chat style deltas without treating contacts' words as owner identity |
| Style analysis | Length, emoji, punctuation, capitalization, common phrases, response behavior |
| Dataset workbench | Gradio UI, participant picker, consent gate, ZIP artifact download |
| Training data | Canonical rows, Alpaca SFT, chat-message SFT, TRL DPO, OpenAI DPO, eval JSONL |
| Recommendations | Prompt card, RAG memory, QLoRA SFT, DPO/KTO, synthetic augmentation, holdout checks |
| Sample text | Method ranking, what works, gaps, style card JSON, prompt snippet, export JSON |
| Authenticity cues | Laughter, ellipsis, emoji, repeated punctuation, elongation, lowercase starts, all-caps, particles |
| Code-switching | Script mix, romanized markers, switch rate, generation rules |
| Character files | Canonical character JSON, Character Card v2, persona Markdown |
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

## General Digital Self

The recommended architecture is layered:

1. **Source events:** selected local messages remain the auditable evidence base.
2. **Identity:** owner-confirmed interview claims and clearly labeled behavioral
   candidates live in a versioned profile.
3. **Memory:** relevant events are retrieved by owner, source, relationship, and
   point in time.
4. **Relationship context:** communication style can vary by relationship without
   flattening everyone into one average voice.
5. **Optional tuning:** a LoRA adapter may improve surface style later, but it is
   never the source of truth for values, preferences, or current facts.

Create and complete the private owner interview:

```bash
living-brain self interview \
  --owner-name "Your Name" \
  --output ./private/self-interview.yaml
```

List chats from WhatsApp for Mac. This opens the local database in SQLite
read-only mode and does not print message bodies:

```bash
living-brain self chats --source whatsapp-mac
```

The native Mac schema is private and may change between WhatsApp releases. For
a more stable linked-device mirror, point the same commands at the third-party
[`wacli`](https://github.com/steipete/wacli) SQLite store:

```bash
living-brain self chats \
  --source wacli \
  --path ~/.wacli/wacli.db
```

Build from several explicitly selected source chat IDs:

```bash
living-brain self build \
  --source whatsapp-mac \
  --chat "first-source-chat-id" \
  --chat "second-source-chat-id" \
  --owner-name "Your Name" \
  --interview ./private/self-interview.yaml \
  --output ./private/digital-self.json

living-brain self validate ./private/digital-self.json
```

Use `--all-chats` only after reviewing the source list. A stable pseudonym key is
created at `~/.config/living-brain/pseudonym.key` by default; use `--key-file`
to place it elsewhere. The profile contains owner-message hashes and aggregate
style metrics, not owner message bodies. Third-party text is excluded unless
`--include-third-party-context` is explicitly supplied, and even then it cannot
support identity claims.

Export the private held-out evaluation suite and a separate text-free summary:

```bash
living-brain self evaluate \
  --source whatsapp-mac \
  --chat "first-source-chat-id" \
  --chat "second-source-chat-id" \
  --owner-name "Your Name" \
  --interview ./private/self-interview.yaml \
  --profile ./private/digital-self.json \
  --output ./private/self-evaluation.json \
  --summary-output ./private/self-evaluation-summary.json
```

The private suite may contain held-out contact prompts and owner replies. It is
written with owner-only permissions. The summary contains counts and coverage
only, so it can be inspected without reproducing dialogue.

## Digital Brain Proof Of Concept

The research-backed v2 PoC treats the canonical self as typed, versioned state
with provenance, confidence, time, sensitivity, ownership, and context scope. It
does not treat an LLM, vector index, prompt, or style adapter as the person.

Run the complete workflow on synthetic data only:

```bash
living-brain brain guide \
  --demo \
  --workspace ./private/brain-demo \
  --as-of 2026-07-11T12:00:00+00:00
```

The command performs source selection, coverage analysis, adaptive interview,
versioned build, inspection, owner correction, grounded simulation, and eight-axis
evaluation without reading WhatsApp or calling an external model. Reopen the final
state with `living-brain brain coverage` or `living-brain brain inspect`.

The evidence snapshot contains 200 reviewed seat records and 187 unique papers
after deduplication: 100 full-text and 87 abstract inspections. Start with the
[`research council`](research/digital-self-council/README.md), the
[`integrated synthesis`](research/digital-self-council/synthesis.md), and the
[`implementation and verification guide`](docs/digital-brain-poc.md).

This remains a bounded proof of concept, not a complete human replica. Real local
sources currently build `digital_self.v1` first and migrate with
`living-brain brain migrate`; automated event-to-state extraction, complete
deletion propagation, and autonomous representation are intentionally absent.

## Persona Dataset Quick Start

The export-based workbench remains available when the goal is a portable style
dataset or a fine-tuning artifact.

Export a WhatsApp chat as text:

1. Open a WhatsApp chat.
2. Choose **Export Chat**.
3. Export **Without Media**.
4. Save the `.txt` file locally.

Parse and inspect the chat:

```bash
living-brain parse chat.txt --your-name "Your Name" --output processed
```

Slash dates use month/day order by default. For exports that use day/month
order, pass `--date-order dmy`.

This writes:

- `processed.json` with style metrics.
- `processed.jsonl` with parsed messages.

Launch the persona dataset workbench:

```bash
living-brain workbench --port 7861
```

Open the local URL, upload the `.txt` export, choose a participant, confirm
consent, and download the generated ZIP. Other participants' messages are
withheld by default; including them requires a separate permission confirmation.

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
- `character_card_v2.json`
- `persona.md`
- `recommendation.md`

The canonical rows preserve source metadata, context, target replies, privacy
redactions, quality labels, splits, and synthetic flags. SFT rows use observed
target replies. DPO rows pair the observed reply with a labeled synthetic
negative, such as a too-formal variant. Splits are assigned by whole conversation:
SFT and DPO files contain training rows only, while `eval.jsonl` contains only
available validation and test rows. Small imports without enough conversation
groups intentionally produce no held-out rows.

Character-card and persona exports contain aggregate style descriptions, not
source dialogue, catchphrases, or other participants' display names.

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

Conversation memories are stored as overlapping chunks using the configured
`chunk_size` and `chunk_overlap`. Retrieved memories and facts are escaped,
bounded by `max_context_chars` and `max_facts`, and presented to the model as
untrusted historical data rather than instructions.

## Python API

```python
from living_brain.core.config import Config
from living_brain.inference.orchestrator import Orchestrator

config = Config(persona_name="Alice")
orchestrator = Orchestrator(
    config=config,
    profile_path="private/digital-self.json",
    adapter_name="alice_style",  # Optional surface-style layer.
)

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
  brain/         typed state, migration, consolidation, simulation, evaluation
  core/          configuration
  identity/      self model, interviews, read-only sources, prompt, evaluation
  ingest/        WhatsApp parsing, style analysis, dataset builders
  inference/     Gradio chat, dataset UI, orchestrator
  memory/        ChromaDB store, facts, retrieval, consolidation
  research/      corpus validation, council orchestration, evidence-map checks
  style/         dataset formatting, adapter training, adapter management
  main.py        CLI entry point
docs/
  digital-brain-poc.md
  persona-methods-study.md
  persona-authenticity-source-corpus.jsonl
research/
  digital-self-council/  public paper corpus, reviews, synthesis, manifest
tests/
  test_pr_review_regressions.py
```

## Privacy And Safety

- Use the workbench only for your own messages, an organization voice you manage,
  or someone who gave explicit consent.
- Keep raw WhatsApp exports local.
- Treat the native WhatsApp Mac reader as version-sensitive and verify its schema
  after app upgrades.
- Treat `wacli` as a third-party linked device with best-effort history, not an
  official WhatsApp API or guaranteed complete archive.
- Keep the pseudonym key, interview, profile, and private evaluation suite out of
  source control.
- Treat generated brain snapshots, guided-run artifacts, and council runtime
  directories as private local data; only the public research corpus is committed.
- Brain snapshots can contain authorized owner evidence payloads. Use the redacted
  inspection command for review; do not treat the canonical JSON as a safe export.
- Leave third-party context disabled unless every included participant gave permission.
- Review every generated dataset before training.
- Treat synthetic rows as synthetic.
- Do not train SFT on unreviewed synthetic replies as if they were real messages.
- Keep held-out real replies for distribution and memorization checks.
- Do not invent private facts.
- Label AI-assisted drafts when they leave private testing.
- Never grant generated output authority to send, promise, transact, or represent
  the owner. High-stakes or under-supported cases require the owner.

Core parsing and dataset generation do not require external APIs.

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run the regression tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests -q -o addopts=''
```

Compile every supported import path:

```bash
python -m compileall -q living_brain tests
```

CI runs the regression suite, CLI smoke tests, and focused Ruff/mypy checks on
Python 3.10, 3.11, and 3.12. Generated workbench downloads are kept in a private
temporary directory and removed when the workbench is cleaned up or exits.

Run the focused lint checks locally:

```bash
uvx ruff check tests \
  living_brain/brain \
  living_brain/identity \
  living_brain/research \
  living_brain/main.py \
  living_brain/ingest/persona_dataset.py \
  living_brain/ingest/whatsapp_parser.py \
  living_brain/memory/fact_store.py \
  living_brain/memory/vector_store.py \
  living_brain/inference/orchestrator.py
```

## License

MIT License. See [`LICENSE`](LICENSE).
