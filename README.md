# Digital Self

An experimental, local-first toolkit for building inspectable representations
of one person from authorized WhatsApp conversations.

**Website:** [whatsapp-llm.pages.dev](https://whatsapp-llm.pages.dev/) ·
**Documentation:** [whatsapp-llm.pages.dev/docs](https://whatsapp-llm.pages.dev/docs/)

The project explores two scopes:

| Approach | Input | Result |
| --- | --- | --- |
| **Mini-Me** | One selected person's messages, usually from one chat | A relationship-scoped writing-style dataset with held-out evaluation |
| **Digital Self** | Owner messages across several selected chats, plus an optional interview | A versioned profile with evidence, time, uncertainty, relationships, and optional typed state |

Neither approach reconstructs a person or a mind. The output is an experimental
model of selected evidence that the owner must inspect, correct, and judge.

The installed command remains `living-brain` for compatibility.

## Current State

The repository currently supports:

- read-only chat discovery and ingestion from WhatsApp for Mac
- ingestion from a local third-party `wacli` linked-device mirror
- explicit chat selection, stable pseudonyms, and owner interviews
- `digital_self.v1` profiles with evidence hashes, provenance, confidence,
  temporal validity, contradictions, and relationship-specific style
- private held-out evaluation rows and separate text-free summaries
- migration to typed state with coverage, correction, scoped simulation, and
  independent evaluation axes
- a deterministic synthetic walkthrough that writes 12 auditable artifacts
- a narrow Mini-Me workbench for consented, single-person style datasets

## Install

```bash
git clone https://github.com/Amal-David/whatsapp-llm
cd whatsapp-llm
pip install -e .
```

## Build A Multi-Chat Digital Self

List available chats without printing message bodies:

```bash
living-brain self chats --source whatsapp-mac
```

Create a private owner interview:

```bash
living-brain self interview \
  --owner-name "Your Name" \
  --output ./private/self-interview.yaml
```

Build and validate a profile from explicitly selected chats:

```bash
living-brain self build \
  --source whatsapp-mac \
  --chat "first-chat-id" \
  --chat "second-chat-id" \
  --owner-name "Your Name" \
  --interview ./private/self-interview.yaml \
  --output ./private/digital-self.json

living-brain self validate ./private/digital-self.json
```

Use `--all-chats` only after reviewing the source list. To use a `wacli` mirror,
pass `--source wacli --path ~/.wacli/wacli.db`. The `self evaluate` command
creates private held-out rows and a separate text-free summary.

## Build A Mini-Me

Mini-Me is the narrower workflow for learning how one selected person tends to
write in a particular conversational setting.

```bash
living-brain workbench --port 7861
```

Upload an authorized WhatsApp text export, select the participant, confirm
consent, and review the generated archive. The workbench produces canonical
examples, style summaries, training-ready rows, preference pairs, and held-out
evaluation rows. Third-party messages are withheld by default.

Treat the result as relationship-scoped unless evidence from other chats shows
that the same behavior generalizes.

## Inspect The Typed-State Workflow

Run the complete deterministic walkthrough with synthetic data only:

```bash
living-brain brain guide \
  --demo \
  --workspace ./private/digital-self-demo \
  --as-of 2026-07-11T12:00:00+00:00
```

The walkthrough performs source selection, initial state creation, coverage
analysis, adaptive interview, versioning, inspection, owner correction,
simulation, and evaluation. It reads no WhatsApp data and calls no external
model.

The 12 generated files are workflow stages and receipts, not 12 models or a
single replica score. Their exact purpose is documented in the
[`Digital Self explainer`](docs/digital-self-explainer.md).

## Evidence And Evaluation

The owner selects local sources. The system normalizes messages, separates owner
evidence from third-party context, and records provenance. Observed behavior
remains a candidate rather than a fact; claims and style stay scoped by time and
relationship.

Owner message bodies are not copied into the portable `digital_self.v1` profile.
The profile stores hashes, aggregates, provenance, and relationship-safe
metadata. Private evaluation rows may contain conversation text and are written
with owner-only permissions; their summary contains counts and coverage only.

Mini-Me checks held-out replies for style, context, privacy, invention, and
memorization. The typed-state evaluator separately checks behavior, time,
relationships, attribution, decisions, calibration, privacy, authority, and
explicit owner judgment. Privacy failures block a pass. There is no single
"percent cloned" score.

## Limits

- WhatsApp conversations are partial, relationship-specific evidence, and the
  Mac app uses a private schema that may change.
- `wacli` is a third-party linked-device mirror, not an official WhatsApp API or
  a guaranteed complete archive.
- Automatic raw-message-to-typed-state extraction and complete deletion
  propagation are not finished.
- The synthetic walkthrough proves software contracts, not real-person fidelity.
- The project has no authority to send, promise, transact, or represent the
  owner.

Keep raw chats, pseudonym keys, interviews, profiles, evaluation rows, and typed
state snapshots private and out of source control.

## Documentation

- [`Digital Self explainer`](docs/digital-self-explainer.md): extraction,
  conversion, Mini-Me artifacts, evaluation, and the 12 guided-run files
- [`Typed Digital Self PoC`](docs/digital-brain-poc.md): schemas, migration,
  correction, simulation, and verification
- [`Research synthesis`](research/digital-self-council/synthesis.md): evidence,
  constraints, architecture, and non-goals
- [`Research council README`](research/digital-self-council/README.md): corpus
  structure and reproducibility workflow

## Development

```bash
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests -q -o addopts=''
python -m compileall -q living_brain tests
```

CI runs tests, compilation, CLI smoke checks, package builds, Ruff, and mypy on
Python 3.10, 3.11, and 3.12.

## License

MIT. See [`LICENSE`](LICENSE).
