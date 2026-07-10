# General Digital-Self Methods Study

## Objective

Build a local system that uses authorized WhatsApp conversations and explicit
owner self-report to simulate one person's identity, memory, decisions, and
relationship-conditioned communication. Portable style datasets remain useful,
but they are one optional layer rather than the digital self itself. The product
goal is owner-controlled simulation and drafting, not autonomous representation,
deceptive impersonation, or detector evasion.

## Recommended Digital-Self Architecture

The implementation uses a layered hybrid instead of asking one fine-tuned model
to contain everything:

| Layer | Owns | Must Not Own |
|---|---|---|
| Source events | Immutable local message events, timestamps, source IDs, content hashes | Inferred identity |
| Identity model | Versioned claims, provenance, confidence, validity windows, supersession, conflicts | Raw contact dialogue by default |
| Relationship model | Pseudonymous per-relationship communication deltas | A contact's private facts or words as owner identity |
| Temporal memory | Owner/source/relationship-scoped retrieval bounded by the requested point in time | Current truth without validity checks |
| Surface style | Aggregate writing metrics and an optional LoRA adapter | Values, authority, current preferences, or factual memory |
| Evaluation | Held-out chat/time groups, interview retests, blind owner comparisons | Training rows or configuration-revealing ratings |

WhatsApp evidence is strong for observed behavior but incomplete for motives,
values, private goals, and the reasons behind decisions. A guided owner interview
therefore supplies confirmed claims while message-derived style remains a
candidate inference. Conflicting current claims remain visible; the runtime does
not collapse them into a false certainty.

This approach borrows the useful local-memory and self-modeling direction of
[Second-Me](https://github.com/mindverse/Second-Me), but keeps the canonical self
model portable and inspectable. Parameter training is optional and must beat the
profile-plus-retrieval baseline in blinded owner evaluation before becoming the
default.

## Local Source Acquisition

Three adapters share one normalized, read-only message contract:

- `whatsapp-mac` opens WhatsApp for Mac's private `ChatStorage.sqlite` in SQLite
  `mode=ro`, enables `query_only`, and reads inside a transaction. This avoids
  manual text exports but is version-sensitive because WhatsApp does not publish
  that schema.
- `wacli` reads the local SQLite mirror produced by the third-party
  [`wacli`](https://github.com/steipete/wacli) linked device. Its upstream history
  sync is explicitly best-effort, so completeness must be measured rather than
  assumed.
- `json` is a portable fixture/interchange source for testing and offline tools.

All chat, relationship, and sender identifiers are transformed with a stable
keyed HMAC before they enter the profile. Source inspection shows chat names and
source IDs locally so the owner can select chats; profile artifacts retain only
pseudonymous IDs. Owner message bodies become hashes in identity evidence.
Third-party text is excluded by default and can never support an owner claim.

## Style And Tuning Ranking

This ranking applies to the surface-style layer after the self model and
evaluation split exist. It is not a ranking of complete digital-self
architectures.

| Rank | Method | Dataset Needed | When To Use | Recommendation |
|---:|---|---|---|---|
| 1 | Prompt-only style card with few-shot exemplars | Style summary plus 5-20 representative replies | Any small import, early validation, reversible profiles | Default MVP |
| 2 | RAG persona memory | Editable examples, traits, relationship context, retrieved target replies | 100-500 messages or more | Best product default after import |
| 3 | QLoRA SFT adapter | Context messages mapped to the target person's next reply | 500-2,000 clean target replies minimum; 5,000+ better | Advanced local/offline training path |
| 4 | SFT plus DPO | Prompt, chosen target-style reply, rejected non-style reply | 200-1,000 preference pairs after SFT | Add when the UI collects comparisons |
| 5 | KTO | Prompt, response, desirable/undesirable label | 500+ unary feedback labels | Good if feedback is thumbs-up/down rather than pairwise |
| 6 | ORPO | Prompt plus chosen/rejected pair | 1,000+ preference pairs | Experiment after DPO baseline |
| 7 | Persona adapters / multi-LoRA | One adapter per consenting person or cluster | Multi-person deployment | Useful later for adapter routing |
| 8 | IPO / self-judged preference methods | Multiple generated replies plus judge choices | Strong judge and careful audits | Research-only for this use case |
| 9 | DRPO / robust DPO variants | Preference rows plus subgroup or distribution metadata | Large shifted preference datasets | Not default for one person |
| 10 | GRPO / RL with rewards | Multiple rollouts and a verifiable reward | Math, code, or tool tasks | Avoid for conversational style |
| 11 | Full fine-tuning | Large target corpus and GPU budget | Rare, high-control settings | Avoid for small UI |

The staged product path is:

1. Consent, explicit chat selection, local normalization, and pseudonymization.
2. Owner interview plus a versioned, inspectable identity profile.
3. Relationship and temporal style summaries from training groups only.
4. Profile-only and retrieval-backed baselines on held-out chat/time groups.
5. SFT-ready examples and preference feedback only when the baseline justifies tuning.
6. Blinded owner comparisons before any adapter becomes the default.

## Evidence Corpus

The broader authenticity pass is captured in `docs/persona-authenticity-source-corpus.jsonl`.
It is machine-checkable and currently contains 120 unique source rows:

- 35 human texting / CMC / WhatsApp sources
- 35 code-switching sources
- 25 paralinguistic and graphicon sources
- 23 low-data style-transfer, personalization, alignment, and augmentation sources
- 2 evaluation/privacy sources

Source rows include `id`, `category`, `title`, `year`, `url`, `source_type`,
`key_finding`, and `product_implication`. The corpus is evidence for product
decisions, not a claim that any single metric proves humanness.

The Lissin naturalness report is the closest internal analog: it shows that
paralinguistic markup matters, but engine/model choice, evaluation design, and
human validation can move results as much as markup density. This workbench
therefore recommends authentic texting mechanics for authorized drafting, not
"undetectable AI" or deceptive impersonation.

## Authentic Human Texting Model

The main finding from the 100+ source pass is that authentic text messaging is
not mainly "better prose." It is a distribution over small interactional moves:

- adjacency and reply obligation: which prior bubble is being answered
- burst shape: one idea split over several short bubbles versus one polished block
- repair: typo corrections, "wait", "I mean", and revised turns
- phatic work: acknowledgments, presence, soft confirmations, and low-effort care
- relationship-specific shorthand: openings, closings, particles, inside-register terms
- typography: casing, ellipsis, repeated punctuation, elongation, abbreviations
- graphicons: emoji, stickers, GIF/media placeholders, reactions, and placement
- code-switching: matrix language, embedded markers, switch position, and function
- imperfection budget: enough messiness to match the source, not random noise

For the UI, that means the useful object is a participant-level style
distribution by relationship and context, not a single averaged "persona tone."

## Method Notes

LoRA freezes the base model and trains low-rank matrices, making persona adapters easier to store, delete, and route than full model copies. QLoRA adds 4-bit quantization so local SFT is practical on smaller GPUs. DPO is the clean baseline for chosen/rejected preference tuning because it avoids a separate reward model. ORPO, KTO, IPO, DRPO, and GRPO matter as later experiments, but they should not be the first UI path for WhatsApp personas.

For this repo, the existing `DataFormatter` already supports Alpaca, ChatML, and Llama-style SFT exports. The workbench extends that idea with a canonical intermediate format so the same parse can produce:

- prompt/character card artifacts
- Alpaca SFT JSONL
- conversational SFT JSONL
- TRL-style DPO JSONL
- OpenAI-style DPO JSONL
- eval JSONL
- Character Card v2 and Markdown persona files

## Dataset Contracts

The contracts below support style training and portable character exports. The
canonical digital-self profile is separate: it stores claims and evidence
metadata, not SFT prompt/target rows.

### Canonical Row

```json
{
  "schema_version": "whatsapp_persona.v1",
  "example_id": "sha256:...",
  "split": "train",
  "split_group_id": "conv_0001",
  "persona_id": "alice",
  "context": [
    {"role": "other", "speaker": "Bob", "content": "are you coming later?"}
  ],
  "target": {"role": "self", "speaker": "Alice", "content": "probably 9ish"},
  "privacy": {
    "redaction_version": "basic_redaction_v1",
    "redactions": ["URL", "EMAIL"],
    "risk": "low"
  },
  "quality": {
    "synthetic": false,
    "drop_reasons": []
  }
}
```

### SFT Row

```json
{
  "system": "You are Alice. Respond in Alice's observed WhatsApp style. Do not reveal private details.",
  "instruction": "Continue this conversation as Alice.",
  "input": "Bob: are you coming later?",
  "output": "probably 9ish"
}
```

### DPO Row

```json
{
  "prompt": [
    {"role": "system", "content": "Respond as Alice in WhatsApp style."},
    {"role": "user", "content": "Bob: are you coming later?"}
  ],
  "chosen": [{"role": "assistant", "content": "probably 9ish"}],
  "rejected": [{"role": "assistant", "content": "I will respond in a clear and formal manner: probably 9ish"}],
  "metadata": {
    "synthetic": true,
    "negative_source": "formality_mutation"
  }
}
```

Synthetic negatives are only training aids for preference methods. They should be labeled and should not be mixed into SFT as ground truth.

## Sample Text Recommender

The Sample Text tab handles cases where the user has only pasted examples rather
than a full WhatsApp export. It produces:

- a method ranking across prompt card, RAG persona memory, QLoRA SFT, DPO/KTO,
  and synthetic augmentation
- "what works" signals from the sample
- authenticity gaps where the sample cannot support a reliable claim
- code-switching guidance with primary script, secondary scripts, mixed-message
  rate, romanized marker rate, and generation rules
- paralinguistic tags such as `LAUGHTER`, `ELLIPSIS`, `EMOJI`,
  `REPEATED_PUNCTUATION`, `ELONGATION`, `LOWERCASE_START`, `ALL_CAPS`, and
  `DISCOURSE_PARTICLE`
- low-data augmentation recipes with explicit synthetic provenance
- a style card JSON object and prompt snippet for immediate use

The method ladder is deliberately conservative:

- under 20 messages: prompt card plus manual examples
- 20-99 messages: prompt card plus retrieved exemplars
- 100-499 messages: RAG persona memory
- 500-1,999 messages: QLoRA SFT candidate with strict holdout
- 2,000+ messages: QLoRA SFT, then preference tuning if labels exist

Preference training should follow the labels available. DPO/ORPO need
chosen-rejected pairs. KTO needs unary desirable/undesirable labels. DRPO-style
robust variants need subgroup or distribution metadata and are not a default for
one-person WhatsApp datasets.

## Style Features

The useful unit is a distribution, not one average. Compute features per participant, chat, time window, and relationship:

- message length percentiles and one-word reply rate
- consecutive-message burst size
- response latency distribution
- emoji density, diversity, and placement
- punctuation, ellipses, repeated punctuation, all-caps ratio
- laughter, fillers, backchannels, and discourse particles
- vowel elongation and repeated letters
- code-switching markers by script, token inventory, and switch position
- media placeholder frequency and media-only turns

Use Jensen-Shannon distance for categorical distributions, KS or Wasserstein for length/timing distributions, PSI for coarse drift, and MMD for multivariate feature vectors. In this product, "distribution-free" means nonparametric checks such as KS or MMD; it must not mean hiding synthetic output or bypassing disclosure.

## Code-Switching

Code-switching can be intra-message, inter-message, or tag-like. The initial implementation uses lightweight script and token heuristics because the repo has no language ID dependency. Future upgrades can add token-level language ID, matrix-language estimation, romanized Indic markers, and relationship-specific switch rates.

Recommended extraction:

- script mix by participant: Latin, Devanagari, Arabic, Bengali, Tamil, Telugu, Kannada, Malayalam, CJK, other
- switch rate per message and per 100 tokens
- common switch particles such as `haan`, `acha`, `arre`, `yaar`, `lol`, `hmm`
- switch context around apologies, requests, jokes, logistics, and emphasis

Generation should keep the matrix language stable and move only the observed
markers/functions. Good code-switching augmentation changes position and
function within the person's observed distribution; it does not translate whole
messages or sprinkle second-language words at random.

## Low-Data Augmentation

Synthetic data is useful for coverage and preference comparisons, but it must be
kept separate from real target turns. Recommended recipes:

- intent-preserving paraphrase: same fact, same intent, same length band
- tag-controlled variants: vary only observed tags within measured rates
- context-style pairing: reuse real contexts without inventing private facts
- negative formality mutation: generate too-formal, too-long, generic, or
  context-ignoring rejected responses for DPO/KTO review queues
- active learning: ask the owner to write or approve missing reply types

Never train SFT on unreviewed synthetic replies as if they were real messages.
Always hold out real target replies for distribution checks and memorization
guards.

## Character File Strategy

Use an internal WhatsApp Participant Character v1 spec as canonical truth, then export pure transforms:

1. `canonical_character.json`
2. `character_card_v2.json`
3. `persona.md`

Character Card v2 is a portable prompt-card schema with `spec`, `spec_version`, and a nested `data` payload. The canonical format should keep provenance, confidence, privacy policy, and derived style metrics that external prompt-card formats do not own cleanly.

## Safety And Product Constraints

The workbench should support only:

- the uploader's own style
- an organization/team voice the uploader administers
- a person who gave explicit consent

It should not optimize for "undetectable AI" or "make people believe this person wrote it." The system prompt identifies the output as a simulation, denies live authority for the owner, and forbids autonomous message sending or commitments. Drafts should be labeled, reviewed, and never auto-sent. Other participants' messages may be used as context for a target reply dataset, but they should not become target style examples unless separately consented. Raw WhatsApp quotes should stay out of shared character-card exports by default.

## General Digital-Self Evaluation

Evaluation rows have stable IDs derived from their held-out source turn or
explicit interview question, never from the inference configuration. This lets a
future adapter join the comparison without changing the test set.

The required evaluation coverage is:

- latest preference and point-in-time correctness
- superseded beliefs or behavior
- unresolved contradiction handling
- grounding in confirmed claims or relevant evidence
- abstention when the profile is insufficient
- relationship leakage and third-party privacy

The baseline configurations are generic, profile-only, and
profile-plus-retrieval. Candidate outputs are exported to an A/B owner rating
sheet with random-looking A/B assignment; the configuration mapping is kept in a
separate answer key. Automatic summaries contain only counts and coverage. The
private suite may contain held-out dialogue and is written with owner-only
filesystem permissions.

## Sources

- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- DPO: https://arxiv.org/abs/2305.18290
- ORPO: https://arxiv.org/abs/2403.07691
- KTO: https://arxiv.org/abs/2402.01306
- GRPO / DeepSeekMath: https://arxiv.org/abs/2402.03300
- Distributionally Robust DPO: https://arxiv.org/abs/2502.01930
- TRL dataset formats: https://huggingface.co/docs/trl/main/en/dataset_formats
- TRL SFT trainer: https://huggingface.co/docs/trl/en/sft_trainer
- TRL DPO trainer: https://huggingface.co/docs/trl/en/dpo_trainer
- OpenAI supervised fine-tuning: https://developers.openai.com/api/docs/guides/supervised-fine-tuning
- OpenAI direct preference optimization: https://developers.openai.com/api/docs/guides/direct-preference-optimization
- Character Card v2: https://github.com/malfoyslastname/character-card-spec-v2/blob/main/spec_v2.md
- Second-Me repository: https://github.com/mindverse/Second-Me
- AI-native Memory 2.0 / Second Me: https://arxiv.org/abs/2503.08102
- wacli repository: https://github.com/steipete/wacli
- Lissin naturalness report: https://lissin-naturalness-report.pages.dev/
