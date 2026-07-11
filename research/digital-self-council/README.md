# Digital-Self Research Council

This directory is the public, reproducible evidence layer for the digital-brain
architecture. It contains no WhatsApp data and no generated owner profile.

## Why A Council

Representing a person is not one research problem. A dialogue model can imitate
surface language while failing at autobiographical continuity, values, decisions,
relationship boundaries, temporal change, or calibrated uncertainty. The council
keeps those disciplines separate long enough to expose disagreement, then merges
them through one evidence contract.

The ten seats and their boundaries are defined in [`council.yaml`](council.yaml).
The ring cross-review prevents each seat from approving its own interpretation:

| Seat | Cross-reviews |
| --- | --- |
| Psychometrics | Narrative identity |
| Narrative identity | Cognitive memory |
| Cognitive memory | Persona dialogue |
| Persona dialogue | Social relationships |
| Social relationships | Longitudinal modeling |
| Longitudinal modeling | Values, decisions, and emotion |
| Values, decisions, and emotion | Evaluation fidelity |
| Evaluation fidelity | Privacy and identity safety |
| Privacy and identity safety | Digital twins and agents |
| Digital twins and agents | Psychometrics |

## Evidence Standard

A title in a bibliography is not an evidence record. Every counted paper must
conform to [`paper.schema.json`](paper.schema.json) and include:

- a canonical paper identifier and primary HTTPS URL
- bibliographic metadata and council taxonomy
- the research question, method, population/data, and modalities
- one paraphrased finding and concrete architecture implications
- limitations, evidence strength, relevance, and inspection depth
- extraction and independent review provenance

The counted corpus must contain 100-200 unique primary papers. The target is 150.
Published and preprint versions of the same work count once. Books and theory
chapters can be retained as uncounted context but cannot satisfy the paper floor.

## Current Snapshot

The committed run contains 200 cross-reviewed seat records and 187 unique papers
after reconciling 13 duplicate groups. Inspection depth is 100 full text and 87
abstract. All ten seats contribute 20 reviewed papers before deduplication.

`corpus-manifest.json` pins the council and schema contracts, every seat artifact,
every review, the merged corpus, and the evidence map by SHA-256. The runtime run
directory is deliberately not committed because it contains absolute local paths;
the public manifest is the portable reproducibility authority.

## Directory Contract

The completed council run uses this shape:

```text
research/digital-self-council/
  council.yaml
  paper.schema.json
  seats/
    psychometrics.jsonl
    narrative_identity.jsonl
    ...
  seat-notes/
    psychometrics.md
    ...
  reviews/
    psychometrics.reviewed-by-digital_twins_agents.md
    ...
  corpus.jsonl
  corpus-report.json
  corpus-summary.json
  corpus-manifest.json
  taxonomy.md
  evidence-map.json
  contradictions.md
  elicitation-map.md
  synthesis.md
  synthesis-drafts/
    01-construct-validity.md
    ...
```

Seat files may overlap during discovery. `corpus.jsonl` is the deduplicated
authority. Alternate DOI, arXiv, ACL, PubMed, title/year, and ISBN identifiers are
checked across records so an alias collision cannot silently inflate the count.

## Validation

The validator is intentionally local and deterministic:

```python
from living_brain.research import load_jsonl, validate_corpus

records = load_jsonl("research/digital-self-council/corpus.jsonl")
report = validate_corpus(records)
print(report.to_dict())
```

This validates structure, provenance, deduplication, source type, and corpus size.
It does not claim that a URL is currently reachable or that an extracted finding
is correct. URL rechecking and cross-review remain separate completion gates.

Validate the grounded architecture claims and rebuild the portable manifest:

```python
import json
from pathlib import Path

from living_brain.research import (
    build_corpus_manifest,
    load_jsonl,
    validate_evidence_map,
)

root = Path("research/digital-self-council")
corpus = load_jsonl(root / "corpus.jsonl")
evidence_map = json.loads((root / "evidence-map.json").read_text())
print(validate_evidence_map(evidence_map, corpus))
print(build_corpus_manifest(root))
```

## Reproduce A Council Run

1. Initialize a private local run:

   ```bash
   living-brain research init \
     --run-dir .research-runs/digital-self \
     --run-id digital-self-v1 \
     --council research/digital-self-council/council.yaml \
     --schema research/digital-self-council/paper.schema.json
   ```

2. Give each agent one bounded seat using `prompts/seat.md`. The agent owns only
   its seat JSONL and seat note and must leave reviewer fields null.
3. Register each extraction with `research validate-seat`, including agent ID and
   the actual query strategy. Partial or invalid seats remain failed or pending.
4. Give each completed seat to the previous seat in the review ring using
   `prompts/cross-review.md`. The reviewer corrects the artifact, writes one review,
   and stamps review provenance only after approval.
5. Register reviewed seat files again so the run pins their final hashes.
6. Run `research merge`. It refuses partial, changed, unreviewed, self-reviewed,
   or sub-100 evidence and preserves every duplicate source group in the report.
7. Rebuild `corpus-summary.json`, `evidence-map.json`, and
   `corpus-manifest.json`; run citation, URL, schema, test, lint, and type checks.

Council agents use public research only. A research refresh never requires or
permits access to WhatsApp messages, owner profiles, local databases, or provider
credentials.

## Synthesis Rules

The synthesis must distinguish:

- **validated:** supported by direct evidence across relevant studies
- **plausible:** a defensible design hypothesis with partial or indirect evidence
- **speculative:** useful to test, but not established by the reviewed literature

Architecture claims cite paper record IDs rather than free-form links. Negative
and contradictory evidence stays visible. WhatsApp-only observability gaps become
adaptive owner-interview questions, not confident psychological inference.

## Privacy Boundary

Research agents use public research sources only. They must never read local
message databases, chat exports, digital-self profiles, pseudonym keys, private
evaluation rows, or council provider credentials. Public paper evidence and
private identity evidence remain different systems with different ownership.
