# Cross-Review: persona_dialogue by cognitive_memory

## Review Metadata

- Subject seat: `persona_dialogue`
- Reviewer seat: `cognitive_memory`
- Reviewer agent: `019f4f06-f7d6-7190-87d7-80a1806aedcf`
- Review timestamp: `2026-07-11T03:48:06Z`
- Extractor recorded on all papers: `council-persona_dialogue`
- Final decision: **APPROVE after required corrections**

All 20 records were checked against primary paper sources. Nineteen records were
reconciled to ACL Anthology paper pages and one to the official AAAI proceedings
page. The 13 `full_text` records were checked in their proceedings PDFs; the seven
`abstract` records were checked against their official abstracts and bibliographic
metadata without upgrading their inspection depth. No private data, WhatsApp
content, profile, message database, or credential was accessed.

## 1. Reviewed Record IDs

| Record | Depth | Disposition |
| --- | --- | --- |
| [paper:persona-dialogue-li-2016] | `full_text` | Approved |
| [paper:persona-dialogue-zhang-2018] | `full_text` | Approved |
| [paper:persona-dialogue-mazare-2018] | `full_text` | Approved |
| [paper:persona-dialogue-madotto-2019] | `full_text` | Approved |
| [paper:persona-dialogue-welleck-2019] | `full_text` | Approved |
| [paper:persona-dialogue-kim-2020] | `abstract` | Approved |
| [paper:persona-dialogue-majumder-2020] | `abstract` | Approved |
| [paper:persona-dialogue-xu-2022-goldfish] | `full_text` | Approved after architecture correction |
| [paper:persona-dialogue-xu-2022-lemon] | `full_text` | Approved |
| [paper:persona-dialogue-chen-2023-orig] | `abstract` | Approved |
| [paper:persona-dialogue-gao-2023-peacok] | `abstract` | Approved |
| [paper:persona-dialogue-shao-2023-character-llm] | `full_text` | Approved |
| [paper:persona-dialogue-zhong-2024-memorybank] | `full_text` | Approved after finding correction |
| [paper:persona-dialogue-maharana-2024-locomo] | `full_text` | Approved |
| [paper:persona-dialogue-wang-2024-rolellm] | `full_text` | Approved after metadata correction |
| [paper:persona-dialogue-wang-2024-incharacter] | `full_text` | Approved |
| [paper:persona-dialogue-ahn-2024-timechara] | `full_text` | Approved |
| [paper:persona-dialogue-salemi-2024-lamp] | `abstract` | Approved |
| [paper:persona-dialogue-wang-2024-emg-rag] | `abstract` | Approved after architecture/limitation correction |
| [paper:persona-dialogue-arimoto-2024] | `abstract` | Approved |

## 2. Identifier Or Metadata Corrections

- [paper:persona-dialogue-wang-2024-rolellm]: corrected three author names to
  match the authoritative proceedings PDF: `Noah Wang` to `Zekun Moore Wang`,
  `Z.y. Peng` to `Zhongyuan Peng`, and `Wenhao Huang` to `Stephen W. Huang`.
  Author order and the other 14 author names were already correct.
- The canonical ACL ID `2024.findings-acl.878`, DOI
  `10.18653/v1/2024.findings-acl.878`, title, year, venue, and primary URL for
  [paper:persona-dialogue-wang-2024-rolellm] were correct.
- All other canonical IDs and primary URLs identified the claimed papers. Titles,
  years, venues, and publication types were consistent with the primary pages.
- The empty DOI aliases on [paper:persona-dialogue-shao-2023-character-llm] and
  [paper:persona-dialogue-arimoto-2024] were retained because their official ACL
  entries do not list a DOI.

## 3. Overclaims And Required Wording Changes

- [paper:persona-dialogue-zhong-2024-memorybank]: removed the causal statement
  that adding MemoryBank improved retrieval because the reported quantitative
  table does not provide a no-MemoryBank ablation. The corrected finding says
  that MemoryBank-equipped variants retrieved prior interactions in the simulated
  probes.
- [paper:persona-dialogue-zhong-2024-memorybank]: replaced the claim of improved
  empathy "measures" with the narrower source-supported statement that the paper
  presents a qualitative side-by-side example interpreted by its authors as
  favoring psychologically tuned SiliconFriend over ChatGLM.
- [paper:persona-dialogue-wang-2024-emg-rag]: retained the abstract-supported
  approximately 10% task improvement and reported smartphone usability claim,
  but made the uninspected provenance and source-deletion implementation details
  explicit in the limitation.
- No inspection-depth changes were warranted. Quantitative and causal wording in
  the other 17 findings remained within the inspected primary evidence.

## 4. Missing Counterevidence Or Foundational Work

No corpus-blocking foundational omission was found. The slice contains the early
speaker-embedding model [paper:persona-dialogue-li-2016], PersonaChat
[paper:persona-dialogue-zhang-2018], and Dialogue NLI
[paper:persona-dialogue-welleck-2019], followed by retrieval, memory, robustness,
roleplay, temporal, and realism studies.

Relevant counterevidence is present rather than averaged away:

- Self-profile access improved consistency but not engagement
  [paper:persona-dialogue-zhang-2018].
- Web-mined first-person statements can be temporary or contradictory, and domain
  transfer required adaptation [paper:persona-dialogue-mazare-2018].
- BLEU and perplexity moved against human consistency judgments in the
  meta-learning experiment [paper:persona-dialogue-madotto-2019].
- Persona fine-tuning reduced open-domain coherence in self-play
  [paper:persona-dialogue-xu-2022-lemon].
- Synthetic LoCoMo and acted Multi-Session Chat do not establish natural
  long-term relationship fidelity [paper:persona-dialogue-maharana-2024-locomo]
  [paper:persona-dialogue-arimoto-2024].
- Role benchmarks leave multi-turn drift, static-label error, and point-in-time
  leakage unresolved [paper:persona-dialogue-wang-2024-rolellm]
  [paper:persona-dialogue-wang-2024-incharacter]
  [paper:persona-dialogue-ahn-2024-timechara].

The remaining evidence gap is substantive, not a missing citation that can be
patched into this seat: none of these papers validates faithful, longitudinal
simulation of a consenting living owner across natural relationships. The seat
note states that boundary clearly.

## 5. Duplicate Candidates

- No duplicate candidate was found.
- The corpus has 20 unique record IDs, 20 unique canonical IDs, and 20 unique
  normalized title-year pairs.
- No preprint and published version are both counted. Versioned ACL PDFs for
  [paper:persona-dialogue-shao-2023-character-llm] resolve to one canonical ACL
  record and are not separate papers.
- The roleplay papers and the long-memory papers overlap conceptually but ask
  distinct questions and use distinct datasets or evaluations.
- No separate duplicate report was present for this seat at review time.

## 6. Architecture Claims Downgraded Or Rejected

- Rejected `immutable episodic log` as an implication of
  [paper:persona-dialogue-xu-2022-goldfish]. The paper supports retrieval and
  summarization memory, not immutability; immutability would also conflict with
  explicit retention and deletion controls. The record and synthesis now specify
  a source-linked episodic record under retention controls.
- Downgraded provenance and source-level deletion from implied EMG-RAG paper
  features to requirements that a digital-brain implementation must add
  [paper:persona-dialogue-wang-2024-emg-rag]. At abstract depth, those
  implementation properties could not be verified. Node- and edge-level
  correction/deletion remains supported by the editable-graph framing.
- Retained the warning that MemoryBank's Ebbinghaus-inspired score is a heuristic
  policy device, not evidence of human-memory equivalence
  [paper:persona-dialogue-zhong-2024-memorybank].
- Other architecture implications were accepted as bounded design consequences
  because their records distinguish empirical findings from owner-control,
  provenance, abstention, and evaluation requirements.

## 7. Final Decision

**APPROVE after required corrections.** The four affected records and the two
matching synthesis statements were corrected before review provenance was set.
All 20 records now carry reviewer
`019f4f06-f7d6-7190-87d7-80a1806aedcf` and the single shared UTC timestamp
`2026-07-11T03:48:06Z`. The reviewer is distinct from the recorded extractor.

Final enforced validation:

```text
{'total_records': 20, 'counted_primary_papers': 20, 'unique_canonical_papers': 20, 'reviewed_records': 20, 'seat_counts': {'persona_dialogue': 20}, 'publication_type_counts': {'conference': 20}, 'inspection_depth_counts': {'abstract': 7, 'full_text': 13}}
```
