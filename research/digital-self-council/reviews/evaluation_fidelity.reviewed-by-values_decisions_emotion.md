# Cross-review: `evaluation_fidelity`

- Subject seat: `evaluation_fidelity`
- Reviewer seat: `values_decisions_emotion`
- Reviewer agent: `019f4f07-351c-7361-b187-acee575e3fe6`
- Records reviewed: 20 counted primary papers
- Inspection-depth mix: 19 `full_text`, 1 `abstract`
- Decision: **APPROVE AFTER CORRECTIONS**

The review used publisher paper pages, official proceedings, ACL Anthology, PMLR, OpenReview, arXiv, USENIX, Nature, and PMC/DOI records. It did not use blogs, product documentation, code repositories, private data, or a second publication of the same work as independent evidence.

## 1. Reviewed record IDs

| Record | Depth | Primary-source check | Outcome |
| --- | --- | --- | --- |
| `paper:persona-chat-2018` | `full_text` | ACL Anthology paper page and PDF | Approved after correcting the engagement claim. |
| `paper:dialogue-nli-2019` | `full_text` | ACL Anthology paper page and PDF | Approved as written. |
| `paper:incharacter-2024` | `full_text` | ACL Anthology paper page and PDF | Approved after identifying what the 80.7% result was measured against. |
| `paper:charactereval-2024` | `full_text` | ACL Anthology paper page and PDF | Approved as written. |
| `paper:timechara-2024` | `full_text` | ACL Anthology paper page and PDF | Approved after separating the 600-instance model comparison from the 50-instance human judge validation. |
| `paper:sotopia-2024` | `full_text` | Official ICLR proceedings and paper PDF | Approved as written. |
| `paper:llm-user-sim-crs-2024` | `full_text` | ACL Anthology paper page and PDF | Approved after describing the four sources as movie datasets rather than recommendation platforms. |
| `paper:neural-user-simulation-2018` | `full_text` | ACL Anthology paper page and PDF | Approved; the record follows the official Anthology metadata title. |
| `paper:turing-experiments-2023` | `full_text` | PMLR proceedings page and PDF | Approved as written. |
| `paper:out-of-one-many-2023` | `abstract` | Cambridge publisher abstract and DOI metadata | Approved at the stated abstract-only depth; no full-text inference was added. |
| `paper:synthetic-replacements-2024` | `full_text` | Cambridge publisher full article and DOI metadata | Approved as written. |
| `paper:self-report-agents-2024` | `full_text` | Current arXiv paper page and PDF | Approved as written; retained as a preprint and not treated as peer-reviewed. |
| `paper:llms-more-rational-2025` | `full_text` | Official ICLR proceedings and paper PDF; arXiv identifier cross-check | Approved after replacing the title/year fallback with the verified arXiv identifier. |
| `paper:emulate-personality-2025` | `full_text` | Nature publisher article and DOI metadata | Approved as written. |
| `paper:personality-temporal-stability-2024` | `full_text` | PMC full text, PubMed metadata, and DOI | Approved as written. |
| `paper:linguistic-calibration-2022` | `full_text` | ACL Anthology/TACL paper page and PDF | Approved as written. |
| `paper:extracting-training-data-2021` | `full_text` | USENIX proceedings page and paper PDF | Approved as written. |
| `paper:quantifying-memorization-2023` | `full_text` | ICLR OpenReview paper and arXiv version | Approved as written. |
| `paper:pii-leakage-2023` | `full_text` | IEEE DOI registration and arXiv paper | Approved after correcting the IEEE DOI and primary URL. |
| `paper:aligned-production-extraction-2025` | `full_text` | ICLR OpenReview paper and arXiv version | Approved after distinguishing baseline token generation from the targeted attack. |

## 2. Identifier or metadata corrections

- `paper:pii-leakage-2023`: corrected the non-resolving DOI `10.1109/SP46215.2023.00154` to IEEE's registered DOI `10.1109/SP46215.2023.10179300`; the canonical identifier, primary URL, and DOI search query now agree. The arXiv alias `2302.00539` independently identifies the same paper.
- `paper:llms-more-rational-2025`: replaced the title/year fallback canonical ID with verified `arxiv:2406.17055` and retained the title/year string as an alternate ID. The primary URL remains the official ICLR 2025 proceedings page.
- `paper:neural-user-simulation-2018`: the official ACL Anthology metadata title uses “of Spoken Dialogue Systems,” while the PDF title uses “for Spoken Dialogue Systems” and exposes a slightly different author presentation. The record retains the official Anthology record metadata; this source-level discrepancy is documented rather than silently blended.
- `paper:aligned-production-extraction-2025`: arXiv `2311.17035` began under an earlier title and author presentation, while the record follows the final ICLR 2025 paper metadata and OpenReview primary page. These are versions of one work, not two papers.
- The other 16 records' canonical identifiers and all 20 primary HTTPS URLs identify the claimed papers. Publication year, venue, publication type, and author lists were consistent with the selected primary record, subject to the two documented version wrinkles above.

## 3. Overclaims and required wording changes

- `paper:persona-chat-2018`: rejected the original statement that own-persona conditioning improved engagement. Persona-Chat-trained models were more engaging than Twitter/OpenSubtitles baselines, but within Persona-Chat the own-persona condition improved profile detectability and consistency without improving engagingness. The corrected finding also retains the high human-rating variance and weaker other-person effect.
- `paper:incharacter-2024`: narrowed “80.7 percent alignment” to the reported dimensional result against Personality Database 16P labels. It must not be read as 80.7% agreement with independent human owner judgments.
- `paper:timechara-2024`: corrected the method and sample wording. The 600-instance model comparison was judged primarily by GPT-4; human validation covered 50 instances with 27 qualified annotators.
- `paper:llm-user-sim-crs-2024`: changed “four recommendation platforms” to “four real-world movie datasets,” matching ReDial, Reddit, IMDb, and MovieLens as used by the study.
- `paper:aligned-production-extraction-2025`: changed the data description so the 50-million-token run is attributed to baseline ChatGPT probing rather than conflated with the targeted attack. The paper's attack result remains conservatively paraphrased as more than ten thousand recovered examples at roughly USD 200.
- The findings for `paper:dialogue-nli-2019`, `paper:charactereval-2024`, `paper:sotopia-2024`, `paper:neural-user-simulation-2018`, `paper:turing-experiments-2023`, `paper:out-of-one-many-2023`, `paper:synthetic-replacements-2024`, `paper:self-report-agents-2024`, `paper:emulate-personality-2025`, `paper:personality-temporal-stability-2024`, `paper:linguistic-calibration-2022`, `paper:extracting-training-data-2021`, `paper:quantifying-memorization-2023`, and `paper:pii-leakage-2023` did not exceed their stated inspection depth.

The seat note already expresses these conclusions cautiously and does not repeat the corrected errors. No factual edit to `seat-notes/evaluation_fidelity.md` was required.

## 4. Missing counterevidence or foundational work

- A useful non-blocking counterevidence candidate is Song et al., *Have Large Language Models Developed a Personality?* (`arXiv:2305.14693`). Its option-order symmetry and situation-insensitivity tests directly challenge machine self-assessment, complementing the interviewer/assessor caveats in `paper:incharacter-2024`, demographic sensitivity in `paper:emulate-personality-2025`, and test-retest instability in `paper:personality-temporal-stability-2024`. It was not added during cross-review because this pass corrects and judges the submitted seat rather than expanding extraction.
- The positive population-simulation evidence in `paper:out-of-one-many-2023` is inspected only at abstract depth. It is appropriately counterbalanced by the full-text subgroup, variance, prompt, and temporal failures in `paper:synthetic-replacements-2024`; neither paper establishes owner-specific prediction.
- `paper:self-report-agents-2024` provides broad individual simulation evidence but remains a preprint, while `paper:llms-more-rational-2025` and `paper:llm-user-sim-crs-2024` show that plausible-seeming simulations can be systematically too rational, popularity-biased, generic, or internally inconsistent.
- The seat has no direct benchmark built from held-out longitudinal messaging behavior plus owner adjudication. `paper:timechara-2024` supplies temporal-state logic using fiction, and `paper:persona-chat-2018` supplies blinded dialogue ratings, but neither closes that observability gap.
- LLM-judge reliability appears inside `paper:sotopia-2024`, `paper:charactereval-2024`, `paper:incharacter-2024`, and `paper:timechara-2024`; a dedicated cross-model evaluator-bias benchmark would strengthen the seat. This is a future corpus gap, not a reason to reject the current records.

## 5. Duplicate candidates

The cross-seat alias scan covered all 10 current seat files and found no duplicate canonical paper inside `evaluation_fidelity`, but found nine cross-seat collisions:

| `evaluation_fidelity` record | Other seat record | Collision evidence |
| --- | --- | --- |
| `paper:persona-chat-2018` | `persona_dialogue` / `paper:persona-dialogue-zhang-2018` | Same ACL ID, DOI, and title. |
| `paper:dialogue-nli-2019` | `persona_dialogue` / `paper:persona-dialogue-welleck-2019` | Same ACL ID, DOI, and title. |
| `paper:incharacter-2024` | `persona_dialogue` / `paper:persona-dialogue-wang-2024-incharacter` | Same ACL ID, DOI, and title. |
| `paper:timechara-2024` | `persona_dialogue` / `paper:persona-dialogue-ahn-2024-timechara` | Same ACL ID, DOI, and title. |
| `paper:turing-experiments-2023` | `digital_twins_agents` / `paper:turing-experiments-2023` | Same arXiv ID and title. |
| `paper:out-of-one-many-2023` | `digital_twins_agents` / `paper:out-of-one-many-2023` | Same DOI, arXiv ID, and title. |
| `paper:self-report-agents-2024` | `digital_twins_agents` / `paper:generative-agents-1000-2024` | Same arXiv ID `2411.10109`; title changed across versions. |
| `paper:extracting-training-data-2021` | `privacy_identity_safety` / `paper:carlini-2021-training-data-extraction` | Same arXiv ID and title. |
| `paper:pii-leakage-2023` | `privacy_identity_safety` / `paper:lukas-2023-pii-leakage` | Same arXiv ID and title; the corrected IEEE DOI should be propagated during corpus deduplication. |

Per the council charter, seat overlap is allowed during discovery and `corpus.jsonl` is the deduplicated authority. These records therefore remain counted in this seat; the nine aliases must resolve to single papers when the global corpus is assembled. The `paper:self-report-agents-2024` collision demonstrates why identifier-based deduplication is required even when titles differ.

## 6. Architecture claims to downgrade or reject

- Reject “persona conditioning improves engagement” as a general architecture claim. `paper:persona-chat-2018` separates training-corpus engagement gains from the within-corpus own-persona effect.
- Reject 80.7% label alignment as evidence of 80.7% owner fidelity. `paper:incharacter-2024` measures fictional-character dimensional labels and depends on model-mediated interviewing and assessment.
- Reject treating TimeChara's 600 responses as human-scored ground truth. `paper:timechara-2024` used a human subset to validate an automated judge, which remains a separate source of uncertainty.
- Downgrade any claim that aggregate experimental replication implies individual predictive validity. `paper:turing-experiments-2023` and `paper:out-of-one-many-2023` are bounded by the subgroup and covariance failures in `paper:synthetic-replacements-2024`, the excess rationality in `paper:llms-more-rational-2025`, and the preference failures in `paper:llm-user-sim-crs-2024`.
- Reject psychometric consistency as proof that a model has recovered a person's latent internal state. `paper:emulate-personality-2025` shows unusually clean factor structure, while `paper:personality-temporal-stability-2024` shows model- and trait-dependent drift; both use elicited model responses rather than privileged access to internal human states.
- Reject low leakage under ordinary prompting or alignment refusals as a privacy guarantee. `paper:extracting-training-data-2021`, `paper:pii-leakage-2023`, and `paper:aligned-production-extraction-2025` require adversarial extraction, reconstruction, and inference tests. `paper:quantifying-memorization-2023` also means apparent imitation can be contaminated by memorized benchmark text.

No record-level evidence-strength downgrade or removal was required after these wording corrections. The changes narrow unsupported architecture inferences while preserving the papers' directly observed results.

## 7. Final decision

**APPROVE AFTER CORRECTIONS.** All 20 records were checked against primary sources at their declared depth. The corrected JSONL contains 20 unique canonical papers within the seat, with 19 full-text inspections and one honestly marked abstract inspection. Nine cross-seat duplicates are explicitly reserved for global deduplication. The synthesis note requires no factual correction. This approval authorizes applying the shared review timestamp `2026-07-11T04:16:12Z` and the reviewer agent ID to all 20 provenance objects, followed by validation with `require_reviewed=True`.
