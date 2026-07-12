# Cross-Review: `digital_twins_agents`

- Reviewer seat: `privacy_identity_safety`
- Reviewer agent: `019f4f07-4af5-7562-98f7-3f525272ce25`
- Subject extractor: `council-digital_twins_agents`
- Records reviewed: 20 counted primary papers
- Inspection-depth mix reviewed: 17 full text, 3 abstract
- Reviewed at: `2026-07-11T09:06:58Z`
- Decision entered before provenance stamping: **APPROVE after corrections**

The reviewer did not extract these records. Verification used only primary paper sources: publisher or official proceedings pages, DOI-linked papers, ACL Anthology, arXiv, PMLR, and the authors' official Microsoft Research publication pages. No WhatsApp content, private profile, local database, key, or other private data was accessed.

## 1. Reviewed record IDs

| Record | Depth | Disposition |
|---|---|---|
| `paper:mylifebits-2006` | abstract | Approved as written |
| `paper:sensecam-retrospective-2006` | abstract | Approved as written |
| `paper:sensecam-lifelog-memory-2007` | abstract | Approved as written |
| `paper:connected-memories-comet-2022` | full text | Approved as written |
| `paper:social-simulacra-2022` | full text | Approved as written |
| `paper:generative-agents-2023` | full text | Approved after evidence-strength reclassification |
| `paper:generative-agents-1000-2024` | full text | Approved after current-revision metadata and claim corrections |
| `paper:out-of-one-many-2023` | full text | Approved; cross-seat duplicate candidate |
| `paper:turing-experiments-2023` | full text | Approved; cross-seat duplicate candidate |
| `paper:twin-2k-500-2025` | full text | Approved as written |
| `paper:behaviorchain-2025` | full text | Approved after evidence-strength downgrade |
| `paper:twinvoice-2026` | full text | Approved after data-provenance, architecture, and evidence corrections |
| `paper:shopcart-human-behavior-2026` | full text | Approved as written |
| `paper:memgpt-2023` | full text | Approved as written |
| `paper:memorybank-2024` | full text | Approved; cross-seat duplicate candidate |
| `paper:plato-ltm-2022` | full text | Approved after evidence-strength downgrade; cross-seat duplicate candidate |
| `paper:locomo-2024` | full text | Approved; cross-seat duplicate candidate |
| `paper:longmemeval-2025` | full text | Approved after evidence-strength downgrade |
| `paper:secom-2025` | full text | Approved after title and evidence-strength corrections |
| `paper:personalization-legitimizes-risks-2026` | full text | Approved after architecture, limitation, and evidence-strength corrections |

All canonical identifiers and primary URLs identify the claimed papers. The three abstract-depth records limit their claims to information available on official abstract or publication pages. The other 17 records were checked against full papers, including methods, sample or dataset construction, reported results, limitations, and architecture implications.

## 2. Identifier and metadata corrections

1. `paper:generative-agents-1000-2024`: retained canonical arXiv identifier `2411.10109`, publication year 2024, and the primary arXiv URL, but updated the record to the current revision title, **LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals**. Added current-version authors Jonne Kamphorst and Niles Egan and corrected the author order. Rewrote the research question, method, modalities, finding, limitations, and search queries to reflect the current interview-only, survey-only, and combined-agent study rather than the superseded first-version framing.
2. `paper:secom-2025`: retained arXiv identifier `2502.05589` and the official ICLR 2025 primary URL; corrected the title to the published proceedings form, **SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents**.
3. No canonical identifier, alternate identifier, year, venue, publication type, or primary URL required correction for the other 18 records.

## 3. Overclaims and required wording changes

1. `paper:generative-agents-1000-2024`: replaced the superseded claim that interview agents achieved "about 85%" across selected measures with the current paper's bounded General Social Survey result: 83% normalized accuracy for interview-only agents, 82% for survey-only agents, 86% for combined agents, and 74% for demographics-only agents. The record now says explicitly that performance varies by outcome and does not establish a complete or stable digital self.
2. `paper:twinvoice-2026`: removed the description of the interpersonal dimension as private. It uses a public Pushshift Telegram archive, while the social dimension uses public Chinese microblog data and the narrative dimension is fictional. Reworded the research question and synthesis so these separate corpora are not presented as one person's integrated life or as evidence from consented private conversations.
3. `paper:personalization-legitimizes-risks-2026`: removed "purpose checks" from the paper-backed architecture claim because the source evaluates PII-aware sanitization and access control, not a purpose-control mechanism. Removed the asserted self-reinforcing write-back loop because the study does not test longitudinal write-back. The retained claim is the evaluated post-retrieval detection-reflection intervention.
4. The subject note was corrected wherever it repeated these claims. Governance recommendations about write-back and purpose limitation are now labeled as separate requirements needing their own evaluation, not findings of PS-Bench.

## 4. Missing counterevidence or foundational work

### Counterevidence restored

- `paper:generative-agents-1000-2024`: added that economic-game performance did not significantly differ among interview-only, survey-only, and combined agents; the aggregate analysis of five experiments was underpowered; sample-level correlation structure was not reproduced; and the evidence concerns self-reports and controlled responses rather than natural real-world behavior.
- `paper:twinvoice-2026`: added the public-archive and separate-corpora boundary. Strong recognition scores therefore cannot be read as evidence of an integrated, consented relationship twin.
- `paper:behaviorchain-2025`, `paper:plato-ltm-2022`, `paper:longmemeval-2025`, `paper:secom-2025`, and `paper:personalization-legitimizes-risks-2026`: retained their useful controlled findings while making synthetic construction, LLM-generated data or scoring, narrow human validation, and limited external validity decisive in evidence-strength calibration.
- The existing negative evidence from `paper:turing-experiments-2023`, `paper:twin-2k-500-2025`, and `paper:shopcart-human-behavior-2026` is sufficient to prevent aggregate plausibility, survey agreement, or fluent generation from being presented as person-level behavioral fidelity.

### Foundational coverage

No additional foundational paper is required for approval of this bounded seat. `paper:mylifebits-2006` and `paper:sensecam-retrospective-2006` establish early capture and cueing assumptions, while `paper:generative-agents-2023` is field-defining for the memory-reflection-planning architecture. `paper:sensecam-lifelog-memory-2007` supplies useful boundary evidence rather than another foundational architecture claim.

## 5. Duplicate candidates

The target file contains 20 unique canonical papers internally. The following are cross-seat identity matches and should be resolved in the council's deduplication phase; they remain `counted=true` here because cross-review precedes deduplication:

| Target record | Matching record | Identity evidence |
|---|---|---|
| `paper:generative-agents-1000-2024` | `evaluation_fidelity` / `paper:self-report-agents-2024` | arXiv `2411.10109` |
| `paper:out-of-one-many-2023` | `evaluation_fidelity` / `paper:out-of-one-many-2023` | DOI `10.1017/pan.2023.2`, arXiv `2209.06899` |
| `paper:turing-experiments-2023` | `evaluation_fidelity` / `paper:turing-experiments-2023` | arXiv `2208.10264`, same PMLR paper |
| `paper:memorybank-2024` | `persona_dialogue` / `paper:persona-dialogue-zhong-2024-memorybank` | DOI `10.1609/aaai.v38i17.29946` |
| `paper:plato-ltm-2022` | `persona_dialogue` / `paper:persona-dialogue-xu-2022-lemon` | DOI `10.18653/v1/2022.findings-acl.207`, ACL `2022.findings-acl.207` |
| `paper:locomo-2024` | `persona_dialogue` / `paper:persona-dialogue-maharana-2024-locomo` | DOI `10.18653/v1/2024.acl-long.747`, ACL `2024.acl-long.747` |

No preprint and published version are double-counted within the target seat.

## 6. Architecture claims downgraded or rejected

1. `paper:generative-agents-1000-2024`: replaced "interviews rather than demographics" with source- and modality-aware grounding. The current revision shows that structured surveys perform similarly to interviews on the reported GSS metric and that combining them yields a modest improvement; it does not justify a universal interview-first rule.
2. `paper:twinvoice-2026`: rejected "maintain relationship-specific memories and style profiles" as a demonstrated result. The benchmark motivates testing relationship- and channel-conditioned behavior separately, but its three dimensions use different datasets and do not evaluate one integrated relationship-memory architecture.
3. `paper:personalization-legitimizes-risks-2026`: rejected purpose checks and a longitudinal unsafe-response write-back loop as source findings. Retained only the supported post-retrieval safety check, PII-aware sanitization and access control, and detection-reflection intervention.
4. Evidence strength changed from `strong` to `moderate` for `paper:behaviorchain-2025`, `paper:twinvoice-2026`, `paper:plato-ltm-2022`, `paper:longmemeval-2025`, `paper:secom-2025`, and `paper:personalization-legitimizes-risks-2026` because controlled or synthetic evidence and narrow external validity do not meet the council's strong-evidence threshold.
5. `paper:generative-agents-2023` changed from `strong` to `foundational`: its small simulated-world evaluation limits external validity, but the paper is field-defining for the memory-reflection-planning architecture. The final seat mix is 3 foundational, 2 strong, and 15 moderate records.

## 7. Final decision

**APPROVE after corrections.** All required factual and wording corrections above are applied to the target JSONL and, where the same claims appeared, to the subject synthesis. The 20 records are suitable for the reviewed corpus at their stated inspection depths and calibrated evidence strengths. Cross-seat identity matches are explicitly deferred to the council's deduplication phase and are not concealed by this approval.
