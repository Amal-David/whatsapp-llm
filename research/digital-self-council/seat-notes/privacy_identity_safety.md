# Privacy, Identity, and Safety Seat Notes

## Scope and evidence posture

This seat covers privacy in personalized models; consent and third-party data; sensitive or protected-trait inference; identity, likeness, and impersonation; digital remains and posthumous avatars; user control; governance; and technical safety mechanisms. It does not make jurisdiction-specific legal conclusions. The requirements below are conservative engineering and product-governance implications from the evidence, not legal advice.

The corpus contains 20 counted primary papers: 17 inspected at full-text depth and 3 at publisher-abstract depth. No WhatsApp content, private profile, local database, credential, or key was accessed. Search was limited to public primary-paper surfaces.

## Search strategy

Research ran as a set of concept families rather than one broad query:

- Training privacy: differential privacy, membership inference, training-data extraction, PII leakage, deduplication, and machine unlearning.
- Inference-time privacy: contextual integrity, multi-party disclosure, third-party consent, and permission signals for model training.
- Sensitive inference: sexual orientation, location, political identity, gender classification, social graphs, behavioral traces, and attribute-inference defenses.
- Identity safety: voice-clone provenance, facial manipulation detection, representation fidelity, and impersonation.
- Digital remains: post-mortem stewardship, delegate expectation alignment, griefbots, advance consent, and user vulnerability.

Queries combined exact titles, identifiers, venue names, and site restrictions for DOI resolvers, arXiv, USENIX, IEEE, ACM, PMLR, OpenReview, CVF, NeurIPS, First Monday, Springer, Frontiers, PubMed, and PMC. Each record preserves its concrete queries in `provenance.search_queries`.

For every candidate, I checked the primary page, canonical identifier, title, authors, year, venue, method, data, principal result, and stated limitations. Published and preprint versions were counted once; alternate identifiers are retained only as aliases. Full text was used for 17 records. The three abstract-depth records are `paper:scheuerman-2019-computers-see-gender`, `paper:brubaker-2016-legacy-contact`, and `paper:gach-2021-facebook-affairs`; their claims are intentionally limited to publisher-verified metadata and abstracts.

## Inclusion and exclusion

Included papers had to be unique, verifiable primary research or a primary deployed-system design case, directly inform the bounded seat, and provide enough evidence to state a concrete architecture implication and limitation. The set intentionally combines attacks, defenses, audits, human studies, and governance-oriented design work rather than treating privacy as only a training algorithm.

Excluded material included blogs, product pages, news reports, GitHub documentation, secondary explainers, literature reviews, unverifiable citations, legal commentary without a primary study, and duplicate preprint/published versions. Papers were also excluded when only a headline-level claim could be verified or when they would encourage unsupported conclusions about an individual's identity. Public availability or crawlability was never treated as consent (`paper:longpre-2024-consent-crisis`).

## Domain taxonomy

| Domain | Core records | What the domain contributes |
|---|---|---|
| Formal and empirical training privacy | `paper:abadi-2016-deep-learning-dp`; `paper:shokri-2017-membership-inference`; `paper:carlini-2021-training-data-extraction`; `paper:kandpal-2022-dedup-privacy`; `paper:lukas-2023-pii-leakage`; `paper:bourtoule-2021-machine-unlearning` | Privacy budgets, leakage audits, duplication risk, PII-specific attacks, and deletion-aware training structure. |
| Contextual and interdependent consent | `paper:mireshghallah-2024-confaide`; `paper:franz-2022-interdependent-privacy`; `paper:longpre-2024-consent-crisis` | Recipient and purpose constraints, third-party ownership, explicit consent signals, and ambiguity handling. |
| Sensitive and protected-trait inference | `paper:kosinski-2013-private-traits`; `paper:jernigan-2009-gaydar`; `paper:gong-2016-attribute-inference`; `paper:buolamwini-2018-gender-shades`; `paper:scheuerman-2019-computers-see-gender`; `paper:jia-2018-attriguard` | Evidence that innocuous traces and relationships expose sensitive traits, while identity classifiers misrepresent and unevenly harm people. |
| Posthumous control and digital remains | `paper:brubaker-2016-legacy-contact`; `paper:gach-2021-facebook-affairs`; `paper:lu-2026-griefbot-perceptions` | Advance directives, bounded stewardship, expectation mismatch, representational integrity, and vulnerability-aware controls. |
| Likeness and impersonation mechanisms | `paper:san-roman-2024-audioseal`; `paper:rossler-2019-faceforensics` | Proactive provenance for synthetic voice and reactive detection for manipulated faces, with explicit robustness limits. |

## Evidence-backed synthesis

### 1. Personalized models can disclose their source data

**Validated.** Model outputs can reveal whether a record was in training (`paper:shokri-2017-membership-inference`), regenerate verbatim web text and PII (`paper:carlini-2021-training-data-extraction`), and expose PII through extraction, reconstruction, and inference even after automated scrubbing (`paper:lukas-2023-pii-leakage`). Duplication materially amplifies regeneration, and deduplication lowers the measured attack surface (`paper:kandpal-2022-dedup-privacy`). Differentially private training provides a formal contribution bound at a selected privacy unit and budget (`paper:abadi-2016-deep-learning-dp`), while deletion requires a separate lineage and retraining mechanism (`paper:bourtoule-2021-machine-unlearning`).

**Plausible architecture conclusion.** A digital brain should keep raw evidence in an access-controlled store and retrieve only authorized fragments at inference time. Any learned artifact still needs person-level contribution limits, exact and near deduplication, privacy accounting, PII extraction tests, and a data-to-artifact deletion map. No single layer is sufficient.

**Unresolved.** The papers do not establish a universally safe privacy budget, clipping unit, or extraction threshold for a small, relationship-dense WhatsApp archive. Those values need threat modeling and validation against the actual training and retrieval design.

### 2. Privacy is a rule about information flow, not just secrecy

**Validated.** Strong LLMs disclosed private facts to inappropriate recipients in multi-party tasks, and privacy reminders or chain-of-thought prompts did not reliably prevent it (`paper:mireshghallah-2024-confaide`). A salience intervention reduced intended disclosure of other people's address-book data, but the study still demonstrates that the importing user is making a decision about many other subjects (`paper:franz-2022-interdependent-privacy`). Web permission signals are mutable, inconsistent, and purpose-specific (`paper:longpre-2024-consent-crisis`).

**Plausible architecture conclusion.** Authorization must execute before retrieval and again before output. Each memory needs subject, sender, audience, source, collection time, purpose, allowed transformations, retention, and consent state. An LLM must not decide access from a prose prompt. Ambiguous or conflicting permission should produce quarantine or a user question, not silent ingestion.

**Unresolved.** There is no evidence-backed rule for resolving every co-owned message when the account owner and another participant disagree. The system needs an explicit conflict policy and a way to remove one participant's contribution without pretending the remaining conversation is contextually unchanged.

### 3. Sensitive inference is a capability to suppress, not a profile feature to celebrate

**Validated.** Likes can predict sensitive labels (`paper:kosinski-2013-private-traits`); friends can expose a hidden sexual-orientation label (`paper:jernigan-2009-gaydar`); and social-plus-behavioral joins improve inference of cities, employers, and majors (`paper:gong-2016-attribute-inference`). At the same time, commercial gender systems had severe intersectional error disparities (`paper:buolamwini-2018-gender-shades`) and restrictive schemas that could not represent nonbinary identities and performed worse for transgender people (`paper:scheuerman-2019-computers-see-gender`). Small perturbations reduced one tested attribute attack but supplied no universal guarantee (`paper:jia-2018-attriguard`).

**Plausible architecture conclusion.** Do not infer or persist sexuality, religion, caste, race, ethnicity, health status, disability, political affiliation, or gender identity from messages, contacts, voice, face, or behavior. If the person explicitly self-describes an identity for a clear purpose, store their wording separately from model-derived claims, allow correction and deletion, and prevent proxy features from reconstituting a removed field.

**Unresolved.** A useful audit must test both explicit sensitive labels and proxy recovery from embeddings, summaries, relationship graphs, nearest neighbors, and generated answers. The corpus does not provide one complete benchmark for that combined surface.

### 4. Likeness safety needs positive provenance and negative detection

**Validated.** AudioSeal shows that a compliant speech generator can add a robust, localized watermark with fast detection, but white-box knowledge enables stronger removal attacks (`paper:san-roman-2024-audioseal`). FaceForensics++ shows that in-domain detectors can outperform people on known manipulations, while performance falls with compression and does not establish future-generator generalization (`paper:rossler-2019-faceforensics`).

**Plausible architecture conclusion.** Synthetic voice or face output needs explicit source-person consent, a visible disclosure, signed generation provenance, an embedded watermark where possible, and a takedown or contest route. A detector score is only one signal. Watermark absence cannot prove authenticity, and detector success cannot prove authorization.

**Unresolved.** Robust provenance across editing, re-recording, model changes, and cross-platform forwarding remains open. False-positive handling is especially important because an incorrect impersonation accusation can itself be harmful.

### 5. Posthumous access is not posthumous personation

**Validated.** Legacy Contact separates limited stewardship from logging in as the deceased (`paper:brubaker-2016-legacy-contact`). Users and selected stewards can remain confident while misunderstanding what the feature actually permits, indicating that ordinary click-through setup is inadequate (`paper:gach-2021-facebook-affairs`). Public interviews about griefbots identify conditional comfort and the same feature's risks of dependency, inauthenticity, prolonged grief, inequality, and misrepresentation (`paper:lu-2026-griefbot-perceptions`).

**Plausible architecture conclusion.** Posthumous simulation should default off. A pre-death directive should separately specify preservation, deletion, export, memorial access, text simulation, voice, face, audience, purpose, duration, steward, re-confirmation, and revocation. A steward may administer the directive but should not gain authority to create undisclosed new statements in the deceased person's voice.

**Unresolved.** There is no longitudinal evidence here that establishes when griefbot use helps or harms bereaved people, nor how to resolve conflicts among a prior directive, a steward, family members, and third parties represented in the model.

## Negative and conflicting evidence

- Deduplication reduced measured regeneration but did not protect unique secrets or contextual disclosure (`paper:kandpal-2022-dedup-privacy`; `paper:mireshghallah-2024-confaide`). It is a preprocessing control, not a privacy guarantee.
- Differential privacy bounds contribution under a chosen unit, while unlearning seeks zero contribution from specified records (`paper:abadi-2016-deep-learning-dp`; `paper:bourtoule-2021-machine-unlearning`). A deletion UI cannot honestly claim success without derived-artifact handling.
- Automated PII scrubbing lowered risk but missed entities, and measured differentially private models still leaked some PII in the tested setup (`paper:lukas-2023-pii-leakage`). Scrubbing, privacy-preserving training, and output controls must remain independent.
- Privacy prompting is weak evidence of safety: explicit reminders and reasoning prompts did not reliably stop contextual disclosure (`paper:mireshghallah-2024-confaide`).
- High benchmark predictability does not make sensitive inference appropriate or individually true (`paper:kosinski-2013-private-traits`; `paper:jernigan-2009-gaydar`; `paper:gong-2016-attribute-inference`). Identity audits show that classification schemas and errors can erase or misrepresent people (`paper:buolamwini-2018-gender-shades`; `paper:scheuerman-2019-computers-see-gender`).
- AttriGuard improved a tested privacy-utility tradeoff, but it changes public data and can face adaptive detection (`paper:jia-2018-attriguard`). Mutating canonical memories would undermine evidence fidelity.
- Reactive deepfake detection performs well in a bounded benchmark, whereas proactive watermarking depends on generator participation and protected detector material (`paper:rossler-2019-faceforensics`; `paper:san-roman-2024-audioseal`). These controls complement rather than replace each other.
- Griefbots may be experienced as comforting and harmful for the same reason: simulated continued presence (`paper:lu-2026-griefbot-perceptions`). Product success metrics such as session length would be unsafe proxies for benefit.
- A third-party salience nudge reduced intended disclosure but did not obtain consent from the people in the address book (`paper:franz-2022-interdependent-privacy`). Better owner UX does not erase the third party's stake.

## Implications for a digital brain

### Required control plane

| Layer | Minimum control | Evidence basis |
|---|---|---|
| Ingestion | Source manifest; person and third-party scope; purpose-specific consent; duplicate grouping; quarantine for ambiguity | `paper:franz-2022-interdependent-privacy`; `paper:longpre-2024-consent-crisis`; `paper:kandpal-2022-dedup-privacy` |
| Storage | Separate canonical evidence, derived summaries, identity self-descriptions, inferred claims, and synthetic media; encrypt and minimize each independently | `paper:lukas-2023-pii-leakage`; `paper:scheuerman-2019-computers-see-gender` |
| Learning | Person-level contribution accounting; differential privacy where appropriate; no protected-trait objectives; extraction and membership red teams | `paper:abadi-2016-deep-learning-dp`; `paper:shokri-2017-membership-inference`; `paper:carlini-2021-training-data-extraction` |
| Retrieval | Deterministic subject-recipient-purpose authorization before context assembly; least-privilege fields and source citations | `paper:mireshghallah-2024-confaide` |
| Output | PII and sensitive-inference checks; uncertainty; provenance; no unconsented voice, face, or identity simulation | `paper:lukas-2023-pii-leakage`; `paper:san-roman-2024-audioseal`; `paper:rossler-2019-faceforensics` |
| User control | View, correct, exclude, delete, export, revoke, pause, and inspect an audit log at person, relationship, source, and capability levels | `paper:bourtoule-2021-machine-unlearning`; `paper:gach-2021-facebook-affairs` |
| Posthumous state | Advance directive; narrow steward role; default-off simulation; periodic re-confirmation; dispute and hard-pause states | `paper:brubaker-2016-legacy-contact`; `paper:gach-2021-facebook-affairs`; `paper:lu-2026-griefbot-perceptions` |

### Hard design boundaries

1. No inferred sensitive or protected traits are promoted into durable profile facts.
2. No participant's data is considered authorized merely because the archive owner can export it.
3. No prompt instruction substitutes for retrieval authorization.
4. No deletion is reported complete until derived indexes, adapters, checkpoints, summaries, and caches are handled and audited.
5. No synthetic likeness is released without positive authorization and provenance; detection alone is insufficient.
6. No posthumous simulation is enabled by ordinary account inheritance or inactivity.
7. No generated statement is represented as something the source person actually said unless it is a cited verbatim record.

## WhatsApp observability gaps

This research did not inspect a WhatsApp archive. Even with an owner-authorized export, the following facts are generally not observable or cannot be safely inferred from message content alone:

- Whether each correspondent or group participant consents to model training, long-term memory, retrieval, summarization, or persona simulation.
- Whether consent changed after sending, or whether deletion, disappearing-message, view-once, edit, block, or retention actions are absent from the export.
- The intended audience, social context, or purpose of a message after forwarding, quoting, screenshots, cross-posting, or off-platform discussion.
- Whether a phone number, display name, shared device, forwarded identity, or imported contact reliably maps to the person the system thinks it does.
- Whether text about sexuality, religion, health, caste, politics, disability, or gender is self-description, quotation, speculation, sarcasm, abuse, or someone else's information.
- Whether a photo, voice note, sticker, or video is authentic, generated, edited, consensually shared, or authorized for future synthesis.
- Whether all relevant media, deleted messages, call context, reactions, receipts, group-history changes, and external relationship events are present.
- Whether a participant is deceased, what posthumous wishes they expressed, who may act as steward, or whether others dispute those wishes.
- Whether the owner has authority to disclose relationship edges and jointly produced conversational context for a new purpose.
- Whether end-to-end transport encryption still protects data after export, indexing, training, logging, backup, or generated disclosure; those are separate system boundaries.

The safe response to these gaps is explicit owner configuration, separate third-party handling, abstention, and auditable uncertainty. It is not silent trait inference or a claim that archive possession resolves consent.

## Open questions

1. What is the privacy unit for this system: message, person, relationship, group, time window, or a combination, and how will repeated forwards be grouped?
2. Which capabilities can the owner authorize alone, and which require exclusion, notification, or affirmative permission from another represented person?
3. How will conflicting permissions be enforced when one co-author wants a shared exchange retained and another wants it removed?
4. Can useful personalization remain retrieval-based and on-device, avoiding weight updates from private text altogether?
5. What red-team suite will test membership, extraction, reconstruction, contextual disclosure, proxy inference, and cross-person leakage before every release?
6. How will deletion verification cover raw evidence, summaries, embeddings, caches, backups, checkpoints, adapters, audit logs, and previously exported artifacts?
7. What is the false-positive and appeal policy for impersonation or sensitive-inference warnings?
8. How will explicit self-described identity be corrected or deleted without being reconstructed from retained proxies?
9. What advance directive and re-confirmation cadence are adequate before any posthumous text, voice, or face simulation is technically possible?
10. Which griefbot safeguards can be evaluated without turning engagement, emotional dependency, or clinical inference into optimization targets?
11. How will the system visibly distinguish verbatim evidence, grounded summary, uncertain inference, roleplay, and synthetic persona output?
12. Who audits consent lineage and privacy budgets, and what evidence must exist before the product can claim that a revocation or deletion took effect?

## Bottom line

The evidence supports a layered design: minimize ingestion; represent consent and audience explicitly; keep canonical evidence separable from derived models; suppress sensitive inference; test leakage; make deletion executable; require positive provenance for likeness; and keep posthumous simulation off without a specific advance directive. The central safety property is not that the model is good at keeping secrets. It is that unauthorized information never reaches a model or recipient in the first place (`paper:mireshghallah-2024-confaide`).
