# Longitudinal Modeling Seat Notes

Task: T-0407
Seat: `longitudinal_modeling`
Extractor: `council-longitudinal_modeling`

## Evidence Status

This seat contains 20 counted primary papers: 19 were inspected at full-text depth and one at abstract depth. No metadata-only record is counted. All 20 records were cross-reviewed by the `social_relationships` seat under reviewer `019f4f07-10e4-7660-9c2f-c6ea34490a72` at `2026-07-11T08:58:31Z`.

The synthesis uses three claim labels:

- **Validated**: directly supported within the inspected papers and their reported settings.
- **Plausible**: an architecture implication supported by converging evidence but not tested as an end-to-end digital-brain design.
- **Speculative**: a useful open design hypothesis that still needs direct evaluation.

## Search Strategy

Search proceeded through six evidence lanes:

1. Temporal collaborative filtering and sequential recommendation: long-term versus short-term state, elapsed-time features, and event-driven user trajectories (`paper:koren-2009-temporal-dynamics`, `paper:xiang-2010-temporal-preference-fusion`, `paper:rendle-2010-fpmc`, `paper:wu-2017-recurrent-recommender`, `paper:dai-2016-deepcoevolve`, `paper:kumar-2019-jodie`, `paper:li-2020-tisasrec`).
2. Incremental recommendation: learning new interactions while preserving earlier utility (`paper:xu-2020-graphsail`, `paper:mi-2020-ader`, `paper:mi-2020-man`).
3. Active learning and preference elicitation: information value, answerability, and user effort (`paper:rashid-2002-getting-to-know-you`, `paper:boutilier-2003-active-cf`, `paper:harpale-2008-personalized-active-learning`, `paper:rashid-2008-information-theoretic-elicitation`).
4. Owner-visible profiles and corrections: inspection, direct manipulation, explanations, and overrides (`paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`, `paper:bakalov-2013-controllable-user-models`).
5. Catastrophic-forgetting safeguards: parameter constraints, episodic replay, and regression metrics (`paper:kirkpatrick-2017-ewc`, `paper:lopez-paz-2017-gem`).
6. Lifelong language learning: sparse textual replay and temporary local adaptation (`paper:masson-dautume-2019-episodic-language-memory`).

Queries combined exact titles with venue, DOI, method, and dataset terms. The concrete queries are retained in each record's provenance. Verification was limited to publisher or official proceedings pages, DOI records, arXiv, and PMC. When a paper had both a preprint and a published version, it was counted once under one canonical identifier and the other identifier was retained only as an alias.

## Inclusion And Exclusion

Included papers had to satisfy all of the following:

- Primary research with a verifiable canonical identifier and primary HTTPS URL.
- Direct relevance to longitudinal state, temporal preference, incremental personalization, elicitation, owner correction, forgetting, or update evaluation.
- Enough inspected evidence to paraphrase the method, data, finding, and limitations without relying on a secondary summary.
- A distinct contribution rather than a duplicate preprint and proceedings version.

Excluded material included surveys used only for discovery, blogs, product pages, GitHub documentation, secondary explainers, unverifiable citations, and papers whose relevance was only generic personalization. Papers about population-level concept drift without a user-model or update-policy connection were also excluded. The one abstract-depth record remains counted because its publisher-verified abstract states the method, domains, and comparative result, but its limitations explicitly mark the uninspected details (`paper:xiang-2010-temporal-preference-fusion`).

## Domain Taxonomy

| Domain | Evidence contribution | Records |
| --- | --- | --- |
| Temporal decomposition | Separates enduring preference, slow drift, and transient or session state | `paper:koren-2009-temporal-dynamics`, `paper:xiang-2010-temporal-preference-fusion`, `paper:rendle-2010-fpmc` |
| Dynamic trajectories | Updates latent user and item state at events and sometimes projects state through elapsed time | `paper:wu-2017-recurrent-recommender`, `paper:dai-2016-deepcoevolve`, `paper:kumar-2019-jodie`, `paper:li-2020-tisasrec` |
| Continual recommendation | Preserves old utility through distillation, exemplars, or nonparametric memory while learning new interactions | `paper:xu-2020-graphsail`, `paper:mi-2020-ader`, `paper:mi-2020-man` |
| Active elicitation | Selects questions by expected value, information gain, answerability, effort, or coverage | `paper:rashid-2002-getting-to-know-you`, `paper:boutilier-2003-active-cf`, `paper:harpale-2008-personalized-active-learning`, `paper:rashid-2008-information-theoretic-elicitation` |
| Owner correction and control | Exposes profiles, source weights, edits, explanations, overrides, or disable controls | `paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`, `paper:bakalov-2013-controllable-user-models` |
| Stability-plasticity safeguards | Constrains updates with parameter importance, replay anchors, and temporal regression metrics | `paper:kirkpatrick-2017-ewc`, `paper:lopez-paz-2017-gem`, `paper:masson-dautume-2019-episodic-language-memory` |

## Evidence-Backed Synthesis

### One profile is not enough

**Validated.** In recommendation benchmarks, models that distinguish stationary preference from slow change and immediate context outperform undifferentiated static models. This pattern appears in explicit temporal factorization, long/short graph fusion, personalized transition models, recurrent trajectories, and interval-aware attention (`paper:koren-2009-temporal-dynamics`, `paper:xiang-2010-temporal-preference-fusion`, `paper:rendle-2010-fpmc`, `paper:wu-2017-recurrent-recommender`, `paper:li-2020-tisasrec`).

**Plausible.** A digital brain should maintain at least four separately governed views: enduring claims, slowly evolving preferences, episode or project context, and request-local state. Promotion between layers should be explicit. A recent interaction can influence a response without becoming a durable identity assertion.

### Time is evidence about context, not meaning

**Validated.** Relative intervals, event order, and time projection can improve future-action prediction (`paper:kumar-2019-jodie`, `paper:li-2020-tisasrec`). However, the temporal signal is not universal: Amazon Beauty showed no obvious sequential pattern in the TiSASRec analysis (`paper:li-2020-tisasrec`).

**Plausible.** Update logic should first test whether a domain exhibits stable temporal signal. Timestamp recency should adjust confidence or retrieval priority, not independently determine that an owner changed their mind. Silence and inactivity must remain missing evidence rather than negative preference.

### Fast memory and slow consolidation serve different jobs

**Validated.** Incremental recommenders can reduce forgetting with graph-structure distillation, bounded exemplars, or a nonparametric memory path (`paper:xu-2020-graphsail`, `paper:mi-2020-ader`, `paper:mi-2020-man`). Sparse replay and retrieval-time local adaptation also complement each other in lifelong language benchmarks (`paper:masson-dautume-2019-episodic-language-memory`).

**Plausible.** New observations should first enter an inspectable episodic store. Request-local adaptation can use those observations without changing durable parameters. Consolidation should happen only after repeated evidence, an owner declaration, or a measured benefit that passes retention tests.

### Preservation pressure should respond to measured drift

**Validated.** A fixed anti-forgetting mechanism is not always beneficial. ADER found that ordinary fine-tuning forgot little in the more stable YOOCHOOSE stream and that EWC added only marginal value there, while stronger preservation helped more on the dynamic DIGINETICA stream (`paper:mi-2020-ader`). GraphSAIL's local-structure term also hurt one model-data pairing in an ablation (`paper:xu-2020-graphsail`).

**Plausible.** A digital brain should estimate drift and regression risk per profile component. It should not apply global decay, replay, or parameter locking uniformly. Stable owner commitments deserve stronger retention than low-confidence, time-bounded inferences.

### Ask questions that can change a decision

**Validated.** Information-value strategies generally outperform random or uncertainty-only elicitation, but answerability and effort materially affect their practical value (`paper:rashid-2002-getting-to-know-you`, `paper:boutilier-2003-active-cf`, `paper:harpale-2008-personalized-active-learning`, `paper:rashid-2008-information-theoretic-elicitation`). Pure entropy performed poorly in an early pilot, and personalized answerability sharply reduced failed queries (`paper:rashid-2002-getting-to-know-you`, `paper:harpale-2008-personalized-active-learning`).

**Plausible.** Ask for owner input only when the answer is likely to alter a consequential response, resolve a conflict, or prevent a durable misupdate. The query score should combine expected decision change, answerability, disclosure cost, urgency, and coverage. A skip or "not sure" should update the query policy, not the person's preference.

### Control is desired, but direct editing can make a model worse

**Validated.** Users valued visibility and control in all three profile-control studies, but objective effects conflict. Editable profiles reduced precision and recall in YourNews (`paper:ahn-2007-open-user-profiles`). TasteWeights reported higher satisfaction and relevance but acknowledged that interaction supplied extra feedback, confounding control with additional data (`paper:bostandjiev-2012-tasteweights`). The professional literature study improved subjective control measures but had seven participants, no control group, and an ordered two-phase design (`paper:bakalov-2013-controllable-user-models`).

**Plausible.** Owner authority should be absolute while edits remain operationally safe: preview the behavioral effect, distinguish temporary suppression from durable deletion, validate the proposed delta, and make every change reversible. The system should never reject an owner's correction merely because an offline metric declines, but it may surface the predicted consequences and ask which scope was intended.

### Forgetting safeguards need semantic governance

**Validated.** EWC, GEM, and episodic replay reduce benchmark forgetting through parameter constraints or retained examples (`paper:kirkpatrick-2017-ewc`, `paper:lopez-paz-2017-gem`, `paper:masson-dautume-2019-episodic-language-memory`). None of these methods supplies an owner-readable reason for retaining a belief, a consent policy for replay data, or a deletion protocol.

**Plausible.** Retention tests should use owner-approved anchors and temporal slices, but replay membership itself must be auditable. A technically useful memory item may still need expiry or deletion. Stability is not the same as legitimacy.

## Negative And Conflicting Evidence

- Temporal modeling is not automatically useful. TiSASRec found weak visible sequential structure in one Amazon domain (`paper:li-2020-tisasrec`), and Koren found that indiscriminate age-based downweighting can discard durable signal (`paper:koren-2009-temporal-dynamics`).
- Continual-learning machinery can be unnecessary or counterproductive in stable streams. Fine-tuning was already competitive on YOOCHOOSE, EWC's gain was marginal there, and one GraphSAIL preservation component hurt a PinSage setting (`paper:mi-2020-ader`, `paper:xu-2020-graphsail`).
- More memory is not a free win. MAN's performance dropped as memory shrank, while its best large-memory setup implies storage and deletion costs (`paper:mi-2020-man`). Lifelong language replay similarly stores raw examples that may be sensitive (`paper:masson-dautume-2019-episodic-language-memory`).
- Direct manipulation has mixed outcomes. YourNews provides objective negative evidence, whereas TasteWeights and the biochemical portal provide mainly short-term subjective positive evidence with important confounds (`paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`, `paper:bakalov-2013-controllable-user-models`).
- General continual-learning benchmarks remain indirect. EWC did not match separate Atari agents, GEM relies on task descriptors, and both are evaluated on clean task sequences unlike overlapping personal change (`paper:kirkpatrick-2017-ewc`, `paper:lopez-paz-2017-gem`).

## Implications For A Digital Brain

The following architecture is an evidence-informed proposal, not a directly validated end-to-end system.

### Separate evidence from derived state

- Append observations to a provenance-bearing evidence ledger; do not silently rewrite the durable profile.
- Build versioned derived views for enduring, slow-changing, episodic, and request-local state (`paper:koren-2009-temporal-dynamics`, `paper:rendle-2010-fpmc`, `paper:kumar-2019-jodie`).
- Label each derived claim with source, observation time, valid-time interval, confidence, inference policy, and supersession status.

### Use an auditable update policy

1. **Observe:** append the event and its context without treating behavior as owner endorsement.
2. **Propose:** generate a typed delta targeting one temporal layer, with confidence, expected benefit, expiry, and affected behaviors.
3. **Evaluate:** compare the proposal against future-time validation, owner-approved replay anchors, and per-slice retention metrics (`paper:xu-2020-graphsail`, `paper:lopez-paz-2017-gem`).
4. **Elicit when needed:** ask only when uncertainty is consequential and the question is answerable at reasonable cost (`paper:boutilier-2003-active-cf`, `paper:harpale-2008-personalized-active-learning`).
5. **Preview and commit:** show high-impact owner-facing changes, then store the accepted version, diff, rationale, policy version, and rollback pointer (`paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`).
6. **Expire, supersede, or delete:** enforce validity windows and owner deletion across evidence, replay sets, indexes, and derived state.

### Evaluate longitudinally

- Use chronological rather than random holdouts for update quality (`paper:koren-2009-temporal-dynamics`, `paper:wu-2017-recurrent-recommender`).
- Report new-period utility, old-period retention, backward transfer, and forward transfer instead of a single aggregate score (`paper:lopez-paz-2017-gem`).
- Stratify by stable versus drifting domains, frequent versus rare evidence, and owner-declared versus inferred facts (`paper:mi-2020-ader`, `paper:mi-2020-man`).
- Track correction success separately from recommendation accuracy. An owner-authorized deletion is successful even when a predictive metric falls.

### Keep owner correction typed and reversible

At minimum, correction controls should distinguish: factually wrong, no longer true, true only in a context, do not use for personalization, hide temporarily, and delete. The distinction responds to the harm observed when broad profile removals were used to solve narrower novelty or redundancy problems (`paper:ahn-2007-open-user-profiles`). Source-weight controls and immediate previews can expose narrower interventions (`paper:bostandjiev-2012-tasteweights`).

## WhatsApp Observability Gaps

No WhatsApp content, private profile, local database, or key was accessed for this seat. The following are observability limits, not findings about any person's messages.

- A sent message is an action in a social context, not a direct scalar preference. Recommendation papers can model event order, but they do not validate translating conversational behavior into identity claims (`paper:rendle-2010-fpmc`, `paper:dai-2016-deepcoevolve`).
- Message timing conflates availability, urgency, timezone, work routines, relationship norms, and platform access. Elapsed time can help prediction, but it cannot identify the cause of a gap (`paper:kumar-2019-jodie`, `paper:li-2020-tisasrec`).
- Silence is missing-not-at-random. It must not be interpreted as dislike, disagreement, or drift without corroboration.
- Audience matters. The same statement can be performative, humorous, quoted, delegated, or constrained by a group. This seat does not provide a validated method for resolving that pragmatics gap.
- Explicit owner correction is sparse. A system needs low-friction clarification and correction controls rather than assuming later behavior implicitly repairs an earlier inference (`paper:rashid-2008-information-theoretic-elicitation`, `paper:bakalov-2013-controllable-user-models`).
- Offline preferences and decisions are unobserved. A conversational archive cannot establish that absence of discussion means absence of interest.
- Edits, deletions, missing media, partial exports, and changing platform affordances can make the observed history incomplete. Derived state should therefore retain uncertainty and provenance.
- Messages involve other people. Replay and long-term memory policies must account for third-party content, not only the owner's personalization benefit (`paper:bostandjiev-2012-tasteweights`, `paper:masson-dautume-2019-episodic-language-memory`).

## Open Questions

1. What evidence threshold should move a claim from request-local to episodic, slow-changing, or enduring state?
2. How should change-point detection incorporate owner declarations when behavior and explicit correction conflict?
3. Which past examples are legitimate replay anchors, and how can rare owner-important commitments be protected without retaining excessive raw data (`paper:mi-2020-ader`)?
4. How should the system distinguish a changed preference from a changed opportunity, audience, or conversational role?
5. What is the right query budget, and how should disclosure sensitivity enter expected-value-of-information calculations (`paper:boutilier-2003-active-cf`)?
6. Can owner corrections be expressed as typed constraints that survive model replacement while remaining easy to inspect and delete?
7. Which update classes require preview or explicit confirmation, and which low-impact updates can be safely automatic?
8. How should rollback propagate through embeddings, caches, retrieval indexes, summaries, and replay memories?
9. What longitudinal evaluation can measure owner-recognized fidelity, not merely next-action prediction?
10. How should a digital brain document uncertainty when a model performs well but its latent trajectory is not semantically interpretable (`paper:wu-2017-recurrent-recommender`, `paper:kumar-2019-jodie`)?

## Record Index

- Temporal and sequential models: `paper:koren-2009-temporal-dynamics`, `paper:xiang-2010-temporal-preference-fusion`, `paper:rendle-2010-fpmc`, `paper:wu-2017-recurrent-recommender`, `paper:dai-2016-deepcoevolve`, `paper:kumar-2019-jodie`, `paper:li-2020-tisasrec`.
- Incremental recommendation: `paper:xu-2020-graphsail`, `paper:mi-2020-ader`, `paper:mi-2020-man`.
- Active elicitation: `paper:rashid-2002-getting-to-know-you`, `paper:boutilier-2003-active-cf`, `paper:harpale-2008-personalized-active-learning`, `paper:rashid-2008-information-theoretic-elicitation`.
- Owner correction and control: `paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`, `paper:bakalov-2013-controllable-user-models`.
- Continual-learning safeguards: `paper:kirkpatrick-2017-ewc`, `paper:lopez-paz-2017-gem`, `paper:masson-dautume-2019-episodic-language-memory`.
