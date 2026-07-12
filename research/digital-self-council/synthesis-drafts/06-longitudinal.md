# Longitudinal Updating and Forgetting

## Evidence Scope

This synthesis treats longitudinal updating as a governance problem as well as a prediction problem. The direct evidence base is the 20-paper `longitudinal_modeling` seat: temporal and sequential recommendation, event-driven trajectories, incremental recommendation, active elicitation, editable user profiles, and continual-learning safeguards. The strongest recurring result is that enduring signal, slow change, and immediate context can all matter, but the optimal split and preservation pressure vary by domain (`paper:koren-2009-temporal-dynamics`, `paper:rendle-2010-fpmc`, `paper:wu-2017-recurrent-recommender`, `paper:li-2020-tisasrec`, `paper:mi-2020-ader`).

The direct seat contains 19 full-text inspections and one abstract inspection. The abstract-only record, `paper:xiang-2010-temporal-preference-fusion`, supports only the published claim that long- and short-term fusion improved its two recommendation tasks; it cannot establish implementation details or owner-recognized preference states. Adjacent evidence is less uniform in depth: much of the cognitive-memory, narrative-identity, psychometrics, social-relationship, and values literature was inspected at abstract depth, while the key conversational-memory, continual-learning, privacy, and evaluation records used below were mostly inspected in full text. Synthesis does not upgrade an abstract-derived result into full-text evidence (`paper:xiang-2010-temporal-preference-fusion`, `paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`, `paper:psychometrics_fleeson_2001`, `paper:bardi-schwartz-2003-values-behavior`, `paper:andersen-chen-2002-relational-self`).

Recommendation events, shopping sessions, ratings, synthetic conversations, fictional characters, and laboratory memory tasks are useful analogues, not direct validation of a living owner's digital self. Their results constrain update mechanics, temporal evaluation, and failure tests; they do not prove that a latent trajectory is an interpretable identity or that retained evidence is consented and legitimate (`paper:kumar-2019-jodie`, `paper:mi-2020-ader`, `paper:lopez-paz-2017-gem`, `paper:locomo-2024`, `paper:timechara-2024`).

The synthesis therefore uses three statuses:

- **VALIDATED** means directly supported in the cited papers' own settings, with transfer limits kept visible.
- **PLAUSIBLE** means a design hypothesis supported by converging evidence but not tested as an end-to-end digital-brain architecture.
- **SPECULATIVE** means a useful claim that still lacks direct evidence and must be treated as an evaluation target, not a product fact.

## VALIDATED Constraints

1. **Do not collapse time into one profile.** Static preference, gradual drift, immediate sequence, and elapsed-time effects each improve prediction in some domains, while traits are better represented as distributions of context-sensitive states than as deductions from isolated acts (`paper:koren-2009-temporal-dynamics`, `paper:xiang-2010-temporal-preference-fusion`, `paper:rendle-2010-fpmc`, `paper:psychometrics_fleeson_2001`, `paper:psychometrics_sherman_2015`).

2. **Temporal signal is domain-specific.** TiSASRec found no obvious sequential pattern for Amazon Beauty, plain fine-tuning forgot little on the more stable YOOCHOOSE stream, and a GraphSAIL preservation component hurt one model-data pairing; time-aware modeling and anti-forgetting machinery must earn their complexity per domain (`paper:li-2020-tisasrec`, `paper:mi-2020-ader`, `paper:xu-2020-graphsail`).

3. **Recency is not a semantic update rule.** Indiscriminate age downweighting can discard enduring signal, and timestamps cannot distinguish changed preference from changed opportunity, role, frame, audience, or availability (`paper:koren-2009-temporal-dynamics`, `paper:li-2020-tisasrec`, `paper:tversky-kahneman-1981-framing`, `paper:andersen-chen-2002-relational-self`).

4. **Observed behavior is not direct preference or value ground truth.** Purchases and clicks are actions under constraints; values relate unevenly to behavior; choice depends on frames and option sets; and the standard free-choice paradigm can manufacture apparent preference change without latent change (`paper:rendle-2010-fpmc`, `paper:bardi-schwartz-2003-values-behavior`, `paper:tversky-simonson-1993-context-preferences`, `paper:chen-risen-2010-free-choice`).

5. **Continuity and change can coexist.** Longitudinal narrative studies find both stable remembered events and changing interpretation, while current self-concept and elicitation can alter immediate autobiographical description. A changed meaning must not silently rewrite the source event (`paper:mcadams-et-al-2006-continuity-change`, `paper:kober-habermas-2017-personal-past-stability`, `paper:jardmo-et-al-2023-repeated-narratives`, `paper:libby-eibach-2002-looking-back`, `paper:dargembeau-garcia-jimenez-2024-working-self`).

6. **Fast acquisition and slow consolidation solve different problems.** Complementary-learning accounts separate rapid episode acquisition from slower interleaved generalization, and continual systems obtain gains from exemplars, episodic constraints, nonparametric memory, or sparse replay. None of those methods makes replay consent, semantic legitimacy, or deletion automatic (`paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`, `paper:cognitive-memory-schapiro-et-al-2017`, `paper:mi-2020-ader`, `paper:lopez-paz-2017-gem`, `paper:masson-dautume-2019-episodic-language-memory`).

7. **Retrieval is itself an update pressure.** Repeated retrieval can strengthen a target, retrieval can impair related competitors, and reminders can integrate later material into an earlier memory. A retrieval or consolidation system must therefore test both target gain and collateral loss instead of assuming that frequently recalled content is more true (`paper:cognitive-memory-karpicke-roediger-2008`, `paper:cognitive-memory-anderson-bjork-bjork-1994`, `paper:cognitive-memory-hupbach-et-al-2007`).

8. **Active elicitation should optimize decision value, answerability, and burden, not uncertainty alone.** Expected-value and information-gain approaches generally beat random or entropy-only selection, while personalized answerability reduces failed questions and aggressive candidate pruning loses decision quality (`paper:rashid-2002-getting-to-know-you`, `paper:boutilier-2003-active-cf`, `paper:harpale-2008-personalized-active-learning`, `paper:rashid-2008-information-theoretic-elicitation`).

9. **Owner control and predictive accuracy are separate objectives.** Users preferred visibility and control, but profile editing reduced objective precision and recall in one small study; the positive TasteWeights result confounded control with extra feedback; and the seven-person controllable-profile study did not establish objective or durable gains (`paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`, `paper:bakalov-2013-controllable-user-models`).

10. **Long memory still fails on time, updates, and abstention.** LongMemEval reports large degradation over sustained histories even with structured memory improvements, LoCoMo remains far below human performance especially on temporal and adversarial questions, and point-in-time roleplay models frequently leak future knowledge (`paper:longmemeval-2025`, `paper:locomo-2024`, `paper:timechara-2024`).

11. **There is no validated universal decay curve for a digital self.** The Ebbinghaus replication is a one-person, 69-list study with a possible non-monotonic 24-hour deviation; MemoryBank's curve is explicitly exploratory; and old ratings can remain useful. A fixed exponential timer is not an evidence-backed deletion policy (`paper:cognitive-memory-murre-dros-2015`, `paper:memorybank-2024`, `paper:koren-2009-temporal-dynamics`).

12. **Deletion and privacy require lineage beyond the source row.** Machine-unlearning work structures training so affected artifacts can be retrained more cheaply, while extraction and deduplication studies show that learned artifacts can reproduce source data and that preprocessing defenses remain incomplete. A deletion claim must cover evidence, derivatives, indexes, replay, caches, and learned artifacts (`paper:bourtoule-2021-machine-unlearning`, `paper:carlini-2021-training-data-extraction`, `paper:kandpal-2022-dedup-privacy`).

13. **Confidence is not evidence quality.** Human judgments of learning can be miscalibrated, and conversational agents need an independently evaluated calibration mechanism. Owner-specific autobiographical calibration remains unvalidated even though linguistic calibration helps in factual question answering (`paper:cognitive-memory-koriat-bjork-2005`, `paper:linguistic-calibration-2022`).

## PLAUSIBLE Design Hypotheses

### Temporal scopes

The following scopes are an engineering hypothesis, not a set of durations established by the papers. Durations are configurable priors; evidence type, domain, relationship, and owner instruction determine promotion or expiry (`paper:koren-2009-temporal-dynamics`, `paper:rendle-2010-fpmc`, `paper:kuppens-2010-dynaffect`, `paper:andersen-chen-2002-relational-self`).

| Scope | Default horizon | Write rule | Exit rule | Evidence basis |
| --- | --- | --- | --- | --- |
| Request-local state | One turn to several hours | Current request, explicit temporary instruction, or retrieved context; never auto-promote | End of task, explicit reset, or short expiry | Immediate transitions and event state matter, but do not establish durable preference (`paper:rendle-2010-fpmc`, `paper:dai-2016-deepcoevolve`). |
| Episode or project state | Hours to weeks | Repeated evidence within one bounded episode, goal, trip, health event, or project | Completion, abandonment, owner correction, or explicit archival | Short histories and session state can improve prediction, while active goals can temporarily suppress competitors without erasing them (`paper:xiang-2010-temporal-preference-fusion`, `paper:shah-2002-goal-shielding`). |
| Slow-changing domain state | Weeks to months | Repeated cross-episode evidence, owner declaration, or a validated change point in one domain | Versioned supersession or owner-scoped correction | Slow drift and domain-specific temporal effects are useful in some recommendation settings (`paper:koren-2009-temporal-dynamics`, `paper:li-2020-tisasrec`). |
| Durable owner-confirmed state | Months to years, never assumed permanent | Explicit owner confirmation plus corroborating history for values, commitments, self-descriptions, and enduring preferences | Explicit change, revocation, deletion, or a newer owner-confirmed version | Stable summaries require aggregation, while state variability and changing narrative interpretation remain possible (`paper:psychometrics_epstein_1983`, `paper:psychometrics_fleeson_2001`, `paper:kober-habermas-2017-personal-past-stability`). |
| Historical evidence and versions | Retained under consent and retention policy | Append source events, corrections, policy decisions, and version links with provenance | Policy-driven deletion; otherwise historical rather than current | Source monitoring, reconsolidation, and point-in-time errors require preserving source and version boundaries (`paper:cognitive-memory-hashtroudi-johnson-chrosniak-1989`, `paper:cognitive-memory-hupbach-et-al-2007`, `paper:timechara-2024`). |

### Drift and promotion

A component should be considered drifting only when evidence changes within the same semantic domain and comparable context. The detector should compare recent and historical windows, retain uncertainty, and condition on relationship, audience, option set, opportunity, and affect; a cross-context difference is not automatically a temporal change (`paper:koren-2009-temporal-dynamics`, `paper:li-2020-tisasrec`, `paper:tversky-simonson-1993-context-preferences`, `paper:bell-1984-audience-design`, `paper:kuppens-2010-dynaffect`).

Promotion should require one of three paths: an explicit owner declaration; repeated, source-diverse evidence that survives context controls; or a measured prediction benefit that passes old-state retention and owner-recognition tests. No number of repeated model-generated summaries should count as independent evidence (`paper:psychometrics_epstein_1983`, `paper:psychometrics_sherman_2015`, `paper:longmemeval-2025`, `paper:cognitive-memory-hupbach-et-al-2007`).

### Active elicitation

A candidate question should be ranked by expected decision change, answerability, burden, disclosure sensitivity, urgency, and evidence coverage. The system should ask when the answer could prevent a consequential response, settle a high-impact conflict, authorize a durable update, or disambiguate deletion scope; uncertainty without action relevance is insufficient (`paper:boutilier-2003-active-cf`, `paper:harpale-2008-personalized-active-learning`, `paper:rashid-2008-information-theoretic-elicitation`, `paper:franz-2022-interdependent-privacy`).

`Skip`, `not sure`, and `do not ask again` are query-policy outcomes, not negative preferences. Missing exposure is not inability to answer, and silence is not an owner correction (`paper:harpale-2008-personalized-active-learning`, `paper:rashid-2002-getting-to-know-you`).

### Owner correction

Corrections should be typed as `factually_wrong`, `changed_over_time`, `context_only`, `wrong_subject_or_source`, `temporary_suppression`, `do_not_personalize`, or `delete`. Each correction should support scope, preview, immediate effect, reversal, and propagation status because broad removal can damage useful state while narrow controls may address the actual problem (`paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`, `paper:persona-dialogue-wang-2024-emg-rag`).

Owner authority should override predictive optimization. The system may explain expected consequences or ask which scope was intended, but an authorized correction or deletion is successful even if recommendation accuracy falls (`paper:ahn-2007-open-user-profiles`, `paper:bourtoule-2021-machine-unlearning`).

### Continual learning and replay

Use a fast, inspectable episodic or retrieval path for new evidence and a slower consolidation path for derived claims or learned parameters. Consolidation should interleave owner-approved anchors, test rare commitments as well as frequent behavior, and occur only after the proposed update passes chronological and cross-context checks (`paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`, `paper:mi-2020-ader`, `paper:lopez-paz-2017-gem`, `paper:masson-dautume-2019-episodic-language-memory`).

Replay eligibility should be an explicit, revocable property of an evidence item. Frequency-based exemplar allocation can underprotect rare but owner-important commitments, while large or unbounded memory increases privacy, storage, and deletion costs (`paper:mi-2020-ader`, `paper:mi-2020-man`, `paper:masson-dautume-2019-episodic-language-memory`).

### Supersession, replayability, and decay

Supersession should create a new version linked to the prior version, close the prior version's current-validity interval, and preserve both for historical queries unless retention or deletion policy requires removal. Relationship- or role-scoped claims may coexist rather than supersede globally (`paper:longmemeval-2025`, `paper:timechara-2024`, `paper:andersen-chen-2002-relational-self`).

Every current state should be replayable from authorized source events, owner corrections, update-policy version, model or adapter version, retrieval/index version, and ordered commit log. Learned weights may accelerate behavior, but they should not be the only canonical location of a personal fact because they are difficult to inspect, correct, time-slice, and delete (`paper:kirkpatrick-2017-ewc`, `paper:bourtoule-2021-machine-unlearning`, `paper:persona-dialogue-xu-2022-goldfish`, `paper:persona-dialogue-wang-2024-emg-rag`).

Decay should first reduce retrieval priority or confidence, then move an item to a reversible archive; it should not erase source evidence or infer that the owner forgot. Owner-confirmed commitments, legal records, correction history, and provenance should follow explicit retention rules rather than unattended recency decay, while an authorized deletion overrides archival value (`paper:cognitive-memory-murre-dros-2015`, `paper:koren-2009-temporal-dynamics`, `paper:memorybank-2024`, `paper:bourtoule-2021-machine-unlearning`).

## SPECULATIVE Claims

1. The proposed turn/hour, episode/week, domain/month, and durable/year horizons may be useful initialization priors, but no included paper validates those exact boundaries for personal messaging. They must be learned or configured per owner and domain (`paper:koren-2009-temporal-dynamics`, `paper:li-2020-tisasrec`).

2. A hybrid drift trigger combining predictive residuals, semantic contradiction, context matching, and owner recognition may distinguish real preference change from role or opportunity change better than recency alone, but that combined detector has not been evaluated (`paper:kumar-2019-jodie`, `paper:tversky-simonson-1993-context-preferences`, `paper:andersen-chen-2002-relational-self`).

3. Owner-approved semantic replay anchors may preserve identity-relevant commitments with less privacy exposure than raw-example replay, but the continual-learning papers evaluate task examples rather than consented semantic anchors (`paper:mi-2020-ader`, `paper:lopez-paz-2017-gem`, `paper:masson-dautume-2019-episodic-language-memory`).

4. Typed correction events may survive model replacement and make owner trust more durable than direct profile editing, but existing control studies are small, short, and do not test cross-model replay (`paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`, `paper:bakalov-2013-controllable-user-models`).

5. Relationship-conditioned decay may preserve useful shared shorthand while preventing one relationship's language from becoming a global persona, but partner-specific common-ground studies do not validate long-horizon expiry rules (`paper:metzing-brennan-2003-conceptual-pacts-broken`, `paper:andersen-chen-2002-relational-self`).

6. A digital brain can perhaps become more faithful by asking fewer, higher-value owner questions, but movie-rating elicitation does not establish optimal burden or disclosure policy for autobiographical and relationship-sensitive questions (`paper:boutilier-2003-active-cf`, `paper:harpale-2008-personalized-active-learning`, `paper:franz-2022-interdependent-privacy`).

## Contradictions and Boundaries

| Tension | Evidence that pushes one way | Evidence that limits or contradicts it | Boundary for this program |
| --- | --- | --- | --- |
| Favor recent evidence | Time and interval features improve future-event prediction in several domains (`paper:kumar-2019-jodie`, `paper:li-2020-tisasrec`). | Age downweighting can discard enduring signal, and one domain showed weak sequential structure (`paper:koren-2009-temporal-dynamics`, `paper:li-2020-tisasrec`). | Recency is a feature, never a standalone supersession or deletion rule. |
| Preserve old capability | Distillation, replay, parameter constraints, and episodic memory reduce benchmark forgetting (`paper:xu-2020-graphsail`, `paper:kirkpatrick-2017-ewc`, `paper:lopez-paz-2017-gem`). | Fine-tuning forgot little in one stable stream; one preservation term hurt; separate Atari agents still outperformed one EWC agent (`paper:mi-2020-ader`, `paper:xu-2020-graphsail`, `paper:kirkpatrick-2017-ewc`). | Apply preservation per component after measured regression risk, not globally. |
| Retain more memory | Large or nonparametric memories improve new and infrequent-item performance and later-session recall (`paper:mi-2020-man`, `paper:persona-dialogue-xu-2022-goldfish`). | Memory consumes storage, complicates deletion, can reproduce sensitive text, and can legitimize harmful intent (`paper:masson-dautume-2019-episodic-language-memory`, `paper:carlini-2021-training-data-extraction`, `paper:personalization-legitimizes-risks-2026`). | Retention requires purpose, authorization, minimization, and deletion lineage. |
| Let owners edit directly | Owners report greater transparency and control (`paper:bostandjiev-2012-tasteweights`, `paper:bakalov-2013-controllable-user-models`). | Broad editing reduced objective metrics in YourNews, and positive studies were confounded or very small (`paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`, `paper:bakalov-2013-controllable-user-models`). | Keep authority absolute, but make edits typed, previewed, scoped, and reversible. |
| Repetition implies importance | Retrieval practice strengthens later access and repeated narratives retain content (`paper:cognitive-memory-karpicke-roediger-2008`, `paper:kober-habermas-2017-personal-past-stability`). | Retrieval can impair competitors, reminders can contaminate memory, and repeated narrative meaning can change (`paper:cognitive-memory-anderson-bjork-bjork-1994`, `paper:cognitive-memory-hupbach-et-al-2007`, `paper:jardmo-et-al-2023-repeated-narratives`). | Repetition raises retrieval salience, not truth or current-validity status. |
| Stable output implies stable person | Some models and self-report agents show encouraging agreement or consistency (`paper:generative-agents-1000-2024`). | Model personality stability varies by model and instrument, synthetic means can hide variance and relation failures, and performance varies by behavioral domain (`paper:personality-temporal-stability-2024`, `paper:synthetic-replacements-2024`, `paper:generative-agents-1000-2024`). | Report domain- and time-specific fidelity, never one stability score. |
| Long-memory benchmarks validate personal memory | LoCoMo, LongMemEval, and segmented retrieval expose concrete temporal and update failures (`paper:locomo-2024`, `paper:longmemeval-2025`, `paper:secom-2025`). | Their histories, questions, or scoring are substantially constructed or model-mediated and omit deletion and social ownership (`paper:locomo-2024`, `paper:longmemeval-2025`, `paper:secom-2025`). | Use them as diagnostic unit tests, not evidence of owner fidelity. |

## Architecture Consequences

### State and event model

The canonical source layer should be append-only under normal operation and deletion-aware under policy. A source event records subject, speaker, audience, source locator, observed time, claimed event time, relationship or role, modality, consent and purpose scope, and extraction uncertainty; source-monitoring failures and interdependent privacy make those fields part of correctness, not optional metadata (`paper:cognitive-memory-hashtroudi-johnson-chrosniak-1989`, `paper:franz-2022-interdependent-privacy`, `paper:mireshghallah-2024-confaide`).

A derived claim should minimally contain the following audit and validity fields (`paper:longmemeval-2025`, `paper:cognitive-memory-hashtroudi-johnson-chrosniak-1989`, `paper:persona-dialogue-wang-2024-emg-rag`):

```text
claim_id
subject_id
claim_type
domain
temporal_scope
relationship_scope
source_event_ids
observed_at
valid_from / valid_to
confidence
owner_status
status: proposed | active | superseded | archived | deleted
supersedes / superseded_by
inference_policy_version
model_and_index_versions
replay_eligibility
retention_and_deletion_scope
```

The separation of source, derived state, temporal validity, and editable status is a plausible response to update failures, source confusion, point-in-time leakage, and graph correction evidence; the exact schema is not directly validated as a whole (`paper:longmemeval-2025`, `paper:cognitive-memory-hashtroudi-johnson-chrosniak-1989`, `paper:timechara-2024`, `paper:persona-dialogue-wang-2024-emg-rag`).

### Auditable update policy

1. **Observe:** append the authorized event and observable context without treating behavior as endorsement or the assistant's output as owner evidence (`paper:rendle-2010-fpmc`, `paper:bardi-schwartz-2003-values-behavior`).
2. **Propose:** emit a typed delta with target temporal scope, affected domain and relationships, evidence, confidence, expected benefit, expiry, and alternatives (`paper:koren-2009-temporal-dynamics`, `paper:andersen-chen-2002-relational-self`).
3. **Test:** score the proposal on future-time utility, old-state retention, rare owner-approved anchors, source attribution, cross-context leakage, and calibration (`paper:lopez-paz-2017-gem`, `paper:mi-2020-ader`, `paper:linguistic-calibration-2022`).
4. **Elicit:** ask only when the expected decision or safety effect justifies answer burden and disclosure cost; keep `skip` separate from preference evidence (`paper:boutilier-2003-active-cf`, `paper:harpale-2008-personalized-active-learning`, `paper:franz-2022-interdependent-privacy`).
5. **Commit:** write the accepted version, diff, policy and model versions, rationale, preview receipt, and rollback pointer; owner corrections take effect independently of predictive score (`paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`).
6. **Consolidate:** generalize only from authorized evidence, interleave approved anchors, and test target gain plus competitor loss before updating a semantic view or adapter (`paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`, `paper:cognitive-memory-anderson-bjork-bjork-1994`, `paper:masson-dautume-2019-episodic-language-memory`).
7. **Supersede or archive:** close current validity without destroying historical provenance; preserve context-specific alternatives when both can be true (`paper:longmemeval-2025`, `paper:andersen-chen-2002-relational-self`).
8. **Delete:** revoke retrieval and replay first, then propagate through sources, summaries, graphs, embeddings, indexes, caches, adapters, checkpoints, and backups; verify residual extraction and record only the minimum audit proof allowed by policy (`paper:bourtoule-2021-machine-unlearning`, `paper:carlini-2021-training-data-extraction`, `paper:kandpal-2022-dedup-privacy`).

### Catastrophic-forgetting safeguards

- Keep owner-approved semantic anchors for rare commitments, safety rules, correction history, and identity statements, but store replay eligibility and purpose separately from ordinary retention (`paper:mi-2020-ader`, `paper:lopez-paz-2017-gem`, `paper:masson-dautume-2019-episodic-language-memory`).
- Measure stability-plasticity per component and domain; increase replay or parameter protection only where chronological regressions demonstrate need (`paper:mi-2020-ader`, `paper:xu-2020-graphsail`, `paper:kirkpatrick-2017-ewc`).
- Preserve a retrieval-first path for mutable facts and episodes; reserve versioned learned adapters for repeatedly validated, low-risk behavior patterns that can be rolled back and unlearned (`paper:persona-dialogue-xu-2022-goldfish`, `paper:persona-dialogue-xu-2022-lemon`, `paper:bourtoule-2021-machine-unlearning`).
- Run safety checks after retrieval because benign personal context can make harmful intent appear legitimate; a stateless pre-retrieval check is insufficient in the tested personalization setting (`paper:personalization-legitimizes-risks-2026`).
- Never use generated summaries, predictions, or prior assistant replies as independent autobiographical evidence. Constructive consolidation and synthetic-memory benchmarks show how derived text can amplify its own assumptions (`paper:cognitive-memory-hupbach-et-al-2007`, `paper:memorybank-2024`, `paper:longmemeval-2025`).

## Evaluation Tests

1. **Chronological prequential test:** update only on evidence available at time `t`, predict `t+1`, and retain complete model, index, policy, and prompt versions. Compare against static, recency-only, retrieval-only, and no-personalization baselines (`paper:koren-2009-temporal-dynamics`, `paper:wu-2017-recurrent-recommender`, `paper:synthetic-replacements-2024`).

2. **Temporal-scope ablation:** remove request-local, episodic, slow, and durable layers one at a time; measure future utility, owner recognition, unsupported promotion, and cross-layer leakage. A useful temporal layer must beat a simpler model in its target domain (`paper:rendle-2010-fpmc`, `paper:li-2020-tisasrec`, `paper:psychometrics_fleeson_2001`).

3. **Drift discrimination test:** construct matched cases for true owner-declared change, temporary mood, role change, new audience, option-set change, missing opportunity, and silence. Score change detection, false supersession, abstention, and time-to-owner-confirmation (`paper:tversky-simonson-1993-context-preferences`, `paper:kuppens-2010-dynaffect`, `paper:andersen-chen-2002-relational-self`, `paper:bell-1984-audience-design`).

4. **Stability-plasticity matrix:** report new-period utility, old-period retention, backward transfer, forward transfer, rare-anchor survival, and per-domain regressions rather than one aggregate score (`paper:lopez-paz-2017-gem`, `paper:mi-2020-ader`, `paper:mi-2020-man`).

5. **Supersession and point-in-time test:** ask the same question before and after a controlled update, require the correct historical answer at each cutoff, and test explicit abstention when validity is ambiguous (`paper:timechara-2024`, `paper:longmemeval-2025`).

6. **Owner-correction contract test:** apply every correction type, verify immediate behavioral effect, inspect all affected derived objects, roll it back, and confirm unrelated scopes remain unchanged. Report correction success separately from predictive loss (`paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`, `paper:persona-dialogue-wang-2024-emg-rag`).

7. **Active-elicitation test:** measure decision improvement per answered question, unanswerable rate, skip rate, subjective and elapsed burden, coverage concentration, privacy sensitivity, and regret from unasked multi-question combinations (`paper:rashid-2002-getting-to-know-you`, `paper:boutilier-2003-active-cf`, `paper:harpale-2008-personalized-active-learning`, `paper:rashid-2008-information-theoretic-elicitation`).

8. **Retrieval-interference test:** after strengthening one memory or consolidating one claim, test related alternatives, source attribution, and later correction. Reject an update that improves the target while silently suppressing an owner-important competitor (`paper:cognitive-memory-karpicke-roediger-2008`, `paper:cognitive-memory-anderson-bjork-bjork-1994`, `paper:cognitive-memory-hupbach-et-al-2007`).

9. **Decay and archive test:** compare no decay, calibrated down-ranking, reversible archival, and heuristic exponential decay across stable and drifting domains. Measure old-signal loss, stale retrieval, owner-reported forgetting mismatch, and restoration fidelity (`paper:koren-2009-temporal-dynamics`, `paper:cognitive-memory-murre-dros-2015`, `paper:memorybank-2024`).

10. **Replayability and rollback test:** reconstruct every state version from authorized events and policy logs, compare hashes and outputs, then restore the prior state after a failed update. A model-only state that cannot be reconstructed fails the audit requirement (`paper:kirkpatrick-2017-ewc`, `paper:bourtoule-2021-machine-unlearning`).

11. **Deletion propagation test:** delete one person's contribution and verify absence from retrieval, summaries, graphs, embeddings, replay sets, caches, adapters, checkpoints, and prompted outputs; run extraction and membership-style attacks on the residual system (`paper:bourtoule-2021-machine-unlearning`, `paper:carlini-2021-training-data-extraction`, `paper:kandpal-2022-dedup-privacy`).

12. **Calibration and abstention test:** maintain separate reliability curves for source facts, owner-confirmed claims, model inferences, drift proposals, and future predictions; penalize confident unsupported promotion more heavily than useful abstention (`paper:linguistic-calibration-2022`, `paper:longmemeval-2025`).

13. **Model-update drift test:** rerun a frozen longitudinal suite after base-model, retriever, embedding, prompt, or policy changes; compare means, variance, subgroup errors, relationship leakage, and correction behavior, not just average accuracy (`paper:synthetic-replacements-2024`, `paper:personality-temporal-stability-2024`).

14. **Post-retrieval safety test:** pair benign memories with harmful requests, vary semantic alignment and persona cues, and confirm that safety does not weaken as memory becomes more personally relevant (`paper:personalization-legitimizes-risks-2026`).

## WhatsApp Observability Gaps

No private WhatsApp data was accessed for this synthesis. Even with lawful, informed, and revocable access, a message archive is a partial record of sent communication, not a ground-truth longitudinal self (`paper:bardi-schwartz-2003-values-behavior`, `paper:andersen-chen-2002-relational-self`, `paper:locomo-2024`).

- A message timestamp records transmission, not when a belief became true, stopped being true, or was privately reconsidered; elapsed time also mixes availability, routine, urgency, and platform access (`paper:kumar-2019-jodie`, `paper:li-2020-tisasrec`).
- A sent message is an action under a frame, audience, and relationship, not a direct value, motive, or durable preference measurement (`paper:bardi-schwartz-2003-values-behavior`, `paper:tversky-kahneman-1981-framing`, `paper:andersen-chen-2002-relational-self`).
- Silence and non-mention cannot distinguish forgetting, irrelevance, privacy, lack of opportunity, changed channel, or an uncued memory (`paper:cognitive-memory-godden-baddeley-1975`, `paper:cognitive-memory-murre-dros-2015`).
- Direct chat or group membership does not reveal the imagined or downstream audience, and past disclosure does not authorize reuse for a new purpose (`paper:bell-1984-audience-design`, `paper:vitak-2012-context-collapse-privacy`, `paper:lampinen-et-al-2011-were-in-it-together`).
- Repeated phrases may be shared shorthand, quotation, irony, politeness, or accommodation; they do not by themselves establish a global style or relationship state (`paper:metzing-brennan-2003-conceptual-pacts-broken`, `paper:andersen-chen-2002-relational-self`).
- Text classifiers estimate how language may be read, not what the owner felt, and mixed or masked affect is not recoverable from one latest label (`paper:demszky-2020-goemotions`, `paper:kuppens-2010-dynaffect`).
- Offline options, outcomes, completed intentions, and unshared corrections are missing, so retrospective fit cannot substitute for held-out behavioral validation (`paper:tversky-simonson-1993-context-preferences`, `paper:generative-agents-1000-2024`).
- Edits, deletions, disappearing messages, calls, voice notes, media context, forwards, and partial exports can break source and validity chains; missing context must remain explicit uncertainty (`paper:cognitive-memory-hashtroudi-johnson-chrosniak-1989`, `paper:locomo-2024`).
- Messages are jointly produced and can contain third-party facts. Archive possession does not establish consent to retention, replay, inference, or simulation (`paper:franz-2022-interdependent-privacy`, `paper:mireshghallah-2024-confaide`).
- Sparse explicit corrections mean later behavior cannot be assumed to repair an earlier inference. The system needs low-burden confirmation, versioning, and an inspectable correction path (`paper:rashid-2008-information-theoretic-elicitation`, `paper:bakalov-2013-controllable-user-models`).

## Highest-Value Owner Questions

1. **Which facts, values, preferences, and commitments should never change without your explicit confirmation?** This defines durable anchors without treating statistical stability as owner authority (`paper:psychometrics_epstein_1983`, `paper:ahn-2007-open-user-profiles`).
2. **When you say something changed, do you mean it was previously wrong, became false at a date, applies only in one context, should be hidden, should not personalize, or must be deleted?** These intents require different supersession and propagation behavior (`paper:ahn-2007-open-user-profiles`, `paper:bourtoule-2021-machine-unlearning`).
3. **Which domains change quickly for you, and which should require repeated evidence over months?** Temporal usefulness and forgetting pressure vary by domain rather than following one global clock (`paper:li-2020-tisasrec`, `paper:mi-2020-ader`).
4. **Which people, roles, or groups make the same statement mean something different?** Relationship-conditioned behavior can coexist without globally changing the owner core (`paper:andersen-chen-2002-relational-self`, `paper:bell-1984-audience-design`).
5. **Which past examples may be retained as replay anchors, and which must never enter training or long-term memory?** Replay utility does not override consent, minimization, or deletion (`paper:masson-dautume-2019-episodic-language-memory`, `paper:franz-2022-interdependent-privacy`).
6. **When should the system ask before updating, and what question burden is acceptable?** Expected decision value and answerability matter, but autobiographical burden is not established by recommender studies (`paper:boutilier-2003-active-cf`, `paper:harpale-2008-personalized-active-learning`).
7. **Should old, low-use memories be down-ranked, archived, reconfirmed, or retained unchanged, and for which categories?** The evidence does not justify a universal forgetting curve (`paper:cognitive-memory-murre-dros-2015`, `paper:memorybank-2024`).
8. **What should happen when your explicit correction conflicts with repeated observed behavior?** Owner authority and predictive fit are distinct objectives and should not be silently averaged (`paper:ahn-2007-open-user-profiles`, `paper:bardi-schwartz-2003-values-behavior`).
9. **Which historical versions should remain available for point-in-time recall, and which should be permanently removed?** Supersession and deletion serve different purposes (`paper:timechara-2024`, `paper:bourtoule-2021-machine-unlearning`).
10. **What proof would make you trust that a correction, rollback, or deletion propagated everywhere?** Learned artifacts and duplicate exposure make source-row deletion insufficient (`paper:carlini-2021-training-data-extraction`, `paper:kandpal-2022-dedup-privacy`, `paper:bourtoule-2021-machine-unlearning`).

## Compact Claim Table

| Claim | Status | Supporting IDs | Contradicting or limiting IDs | Confidence rationale |
| --- | --- | --- | --- | --- |
| Longitudinal state needs more than one temporal component. | VALIDATED | `paper:koren-2009-temporal-dynamics`; `paper:rendle-2010-fpmc`; `paper:psychometrics_fleeson_2001` | `paper:li-2020-tisasrec` | High for separating scopes; exact scopes and durations are not validated. |
| Recency alone should not supersede or delete an older claim. | VALIDATED | `paper:koren-2009-temporal-dynamics`; `paper:tversky-simonson-1993-context-preferences` | `paper:kumar-2019-jodie`; `paper:li-2020-tisasrec` | High: time helps prediction, but old evidence can remain useful and context can explain change. |
| Fast episodic intake plus slow consolidation is a defensible learning pattern. | PLAUSIBLE | `paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`; `paper:cognitive-memory-schapiro-et-al-2017`; `paper:masson-dautume-2019-episodic-language-memory` | `paper:mi-2020-ader` | Medium: convergent computational evidence, but no end-to-end owner model and preservation can be unnecessary in stable streams. |
| Active elicitation should consider decision value, answerability, and burden. | VALIDATED | `paper:boutilier-2003-active-cf`; `paper:harpale-2008-personalized-active-learning`; `paper:rashid-2008-information-theoretic-elicitation` | `paper:rashid-2002-getting-to-know-you` | High in recommender settings; transfer to sensitive autobiographical questions remains uncertain. |
| Owner corrections should be typed, scoped, previewed, and reversible. | PLAUSIBLE | `paper:ahn-2007-open-user-profiles`; `paper:bostandjiev-2012-tasteweights`; `paper:persona-dialogue-wang-2024-emg-rag` | `paper:bakalov-2013-controllable-user-models` | Medium-low: owner demand is clear, but studies are small and objective effects conflict. |
| Supersession should preserve historical versions while changing current validity. | PLAUSIBLE | `paper:longmemeval-2025`; `paper:timechara-2024`; `paper:kober-habermas-2017-personal-past-stability` | `paper:bourtoule-2021-machine-unlearning` | Medium: temporal errors and narrative change support versioning, while deletion can require true removal. |
| A universal exponential decay policy is unsupported. | VALIDATED | `paper:cognitive-memory-murre-dros-2015`; `paper:koren-2009-temporal-dynamics`; `paper:memorybank-2024` | None in the merged corpus directly validates a universal digital-self curve. | High for rejection of universality; low for choosing a replacement policy. |
| Replay should be owner-approved, auditable, and revocable. | PLAUSIBLE | `paper:mi-2020-ader`; `paper:lopez-paz-2017-gem`; `paper:masson-dautume-2019-episodic-language-memory` | `paper:carlini-2021-training-data-extraction`; `paper:bourtoule-2021-machine-unlearning` | Medium: replay reduces forgetting, but privacy and deletion governance are outside the continual-learning studies. |
| Every state version should be reconstructable without treating weights as canonical memory. | PLAUSIBLE | `paper:longmemeval-2025`; `paper:persona-dialogue-wang-2024-emg-rag`; `paper:bourtoule-2021-machine-unlearning` | `paper:kirkpatrick-2017-ewc` | Medium: audit and correction needs support it, but no cited paper validates the complete reconstruction design. |
| Owner-approved semantic anchors can protect rare identity commitments with less privacy cost than raw replay. | SPECULATIVE | `paper:mi-2020-ader`; `paper:lopez-paz-2017-gem` | `paper:masson-dautume-2019-episodic-language-memory`; `paper:carlini-2021-training-data-extraction` | Low: neither the semantic-anchor representation nor its privacy advantage has been tested. |
| WhatsApp-only evidence can recover a stable, context-free owner model. | SPECULATIVE and unsupported | None | `paper:bardi-schwartz-2003-values-behavior`; `paper:andersen-chen-2002-relational-self`; `paper:demszky-2020-goemotions`; `paper:locomo-2024` | Very low: observable messages omit options, offline outcomes, audiences, motives, and internal state. |
| Catastrophic-forgetting safeguards must be adaptive and evaluated per component. | PLAUSIBLE | `paper:mi-2020-ader`; `paper:xu-2020-graphsail`; `paper:kirkpatrick-2017-ewc`; `paper:lopez-paz-2017-gem` | `paper:mi-2020-man` | Medium-high for adaptive testing; low for any single optimal safeguard. |
