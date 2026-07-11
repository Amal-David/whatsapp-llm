# Evaluation Fidelity Seat Notes

## Scope and corpus status

This seat covers evaluation of persona dialogue, role-playing agents, and user simulators, with emphasis on behavioral fidelity, individual predictive validity, owner and human judgments, psychometric reliability, calibration, temporal and relationship tests, memorization, and privacy leakage. The corpus contains 20 counted primary papers: 19 inspected at full-text depth and one at abstract depth. No record is marked reviewed or cross-reviewed.

The research used only public primary paper sources. It did not inspect WhatsApp messages, private profiles, local databases, credentials, or any other owner data.

## Search strategy

Searches were run in several deliberately separate families so that conversational resemblance would not dominate the evidence base:

- Persona and roleplay evaluation: ACL Anthology searches for Persona-Chat, Dialogue NLI, InCharacter, CharacterEval, and TimeChara, plus official ICLR proceedings for SOTOPIA (`paper:persona-chat-2018`, `paper:dialogue-nli-2019`, `paper:incharacter-2024`, `paper:charactereval-2024`, `paper:timechara-2024`, `paper:sotopia-2024`).
- User-simulator validity: ACL and SIGDIAL searches combining *user simulator*, *human evaluation*, *real users*, *preference agreement*, and *downstream policy* (`paper:llm-user-sim-crs-2024`, `paper:neural-user-simulation-2018`).
- Human-behavior simulation: ICML, ICLR, Cambridge University Press, Nature, Royal Society, and arXiv searches combining *human simulation*, *predictive validity*, *test-retest*, *variance*, *personality*, and *decision-making* (`paper:turing-experiments-2023`, `paper:out-of-one-many-2023`, `paper:synthetic-replacements-2024`, `paper:self-report-agents-2024`, `paper:llms-more-rational-2025`, `paper:emulate-personality-2025`, `paper:personality-temporal-stability-2024`).
- Confidence and uncertainty: ACL/TACL searches for conversational linguistic calibration (`paper:linguistic-calibration-2022`).
- Memorization and privacy leakage: USENIX Security, IEEE Security and Privacy, ICLR/OpenReview, and arXiv searches for training-data extraction, memorization scaling, PII extraction, reconstruction, inference, and attacks against aligned production models (`paper:extracting-training-data-2021`, `paper:quantifying-memorization-2023`, `paper:pii-leakage-2023`, `paper:aligned-production-extraction-2025`).

Exact-title and identifier searches were used to verify metadata. DOI, ACL, arXiv, PubMed, and title-year identifiers were checked for duplicate versions. When a conference or journal version was available, its record was retained and a duplicate preprint was omitted. The Park et al. record uses the current arXiv v3 title while retaining the original 2024 arXiv year and explicitly noting the 2026 revision (`paper:self-report-agents-2024`).

## Inclusion and exclusion

Included papers had to contribute a concrete evaluation construct, empirical benchmark, validation design, or attack relevant to person-like agents. A paper also needed a verifiable primary HTTPS page and enough inspected evidence to paraphrase its method, result, and limitations without inference from secondary coverage.

Included evidence spans:

- Turn-level persona consistency and recognizability (`paper:persona-chat-2018`, `paper:dialogue-nli-2019`).
- Multidimensional character and social-role judgment (`paper:incharacter-2024`, `paper:charactereval-2024`, `paper:sotopia-2024`).
- Temporal knowledge boundaries (`paper:timechara-2024`).
- User-simulator transfer to preferences, feedback, policies, and real users (`paper:llm-user-sim-crs-2024`, `paper:neural-user-simulation-2018`).
- Aggregate and individual human-behavior prediction (`paper:turing-experiments-2023`, `paper:out-of-one-many-2023`, `paper:synthetic-replacements-2024`, `paper:self-report-agents-2024`, `paper:llms-more-rational-2025`).
- Psychometric structure and temporal reliability (`paper:emulate-personality-2025`, `paper:personality-temporal-stability-2024`).
- Confidence calibration and privacy red-teaming (`paper:linguistic-calibration-2022`, `paper:extracting-training-data-2021`, `paper:quantifying-memorization-2023`, `paper:pii-leakage-2023`, `paper:aligned-production-extraction-2025`).

Excluded material included blogs, product pages, GitHub documentation, benchmark leaderboards without a primary paper, papers whose identifiers or claims could not be verified, and duplicate preprint/published versions. Papers that merely introduced a role-playing model without a fidelity-validation contribution were also excluded. No automatic similarity metric was accepted as stand-alone proof of person replication.

## Domain taxonomy

| Evaluation domain | What must be measured | Primary records |
|---|---|---|
| Persona dialogue | Profile entailment, contradiction, recognizability, engagement, and paraphrase robustness | `paper:persona-chat-2018`, `paper:dialogue-nli-2019` |
| Character roleplay | Knowledge, personality, behavior, style, and assessor-to-human agreement | `paper:incharacter-2024`, `paper:charactereval-2024` |
| Social and relationship behavior | Goal completion, rapport, norms, strategic communication, and relationship conditioning | `paper:sotopia-2024` |
| Temporal fidelity | Information available at the target time, model-version drift, and repeated-session stability | `paper:timechara-2024`, `paper:synthetic-replacements-2024`, `paper:personality-temporal-stability-2024` |
| User-simulator validity | Preference agreement, coherent feedback, simulator exploitation, and transfer to real users | `paper:llm-user-sim-crs-2024`, `paper:neural-user-simulation-2018` |
| Behavioral prediction | Individual agreement, aggregate distributions, variance, covariance, subgroup error, and repeated-choice noise | `paper:turing-experiments-2023`, `paper:out-of-one-many-2023`, `paper:synthetic-replacements-2024`, `paper:self-report-agents-2024`, `paper:llms-more-rational-2025` |
| Psychometric validity | Convergence, discrimination, reliability, factor structure, criterion validity, and profile complexity | `paper:incharacter-2024`, `paper:emulate-personality-2025`, `paper:personality-temporal-stability-2024` |
| Calibration | Correspondence between expressed confidence and empirical correctness | `paper:linguistic-calibration-2022` |
| Memorization and privacy | Verbatim extraction, duplication/context effects, PII extraction/reconstruction/inference, and alignment bypass | `paper:extracting-training-data-2021`, `paper:quantifying-memorization-2023`, `paper:pii-leakage-2023`, `paper:aligned-production-extraction-2025` |

## Evidence-backed synthesis

### Fidelity is a vector, not a scalar

The papers repeatedly separate constructs that a single similarity score would blur. Persona-Chat distinguishes engagingness, consistency, fluency, and profile recognizability, while Dialogue NLI isolates entailment and contradiction (`paper:persona-chat-2018`, `paper:dialogue-nli-2019`). CharacterEval expands the target to knowledge, personality, behavior, and linguistic style; SOTOPIA shows that even an evaluator that tracks goal completion can agree less well with humans on social dimensions (`paper:charactereval-2024`, `paper:sotopia-2024`). A digital-brain report should therefore expose a profile of scores and failure examples, never a single “is like the owner” percentage.

### Predictive validity needs held-out real behavior

User-simulation work provides the clearest validation hierarchy. A simulator can produce plausible dialogue yet disagree with human preferences, generate generic requests, or contradict its own feedback (`paper:llm-user-sim-crs-2024`). A stronger test asks whether a policy trained against the simulator succeeds with real users, but the policy can still exploit simulator-specific behavior as optimization continues (`paper:neural-user-simulation-2018`). For a digital brain, the strongest available target is therefore a preregistered prediction of an owner's later, held-out response or action, compared with simple baselines and the owner's own retest reliability.

Interview-grounded agents provide direct evidence for that design: they improved held-out survey prediction over demographics-only agents when scores were normalized by participants' two-week test-retest agreement (`paper:self-report-agents-2024`). The same paper also supplies a boundary condition: the advantage did not clearly transfer to economic games, and the experiment-level test lacked power. Behavioral domains must remain separate rather than being averaged into a general fidelity claim.

### Human judgments are necessary but construct-bound

Human raters are indispensable for style, recognizability, relationship appropriateness, and whether an answer feels characteristic. They are not a substitute for observed outcomes. Persona-Chat reported substantial judge variance (`paper:persona-chat-2018`). InCharacter and CharacterEval improve evaluation structure, but both still depend on static fictional-character labels and evaluator pipelines (`paper:incharacter-2024`, `paper:charactereval-2024`). SOTOPIA further shows that automated-judge agreement varies by dimension (`paper:sotopia-2024`). Owner judgments should be reported alongside, not merged with, behavioral prediction and contradiction rates.

### Matching means can hide a nonhuman distribution

Several papers reproduce average human patterns while failing at variance or individual behavior. The Turing Experiments replicated multiple classic effects, yet the wisdom-of-crowds condition produced unusually accurate, near-identical estimates instead of the error diversity that makes the human phenomenon work (`paper:turing-experiments-2023`). Synthetic political respondents could resemble human means while compressing variance and changing regression and subgroup relationships (`paper:synthetic-replacements-2024`). Risky-choice models predicted people as more expected-value-rational than they actually were, particularly under chain-of-thought prompting (`paper:llms-more-rational-2025`). Evaluation must include dispersion, covariance, repeated-choice noise, subgroup calibration, and excess-rationality checks.

### Psychometric reliability is not enough

InCharacter demonstrates that open-ended interviewing can be more faithful to intended character labels than direct self-report and can be repeatable across runs (`paper:incharacter-2024`). Yet high consistency can itself be artificial: GPT-4 personality emulations produced very high factor loadings and weakened human-like intertrait dependencies (`paper:emulate-personality-2025`). Repeated model testing also found stability to be model-, trait-, and instrument-dependent, with socially desirable response profiles (`paper:personality-temporal-stability-2024`). A defensible psychometric suite therefore needs test-retest reliability, convergent and discriminant validity, human-relative factor structure, criterion prediction, and robustness to richer context.

### Time and relationships require explicit test axes

TimeChara shows that a globally true fact can still be a fidelity error when it was unavailable to the represented person at the requested time (`paper:timechara-2024`). Bisbee et al. show another time problem: identical survey prompts changed after a deployed model update (`paper:synthetic-replacements-2024`). Relationship fidelity adds a separate interaction problem: short roleplay can measure rapport and social goals, but it does not establish persistence across months or asymmetric histories between two people (`paper:sotopia-2024`). Tests must bind every claim to an evidence timestamp, relationship, and model version.

### Confidence must track evidence quality

Conversational models can sound certain regardless of correctness. Linguistic calibration improved when generation was conditioned on a learned correctness signal, without requiring a reduction in answer accuracy in the studied factual setting (`paper:linguistic-calibration-2022`). For a digital brain, confidence should be tested separately for directly observed messages, owner-confirmed facts, model inferences, and future-behavior predictions. Calibration in factual question answering is evidence for the method, not evidence that autobiographical or preference confidence is already solved.

### Memorization can masquerade as fidelity

Exact reproduction of a private message may look personally authentic while actually demonstrating unsafe memorization. GPT-2 extraction recovered rare verbatim sequences and personal identifiers (`paper:extracting-training-data-2021`). Memorization grows with duplicate exposure, model size within a family, and longer prompt context (`paper:quantifying-memorization-2023`). PII-focused attacks show that scrubbing and differential privacy reduce but do not eliminate extraction, reconstruction, or inference, with important privacy-utility trade-offs (`paper:pii-leakage-2023`). Production alignment is also not a privacy certificate: targeted attacks bypassed ordinary chat behavior and extracted thousands of training examples from the tested deployment (`paper:aligned-production-extraction-2025`). Privacy red-teaming belongs inside fidelity evaluation because memorized specificity is a false positive for personhood.

## Negative and conflicting evidence

- **Aggregate optimism versus distributional failure.** Argyle et al. report encouraging aggregate and subgroup correspondence for backstory-conditioned political samples (`paper:out-of-one-many-2023`). Bisbee et al. find that similar synthetic-survey designs can compress variance, distort regression and subgroup patterns, and drift after model updates (`paper:synthetic-replacements-2024`). The conflict is resolved by requiring individual, distributional, covariance, and temporal tests rather than accepting aggregate means.
- **Replication versus contamination and variance collapse.** Classic effects were reproduced in several Turing Experiments, but known paradigms may have appeared in training and one experiment showed nonhuman hyper-accuracy (`paper:turing-experiments-2023`). Novel held-out variants and variance checks are mandatory.
- **Perceived plausibility versus actual choice.** LLM predictions can appear reasonable or model people's judgments while overpredicting economically rational behavior (`paper:llms-more-rational-2025`). Human evaluators saying “this sounds like a person” cannot establish predictive validity.
- **Trait alignment versus human psychometric structure.** Interview and questionnaire methods can produce strong target-trait agreement (`paper:incharacter-2024`, `paper:emulate-personality-2025`), while factor purity, social desirability, and weak temporal stability reveal nonhuman response processes (`paper:emulate-personality-2025`, `paper:personality-temporal-stability-2024`).
- **Synthetic transfer versus simulator exploitation.** Neural simulation improved a real-user dialogue-policy result, but longer optimization exposed simulator-specific overfitting (`paper:neural-user-simulation-2018`). A digital brain used as an environment needs continuing real-human validation.
- **Defense gains versus residual leakage.** Differential privacy and scrubbing can materially reduce measured PII leakage (`paper:pii-leakage-2023`), but exact and contextual leakage remain, and alignment can hide rather than remove extractability (`paper:aligned-production-extraction-2025`). Report residual risk rather than a binary safe label.

## Implications for a digital brain

The evidence supports a staged evaluation harness with explicit baselines and veto conditions:

1. **Freeze the evaluation unit.** Record owner, relationship, evidence cutoff, model version, prompt, retrieval snapshot, and decoding settings for every run (`paper:timechara-2024`, `paper:synthetic-replacements-2024`).
2. **Split construction from evaluation.** Hold out complete conversations, later dates, instruments, and behavioral outcomes. Compare the full system with demographics-only, recency-only, retrieval-only, and generic-LLM baselines (`paper:self-report-agents-2024`).
3. **Run claim-level checks.** Score supported, unsupported, contradicted, and future-leaking claims; test paraphrases so quotation is not rewarded (`paper:dialogue-nli-2019`, `paper:timechara-2024`).
4. **Run owner and human judgments.** Ask the owner to rate recognizability, style, values, and relationship appropriateness with an explicit “insufficient evidence” option. Use non-owner judges only for constructs they can observe and only with consent (`paper:persona-chat-2018`, `paper:charactereval-2024`).
5. **Run predictive tests.** Pre-register later questions or choices, score probability as well as the top prediction, normalize against owner test-retest reliability, and preserve domain-specific results (`paper:self-report-agents-2024`, `paper:llms-more-rational-2025`).
6. **Run distributional and psychometric tests.** Compare variance, covariance, subgroup error, repeated-choice consistency, factor structure, criterion validity, and temporal reliability (`paper:synthetic-replacements-2024`, `paper:emulate-personality-2025`, `paper:personality-temporal-stability-2024`).
7. **Run relationship scenarios.** Test the same intent across different counterparties and histories, then repeat after time has passed. Separate social-goal success from owner-specific behavior (`paper:sotopia-2024`).
8. **Run calibration tests.** Produce reliability curves for factual claims, remembered events, inferred preferences, and predicted actions; penalize confident unsupported claims (`paper:linguistic-calibration-2022`).
9. **Run privacy attacks as hard gates.** Test unprompted extraction, long-context seeding, repeated-text canaries, partial-context reconstruction, candidate inference, and alignment bypass. A severe private-text leak should veto release regardless of behavioral scores (`paper:extracting-training-data-2021`, `paper:quantifying-memorization-2023`, `paper:pii-leakage-2023`, `paper:aligned-production-extraction-2025`).

The report should retain a score vector and examples rather than compute one overall fidelity number. At minimum it should show owner judgment, contradiction rate, temporal leakage, held-out prediction, test-retest-normalized prediction, distributional error, psychometric validity, calibration, relationship consistency, and privacy-attack results.

## WhatsApp observability gaps

WhatsApp-like message history can directly show only a selected, sent communication event and its available metadata. It does not, by itself, observe:

- Messages considered but never sent, edits before sending, private uncertainty, or alternative actions.
- Whether a message expresses a stable belief, politeness strategy, joke, conflict avoidance, copied text, or a one-off mood.
- Offline decisions and behavior needed for predictive validation.
- Relationship context carried through calls, meetings, other apps, shared history, or third parties.
- Whether silence means disinterest, lack of time, technical interruption, emotional regulation, or an unseen event.
- The owner's internal confidence, later regret, changed belief, or correction unless explicitly recorded.
- A complete denominator for habits: observed messages reveal what happened in the sampled channel, not all opportunities in which the owner could have acted.

These are not missing values that a model may silently fill. They are unobserved constructs. The evaluation harness should label them unknown, invite owner confirmation where appropriate, and test uncertainty calibration. Message-level imitation should be reported as *channel-conditioned communication fidelity*, distinct from personality, relationship intent, or behavioral prediction. Time splits must also prevent future messages from leaking into past-person evaluations (`paper:timechara-2024`), while duplicated phrases and long context need special privacy tests (`paper:quantifying-memorization-2023`).

## Open questions

1. What owner test-retest protocol is acceptable for low-frequency or consequential decisions where repeated questioning changes the answer?
2. Which owner judgments can be collected without turning the target into a self-presentation exercise, and when should a close contact's judgment be considered with explicit consent?
3. How should fidelity be scored when an owner has genuinely changed, holds contradictory beliefs, or behaves differently by relationship?
4. What probability-calibration target is appropriate when the ground truth is an owner's later self-report rather than an externally verifiable fact?
5. How can relationship tests preserve the counterpart's privacy and agency without training on or reproducing their private messages?
6. What level of distributional error or temporal drift should block deployment even when owner recognizability is high?
7. Can retrieval-based systems preserve specificity while reducing memorization risk relative to training or fine-tuning on personal text, and how should that comparison be attacked empirically?
8. How should severe low-frequency privacy leaks be combined with average-case metrics? The evidence here favors a hard-gate model, but thresholds require an explicit threat model and owner consent policy.

## Bottom line

The strongest evidence does not support a single test for a faithful digital self. It supports a layered claim: a system may resemble an owner in dialogue, remain consistent with known evidence, predict some held-out responses, preserve human-like psychometric structure, behave appropriately across relationships and time, express calibrated uncertainty, and resist private-data extraction. Each clause needs its own test, baseline, uncertainty, and failure examples. Success on one clause must not be used to infer the others.
