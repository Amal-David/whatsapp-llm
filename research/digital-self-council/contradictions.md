# Contradiction And Uncertainty Register

This register keeps tensions visible instead of averaging them into one confident
persona. Paper IDs resolve against `corpus.jsonl`; claim statuses resolve against
`evidence-map.json`.

## C01: Prediction Versus Construct Validity

- **Positive evidence:** language, social traces, and smartphone behavior can
  predict Big Five or related labels (`paper:psychometrics_mairesse_2007`,
  `paper:psychometrics_park_2015`, `paper:psychometrics_stachl_2020`).
- **Boundary evidence:** measurement error, cross-cultural response behavior, and
  construct drift can dominate the inferred score (`paper:psychometrics_laajaj_2019`,
  `paper:psychometrics_mezquita_2019`).
- **Resolution:** retain predictions as source-scoped candidate claims; never
  convert predictive accuracy into a claim of complete personality recovery.
- **Remaining uncertainty:** owner-specific calibration across language,
  relationship, and time has not been established.

## C02: Trait Stability Versus Situational Variability

- **Stability evidence:** aggregation across observations can improve dispositional
  estimates (`paper:psychometrics_epstein_1983`).
- **Variability evidence:** within-person states vary systematically across
  situations (`paper:psychometrics_fleeson_2001`,
  `paper:psychometrics_sherman_2015`).
- **Resolution:** represent a distribution of states and context-conditioned
  tendencies, plus a separately estimated long-term prior.
- **Remaining uncertainty:** the amount of WhatsApp evidence needed for a stable
  owner prior is unknown.

## C03: Hierarchical Autobiographical Organization Versus Constructed Retrieval

- **Structure evidence:** owner-specific life periods and chapters cue retrieval
  (`paper:conway-bekerian-1987-organization`,
  `paper:chen-mcanally-reese-2013-memory-organization`).
- **Boundary evidence:** the same results can be explained partly by constructing
  a retrieval plan; they do not prove one fixed latent storage hierarchy.
- **Resolution:** expose owner-editable period and chapter links as retrieval paths,
  without claiming that the graph is the person's literal memory structure.
- **Remaining uncertainty:** graph retrieval must be compared with flat retrieval
  plus generated plans.

## C04: Narrative Coherence Versus Adaptation

- **Positive evidence:** temporal, causal, contextual, and thematic coherence are
  measurable narrative dimensions (`paper:reese-et-al-2011-narrative-coherence`,
  `paper:kober-schmiedek-habermas-2015-coherence-development`).
- **Boundary evidence:** coherence did not reliably increase with repeated
  narration or track improvement in all settings
  (`paper:habermas-de-silveira-2008-global-coherence`,
  `paper:adler-2012-living-into-story`).
- **Resolution:** annotate coherence dimensions; do not manufacture a polished
  story or use coherence as maturity, truth, or mental-health evidence.
- **Remaining uncertainty:** owner-valued coherence may differ from research coding.

## C05: Narrative Repetition Versus Factual Truth

- **Continuity evidence:** events and themes can recur across years
  (`paper:kober-habermas-2017-personal-past-stability`,
  `paper:jardmo-et-al-2023-repeated-narratives`).
- **Reconstruction evidence:** current self-compatibility and perspective alter
  recall (`paper:libby-eibach-2002-looking-back`).
- **Resolution:** repetition raises retrieval salience, not factual confidence;
  keep event claims and later interpretations separate.
- **Remaining uncertainty:** event matching across changed details requires owner
  correction and probabilistic identity links.

## C06: Human-Memory Inspiration Versus Cognitive Equivalence

- **Systems evidence:** memory functions and learning processes are separable
  (`paper:cognitive-memory-vargha-khadem-1997`,
  `paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`).
- **Boundary evidence:** formal architectures and neural simulations model selected
  tasks, not a complete person (`paper:cognitive-memory-anderson-et-al-2004`,
  `paper:cognitive-memory-eliasmith-et-al-2012`).
- **Resolution:** borrow testable separations and failure-aware policies; reject
  claims that the software is biologically or cognitively equivalent.
- **Remaining uncertainty:** which cognitive analogies improve digital-self fidelity
  must be established empirically.

## C07: Forgetting As Utility Versus Forgetting As Loss

- **Utility evidence:** retrieval practice, interference, and accessibility changes
  can improve selection (`paper:cognitive-memory-karpicke-roediger-2008`,
  `paper:cognitive-memory-anderson-bjork-bjork-1994`).
- **Loss evidence:** continual adaptation can catastrophically overwrite prior
  capabilities (`paper:kirkpatrick-2017-ewc`, `paper:lopez-paz-2017-gem`).
- **Resolution:** decay affects retrieval priority only; protected evidence and
  owner corrections remain auditable and replayable.
- **Remaining uncertainty:** optimal decay is domain-, risk-, and owner-dependent.

## C08: Persona Consistency Versus Owner Fidelity

- **Positive evidence:** persona conditioning and consistency checks can improve
  explicit profile adherence (`paper:persona-dialogue-zhang-2018`,
  `paper:persona-dialogue-welleck-2019`, `paper:incharacter-2024`).
- **Boundary evidence:** psychometric behavior can be temporally unstable and
  aggregate simulation can miss subgroup or individual structure
  (`paper:personality-temporal-stability-2024`,
  `paper:synthetic-replacements-2024`).
- **Resolution:** score consistency, owner prediction, temporal stability,
  relationship boundaries, and memorization separately.
- **Remaining uncertainty:** no accepted benchmark establishes person-level fidelity.

## C09: More Context Versus Reliable Long-Term Memory

- **Positive evidence:** external memory and long-context retrieval improve access
  to prior interactions (`paper:memgpt-2023`, `paper:memorybank-2024`).
- **Failure evidence:** long-horizon benchmarks still expose temporal, causal, and
  retrieval failures (`paper:locomo-2024`, `paper:longmemeval-2025`).
- **Resolution:** use typed retrieval, source traces, conflict checks, and explicit
  missing-evidence states rather than treating context length as memory.
- **Remaining uncertainty:** the best memory topology depends on the owner task.

## C10: Relationship Adaptation Versus One Global Persona

- **Adaptation evidence:** audience and partner history alter language and
  self-presentation (`paper:bell-1984-audience-design`,
  `paper:brennan-clark-1996-conceptual-pacts`,
  `paper:ireland-et-al-2011-language-style-matching`).
- **Boundary evidence:** collapsed audiences produce disclosure and privacy failures
  (`paper:marwick-boyd-2011-context-collapse`,
  `paper:vitak-2012-context-collapse-privacy`).
- **Resolution:** allow scoped selves and style policies to coexist; conflict only
  when claims share the same scope and time.
- **Remaining uncertainty:** sparse relationships require stronger abstention.

## C11: Personalization Versus Auditability

- **Adaptation evidence:** temporal and interactive user models benefit from new
  evidence (`paper:koren-2009-temporal-dynamics`, `paper:kumar-2019-jodie`).
- **Control evidence:** editable and inspectable user profiles improve user agency
  (`paper:ahn-2007-open-user-profiles`, `paper:bakalov-2013-controllable-user-models`).
- **Resolution:** updates create versions, diffs, and supersession links; no silent
  profile mutation.
- **Remaining uncertainty:** the owner burden of reviewing frequent updates must be
  measured.

## C12: Value Structure Versus Value-Behavior Gaps

- **Structure evidence:** individual values form recurring motivational relations
  (`paper:schwartz-2012-refined-values`).
- **Boundary evidence:** value-behavior links differ by value and are obscured by
  norms and context (`paper:bardi-schwartz-2003-values-behavior`).
- **Resolution:** distinguish owner declaration, observed behavior, role norm, and
  inferred motive.
- **Remaining uncertainty:** chat alone cannot identify which factor caused an act.

## C13: Coherent Preference Versus Context-Constructed Choice

- **Consistency evidence:** choices can remain orderly after an initial anchor
  (`paper:ariely-2003-coherent-arbitrariness`).
- **Context evidence:** framing, option sets, and elicitation procedure change choice
  (`paper:tversky-kahneman-1981-framing`,
  `paper:tversky-simonson-1993-context-preferences`,
  `paper:chen-risen-2010-free-choice`).
- **Resolution:** learn domain- and frame-conditioned decision policies, not one
  global utility scalar.
- **Remaining uncertainty:** prospective counterfactual evaluation is required.

## C14: Emotion Labels Versus Felt Affect

- **Structure evidence:** appraisal and temporal dynamics provide useful affect
  representations (`paper:smith-ellsworth-1985-appraisal`,
  `paper:kuppens-2010-dynaffect`).
- **Boundary evidence:** emotions can be mixed, context changes interpretation, and
  text labels reflect readers rather than privileged access to the writer
  (`paper:trampe-2015-everyday-emotions`,
  `paper:carroll-russell-1996-face-context`,
  `paper:demszky-2020-goemotions`).
- **Resolution:** classifier output remains an affect hypothesis below direct owner
  report.
- **Remaining uncertainty:** calibration varies by relationship, language, and humor.

## C15: Formal Mentalizing Versus Perspective Failure

- **Positive evidence:** inverse planning can model bounded belief and desire
  judgments (`paper:baker-2017-bayesian-mentalizing`,
  `paper:wu-2018-emotion-mental-states`).
- **Boundary evidence:** people still anchor on their own perspective and misuse
  privileged knowledge (`paper:keysar-2003-tom-limits`,
  `paper:epley-2004-perspective-taking`).
- **Resolution:** store competing hypotheses and use them for clarification, never
  as biographical truth.
- **Remaining uncertainty:** rationality assumptions are rarely observable in chat.

## C16: Aggregate Simulation Versus Individual Prediction

- **Positive evidence:** language models can reproduce some aggregate experimental
  and population patterns (`paper:turing-experiments-2023`,
  `paper:out-of-one-many-2023`).
- **Boundary evidence:** subgroup effects, covariance, preference behavior, and
  rationality can diverge (`paper:synthetic-replacements-2024`,
  `paper:llm-user-sim-crs-2024`, `paper:llms-more-rational-2025`).
- **Resolution:** require owner-level held-out and prospective tests; population fit
  is background evidence only.
- **Remaining uncertainty:** owner judgment itself has test-retest variability.

## C17: Confidence Language Versus Correctness

- **Positive evidence:** linguistic calibration can be measured and improved
  (`paper:linguistic-calibration-2022`).
- **Boundary evidence:** confident style, persona consistency, and evaluator
  preference do not establish evidence coverage.
- **Resolution:** derive confidence from support, source quality, conflict, scope,
  and temporal fit; display abstention reasons.
- **Remaining uncertainty:** confidence thresholds must vary by stakes.

## C18: Personalization Benefit Versus Privacy Loss

- **Benefit evidence:** personal histories can improve retrieval and adaptation
  (`paper:mylifebits-2006`, `paper:memgpt-2023`).
- **Risk evidence:** models leak memorized text, PII, membership, and inferred
  attributes (`paper:carlini-2021-training-data-extraction`,
  `paper:lukas-2023-pii-leakage`, `paper:shokri-2017-membership-inference`,
  `paper:kosinski-2013-private-traits`).
- **Resolution:** local-first storage, source-level consent, minimization, extraction
  tests, and revocation are architectural requirements.
- **Remaining uncertainty:** no single privacy technique covers the full lifecycle.

## C19: Durable Provenance Versus Deletion Rights

- **Audit need:** reproducible state changes require durable evidence identifiers and
  update history.
- **Deletion need:** consent provenance and machine unlearning require removal and
  revocation paths (`paper:longpre-2024-consent-crisis`,
  `paper:bourtoule-2021-machine-unlearning`).
- **Resolution:** preserve tombstones, hashes, and non-content audit events while
  deleting content and excluding it from retrieval, consolidation, and exports.
- **Remaining uncertainty:** legal deletion obligations are jurisdiction-specific
  and outside this research claim.

## C20: Digital Continuity Versus Autonomous Representation

- **Continuity evidence:** lifelogs and personal-agent systems can preserve and
  retrieve aspects of prior activity (`paper:mylifebits-2006`,
  `paper:sensecam-lifelog-memory-2007`).
- **Boundary evidence:** posthumous stewardship, consent, and griefbot perception
  remain unresolved (`paper:brubaker-2016-legacy-contact`,
  `paper:lu-2026-griefbot-perceptions`).
- **Resolution:** the proof of concept is an owner-controlled simulator, never an
  autonomous or posthumous representative.
- **Remaining uncertainty:** future deployment requires a separate governance and
  identity-rights program.
