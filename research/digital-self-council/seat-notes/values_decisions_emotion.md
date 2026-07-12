# Values, Decisions, Emotion, Goals, and Theory of Mind

## Scope and corpus

This seat covers human value and goal representation, decision styles and learned
policies, preference construction, appraisal and affect dynamics, text-based
affective computing, theory of mind, motive uncertainty, and the limits of
inferring internal states from interaction. It contains 20 counted, unique primary
papers: 19 journal articles and one conference paper. No record is marked
cross-reviewed. Full text was inspected for
`paper:trampe-2015-everyday-emotions`, `paper:demszky-2020-goemotions`, and
`paper:wu-2018-emotion-mental-states`; the other 17 records are explicitly
abstract-level inspections.

No WhatsApp content, private profile, local message database, pseudonym key,
evaluation row, or provider credential was accessed for this research.

## Search strategy

Searches combined exact titles, topic terms, and identifier checks on primary
research surfaces only: publisher article pages, DOI-resolved pages, PubMed,
PLOS, ACL Anthology, and official proceedings. Query families included:

- `basic individual values`, `value behavior`, `goal shielding`, and
  `experience sampling desire conflict self-control` for values and active goals;
- `framing`, `context-dependent preferences`, `coherent arbitrariness`,
  `free-choice paradigm`, `strategy selection learning`, and `decision-making
  style` for preferences and policies;
- `cognitive appraisal`, `fear anger risk`, `temporal dynamics of affect`,
  `emotions in everyday life`, `face in context`, and `fine-grained emotion
  text dataset` for affect;
- `limits on theory of mind`, `egocentric anchoring`, `Bayesian mentalizing`,
  and `beliefs desires emotional expressions` for motive inference.

Each JSONL record retains the concrete queries used to locate and recheck it.
Canonical DOI, ACL, and PubMed identifiers were compared against title, author,
year, and venue metadata on a permitted primary surface. A published article and
its preprint were treated as one work; only the published version was retained.

## Inclusion and exclusion

Included papers had to be within the seat boundary, identify a real journal or
conference work, expose enough primary metadata to verify the work, and provide
at least an abstract that supported a bounded paraphrase. Original formal models
were included when they made a testable contribution central to the seat, such as
context-dependent choice (`paper:tversky-simonson-1993-context-preferences`) or
Bayesian mentalizing (`paper:baker-2017-bayesian-mentalizing`).

Excluded material included books and chapters, reviews and meta-analyses, blogs,
product pages, GitHub documentation, clinical profiling work, secondary
explainers, records visible only through an unverifiable citation, and duplicate
preprint/published versions. Candidate findings were also dropped when a primary
page did not support the authors, identifier, method, or claimed result. This
means the seat favors honest abstract-level extraction over invented full-text
specificity.

## Domain taxonomy

| Domain | Records | What the evidence tests |
| --- | --- | --- |
| Values and goals | `paper:schwartz-2012-refined-values`, `paper:bardi-schwartz-2003-values-behavior`, `paper:shah-2002-goal-shielding`, `paper:hofmann-2012-everyday-temptations` | Value structure, uneven value-behavior links, active-goal competition, and desire/conflict/enactment in daily life |
| Preference construction and decision policy | `paper:tversky-kahneman-1981-framing`, `paper:tversky-simonson-1993-context-preferences`, `paper:ariely-2003-coherent-arbitrariness`, `paper:chen-risen-2010-free-choice`, `paper:rieskamp-otto-2006-strategy-selection`, `paper:scott-bruce-1995-decision-style` | Frames, option sets, anchors, measurement artifacts, learned strategy selection, and self-reported styles |
| Appraisal and affect | `paper:smith-ellsworth-1985-appraisal`, `paper:lerner-keltner-2001-fear-anger-risk`, `paper:kuppens-2010-dynaffect`, `paper:trampe-2015-everyday-emotions`, `paper:carroll-russell-1996-face-context`, `paper:demszky-2020-goemotions` | Appraisal dimensions, affect-dependent risk, temporal dynamics, mixed emotion, contextual perception, and text classification |
| Theory of mind and motive uncertainty | `paper:keysar-2003-tom-limits`, `paper:epley-2004-perspective-taking`, `paper:baker-2017-bayesian-mentalizing`, `paper:wu-2018-emotion-mental-states` | Online perspective failures, egocentric priors, inverse planning, and evidence-sensitive belief/desire inference |

## Evidence-backed synthesis

### Values are structured, but behavior is not a direct value sensor

The strongest structural evidence supports a motivational continuum with
compatible and conflicting neighboring values rather than an unordered bag of
labels (`paper:schwartz-2012-refined-values`). That is useful for representation,
but prediction must stay restrained: relations between declared values and
recurrent behavior differ sharply by value domain and can be obscured by social
norms (`paper:bardi-schwartz-2003-values-behavior`). A digital brain should
therefore distinguish an owner-declared value, a behavior that may instantiate a
value, and a contextual norm that may explain the same behavior.

Goals are also stateful. Activating a committed focal goal can suppress access to
competing goals without erasing them (`paper:shah-2002-goal-shielding`). Daily
desire evidence further separates wanting, conflict, resistance, and enactment;
person-level variables were relatively more associated with early components,
while situational and interpersonal variables mattered more later in the episode
(`paper:hofmann-2012-everyday-temptations`). Silence about a goal and failure to
enact a desire are therefore weak evidence that either has ceased to exist.

### Observed choice is jointly produced by person, task, and measurement

Equivalent outcome frames can reverse choice
(`paper:tversky-kahneman-1981-framing`), and adding an option can change the
relative attractiveness of existing options
(`paper:tversky-simonson-1993-context-preferences`). Arbitrary initial anchors
can shift valuation levels while later responses remain coherently ordered
(`paper:ariely-2003-coherent-arbitrariness`). Together, these papers reject the
idea that one isolated choice reveals a context-free utility.

Measurement itself can manufacture an apparent update. The standard free-choice
paradigm can produce post-choice spreading even when latent preferences are
unchanged (`paper:chen-risen-2010-free-choice`). Conversely, repeated feedback can
shift which inference strategy a person selects
(`paper:rieskamp-otto-2006-strategy-selection`). A five-style self-report measure
can describe rational, intuitive, dependent, avoidant, and spontaneous tendencies
(`paper:scott-bruce-1995-decision-style`), but those scores should be priors, not
substitutes for domain-specific choice episodes and outcomes.

### Affect needs appraisal, time, mixtures, and context

Appraisal evidence distinguishes emotions through certainty, control,
responsibility, effort, attention, and pleasantness
(`paper:smith-ellsworth-1985-appraisal`). This helps explain why fear and anger,
despite sharing negative valence, can be associated with opposite risk judgments
through different certainty and control appraisals
(`paper:lerner-keltner-2001-fear-anger-risk`). Affective state should therefore
modify a decision model transiently and through specific appraisals, not become a
permanent owner trait.

Longitudinal evidence supports a dynamic state with a personal home base,
variability, and attractor strength (`paper:kuppens-2010-dynaffect`). Large-scale
smartphone sampling also found frequent mixed positive and negative reports and
different co-occurrence patterns among emotions
(`paper:trampe-2015-everyday-emotions`). A single latest label is an inadequate
state representation.

Perceived expression is not equivalent to internal experience. Situational
descriptions overrode prototypical facial cues across the tested conflicts
(`paper:carroll-russell-1996-face-context`). In text, GoEmotions provides a useful
fine-grained label space, but its fine-label BERT baseline reached only .46 mean
F1, and the labels came from readers viewing decontextualized Reddit comments
(`paper:demszky-2020-goemotions`). A classifier can estimate how text is likely to
be read; it cannot certify what the writer felt.

### Mentalizing can be structured and still fail

Formal inverse-planning models can reproduce human judgments about beliefs,
desires, and percepts in constrained spatial tasks
(`paper:baker-2017-bayesian-mentalizing`). Emotional reactions can add evidence:
observers integrated wanted, expected, caused, and observed outcomes in a way
captured by a Bayesian model (`paper:wu-2018-emotion-mental-states`). The same
study also shows a boundary: belief recovery failed when action and target mental
states were improbable together or when reaction evidence was insufficient.

Human mentalizing is not a gold standard. Adults sometimes interpret instructions
through privileged knowledge despite explicitly knowing that the speaker lacks it
(`paper:keysar-2003-tom-limits`). Perspective estimates also tend to anchor on the
self, stop after reaching a plausible value, worsen under time pressure, and
improve with accuracy incentives (`paper:epley-2004-perspective-taking`). A
digital brain should use mental-state hypotheses to plan clarification, not store
the most convenient hypothesis as biographical truth.

## Negative and conflicting evidence

- **Value structure does not guarantee behavioral fidelity.** The refined value
  continuum is well supported (`paper:schwartz-2012-refined-values`), yet several
  value-behavior relations were only marginal and norms obscured others
  (`paper:bardi-schwartz-2003-values-behavior`).
- **Consistency does not guarantee stability.** Coherent valuations can inherit an
  arbitrary anchor (`paper:ariely-2003-coherent-arbitrariness`), and an apparent
  post-choice change can arise from measurement and selection
  (`paper:chen-risen-2010-free-choice`).
- **Emotion categories are useful but nonexclusive and observer-dependent.**
  Appraisal patterns differentiate states (`paper:smith-ellsworth-1985-appraisal`),
  while daily reports frequently mix valences
  (`paper:trampe-2015-everyday-emotions`) and context can dominate an expressive
  cue (`paper:carroll-russell-1996-face-context`).
- **Affect-policy effects are conditional.** Fear and anger diverged in the studied
  risk tasks (`paper:lerner-keltner-2001-fear-anger-risk`), but everyday enactment
  also depends heavily on situational and interpersonal conditions
  (`paper:hofmann-2012-everyday-temptations`). The former is not a license to infer
  a fixed risk policy from an emotion label.
- **Theory-of-mind competence and use dissociate.** Structured Bayesian accounts
  fit judgments (`paper:baker-2017-bayesian-mentalizing`,
  `paper:wu-2018-emotion-mental-states`), while adults still make egocentric online
  errors (`paper:keysar-2003-tom-limits`,
  `paper:epley-2004-perspective-taking`). Model coherence is not hidden-state
  verification.

## Implications for a digital brain

### Validated constraints

1. Keep value declarations, observed behavior, norms, and inferred motives as
   distinct evidence types (`paper:schwartz-2012-refined-values`,
   `paper:bardi-schwartz-2003-values-behavior`).
2. Persist each decision's frame, option set, reference point, elicitation order,
   and outcome before updating a preference
   (`paper:tversky-kahneman-1981-framing`,
   `paper:tversky-simonson-1993-context-preferences`,
   `paper:ariely-2003-coherent-arbitrariness`).
3. Represent affect as time-indexed, mixed, and appraisal-bearing, with direct
   owner report outranking classifier output (`paper:kuppens-2010-dynaffect`,
   `paper:trampe-2015-everyday-emotions`,
   `paper:demszky-2020-goemotions`).
4. Preserve a posterior over beliefs and motives and expose uncertainty whenever
   observability, options, or rationality assumptions are missing
   (`paper:baker-2017-bayesian-mentalizing`,
   `paper:wu-2018-emotion-mental-states`).

### Plausible design hypotheses

1. A versioned value graph can encode compatible and conflicting priorities, with
   owner confirmation as the authority and behavior as supporting or
   countervailing evidence (`paper:schwartz-2012-refined-values`).
2. An active-goal workspace can temporarily gate competing goals while retaining
   their history and facilitative links (`paper:shah-2002-goal-shielding`).
3. A policy repertoire can maintain domain-conditioned decision strategies and
   update their weights from owner feedback without globally rewriting style
   (`paper:rieskamp-otto-2006-strategy-selection`,
   `paper:scott-bruce-1995-decision-style`).
4. A mentalizing module can generate clarification questions from competing
   belief-desire hypotheses, while a separate evidence ledger prevents those
   hypotheses from becoming owner facts (`paper:keysar-2003-tom-limits`,
   `paper:epley-2004-perspective-taking`).

### Speculative claims that require evaluation

It is not established that WhatsApp interaction alone can recover a stable value
hierarchy, affect-dynamics parameters, or transferable decision policy. It is also
not established that a model which predicts an owner's next message has recovered
the owner's motive. Those claims require prospective owner-confirmed evaluation,
counterfactual reframing, and calibration against withheld real decisions, not
just retrospective fit (`paper:chen-risen-2010-free-choice`,
`paper:demszky-2020-goemotions`, `paper:baker-2017-bayesian-mentalizing`).

## WhatsApp observability gaps

A generic messaging surface may expose sent text, emoji, timing, reply structure,
and some conversational context. Even with consent, it usually does **not** expose:

- alternatives the owner considered but never mentioned, rejected options, or the
  original frame and anchor (`paper:tversky-kahneman-1981-framing`,
  `paper:ariely-2003-coherent-arbitrariness`);
- whether an expressed desire conflicted with another goal, was resisted, or was
  later enacted offline (`paper:hofmann-2012-everyday-temptations`);
- private norms, role obligations, resource constraints, coercion, habit, or
  audience effects that can separate behavior from values
  (`paper:bardi-schwartz-2003-values-behavior`);
- unexpressed affect, intensity, physiology, mixed feelings, emotion masking, or
  the appraisal that generated a phrase (`paper:smith-ellsworth-1985-appraisal`,
  `paper:trampe-2015-everyday-emotions`);
- what the owner believed another person knew, which outcomes the owner expected,
  or whether an apparently irrational action reflected missing options
  (`paper:keysar-2003-tom-limits`,
  `paper:wu-2018-emotion-mental-states`);
- ground truth that a reader's emotion interpretation matches the writer's
  experience (`paper:carroll-russell-1996-face-context`,
  `paper:demszky-2020-goemotions`).

These gaps should become adaptive owner-interview prompts: Which options were
available? What mattered in the tradeoff? Was the message a report, a joke, a
performance, or venting? What outcome was expected? Was the action freely chosen?
Which other goal was active? The answer should be stored as owner-provided
evidence with time and scope, not backfilled as if it had been observable from the
message.

## Open questions

1. What prospective protocol can distinguish a stable value from role conformity,
   habit, or a one-off goal while minimizing owner burden
   (`paper:bardi-schwartz-2003-values-behavior`)?
2. How should a value graph and active-goal workspace represent genuine changes
   without mistaking temporary goal shielding for abandonment
   (`paper:shah-2002-goal-shielding`)?
3. Which repeated, incentive-compatible choice tasks can estimate frame-robust
   preferences without creating the preference through elicitation
   (`paper:tversky-kahneman-1981-framing`,
   `paper:chen-risen-2010-free-choice`)?
4. Can affect dynamics learned from sparse owner self-report improve decisions
   beyond a simple uncertainty-aware baseline, and how quickly do those dynamics
   drift (`paper:kuppens-2010-dynaffect`)?
5. How should text-affect calibration vary by relationship, language, humor, and
   conversational history when public-comment benchmarks omit those variables
   (`paper:demszky-2020-goemotions`)?
6. What abstention threshold best prevents self-projection and rationality
   assumptions from becoming false motive claims
   (`paper:epley-2004-perspective-taking`,
   `paper:baker-2017-bayesian-mentalizing`)?
7. Which owner-facing evaluation can test whether clarification-driven
   mentalizing is useful without rewarding confident mind-reading
   (`paper:keysar-2003-tom-limits`,
   `paper:wu-2018-emotion-mental-states`)?
