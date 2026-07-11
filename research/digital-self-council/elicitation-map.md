# WhatsApp Evidence Gaps And Adaptive Owner Elicitation

Messaging history is selected behavior for particular audiences. It is useful
evidence, but it is not a complete autobiography, value inventory, decision log,
emotion diary, or grant of authority. The system should expose those gaps and ask
the owner only the questions that materially reduce uncertainty.

## What A Consented Message Source Can Support

Depending on the source adapter, the system may observe owner-authored text,
timestamps, chat and channel, reply structure, message type, attachments, and
repeated interaction patterns. Those signals can support bounded claims about:

- communication behavior in the sampled relationship and period
- explicit self-statements, plans, preferences, memories, and values
- events and people mentioned by the owner
- relationship-specific vocabulary, accommodation, and disclosure behavior
- temporal recurrence, change, and contradiction within the archive

They do not directly reveal:

- untold events, silent relationships, or the owner's full life-period structure
  (`paper:mclean-2005-meaning-telling`,
  `paper:thomsen-berntsen-2008-life-script-chapters`)
- whether a remembered event is factually exact, currently endorsed, or
  self-defining (`paper:libby-eibach-2002-looking-back`,
  `paper:dargembeau-garcia-jimenez-2024-working-self`)
- unchosen options, frames, anchors, incentives, or offline outcomes
  (`paper:tversky-kahneman-1981-framing`,
  `paper:tversky-simonson-1993-context-preferences`,
  `paper:chen-risen-2010-free-choice`)
- whether behavior expressed a value, norm, habit, coercion, role, or temporary
  goal (`paper:bardi-schwartz-2003-values-behavior`,
  `paper:hofmann-2012-everyday-temptations`)
- felt emotion, appraisal, intensity, masking, or mixed affect
  (`paper:trampe-2015-everyday-emotions`,
  `paper:carroll-russell-1996-face-context`,
  `paper:demszky-2020-goemotions`)
- another person's traits, consent, private history, or preferred disclosure scope
  (`paper:franz-2022-interdependent-privacy`,
  `paper:marwick-boyd-2011-context-collapse`)
- a stable global trait from one relationship or period
  (`paper:psychometrics_fleeson_2001`,
  `paper:psychometrics_sherman_2015`)
- the authority to act, send, promise, transact, or represent the owner

## Priority Model

Question selection is a **plausible design hypothesis**, informed by active
elicitation research rather than validated for digital replicas
(`paper:rashid-2008-information-theoretic-elicitation`,
`paper:harpale-2008-personalized-active-learning`,
`paper:bostandjiev-2012-tasteweights`). Each candidate question receives ordinal
scores:

| Factor | Range | Meaning |
| --- | --- | --- |
| Coverage gap | 0-3 | No evidence, weak evidence, conflicting evidence, or adequate evidence |
| Decision impact | 0-3 | How much this answer changes high-value simulation or safety behavior |
| Contradiction | 0-2 | Whether current evidence materially disagrees |
| Staleness | 0-2 | Whether the best evidence may no longer describe the owner |
| Owner request | 0-3 | Whether the owner explicitly wants this dimension represented |
| Sensitivity | 0-3 penalty | Potential harm, protected status, third-party exposure, or emotional burden |
| Burden | 1-3 penalty | Time, cognitive effort, and repetition required |

The queue sorts by expected benefit after penalties, but four rules override the
score:

1. Consent, authority, retention, and disclosure boundaries are asked before
   model-enrichment questions.
2. Protected-trait or clinical inference is never elicited merely to improve
   simulation; the owner may provide relevant information for a specific purpose.
3. A high-sensitivity question is opt-in, explains why it was proposed, supports
   skip and delete, and cannot block ordinary use.
4. Repeated prompts are suppressed unless evidence changed, the owner requested a
   refresh, or a prior answer expired.

## Priority Tiers

### P0: Required Before Simulation

| Gap | Owner-facing question | Stored result | Why first |
| --- | --- | --- | --- |
| Source consent | Which sources and date ranges may be used, and may third-party text be retained or only used transiently? | Source-level consent, retention, and revocation policy | Personalization increases leakage and interdependent privacy risk (`paper:carlini-2021-training-data-extraction`, `paper:franz-2022-interdependent-privacy`) |
| Representation boundary | Is this system only for private owner-controlled simulation, and which actions must it never take? | Authority policy and prohibited actions | Persona performance is not authority or identity fidelity (`paper:incharacter-2024`, `paper:synthetic-replacements-2024`) |
| Disclosure boundary | Which relationships, topics, and memories must never cross audience boundaries? | Relationship/audience disclosure rules | Context collapse creates disclosure failures (`paper:marwick-boyd-2011-context-collapse`, `paper:vitak-2012-context-collapse-privacy`) |
| Sensitive inference | Should behavioral inference of protected or sensitive attributes be disabled entirely? | Default deny policy; explicit exceptions only | Traces can expose sensitive attributes (`paper:kosinski-2013-private-traits`, `paper:gong-2016-attribute-inference`) |

### P1: High-Impact Identity And Decision Gaps

| Dimension | Trigger | Example question | Representation after answer | Evidence basis |
| --- | --- | --- | --- | --- |
| Current goals | A repeated plan is stale, conflicted, or lacks outcome evidence | “Is this still an active goal? What would count as progress, pause, or abandonment?” | Durable goal plus active status, time, conflicts, and owner authority | `paper:shah-2002-goal-shielding`, `paper:hofmann-2012-everyday-temptations` |
| Values | Behavior and explicit statements diverge | “What mattered in this tradeoff, and was your choice driven by a value, obligation, habit, or constraint?” | Owner-declared value evidence; behavior remains separate | `paper:schwartz-2012-refined-values`, `paper:bardi-schwartz-2003-values-behavior` |
| Decision policy | A high-impact choice lacks frame or alternatives | “Which options did you consider, what was the reference point, and what outcome followed?” | Decision episode with options, frame, rationale, and outcome | `paper:tversky-kahneman-1981-framing`, `paper:tversky-simonson-1993-context-preferences` |
| Relationship boundary | The system would use information learned in one chat in another relationship | “May this fact influence replies to this audience, or is it limited to where it was shared?” | Narrow relationship/audience scope | `paper:bell-1984-audience-design`, `paper:lampinen-et-al-2011-were-in-it-together` |
| Contradictory self-claim | Two current claims share scope and disagree | “Do these describe a change, different contexts, an exception, or an incorrect inference?” | Coexisting scopes, supersession, or rejection with owner rationale | `paper:psychometrics_fleeson_2001`, `paper:mcadams-et-al-2006-continuity-change` |
| High-stakes authority | A scenario asks the simulator to promise, transact, advise, or speak as the owner | “Should the system abstain, draft privately, or ask you each time?” | Explicit authority threshold; default abstention | `paper:linguistic-calibration-2022`, `paper:mireshghallah-2024-confaide` |

### P2: Autobiographical And Context Coverage

| Dimension | Trigger | Example question | Representation after answer | Evidence basis |
| --- | --- | --- | --- | --- |
| Life periods | Events cluster without owner-endorsed structure | “What name would you give this period, when did it overlap other chapters, and is it still active?” | Owner-defined overlapping period used as retrieval cue | `paper:conway-bekerian-1987-organization`, `paper:chen-mcanally-reese-2013-memory-organization` |
| Self-images and roles | Repeated “I am” language appears without temporal scope | “Was this identity defining, situational, aspirational, or something you no longer endorse?” | Versioned self-image with formation/retirement estimate and exceptions | `paper:rathbone-moulin-conway-2008-self-centered` |
| Event versus meaning | A repeated story changes interpretation | “What happened, what did it mean then, and what does it mean to you now?” | Event claims separated from narrative versions | `paper:jardmo-et-al-2023-repeated-narratives`, `paper:libby-eibach-2002-looking-back` |
| Untold counter-story | A strong identity inference rests on one theme | “Is there an episode that does not fit this pattern?” | Disconfirming episode and lower claim confidence | `paper:blagov-singer-2004-four-dimensions`, `paper:pals-2006-difficult-experiences` |
| Audience and telling purpose | A story is generalized outside the original chat | “Were you informing, venting, joking, persuading, or performing for this audience?” | Telling purpose and audience provenance | `paper:mclean-2005-meaning-telling`, `paper:bell-1984-audience-design` |
| Preference drift | Older and newer choices disagree | “Did your preference change, or were the situation and available options different?” | Temporal preference versions or context scopes | `paper:koren-2009-temporal-dynamics`, `paper:xiang-2010-temporal-preference-fusion` |

### P3: Calibration And Style Refinement

| Dimension | Trigger | Example question | Representation after answer | Evidence basis |
| --- | --- | --- | --- | --- |
| Affect calibration | Text classifier and direct language disagree | “How were you feeling, if you want to record it, and was the message sincere, masked, humorous, or venting?” | Optional owner report above classifier hypothesis | `paper:demszky-2020-goemotions`, `paper:carroll-russell-1996-face-context` |
| Communication intent | Style differs by relationship or channel | “Should the simulator preserve this difference, and in which contexts?” | Relationship/channel style delta | `paper:ireland-et-al-2011-language-style-matching`, `paper:bell-1984-audience-design` |
| Habit versus preference | Repeated action is observed without explanation | “Is this what you prefer, what was convenient, or simply a routine?” | Procedure distinct from preference/value | `paper:rieskamp-otto-2006-strategy-selection`, `paper:bardi-schwartz-2003-values-behavior` |
| Confidence calibration | The simulator is fluent where evidence is sparse | “Would you rather see an uncertain draft, a clarification question, or an abstention here?” | Owner-selected confidence and abstention policy | `paper:linguistic-calibration-2022` |

## Answer Lifecycle

Every answer is stored as owner-provided evidence, not as retroactive proof that
the archive contained the answer. The record includes question ID, exact prompt,
why it was asked, candidate state it may affect, timestamp, sensitivity, scope,
skip/delete state, and links to any claim it confirms, rejects, contextualizes, or
supersedes. Prompting effects make this provenance necessary
(`paper:dargembeau-garcia-jimenez-2024-working-self`).

Answers can be:

- **confirming:** strengthen a bounded claim
- **contextualizing:** add role, audience, time, or exception scope
- **superseding:** create a new version while preserving prior audit history
- **contradicting:** retain both sides and lower certainty until resolved
- **declined:** suppress the question without inventing an answer
- **revoked:** delete content and remove its influence while retaining a non-content
  audit tombstone

## Coverage Report

The guided experience should show, per taxonomy layer:

- evidence count and date range
- strongest source and inspection/provenance quality
- current confirmed, inferred, conflicted, stale, and unknown items
- relationship and audience coverage
- sensitive or third-party constraints
- held-out evaluation coverage
- next three questions, expected benefit, sensitivity, and burden

“Unknown” is a successful output. The purpose of elicitation is to make the model
more inspectable and useful, not to pressure the owner into completing a fictional
psychological profile.
