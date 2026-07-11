# Cognitive Memory Seat Notes

## Scope and status

This seat covers episodic, semantic, procedural, working, prospective, and source
memory; consolidation, retrieval, forgetting, and metacognition; complementary
learning systems; and inspectable cognitive architectures. It does not claim that
a software system is conscious or cognitively equivalent to a person.

The seat contains 20 unique counted journal papers. Three records were inspected
at full-text depth and 17 at abstract depth. Every record has a DOI as its
canonical identifier. All 20 records were cross-reviewed by the
`narrative_identity` seat under reviewer agent
`019f4f06-ead6-7c73-8a68-dfabd3b272ba` at `2026-07-11T03:02:36Z`; the review
decision and evidence gaps are documented in
`reviews/cognitive_memory.reviewed-by-narrative_identity.md`.

No WhatsApp content, private profile, local database, credential, or key was
accessed during this research.

## Search strategy

Search proceeded by memory function rather than by one broad phrase:

- Episodic versus semantic dissociations and hippocampal dependence.
- Procedural or habit learning versus declarative recollection.
- Working-memory capacity, binding, and buffer architectures.
- Event-based prospective memory, spontaneous retrieval, and monitoring cost.
- Source attribution, source confusability, and metacognitive calibration.
- Retrieval practice, retrieval-induced forgetting, context reinstatement, and
  retention curves.
- Reactivation, reconsolidation, sleep cueing, and offline consolidation.
- Complementary learning systems, ACT-R, and integrated neural cognitive models.

Exact-title searches were combined with DOI and site-restricted PubMed/PMC or
publisher searches. DOI content negotiation was used as a final metadata check
for all 20 records, including title, venue, publication date, and authorship.
PubMed identifiers were retained as aliases where verified. PMC or publisher
full text was used for `paper:cognitive-memory-hupbach-et-al-2007`,
`paper:cognitive-memory-murre-dros-2015`, and
`paper:cognitive-memory-schapiro-et-al-2017`.

The search intentionally included null, cost, and contamination findings, not
only successful memory enhancement. This surfaced no age deficit in one
prospective-memory paradigm (`paper:cognitive-memory-einstein-mcdaniel-1990`),
retrieval-induced impairment of competitors
(`paper:cognitive-memory-anderson-bjork-bjork-1994`), and reminder-driven source
intrusions (`paper:cognitive-memory-hupbach-et-al-2007`).

## Inclusion and exclusion

Included papers met all of these conditions:

- A unique, resolvable DOI and primary HTTPS paper URL were verified.
- The published paper itself was inspected through a publisher, DOI, PubMed, or
  PMC surface.
- The method and result were specific enough to paraphrase without relying on a
  secondary explainer.
- The paper directly constrained a memory component, update rule, retrieval
  process, confidence mechanism, or integrated architecture.
- A preprint and published version were never counted separately.

Excluded material included books and chapters, blogs, product pages, project
documentation, secondary explainers, duplicate versions, and records available
only at metadata depth. Formal papers that introduced an originating model were
retained even when they synthesized prior evidence: the episodic buffer
(`paper:cognitive-memory-baddeley-2000`), complementary learning systems
(`paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`), and ACT-R's
integrated architecture (`paper:cognitive-memory-anderson-et-al-2004`). Their
limitations and non-empirical status are explicit in the records.

## Domain taxonomy

| Domain | Included evidence | Architectural question |
| --- | --- | --- |
| Episodic and semantic memory | `paper:cognitive-memory-vargha-khadem-1997`; `paper:cognitive-memory-hupbach-et-al-2007` | How should event-specific recollection remain distinct from generalized knowledge and later reconstruction? |
| Procedural and habit memory | `paper:cognitive-memory-cohen-squire-1980`; `paper:cognitive-memory-knowlton-mangels-squire-1996` | Which learned policies can change without implying an explicit remembered reason? |
| Working memory and binding | `paper:cognitive-memory-luck-vogel-1997`; `paper:cognitive-memory-baddeley-2000` | What belongs in a bounded active workspace, and how are multimodal items bound without losing provenance? |
| Prospective memory | `paper:cognitive-memory-einstein-mcdaniel-1990`; `paper:cognitive-memory-einstein-et-al-2005` | How are intentions triggered, monitored, externally cued, and marked complete? |
| Source memory | `paper:cognitive-memory-hashtroudi-johnson-chrosniak-1989`; `paper:cognitive-memory-hupbach-et-al-2007` | How are origin, source class, and confusable alternatives represented? |
| Consolidation and updating | `paper:cognitive-memory-hupbach-et-al-2007`; `paper:cognitive-memory-rasch-et-al-2007`; `paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`; `paper:cognitive-memory-schapiro-et-al-2017` | Which records are replayed or generalized, and how are updates versioned and tested for interference? |
| Retrieval and forgetting | `paper:cognitive-memory-godden-baddeley-1975`; `paper:cognitive-memory-anderson-bjork-bjork-1994`; `paper:cognitive-memory-karpicke-roediger-2008`; `paper:cognitive-memory-murre-dros-2015` | How do cue match, practice, competition, and elapsed time alter accessibility? |
| Metacognition | `paper:cognitive-memory-koriat-bjork-2005`; `paper:cognitive-memory-metcalfe-finn-2008`; `paper:cognitive-memory-karpicke-roediger-2008` | When does confidence predict usable recall, and when does it misallocate study or update effort? |
| Integrated architectures | `paper:cognitive-memory-anderson-et-al-2004`; `paper:cognitive-memory-eliasmith-et-al-2012`; `paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`; `paper:cognitive-memory-schapiro-et-al-2017` | Can typed memory and control modules cooperate while keeping state and decisions inspectable? |

## Evidence-backed synthesis

### 1. One undifferentiated long-term store is not supported

Episodic and semantic acquisition can dissociate after early hippocampal injury
(`paper:cognitive-memory-vargha-khadem-1997`). Skill learning can persist despite
amnesia (`paper:cognitive-memory-cohen-squire-1980`), and probabilistic habit
learning can dissociate in the opposite direction from declarative memory in
amnesia and Parkinson's disease
(`paper:cognitive-memory-knowlton-mangels-squire-1996`).

**Validated:** memory performance depends on the type of representation and task;
successful access in one system is not proof of access in another.

**Plausible design implication:** use typed stores for episodes, semantic claims,
procedures, and intentions. Every derived behavior should report which store
supplied it and should not invent a declarative rationale for a learned policy.

### 2. The active workspace should bind structured items but remain bounded

Visual working memory behaved as though capacity was more closely tied to
integrated objects than to each separate feature
(`paper:cognitive-memory-luck-vogel-1997`). The episodic-buffer proposal explains
how a limited temporary system could bind specialized working-memory contents
with long-term knowledge (`paper:cognitive-memory-baddeley-2000`).

**Validated:** temporary retention is capacity constrained, and binding changes
the relevant unit of capacity.

**Plausible design implication:** build the active context from event or entity
bundles whose fields keep source links. Capacity, eviction, and truncation should
be logged. The often-reported four-object result is not a defensible hard-coded
message or token limit.

### 3. Intentions need both triggers and monitoring policies

Prospective memory did not simply track retrospective memory, and external aids
and distinctive targets improved performance
(`paper:cognitive-memory-einstein-mcdaniel-1990`). Five experiments supported
both resource-demanding monitoring and spontaneous cue-triggered retrieval
(`paper:cognitive-memory-einstein-et-al-2005`).

**Validated:** remembering that an action must occur can rely on different
retrieval routes, and cue properties matter.

**Plausible design implication:** an intention record needs an action, trigger,
time or event condition, monitoring mode, reminder route, completion evidence,
and expiry policy. The system should disclose whether an intention surfaced by
matching a cue, scheduled monitoring, or owner intervention.

### 4. Provenance is a memory function, not optional metadata

Source errors were not uniform: older adults had more difficulty discriminating
same-class sources but not internal from external sources in general
(`paper:cognitive-memory-hashtroudi-johnson-chrosniak-1989`). Reactivating an
episode before related new learning produced delayed intrusions from the new
list (`paper:cognitive-memory-hupbach-et-al-2007`).

**Validated:** content can be available while its origin is uncertain, and source
confusion depends on similarity among candidate origins.

**Plausible design implication:** every episode and semantic claim should carry
source identity, source class, acquisition context, direct-versus-inferred
status, and competing source hypotheses. A claim without provenance should not
be promoted merely because its text is fluent or frequently retrieved.

### 5. Consolidation is selective, constructive, and potentially contaminating

Learning-associated odor cues presented during slow-wave sleep selectively
improved declarative retention in the studied paradigm
(`paper:cognitive-memory-rasch-et-al-2007`). A reminder could reopen an episodic
memory to delayed integration of later material
(`paper:cognitive-memory-hupbach-et-al-2007`). Complementary-learning-systems
models assign rapid item storage and gradual interleaved structure learning to
different processes
(`paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`). A later model
split episode separation and regularity learning across pathways even within the
hippocampal model (`paper:cognitive-memory-schapiro-et-al-2017`).

**Validated:** in the studied human paradigms, reactivation can affect later
retention and can also introduce source errors.

**Plausible design implication:** consolidation should be an auditable job that
selects records, replays evidence, proposes abstractions, tests old and new
memories, and emits a new version. It must never rewrite the sole copy of an
episode.

### 6. Retrieval changes future accessibility in more than one direction

Matching learning and recall environments improved free recall in divers
(`paper:cognitive-memory-godden-baddeley-1975`). Repeated retrieval greatly
improved delayed memory for practiced vocabulary
(`paper:cognitive-memory-karpicke-roediger-2008`), yet selective retrieval
impaired related unpracticed category members
(`paper:cognitive-memory-anderson-bjork-bjork-1994`). A close forgetting-curve
replication found rapid early decline, slower later decline, and a possible
non-monotonic point near 24 hours
(`paper:cognitive-memory-murre-dros-2015`).

**Validated:** retrieval can strengthen targets, context can gate access, and
selective retrieval can impose costs on competitors.

**Plausible design implication:** record retrieval events as state-changing
operations. Evaluate target benefit and competitor loss separately, broaden
retrieval when one claim dominates, and distinguish archived, low-ranked,
temporarily inaccessible, contradicted, and deleted states.

### 7. Confidence cannot govern memory maintenance by itself

Judgments made while both cue and target were visible overestimated later
cue-only accessibility (`paper:cognitive-memory-koriat-bjork-2005`). Misleading
judgments of learning causally steered what people chose to restudy
(`paper:cognitive-memory-metcalfe-finn-2008`). University students also failed to
predict the delayed advantage of retrieval practice
(`paper:cognitive-memory-karpicke-roediger-2008`).

**Validated:** metacognitive judgments can be systematically miscalibrated and
can cause poor allocation decisions.

**Plausible design implication:** confidence should be calibrated against
cue-reduced retrieval tests, source accuracy, contradiction resolution, and
owner verification. Confidence, retrieval fluency, evidence strength, and truth
must remain separate fields.

### 8. Inspectability favors typed modules and explicit control flow

ACT-R coordinates specialized modules through bounded buffers and explicit rule
selection (`paper:cognitive-memory-anderson-et-al-2004`). Spaun demonstrated that
one integrated model could perform multiple perception, memory, reasoning, and
action tasks through shared interfaces
(`paper:cognitive-memory-eliasmith-et-al-2012`). Complementary-learning-systems
work supplies a concrete reason for fast and slow learning paths
(`paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`;
`paper:cognitive-memory-schapiro-et-al-2017`).

**Plausible design implication:** an inspectable digital brain should expose
module inputs and outputs, active buffer contents, retrieved candidates, policy
scores, selected action, memory mutations, and evaluation results.

**Speculative:** reproducing those organizational principles will improve
personal fidelity. None of these papers establishes that ACT-R, Spaun, or a
biological memory analogy is sufficient for representing a particular person.

## Negative and conflicting evidence

1. **Retrieval benefit versus retrieval cost.** Retrieval strengthened practiced
   targets in `paper:cognitive-memory-karpicke-roediger-2008` but impaired related
   unpracticed items in `paper:cognitive-memory-anderson-bjork-bjork-1994`. A
   digital brain therefore cannot treat “retrieve more” as uniformly beneficial.

2. **Consolidation versus contamination.** Cueing can strengthen retention
   (`paper:cognitive-memory-rasch-et-al-2007`), but reactivation can also blend
   later information into an earlier episode
   (`paper:cognitive-memory-hupbach-et-al-2007`). Replay requires provenance and
   regression tests, not just salience scoring.

3. **Monotonic decay is too simple.** `paper:cognitive-memory-murre-dros-2015`
   broadly reproduced a forgetting curve but reported a possible upward
   deviation near 24 hours. Its one-person design prevents hard-coding that
   shape, and it did not establish sleep as the cause.

4. **Age effects depend on task demands.** Contrary to a broad deficit
   prediction, `paper:cognitive-memory-einstein-mcdaniel-1990` found no age
   difference in its prospective tasks. Likewise,
   `paper:cognitive-memory-hashtroudi-johnson-chrosniak-1989` found a selective,
   not general, source-monitoring deficit. Architecture should model cue and
   source confusability instead of demographic shortcuts.

5. **Semantic competence does not imply intact episodes.** The developmental
   cases in `paper:cognitive-memory-vargha-khadem-1997` acquired substantial
   factual and language knowledge despite severe everyday episodic amnesia. A
   fluent semantic persona can therefore conceal weak autobiographical grounding.

6. **Model integration is not person replication.** ACT-R and Spaun organize
   multiple functions (`paper:cognitive-memory-anderson-et-al-2004`;
   `paper:cognitive-memory-eliasmith-et-al-2012`), but selected task fits do not
   establish autobiographical fidelity, consciousness, or cognitive equivalence.

## Implications for an inspectable digital brain

The evidence supports the following architecture as a testable engineering
hypothesis, not a claim that software implements human memory:

1. **Immutable episode ledger.** Store event time, participants or entities,
   content references, context, source, confidence, and later corrections. Do not
   overwrite the source episode during abstraction or reconsolidation
   (`paper:cognitive-memory-hupbach-et-al-2007`).

2. **Versioned semantic store.** Maintain generalized claims separately, each
   linked to supporting and contradicting episodes. Promote updates through slow,
   interleaved evaluation
   (`paper:cognitive-memory-vargha-khadem-1997`;
   `paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`).

3. **Procedure and policy store.** Represent learned routines independently from
   declarative reasons, with explicit provenance and owner confirmation for
   consequential policies (`paper:cognitive-memory-cohen-squire-1980`;
   `paper:cognitive-memory-knowlton-mangels-squire-1996`).

4. **Bounded working workspace.** Bind the current input, selected episodes,
   semantic claims, goals, and candidate actions into structured bundles. Log
   omitted and evicted items (`paper:cognitive-memory-luck-vogel-1997`;
   `paper:cognitive-memory-baddeley-2000`).

5. **Prospective intention manager.** Store triggers, monitoring policy,
   reminders, completion evidence, and expiry separately from ordinary facts
   (`paper:cognitive-memory-einstein-mcdaniel-1990`;
   `paper:cognitive-memory-einstein-et-al-2005`).

6. **Source-aware retrieval.** Rank with content and encoding context while
   returning source identity, same-class alternatives, and uncertainty
   (`paper:cognitive-memory-godden-baddeley-1975`;
   `paper:cognitive-memory-hashtroudi-johnson-chrosniak-1989`).

7. **Auditable consolidation.** Select replay candidates, generate proposed
   abstractions, test interference, and write a reversible version with a full
   change log (`paper:cognitive-memory-rasch-et-al-2007`;
   `paper:cognitive-memory-schapiro-et-al-2017`).

8. **Adaptive forgetting and archive policy.** Learn retention from utility,
   retrieval history, replacement evidence, and owner preferences. Prefer
   reversible down-ranking or archive states to deletion when provenance or
   identity value remains (`paper:cognitive-memory-murre-dros-2015`;
   `paper:cognitive-memory-anderson-bjork-bjork-1994`).

9. **Independent metacognitive layer.** Calibrate confidence against held-out
   retrieval and source tests, and prevent confidence alone from scheduling
   study, consolidation, or deletion (`paper:cognitive-memory-koriat-bjork-2005`;
   `paper:cognitive-memory-metcalfe-finn-2008`).

10. **End-to-end decision trace.** Preserve module state, retrieved candidates,
    selected action, and every memory mutation so an owner can inspect why a
    response occurred (`paper:cognitive-memory-anderson-et-al-2004`;
    `paper:cognitive-memory-eliasmith-et-al-2012`).

## WhatsApp observability gaps

WhatsApp messages can provide public-style research questions for this project,
but messages alone do not reveal the cognitive state required by these papers.
No private messages were inspected here.

| Gap | What message text may show | What it cannot establish | Required remedy |
| --- | --- | --- | --- |
| Episodic versus semantic status | A description or factual claim | Whether it is a recollected episode, repeated hearsay, or generalized knowledge | Ask the owner for memory type, source episode, and confidence; preserve uncertainty (`paper:cognitive-memory-vargha-khadem-1997`). |
| Procedural memory | Statements about habits or advice | The routine actually performed, its success rate, or an explicit reason for it | Separate claimed procedure from observed or owner-confirmed procedure (`paper:cognitive-memory-cohen-squire-1980`; `paper:cognitive-memory-knowlton-mangels-squire-1996`). |
| Working memory | Nearby conversational context | The owner's active mental contents, omitted alternatives, or attentional capacity | Treat model context as system state, not inferred human working memory (`paper:cognitive-memory-baddeley-2000`). |
| Prospective memory | “I will” statements and reminders sent in chat | Whether the intention persisted, triggered outside chat, was completed, or was abandoned | Require explicit status, trigger, reminder, and completion evidence (`paper:cognitive-memory-einstein-et-al-2005`). |
| Source memory | Sender, quote markers, and some forwarding metadata | Whether the sender personally witnessed, inferred, remembered, or trusted the claim | Ask for source class and retain competing source hypotheses (`paper:cognitive-memory-hashtroudi-johnson-chrosniak-1989`). |
| Consolidation | Repetition of themes over time | Sleep-dependent replay, offline reactivation, or when a belief became generalized | Model only system consolidation events; do not infer biological consolidation from message timing (`paper:cognitive-memory-rasch-et-al-2007`). |
| Forgetting | Long gaps or failure to mention something | Whether a memory was lost, inaccessible, irrelevant, private, or merely not cued | Use owner tests and reversible archive states; absence of a message is not forgetting (`paper:cognitive-memory-murre-dros-2015`). |
| Retrieval context | Conversation, time, and chat relationship | Which unexpressed environmental or internal cue enabled recall | Store observable context but label missing context explicitly (`paper:cognitive-memory-godden-baddeley-1975`). |
| Metacognition | Hedging, certainty words, or corrections | Calibration accuracy across contexts; expressive style can mimic confidence | Compare confidence with later retrieval, source, and owner-verification outcomes (`paper:cognitive-memory-koriat-bjork-2005`). |
| Reconsolidation and source drift | Later retellings that differ | Whether change reflects correction, reconstruction, strategic framing, or noise | Keep versions and ask the owner to adjudicate consequential changes (`paper:cognitive-memory-hupbach-et-al-2007`). |

These gaps should become owner-controlled interview questions and evaluation
fields, never confident psychological inference from message text.

## Open questions

1. What owner-facing control best distinguishes an episode, semantic claim,
   procedure, intention, and speculation without making capture burdensome?

2. Which evidence should qualify a semantic abstraction for promotion, and how
   much older evidence must be interleaved to test interference
   (`paper:cognitive-memory-mcclelland-mcnaughton-oreilly-1995`)?

3. How should retrieval evaluation measure both target improvement and loss of
   related alternatives (`paper:cognitive-memory-karpicke-roediger-2008`;
   `paper:cognitive-memory-anderson-bjork-bjork-1994`)?

4. What source classes are useful for direct statement, quotation, forwarding,
   owner inference, model inference, and external document, and how should
   same-class ambiguity be displayed
   (`paper:cognitive-memory-hashtroudi-johnson-chrosniak-1989`)?

5. Which memory updates require owner confirmation, and which low-risk updates
   may remain provisional until later evidence arrives
   (`paper:cognitive-memory-hupbach-et-al-2007`)?

6. How should prospective intentions receive completion evidence when the action
   occurs outside observable digital channels
   (`paper:cognitive-memory-einstein-et-al-2005`)?

7. Can retention and retrieval policies be learned per owner and memory class
   without converting non-mention into evidence of forgetting
   (`paper:cognitive-memory-murre-dros-2015`)?

8. What cue-reduced, source-sensitive tests best calibrate memory confidence
   before the system suppresses, archives, or promotes a record
   (`paper:cognitive-memory-koriat-bjork-2005`;
   `paper:cognitive-memory-metcalfe-finn-2008`)?

9. Which minimal module and buffer boundaries provide useful inspection without
   copying biological or ACT-R structure unnecessarily
   (`paper:cognitive-memory-anderson-et-al-2004`)?

10. What ablations can show that an integrated system genuinely uses episodic,
    semantic, procedural, source, and prospective components rather than one
    opaque retrieval shortcut (`paper:cognitive-memory-eliasmith-et-al-2012`;
    `paper:cognitive-memory-schapiro-et-al-2017`)?
