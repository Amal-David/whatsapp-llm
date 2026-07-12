# Persona Dialogue, Character Simulation, and Roleplay

## Seat Snapshot

- Scope: persona-conditioned dialogue, personalized language models, roleplay and character agents, long-horizon consistency and drift, persona datasets, prompting, retrieval, memory, graphs, fine-tuning, and evaluation.
- Counted primary papers: 20.
- Inspection depth: 13 full text; 7 abstract plus bibliographic metadata.
- Publication span: 2016-2024.
- Review state: cross-reviewed and approved after correction by cognitive_memory reviewer `019f4f06-f7d6-7190-87d7-80a1806aedcf` at `2026-07-11T03:48:06Z`.
- Privacy boundary: this research used public paper sources only. No WhatsApp content, private profile, local message database, generated owner profile, or credential was accessed.

## Search Strategy

Search proceeded in five passes:

1. Foundational persona dialogue: exact-title and topic queries for speaker embeddings, PersonaChat, large-scale persona extraction, few-shot adaptation, and Dialogue NLI.
2. Consistency and robustness: queries combining persona dialogue with contradiction, pragmatic reranking, commonsense expansion, input-order sensitivity, and profile grounding.
3. Long-horizon interaction: queries for Multi-Session Chat, long-term persona memory, very long conversational memory, forgetting, summary memory, and retrieval augmentation.
4. Roleplay and character agents: queries for trainable character agents, role-conditioned instruction tuning, personality fidelity interviews, and point-in-time character hallucination.
5. Personalized LMs and structured memory: queries for LaMP, user-history retrieval, persona knowledge graphs, and editable memory graphs.

Discovery queries were restricted to primary research surfaces. Every counted record was reconciled against an ACL Anthology or official AAAI paper page; full-text records were also inspected through the primary PDF. Canonical ACL IDs or DOI values and primary HTTPS URLs were taken from those pages. Published ACL versions were preferred over their preprints, so no preprint/published pair is counted twice.

## Inclusion And Exclusion

Included papers had to make a direct empirical or architectural contribution to at least one of the seat's bounded topics and expose enough primary-source detail to support a concrete finding, limitation, and architecture implication. The set intentionally mixes foundational datasets and models with failure analyses and evaluation papers.

Excluded items included surveys, blogs, product pages, project documentation, GitHub-only descriptions, secondary explainers, unverified citations, duplicate preprints, papers whose only connection was generic agent memory, and papers for which a canonical identifier or primary HTTPS paper page could not be confirmed. No paper was retained merely because its title sounded relevant.

## Domain Taxonomy

| Domain | What it covers | Records |
| --- | --- | --- |
| Explicit and latent persona conditioning | Speaker embeddings, editable text profiles, web-scale profile extraction, and few-shot adaptation | [paper:persona-dialogue-li-2016], [paper:persona-dialogue-zhang-2018], [paper:persona-dialogue-mazare-2018], [paper:persona-dialogue-madotto-2019] |
| Consistency and robustness | Contradiction detection, pragmatic reranking, controlled persona inference, and profile-order sensitivity | [paper:persona-dialogue-welleck-2019], [paper:persona-dialogue-kim-2020], [paper:persona-dialogue-majumder-2020], [paper:persona-dialogue-chen-2023-orig] |
| Long-horizon memory | Multi-session training, dynamic persona stores, forgetting and summary memory, and very-long-dialogue evaluation | [paper:persona-dialogue-xu-2022-goldfish], [paper:persona-dialogue-xu-2022-lemon], [paper:persona-dialogue-zhong-2024-memorybank], [paper:persona-dialogue-maharana-2024-locomo] |
| Structured and retrieval-based personalization | Persona commonsense graphs, user-history retrieval, and editable memory graphs | [paper:persona-dialogue-gao-2023-peacok], [paper:persona-dialogue-salemi-2024-lamp], [paper:persona-dialogue-wang-2024-emg-rag] |
| Roleplay construction and evaluation | Experience fine-tuning, role instruction tuning, personality interviews, and temporal knowledge boundaries | [paper:persona-dialogue-shao-2023-character-llm], [paper:persona-dialogue-wang-2024-rolellm], [paper:persona-dialogue-wang-2024-incharacter], [paper:persona-dialogue-ahn-2024-timechara] |
| Dataset realism and relationship progression | Whether acted persona chats reproduce real long-term interaction patterns | [paper:persona-dialogue-arimoto-2024] |

## Evidence-Backed Synthesis

### What Persona Conditioning Establishes

**Validated:** conditioning dialogue on a speaker representation or explicit profile can improve response specificity and local consistency. This result appears across learned speaker embeddings, editable profiles, mined profiles, and few-shot adaptation [paper:persona-dialogue-li-2016] [paper:persona-dialogue-zhang-2018] [paper:persona-dialogue-mazare-2018] [paper:persona-dialogue-madotto-2019]. At scale, broad conversational pretraining followed by task adaptation was stronger than training only on a small curated persona dataset [paper:persona-dialogue-mazare-2018].

**Not validated:** none of those results demonstrates replication of a real person's identity. Most foundational evaluations reward next-response prediction, short-range consistency, or adherence to an assigned role. PersonaChat's own-profile condition improved consistency but not engagement [paper:persona-dialogue-zhang-2018]. A latent speaker vector is therefore useful as a behavior prior, not an inspectable identity record [paper:persona-dialogue-li-2016].

### Consistency Needs Its Own Control Plane

Explicit persona input does not make a generator contradiction-proof. Dialogue NLI reduced contradictions by reranking candidates against profile claims [paper:persona-dialogue-welleck-2019], while a pragmatic listener improved consistency without retraining the base generator [paper:persona-dialogue-kim-2020]. These results support an independent post-generation consistency stage.

Consistency is also fragile in less obvious ways. Reordering unchanged PersonaChat profile sentences caused large performance swings in GPT-2 and BART, showing that list position can become an accidental behavioral signal [paper:persona-dialogue-chen-2023-orig]. Persona-derived commonsense can make dialogue richer, but those inferences remain hypotheses about an individual [paper:persona-dialogue-majumder-2020] [paper:persona-dialogue-gao-2023-peacok].

### Long Context Is Not Long-Term Memory

**Validated:** later-session dialogue exposes failures that short-chat evaluation misses. Multi-Session Chat showed that retrieval and summarization memory outperform a standard short-context encoder-decoder on long conversations [paper:persona-dialogue-xu-2022-goldfish]. A dynamic system that extracts, stores, updates, and retrieves separate user and bot persona statements also improved long-term consistency and engagement in its constructed task [paper:persona-dialogue-xu-2022-lemon].

Very large context windows do not solve the whole problem. On LoCoMo, long-context and RAG approaches improved performance but still struggled with temporal and causal reasoning and remained behind humans [paper:persona-dialogue-maharana-2024-locomo]. MemoryBank offers a useful external-memory pattern, but its quantitative longitudinal evidence is largely simulated and short-duration [paper:persona-dialogue-zhong-2024-memorybank].

**Plausible architecture hypothesis:** combine a source-linked, retention-controlled episodic record, derived semantic claims, temporal event links, and bounded retrieval. This is better supported than either putting the entire history into a prompt or silently compressing the owner into model weights [paper:persona-dialogue-xu-2022-goldfish] [paper:persona-dialogue-maharana-2024-locomo] [paper:persona-dialogue-wang-2024-emg-rag].

### Prompting, Retrieval, Graphs, And Fine-Tuning Have Different Jobs

Retrieval is the most inspectable mechanism for mutable personal context. LaMP shows broad gains from selecting user-history items for personalized tasks [paper:persona-dialogue-salemi-2024-lamp], and editable memory graphs add explicit correction and selection surfaces [paper:persona-dialogue-wang-2024-emg-rag]. PeaCoK shows how typed persona relations can organize characteristics, routines, goals, experiences, and relationships, but its generic commonsense edges should seed questions rather than become owner facts [paper:persona-dialogue-gao-2023-peacok].

Fine-tuning is useful for stable behavior and style but is harder to inspect, delete, and update. Character-LLM found that character-specific experience tuning and protective examples could improve role fidelity and knowledge boundaries, while also documenting limited data, synthetic experience, and base-model contamination [paper:persona-dialogue-shao-2023-character-llm]. RoleLLM similarly supports role-conditioned tuning but evaluates mainly single-turn question answering [paper:persona-dialogue-wang-2024-rolellm].

The evidence therefore supports retrieval for mutable facts and episodes, explicit prompts for current role and policy, and an optional versioned adapter for stable style. It does not support putting unverified autobiographical claims directly into permanent model weights.

### Evaluation Must Be Multidimensional

No single score establishes fidelity. A useful suite needs at least profile entailment and contradiction [paper:persona-dialogue-welleck-2019], profile-order robustness [paper:persona-dialogue-chen-2023-orig], later-session recall [paper:persona-dialogue-xu-2022-goldfish], temporal and causal reasoning [paper:persona-dialogue-maharana-2024-locomo], open-ended personality or mindset probes [paper:persona-dialogue-wang-2024-incharacter], and point-in-time knowledge boundaries [paper:persona-dialogue-ahn-2024-timechara].

Automatic overlap metrics can actively mislead: the meta-learning study improved human-rated consistency while worsening perplexity and BLEU [paper:persona-dialogue-madotto-2019]. Owner judgment and calibrated abstention are therefore required alongside automatic tests.

## Negative And Conflicting Evidence

- Profile access improved consistency without improving engagement in PersonaChat, so stronger persona expression is not equivalent to a better conversation [paper:persona-dialogue-zhang-2018].
- Web-scale persona extraction captures contradictory and temporary first-person statements, and direct transfer across Reddit and PersonaChat was weak without fine-tuning [paper:persona-dialogue-mazare-2018].
- Ordinary language metrics disagreed with human persona-consistency judgments in the few-shot adaptation experiment [paper:persona-dialogue-madotto-2019].
- Strong persona models remained insensitive to contradiction-bearing words, and identical claim sets produced different behavior when sentence order changed [paper:persona-dialogue-kim-2020] [paper:persona-dialogue-chen-2023-orig].
- In PLATO-LTM, persona-task fine-tuning could reduce open-domain coherence even while memory use improved [paper:persona-dialogue-xu-2022-lemon].
- LoCoMo remains mostly machine-generated, and acted Multi-Session Chat showed intimacy patterns unlike real long-term chats [paper:persona-dialogue-maharana-2024-locomo] [paper:persona-dialogue-arimoto-2024]. This weakens claims that synthetic long-dialogue success transfers directly to real relationships.
- Role-playing evaluation can look strong while missing multi-turn drift, time-bound knowledge, and personality change. RoleLLM is single-turn, InCharacter uses static character labels, and TimeChara still finds substantial future-knowledge leakage [paper:persona-dialogue-wang-2024-rolellm] [paper:persona-dialogue-wang-2024-incharacter] [paper:persona-dialogue-ahn-2024-timechara].

## Implications For A Digital Brain

1. **Use an evidence ledger, not one persona blob.** Each owner claim should retain source reference, extraction time, validity interval, relationship scope, confidence, confirmation status, and supersession links. Latent style representations must remain separate from factual claims [paper:persona-dialogue-li-2016] [paper:persona-dialogue-wang-2024-emg-rag].
2. **Separate memory layers.** Keep recent context, retention-controlled episodic records, derived semantic claims, and generic persona hypotheses distinct. Summaries and graph edges must link back to source evidence [paper:persona-dialogue-xu-2022-goldfish] [paper:persona-dialogue-gao-2023-peacok].
3. **Partition self and relationship memory.** Owner-global claims, interlocutor claims, and relationship-specific behavior need separate namespaces and privacy policies [paper:persona-dialogue-li-2016] [paper:persona-dialogue-xu-2022-lemon].
4. **Make time first-class.** Retrieval should enforce `valid_from`, `valid_to`, and knowledge-cutoff rules; future or superseded memories should not leak into an earlier simulated self [paper:persona-dialogue-maharana-2024-locomo] [paper:persona-dialogue-ahn-2024-timechara].
5. **Prefer retrieval for mutable information.** Use lexical, semantic, and time-aware retrieval with an explicit no-use option. Owner edits and deletions should take effect without retraining [paper:persona-dialogue-salemi-2024-lamp] [paper:persona-dialogue-wang-2024-emg-rag].
6. **Constrain tuning to stable patterns.** A versioned, rollback-capable adapter may learn durable style after enough evidence; changing facts, goals, relationships, and events should stay outside weights [paper:persona-dialogue-madotto-2019] [paper:persona-dialogue-shao-2023-character-llm].
7. **Guard every generated response.** Check grounding, contradiction, time boundary, relationship scope, and unsupported certainty before release [paper:persona-dialogue-welleck-2019] [paper:persona-dialogue-kim-2020] [paper:persona-dialogue-ahn-2024-timechara].
8. **Evaluate by horizon and failure cost.** Maintain separate suites for one-turn style, multi-turn consistency, months-later memory, temporal/causal questions, false recall, owner correction, cross-relationship leakage, and abstention [paper:persona-dialogue-xu-2022-goldfish] [paper:persona-dialogue-maharana-2024-locomo].

## WhatsApp Observability Gaps

WhatsApp text can provide timestamped evidence of what was said to particular people. It does not directly reveal the complete owner, and this research did not inspect any messages.

- **Expression is audience-conditioned.** A person may use different register, disclosure, humor, and opinions with different contacts. Flattening those signals into one global persona erases relationship context [paper:persona-dialogue-li-2016] [paper:persona-dialogue-arimoto-2024].
- **A message is not automatically a durable fact.** First-person text can be a joke, quote, hypothetical, temporary state, old preference, or deliberate performance [paper:persona-dialogue-mazare-2018].
- **Timestamps are not validity intervals.** A message shows when a claim was uttered, not when it became true, stopped being true, or was privately reconsidered [paper:persona-dialogue-ahn-2024-timechara].
- **Silence is not negation.** Missing discussion of a value, relationship, event, or preference provides little evidence that it is absent.
- **Third-party content has ambiguous ownership.** A contact's claim, a forwarded message, quoted text, or a group consensus must not become an owner claim without attribution.
- **Text misses other state.** Offline events, private motives, bodily state, unshared memories, calls, deleted messages, reactions, and media context may be unavailable. Even LoCoMo's image-grounded setup does not reproduce continuity in real personal media [paper:persona-dialogue-maharana-2024-locomo].
- **Interaction frequency is confounded.** More messages can reflect logistics, crisis, or platform habit rather than closeness. Acted chat data also distorts intimacy progression [paper:persona-dialogue-arimoto-2024].
- **Owner correction is indispensable.** Inferred claims and commonsense expansions should become interview questions or tentative hypotheses, never silent profile mutations [paper:persona-dialogue-majumder-2020] [paper:persona-dialogue-gao-2023-peacok].

## Open Questions

1. What is the smallest useful memory unit: message, exchange, episode, event, claim, or relationship-scoped claim bundle?
2. How should retrieval optimize downstream utility while refusing stale, redundant, or privacy-incompatible evidence?
3. What evidence threshold allows a repeated conversational pattern to enter a style adapter, and how is that adapter unlearned?
4. How should the system distinguish genuine drift from audience design, mood, sarcasm, roleplay, or a one-off event?
5. Which conflicts should supersede automatically, which should coexist by time or relationship, and which require an owner interview?
6. How can owner evaluation be sampled without exposing private conversations to external judges or turning familiarity into a single subjective score?
7. What false-memory, wrong-person, future-leakage, and cross-relationship tests should block a release?
8. Can a personalized system become less specific when evidence is weak, instead of filling gaps with category commonsense or base-model knowledge?
9. How should deletion propagate across episodic records, summaries, graph edges, retrieval indexes, cached prompts, and any trained adapter?
10. What longitudinal study design can measure fidelity without assuming that the owner has one static, context-free persona?
