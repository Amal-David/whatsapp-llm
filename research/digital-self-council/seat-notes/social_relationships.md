# Social Relationships Seat Note

- Task: `T-0406`
- Seat: `social_relationships`
- Counted papers: 20
- Inspection depth: 6 full text, 14 abstract, 0 metadata-only
- Review state: cross-reviewed and approved by `persona_dialogue` at `2026-07-11T04:13:24Z`
- Private-data boundary: no WhatsApp content, private profiles, local databases, credentials, or keys were accessed

## Search Strategy

The search used only primary scholarly surfaces: publisher article pages, DOI records, PubMed, arXiv for the paper text corresponding to a counted published version, ACL Anthology, and official proceedings records. Searches were organized around five query families:

1. Relational self, social identity, and roles: `relational self significant other transference`, `social identity anonymity group behavior CMC`, and `relationship boundary work role context`.
2. Audience design and partner-specific common ground: `language style audience design`, `audience design societal norms`, `conceptual pacts lexical choice`, and `partner-specific referring expressions`.
3. Communication accommodation: `language style matching relationship stability`, `linguistic style accommodation social media`, and `sources linguistic alignment conversation`.
4. Self-presentation and context collapse: `self-presentation friends strangers`, `online dating impression management`, `Twitter context collapse imagined audience`, and `invisible audience social networks`.
5. Disclosure and information boundaries: `context collapse privacy disclosures`, `public private disclosure intimacy`, `interpersonal management disclosure SNS`, and `Facebook privacy awareness information sharing`.

Candidate metadata was checked against the primary page or paper before extraction. DOI, ACL, arXiv, and PubMed aliases were recorded only when they referred to the same work. The published version was counted when both a preprint and a proceedings version existed, as in `paper:danescu-niculescu-mizil-et-al-2011-mark-my-words`; its arXiv identifier is retained only as an alternate ID.

## Inclusion And Exclusion

Included papers had to be original journal, conference, or workshop papers within the seat boundary and expose enough primary-source material to verify the research question, method or evidence basis, and at least one restrained finding. The set deliberately includes theoretical papers when they define an architecture-relevant construct, such as relational selves or audience design, but marks their method and evidence limits explicitly (`paper:andersen-chen-2002-relational-self`, `paper:bell-1984-audience-design`).

Excluded material included books and chapters, blogs, product pages, GitHub documentation, dissertations, secondary explainers, papers whose primary identifier or source could not be verified, and duplicate preprint/published versions. Studies centered on inferring contact traits without consent were also excluded. Papers using sensitive social categories were considered only when their contribution concerned contextual identity processes, not as support for protected-trait inference.

Abstract-only records do not make claims beyond the publisher or PubMed abstract. Full-text status was used only where the paper or accepted article was inspected across methods, results, and limitations. No metadata-only record was counted.

## Domain Taxonomy

| Domain | Paper records | Architectural question |
| --- | --- | --- |
| Relational self, social identity, and roles | `paper:andersen-chen-2002-relational-self`; `paper:postmes-spears-sakhel-degroot-2001-social-influence`; `paper:trefalt-2013-between-you-and-me` | Which self-schema, group norm, or role boundary is active in this relationship? |
| Audience design and partner-specific common ground | `paper:bell-1984-audience-design`; `paper:youssef-1993-childrens-linguistic-choices`; `paper:horton-gerrig-2002-speakers-experiences`; `paper:brennan-clark-1996-conceptual-pacts`; `paper:metzing-brennan-2003-conceptual-pacts-broken` | Who is addressed or overhearing, what do they know, and what language has this pair established together? |
| Communication accommodation | `paper:ireland-et-al-2011-language-style-matching`; `paper:danescu-niculescu-mizil-et-al-2011-mark-my-words`; `paper:doyle-frank-2016-sources-linguistic-alignment` | Which style changes are genuinely contingent on the partner's preceding behavior rather than topic or baseline similarity? |
| Self-presentation and context collapse | `paper:tice-et-al-1995-when-modesty-prevails`; `paper:ellison-heino-gibbs-2006-managing-impressions`; `paper:marwick-boyd-2011-context-collapse`; `paper:litt-hargittai-2016-imagined-audience`; `paper:bernstein-et-al-2013-invisible-audience` | Which version of self is appropriate for this intended, permitted, and actual audience? |
| Disclosure and information boundaries | `paper:vitak-2012-context-collapse-privacy`; `paper:bazarova-2012-public-intimacy`; `paper:lampinen-et-al-2011-were-in-it-together`; `paper:acquisti-gross-2006-imagined-communities` | May this information be shared here, with these people, including information that concerns someone else? |

## Evidence-Backed Synthesis

### The self is relational and role-scoped

The strongest conceptual result is that a person should not be represented as one context-free bundle of traits. Significant-other cues can activate linked self-knowledge, affect, and regulation patterns (`paper:andersen-chen-2002-relational-self`). Group context can also make a local norm behaviorally salient even when individual identity cues are reduced (`paper:postmes-spears-sakhel-degroot-2001-social-influence`). Role boundaries are negotiated within specific relationships, and the same tactic can have different substantive and relational outcomes depending on the counterpart (`paper:trefalt-2013-between-you-and-me`).

For a digital brain, this supports a stable owner core plus relationship- and role-scoped overlays. It does not support inventing a separate personality for every contact. The overlay should contain observed context-conditioned behavior, its evidence, and uncertainty, while durable values and facts remain separately represented.

### Audience is structured, plural, and imperfectly known

Audience design distinguishes direct addressees from auditors, overhearers, and reference groups (`paper:bell-1984-audience-design`). That hierarchy is not universal: longitudinal child speech showed that institutional norms and the personal significance of audience members could outweigh a simple audience-role ordering (`paper:youssef-1993-childrens-linguistic-choices`). Speakers also learn when and how to adapt; audience-sensitive production improved with task experience rather than appearing perfectly on the first switch (`paper:horton-gerrig-2002-speakers-experiences`).

Online, the audience problem becomes more severe. People manage collapsed contexts through targeting, concealment, and audience-compatible authenticity (`paper:marwick-boyd-2011-context-collapse`). Their imagined audience changes from post to post even when platform permissions remain stable (`paper:litt-hargittai-2016-imagined-audience`). Their estimate of actual reach can be wrong by several-fold, and visible engagement is a poor reach estimator (`paper:bernstein-et-al-2013-invisible-audience`).

The architecture therefore needs four distinct audience fields: intended, imagined, permitted, and observed. Treating any one as the truth would erase the central uncertainty demonstrated by these papers.

### Interaction history creates partner-specific language

Repeated interaction produces shared conceptualizations and conventional labels that can persist even when shorter wording is available (`paper:brennan-clark-1996-conceptual-pacts`). Comprehension costs when the original partner unexpectedly changes an established label, but not when a new partner introduces the same new label, are consistent with a partner-indexed expectation about wording (`paper:metzing-brennan-2003-conceptual-pacts-broken`). Later eye-tracking work reinterpreted the timing as late recovery from partner-independent preemption (doi:10.1016/j.jml.2006.05.002), so this result does not by itself establish early lookup of partner-specific common ground.

This supports a partner-scoped common-ground store containing aliases, nicknames, shorthand, shared references, and their provenance. The evidence does not justify copying every repeated phrase into permanent memory. A useful record needs a partner, thread or episode, establishment evidence, last confirmation, and confidence that the phrase denotes shared meaning rather than quotation or coincidence.

### Accommodation is real but is not a trait detector

Function-word matching predicted mutual romantic interest and three-month relationship stability in two observational samples (`paper:ireland-et-al-2011-language-style-matching`). Large-scale Twitter replies also showed turn-contingent style accommodation after controls for pair baseline and topic, but convergence was asymmetric and differed by linguistic dimension; crude status proxies were weakly related to stylistic influence (`paper:danescu-niculescu-mizil-et-al-2011-mark-my-words`). A later model comparison found alignment primarily at the lexical level and strongly modulated by discourse act, while warning that aggregate similarity can reflect pre-existing similarity rather than adaptation (`paper:doyle-frank-2016-sources-linguistic-alignment`).

Aggregate language style matching should therefore be stored as symmetric dyadic similarity over a defined window. A directional accommodation feature requires temporally ordered reply analysis against a dyadic baseline and should be conditioned on discourse act. Neither representation is sufficient for inferring closeness, power, approval, personality, or relationship health.

### Self-presentation is audience-conditioned, not simply true or false

People used more modest presentations with friends and more self-enhancing presentations with strangers; relationship-incongruent presentation also carried a memory cost (`paper:tice-et-al-1995-when-modesty-prevails`). Online daters balanced positive or aspirational presentation against credibility and anticipated face-to-face verification (`paper:ellison-heino-gibbs-2006-managing-impressions`). These findings argue against labeling every audience-dependent difference as deception.

A digital brain should distinguish factual claims, aspirations, selective emphasis, uncertainty, and explicitly fictional or playful presentation. It should preserve contradictions by audience and goal until the owner resolves them, rather than choosing the most flattering or most frequent statement as globally true.

### Disclosure is relational, channel-dependent, and co-owned

Audience size and diversity relate to disclosure and privacy-tool use, but privacy concern and settings do not behave as one simple latent privacy preference (`paper:vitak-2012-context-collapse-privacy`). A preregistered close replication (doi:10.1093/joc/jqaf007) reproduced the audience-composition paths with small effects but not the original privacy-concern-to-disclosure relationship, which reversed direction. The same intimate disclosure is interpreted differently when private versus public; public intimate disclosure can be judged less appropriate and reduce liking (`paper:bazarova-2012-public-intimacy`). Privacy management also depends on what others disclose: participants used preventive and corrective, individual and collaborative strategies, and often relied on unspoken reciprocal expectations (`paper:lampinen-et-al-2011-were-in-it-together`). Early Facebook evidence likewise found substantial disclosure among privacy-concerned users and mistaken beliefs about community size and profile visibility (`paper:acquisti-gross-2006-imagined-communities`).

Past disclosure is therefore not consent for future disclosure. Information boundaries need subject, source, audience, channel, purpose, sensitivity, expiry, and consent state. Information about another person should default to co-owned and require explicit permission or redaction.

## Negative And Conflicting Evidence

1. A fixed audience hierarchy is too simple. `paper:bell-1984-audience-design` predicts systematic audience weighting, while `paper:youssef-1993-childrens-linguistic-choices` shows stable audience conditions with changing style and effects of wider social norms. The implementation should learn audience weights and retain counterexamples.
2. Audience knowledge does not guarantee immediate adaptation. `paper:horton-gerrig-2002-speakers-experiences` indicates that speakers may need repeated experience to notice and execute the right adjustment. A digital brain should ask or clarify when common ground is uncertain.
3. Matching is not uniformly positive or diagnostic. `paper:ireland-et-al-2011-language-style-matching` links matching to romantic outcomes, but `paper:doyle-frank-2016-sources-linguistic-alignment` finds strong dependence on lexical item and discourse act, and `paper:danescu-niculescu-mizil-et-al-2011-mark-my-words` finds weak links between style influence and rough status proxies. Relationship scoring from style alone would overclaim.
4. Privacy attitudes, settings, and behavior do not collapse into one construct. `paper:acquisti-gross-2006-imagined-communities` reports concern alongside disclosure and audience misconceptions, while `paper:vitak-2012-context-collapse-privacy` finds different relationships among concern, settings, audience, and disclosure. The close replication at doi:10.1093/joc/jqaf007 retained the audience-composition effects but reversed the concern-disclosure association. Explicit per-information rules are safer than a global privacy score.
5. Disclosure can build intimacy or damage impressions depending on channel. `paper:bazarova-2012-public-intimacy` shows that visibility changes interpretation, and `paper:lampinen-et-al-2011-were-in-it-together` shows that even well-intentioned disclosure can violate another person's boundary. More disclosure is not a monotonic route to relational fidelity.
6. The imagined audience is neither stable nor accurate. `paper:litt-hargittai-2016-imagined-audience` finds message-level fluctuation, while `paper:bernstein-et-al-2013-invisible-audience` finds large reach-estimation errors. Platform or thread membership alone cannot stand in for the audience the owner had in mind.

## Implications For A Digital Brain

### Validated constraints

- Relationship context can change self-presentation, language, norms, and boundary tactics (`paper:andersen-chen-2002-relational-self`; `paper:tice-et-al-1995-when-modesty-prevails`; `paper:trefalt-2013-between-you-and-me`).
- Some shared language is partner-specific and history-dependent (`paper:brennan-clark-1996-conceptual-pacts`; `paper:metzing-brennan-2003-conceptual-pacts-broken`).
- Imagined and actual audiences can diverge, and audience intent can change per message (`paper:litt-hargittai-2016-imagined-audience`; `paper:bernstein-et-al-2013-invisible-audience`).
- Disclosure boundaries are not solely individual because messages can expose information about others (`paper:lampinen-et-al-2011-were-in-it-together`).

### Plausible architecture

1. **Owner core:** explicit facts, values, preferences, and safety rules that are not silently overwritten by one relationship.
2. **Relationship index:** counterpart or group, owner-confirmed role labels, tie history, active role, known boundaries, and uncertainty. No contact traits are inferred as a product capability.
3. **Relational-self layer:** context-conditioned self-schema evidence linked to relationship cues, with temporal validity and provenance.
4. **Audience object:** intended, imagined, permitted, and observed audiences; addressee, auditor, overhearer, and reference-group roles; confidence and mismatch flags.
5. **Partner common ground:** aliases, conceptual pacts, shared events, shorthand, and repair history scoped to a relationship and thread.
6. **Accommodation model:** symmetric windowed matching kept separate from directional turn-level features against a dyadic baseline, with directional features conditioned on discourse act and topic; neither is promoted directly to a personality or closeness label.
7. **Disclosure policy:** information category by audience by channel, plus purpose, consent, expiry, subject ownership, and a confirmation or segmentation gate for mixed audiences. Conservative action under uncertainty is a safety policy, not an inferred behavioral effect.
8. **Decision trace:** every relationship-conditioned generation should expose which evidence, boundary rule, and uncertainty influenced it.

### Speculative hypotheses to test

- A relationship-scoped retrieval layer will improve owner-rated fidelity more than a single global persona without increasing unsupported contact inference.
- Explicit audience objects will reduce boundary violations in group generation compared with relying on chat membership alone.
- Partner-scoped common-ground retrieval will improve naturalness, but only if stale or ambiguous pacts trigger clarification rather than automatic reuse.
- Accommodation constrained by discourse act will sound more authentic than unconditional style matching and will reduce caricature or mimicry.

## WhatsApp Observability Gaps

This seat did not inspect WhatsApp content. Even with owner-authorized messages in a later phase, WhatsApp would expose only partial evidence:

| Potentially observable | Not safely inferable without owner input |
| --- | --- |
| Direct recipient or current group membership | The imagined audience, screenshots, forwarding, device sharing, or who the owner expects will hear about the message |
| Repeated terms, nicknames, reply structure, and local style shifts | Whether repetition is a true conceptual pact, a quotation, irony, coercion, or accidental similarity |
| Explicit role words such as colleague, cousin, or client | The current salience, quality, hierarchy, or emotional meaning of that role |
| Topic withholding or movement to a direct chat | The underlying privacy rule, whether the move was requested, and whether it applies beyond that episode |
| Stated consent or an explicit request not to share | Consent for a new purpose, audience, time, or channel; silence is not consent |
| Group and direct-chat differences | Whether the behavior reflects audience design, platform affordances, task demands, safety concerns, or a temporary state |
| Turn-level language matching | Relationship quality, power, attraction, approval, or personality |
| Messages that mention third parties | The third party's permission, preferred audience, or tolerance for retention in a digital-brain store |

These gaps should become owner questions rather than inferred profile fields. High-value prompts include: "What role does this person play for you now?", "Which topics are private to this relationship?", "Who else might receive or hear this?", "Is this nickname or shorthand still current?", "May information about the other person be retained or reused?", and "When this chat conflicts with another context, which boundary should win?"

## Open Questions

1. How should the system distinguish a durable relational-self pattern from a temporary mood, task role, or politeness accommodation?
2. What minimum evidence and owner confirmation should be required before creating or updating a relationship role?
3. How should conceptual pacts expire, branch across group and direct chats, or survive relationship change?
4. Can discourse-conditioned accommodation improve fidelity without drifting into imitation of contacts or amplifying unhealthy dynamics?
5. How should multilingual code choice and code-switching be represented when audience, topic, and identity cues conflict?
6. What consent protocol is workable for memories that mention multiple people, especially when those people cannot participate in the system?
7. How should intended, permitted, and observed audiences be evaluated when forwarding and offline retelling are unobservable?
8. Which evaluation set can test the same owner's behavior across friends, family, work, strangers, and mixed groups without treating one context as ground truth for all others?
9. How can owner correction update one relationship boundary without silently rewriting global identity or unrelated relationships?
10. What uncertainty threshold should force clarification or abstention before relationship-sensitive generation?

## Bottom Line

The evidence supports a digital brain that is stable at the owner level but explicitly conditional at the relationship, role, audience, channel, and episode levels. The central safety result is equally important: relationship-conditioned behavior is not permission to infer contact traits, and previous disclosure is not permission to repeat information. The architecture should preserve provenance, uncertainty, and information boundaries, then ask the owner when WhatsApp-observable behavior cannot distinguish among plausible social explanations.
