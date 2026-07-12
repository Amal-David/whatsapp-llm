# Psychometrics and Computational Personality

## Scope and result

This seat covers language-based personality inference, computational psychometrics, behavioral traces, state-versus-trait validity, cross-context stability, measurement invariance, and confounding. It contains 20 counted primary papers: 19 journal articles and one conference paper. Ten records were inspected at full-text depth and ten at abstract depth. No paper is marked cross-reviewed, and no WhatsApp content, profile, local database, or credential was accessed.

The central conclusion is deliberately narrower than "messages reveal personality." Repeated language and behavioral traces can predict particular questionnaire or observer criteria in some sampled populations. They do not, by themselves, establish a context-free trait, an inner motive, a causal mechanism, or a complete representation of a person. The measurement target, observation window, relationship, platform, language, population, and elicitation process remain part of every defensible interpretation.

## Search strategy

Searches combined exact-title and identifier lookup with topic queries for personality prediction from language, social-media traces, smartphone sensing, state-trait models, behavioral aggregation, cross-cultural measurement equivalence, acquiescence, and domain adaptation. Discovery was followed by inspection only on accepted primary surfaces: PubMed or PMC, publisher article pages and DOI records, PLOS, the Journal of Artificial Intelligence Research, and ACL Anthology proceedings.

The search moved in five passes:

1. Foundational psychometrics and person-situation work: aggregation, density distributions of states, and simultaneous person/situation effects [paper:psychometrics_epstein_1983; paper:psychometrics_fleeson_2001; paper:psychometrics_sherman_2015].
2. Observable behavior and language as individual-difference signals: lexical reliability, naturalistic speech, thin slices, and computational recognition [paper:psychometrics_pennebaker_king_1999; paper:psychometrics_mehl_2006; paper:psychometrics_borkenau_2004; paper:psychometrics_mairesse_2007].
3. Large-scale social-language and digital-footprint models: blogs, Facebook language and Likes, cross-platform transfer, and smartphone logs [paper:psychometrics_yarkoni_2010; paper:psychometrics_schwartz_2013; paper:psychometrics_park_2015; paper:psychometrics_rieman_2017; paper:psychometrics_kosinski_2013; paper:psychometrics_youyou_2015; paper:psychometrics_stachl_2020].
4. Directly logged WhatsApp usage metadata, selected because it tests what app-use behavior can and cannot support without reading message content [paper:psychometrics_montag_2015].
5. Measurement invariance and response-process confounding across age, language, culture, education, country, gender, and administration mode [paper:psychometrics_soto_2008; paper:psychometrics_nye_2008; paper:psychometrics_rammstedt_2013; paper:psychometrics_laajaj_2019; paper:psychometrics_mezquita_2019].

## Inclusion and exclusion

Included records had a verified canonical DOI or ACL identifier, an accurate primary HTTPS paper URL, enough primary-source detail to extract method and findings honestly, and direct relevance to this seat. The set intentionally includes foundational methodological evidence and modern out-of-sample modeling, positive results and failures, self-report and observer criteria, text and non-text traces, and both supportive and adverse invariance findings.

Excluded material included blogs, product pages, GitHub documentation, reviews used only as discovery aids, books and chapters, unverifiable papers, duplicate preprint/published versions, clinical diagnosis, and protected-trait inference as a proposed capability. Papers were also omitted when a result could not be checked beyond secondary summaries.

Several records are unique publications but not independent datasets. The essay and naturalistic speech corpora in [paper:psychometrics_mairesse_2007] come from [paper:psychometrics_pennebaker_king_1999] and [paper:psychometrics_mehl_2006]. The Facebook studies [paper:psychometrics_schwartz_2013], [paper:psychometrics_park_2015], [paper:psychometrics_kosinski_2013], and [paper:psychometrics_youyou_2015] draw from the myPersonality research ecosystem, and [paper:psychometrics_rieman_2017] transfers a related Facebook language model. They must not be counted as five fully independent replications.

## Domain taxonomy

| Domain | Measurement question | Core records |
| --- | --- | --- |
| Aggregation and trait-state structure | How much repeated evidence is needed, and what is lost by flattening occasions? | [paper:psychometrics_epstein_1983; paper:psychometrics_fleeson_2001; paper:psychometrics_sherman_2015] |
| Observable personality and reputation | What can strangers or sensors judge from sampled behavior? | [paper:psychometrics_mehl_2006; paper:psychometrics_borkenau_2004] |
| Language-based assessment | Which textual features predict self- or observer-rated traits, and under what domain? | [paper:psychometrics_pennebaker_king_1999; paper:psychometrics_mairesse_2007; paper:psychometrics_yarkoni_2010; paper:psychometrics_schwartz_2013; paper:psychometrics_park_2015; paper:psychometrics_rieman_2017] |
| Behavioral traces | What do Likes, phone sensing, and usage metadata predict without semantic content? | [paper:psychometrics_kosinski_2013; paper:psychometrics_youyou_2015; paper:psychometrics_stachl_2020; paper:psychometrics_montag_2015] |
| Invariance and response processes | Is the same score measuring the same construct across people and settings? | [paper:psychometrics_soto_2008; paper:psychometrics_nye_2008; paper:psychometrics_rammstedt_2013; paper:psychometrics_laajaj_2019; paper:psychometrics_mezquita_2019] |

## Evidence-backed synthesis

### Traits are distributions, not message labels

The best-supported state-trait account is hierarchical. Broad dispositions are more reliably represented by aggregates sampled across occasions, while a single act remains narrow and situation-specific [paper:psychometrics_epstein_1983]. Within a person, momentary Big Five-relevant behavior ranges widely, even when the person's central tendency and variability are stable [paper:psychometrics_fleeson_2001]. Real-time behavior and emotion also reflect independent situation effects [paper:psychometrics_sherman_2015]. A digital brain should therefore represent a trait as a posterior distribution over repeated states, not stamp a trait onto each message.

### Prediction depends on the criterion

Language can be a reliable individual-difference signal, but its criterion matters. Early lexical work found replicable language dimensions with mostly modest personality associations [paper:psychometrics_pennebaker_king_1999]. Naturalistic recordings showed that visible traits, especially extraversion, are easier for observers to judge than less visible attributes [paper:psychometrics_mehl_2006]. Thin-slice accuracy improved when episodes were aggregated, yet observer impressions remained limited by nonoverlapping information [paper:psychometrics_borkenau_2004].

The computational result sharpens this distinction. Models trained on observer-rated conversation performed better than models trained on self-report in the small speech corpus; self-report speech models did not beat baseline there [paper:psychometrics_mairesse_2007]. That is evidence about reputation from observable cues, not proof that an interaction model has recovered private identity. A digital brain needs separate views for self-description, observer reputation, and trace-based prediction.

### More text improves reliability, not metaphysical access

Long naturalistic histories can support stable lexical estimates. The blog study averaged more than 100,000 words per author over nearly two years and recovered many language-personality associations, especially for openness [paper:psychometrics_yarkoni_2010]. Large Facebook studies showed that open-vocabulary words, phrases, and topics outperform fixed lexicons for prediction [paper:psychometrics_schwartz_2013], and a separate evaluation found convergence with self-report, informants, external correlates, and six-month stability [paper:psychometrics_park_2015].

These are validated predictive measurements within their study designs. They do not identify one psychological cause for a word pattern. Topic choice, age, gender, education, vocabulary, audience, current events, and platform conventions can all carry the same signal. Even the strong Facebook result should be treated as a complementary assessment calibrated to its source population, not as direct access to enduring motives or a complete self [paper:psychometrics_schwartz_2013; paper:psychometrics_park_2015].

### Cross-context transfer is a separate validation problem

The clearest transportability warning comes from moving a user-level Facebook model to county-level Twitter text. Naive reuse produced implausible outliers and spurious associations driven by local vocabulary; adaptation improved distributions and year-to-year stability, but most target counties still lacked direct personality labels [paper:psychometrics_rieman_2017]. Platform, unit of analysis, time period, language, and local memes are not implementation details. They define a new measurement setting that requires new evidence.

### Behavioral traces are predictive but proxy-rich

Facebook Likes predicted some Big Five scores, especially openness and extraversion, while simultaneously encoding demographics, interests, social structure, and sensitive attributes [paper:psychometrics_kosinski_2013]. Like-based models exceeded a single friend's brief judgment on several reported criteria, but the computer and friend channels retained largely different information and had unequal inputs and instruments [paper:psychometrics_youyou_2015]. Accuracy against a questionnaire target is useful; it does not turn the proxy into the construct.

Smartphone sensing reached moderate out-of-sample correlations for some domains and facets, but agreeableness was not predicted at all; communication metadata and app use were the strongest behavior classes in a young German Android sample [paper:psychometrics_stachl_2020]. Direct WhatsApp-use duration had small trait associations, no clear openness or agreeableness signal, and much stronger age and gender patterns [paper:psychometrics_montag_2015]. These findings support per-trait calibration and explicit abstention. They reject the idea of one universal behavioral personality decoder.

### Invariance is an empirical gate

Questionnaire targets themselves are conditional measurements. Acquiescent responding varies by age and changes apparent coherence and differentiation [paper:psychometrics_soto_2008]. Big Five adjective scales can preserve a broad factor configuration while failing metric and scalar invariance across U.S., Greek, and Chinese samples [paper:psychometrics_nye_2008]. Correcting acquiescence improves factor recovery across countries but does not solve every non-individualistic setting [paper:psychometrics_rammstedt_2013].

Large non-WEIRD survey evidence further shows that administration mode, education, enumerator interaction, translation, response style, and self-selection can jointly blur the intended factors [paper:psychometrics_laajaj_2019]. More encouragingly, a different 50-item instrument achieved reasonable or partial invariance across English and Spanish college samples, countries, and genders when the model allowed cross-loadings [paper:psychometrics_mezquita_2019]. The apparent conflict is informative: invariance is instrument-, population-, model-, and mode-specific, not a permanent property of "the Big Five."

## Negative and conflicting evidence

- More data does not rescue an undefined target. Observer-rated conversational personality was easier to model than self-rated personality in one corpus, demonstrating that identity and reputation are different criteria [paper:psychometrics_mairesse_2007].
- Trait observability is uneven. Openness was unusually legible in blogs and Likes [paper:psychometrics_yarkoni_2010; paper:psychometrics_kosinski_2013], whereas agreeableness was not predictable from smartphone traces and showed no clear association with WhatsApp duration [paper:psychometrics_stachl_2020; paper:psychometrics_montag_2015].
- Six-month language stability supports medium-term reliability, not lifelong immutability or cross-relationship stability [paper:psychometrics_park_2015; paper:psychometrics_fleeson_2001].
- A broad factor pattern can coexist with biased loadings or intercepts. Strict failures in one cross-cultural adjective instrument [paper:psychometrics_nye_2008] and partial support in another questionnaire [paper:psychometrics_mezquita_2019] should not be averaged into a generic verdict.
- Acquiescence correction helps but is incomplete, and face-to-face measurement failures had multiple plausible causes rather than one cultural explanation [paper:psychometrics_rammstedt_2013; paper:psychometrics_laajaj_2019].
- The headline that a computer outjudged humans used many Likes for the model, a brief instrument for friends, and self-report as the central criterion. It supports predictive utility under those conditions, not privileged psychological access [paper:psychometrics_youyou_2015].
- Large Facebook results share a data ecosystem, and the early language classifier reused two earlier corpora. Evidence volume must not be mistaken for independent replication [paper:psychometrics_pennebaker_king_1999; paper:psychometrics_mehl_2006; paper:psychometrics_mairesse_2007; paper:psychometrics_schwartz_2013; paper:psychometrics_park_2015; paper:psychometrics_kosinski_2013; paper:psychometrics_youyou_2015].

## Implications for a digital brain

**Validated constraints**

1. Keep raw observations, context, momentary-state estimates, and trait distributions as separate layers. Aggregate only over a declared domain and retain within-person variability [paper:psychometrics_epstein_1983; paper:psychometrics_fleeson_2001; paper:psychometrics_sherman_2015].
2. Maintain separate self-report, owner-corrected, informant/reputation, and behavioral-prediction views. Do not silently treat one as ground truth for the others [paper:psychometrics_borkenau_2004; paper:psychometrics_mairesse_2007; paper:psychometrics_youyou_2015].
3. Attach every estimate to source platform, language, time window, relationship or audience, evidence count, criterion, calibration population, and uncertainty [paper:psychometrics_park_2015; paper:psychometrics_rieman_2017].
4. Gate cross-context reuse on target-domain reliability, drift, invariance, and known-null checks. An unsupported context returns "unknown," not a forced score [paper:psychometrics_rieman_2017; paper:psychometrics_nye_2008; paper:psychometrics_mezquita_2019].
5. Treat demographic, response-style, education, elicitor, and opportunity variables as confounds to test, not convenient personality features [paper:psychometrics_schwartz_2013; paper:psychometrics_montag_2015; paper:psychometrics_laajaj_2019].

**Plausible design hypotheses**

- A hierarchical model with relationship-conditioned states and a slowly updated trait prior is more defensible than one global persona vector. This follows from aggregation, state distributions, and situation effects, but WhatsApp-specific validation remains to be done [paper:psychometrics_epstein_1983; paper:psychometrics_fleeson_2001; paper:psychometrics_sherman_2015].
- Owner interviews and corrections can serve as active measurement, especially when language evidence is sparse, noninvariant, or contradicted across contexts. The reviewed papers support multimethod validation, but not a particular interview protocol [paper:psychometrics_park_2015; paper:psychometrics_mezquita_2019].
- Per-trait abstention and evidence thresholds should outperform a single all-trait confidence score because observability differs materially by trait and modality [paper:psychometrics_mairesse_2007; paper:psychometrics_stachl_2020].

**Not established**

No reviewed study establishes that private interaction text alone can recover an owner's true personality, values, motives, clinical condition, protected attributes, or behavior in unseen relationships. Predictive agreement with a questionnaire or observer is not proof of a complete digital self.

## WhatsApp observability gaps

This seat did not inspect WhatsApp content. One included study deliberately analyzed app-use duration without message content and found demographic effects larger than most trait associations [paper:psychometrics_montag_2015]. That boundary illustrates how little usage metadata establishes.

Even with explicit consent to analyze messages, a WhatsApp archive would observe only authored or retained communication in selected relationships. It would miss unsent thoughts, silent periods, offline behavior, nonverbal conduct, many situational cues, deleted material, and contexts where the owner uses other channels. Message content is also jointly shaped by the recipient, relationship history, group norms, current topic, code-switching, audience design, platform affordances, and the other participant's prior turn [paper:psychometrics_sherman_2015; paper:psychometrics_rieman_2017].

The archive would not automatically provide a valid criterion. It lacks standardized owner self-report, equivalent informant ratings, known invariance across languages and relationships, and counterfactual behavior outside messaging. A model could learn who is addressed, what topics recur, or when the person writes and mistake those opportunity structures for personality [paper:psychometrics_schwartz_2013; paper:psychometrics_kosinski_2013; paper:psychometrics_laajaj_2019].

For a digital brain, the defensible output is therefore relationship- and time-bounded evidence with uncertainty: "in this context, across these observations, this pattern recurred." The evidence does not license "this is what the person is" without multimethod, owner-approved validation.

## Open questions

1. What minimum number of messages, occasions, relationships, and elapsed time yields stable estimates for each specific construct, and when do gains saturate [paper:psychometrics_epstein_1983; paper:psychometrics_borkenau_2004]?
2. Which temporal model best separates momentary state, recurring situation-response contingency, medium-term phase, and slow trait change [paper:psychometrics_fleeson_2001; paper:psychometrics_park_2015]?
3. Which target should each capability predict: owner self-concept, close-other reputation, future behavior, or relationship-specific behavior, and how should disagreements remain visible [paper:psychometrics_mairesse_2007; paper:psychometrics_youyou_2015]?
4. Can language-derived measurements demonstrate metric and scalar invariance across the owner's languages, code-switching patterns, age periods, and relationships [paper:psychometrics_nye_2008; paper:psychometrics_mezquita_2019]?
5. How can a model distinguish the owner's signal from partner elicitation, topic, group culture, platform conventions, and demographic proxies [paper:psychometrics_sherman_2015; paper:psychometrics_montag_2015; paper:psychometrics_laajaj_2019]?
6. What held-out, future-facing criteria can test behavioral fidelity without circularly rewarding imitation of the same archive used to build the profile [paper:psychometrics_rieman_2017; paper:psychometrics_stachl_2020]?
7. What consent, expiration, correction, and no-inference controls should govern even non-content behavioral traces [paper:psychometrics_kosinski_2013; paper:psychometrics_stachl_2020]?
