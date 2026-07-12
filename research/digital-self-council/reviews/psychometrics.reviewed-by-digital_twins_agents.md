# Psychometrics Cross-Review

- **Subject seat:** `psychometrics`
- **Reviewer seat:** `digital_twins_agents`
- **Reviewer agent:** `019f4f07-5630-7e92-88b1-8534576fbd2e`
- **Reviewed at:** `2026-07-11T08:59:28Z`
- **Decision:** **APPROVE**

All 20 counted records were checked against their primary paper surfaces at the declared inspection depth. Verification used publisher or journal pages, DOI records, PubMed/PMC, ACL Anthology, and official full-text proceedings or article PDFs. No WhatsApp content, private profiles, local databases, or keys were accessed.

## 1. Reviewed Record IDs

| Record ID | Declared depth | Primary-source check | Disposition |
|---|---|---|---|
| `paper:psychometrics_epstein_1983` | abstract | PubMed metadata, DOI, abstract claim | approve |
| `paper:psychometrics_fleeson_2001` | abstract | PubMed metadata, DOI, abstract methods and findings | approve |
| `paper:psychometrics_sherman_2015` | abstract | PubMed metadata, DOI, sample and person/situation effects | approve |
| `paper:psychometrics_pennebaker_king_1999` | abstract | PubMed metadata, DOI, study sequence and factor result | approve |
| `paper:psychometrics_mehl_2006` | abstract | PubMed metadata, DOI, EAR sample and observer/self-report distinction | approve |
| `paper:psychometrics_borkenau_2004` | abstract | PubMed metadata, DOI, episode-sampling result | approve |
| `paper:psychometrics_mairesse_2007` | full_text | JAIR article page and complete PDF, datasets, tasks, baselines, limitations | approve |
| `paper:psychometrics_yarkoni_2010` | full_text | Journal article full text, sample, instruments, reliability, limitations | approve |
| `paper:psychometrics_schwartz_2013` | full_text | PLOS/PMC full text, sample restrictions, features, split-sample evaluation | approve |
| `paper:psychometrics_park_2015` | abstract | PubMed metadata, DOI, development and independent validation samples | approve |
| `paper:psychometrics_rieman_2017` | full_text | ACL Anthology page and PDF, county-year counts, transfer method, failures | approve |
| `paper:psychometrics_kosinski_2013` | full_text | PNAS/PMC full text, sample, 20-item IPIP, SVD and cross-validation | approve |
| `paper:psychometrics_youyou_2015` | full_text | PNAS/PMC full text, samples, instruments, human/computer comparison | approve |
| `paper:psychometrics_stachl_2020` | full_text | PNAS/PMC full text, event count, 300-item instrument, nested evaluation | approve |
| `paper:psychometrics_montag_2015` | full_text | BMC/PMC full text, app-session logging, sample, BFI-10 and effect sizes | approve |
| `paper:psychometrics_soto_2008` | abstract | PubMed metadata, DOI, age, coherence, and acquiescence findings | approve |
| `paper:psychometrics_nye_2008` | abstract | Publisher metadata and abstract, country samples and invariance result | approve |
| `paper:psychometrics_rammstedt_2013` | abstract | Publisher metadata and abstract, 18-country response-style result | approve |
| `paper:psychometrics_laajaj_2019` | full_text | Science Advances/PMC full text, dataset composition and measurement artifacts | approve |
| `paper:psychometrics_mezquita_2019` | full_text | PLOS full text, three-country sample, ESEM and invariance sequence | approve |

## 2. Identifier and Metadata Corrections

No identifier or bibliographic correction was required. The corpus contains 20 unique record IDs and 20 unique canonical papers: 19 DOI records and one ACL Anthology record. Every primary URL is HTTPS and resolves to a primary paper surface.

Several ambiguity-prone details were checked explicitly and are already correct: the plural title in `paper:psychometrics_fleeson_2001`; the 20-item IPIP measure in `paper:psychometrics_kosinski_2013`; and the 2,197-to-2,651 county range in `paper:psychometrics_rieman_2017`.

## 3. Overclaims and Wording Changes

No wording change was required. The records consistently separate association from diagnosis and prediction from construct validity.

- `paper:psychometrics_mairesse_2007` preserves the negative result that self-report speech models did not beat the baseline and distinguishes observer impressions from self-report targets.
- `paper:psychometrics_youyou_2015` qualifies its human-versus-computer comparison with the shorter human rating instrument and partial-information result.
- `paper:psychometrics_stachl_2020` reports uneven predictability, including the lack of successful agreeableness prediction, rather than generalizing across all traits.
- `paper:psychometrics_montag_2015` describes small correlations and does not treat WhatsApp duration as an individual diagnostic.
- `paper:psychometrics_rieman_2017` keeps county-level ecological estimates separate from individual personality inference.
- `paper:psychometrics_fleeson_2001` and `paper:psychometrics_sherman_2015` retain the person-by-situation/state evidence that limits fixed-trait readings of a single observation.

## 4. Missing Counterevidence or Foundational Work

No blocking counterevidence or foundational omission was found for this bounded seat. The corpus includes direct counterweights to its positive prediction results:

- Within-person state variability and situational contribution: `paper:psychometrics_fleeson_2001`, `paper:psychometrics_sherman_2015`, and `paper:psychometrics_borkenau_2004`.
- Weak, trait-specific, or target-dependent prediction: `paper:psychometrics_mehl_2006`, `paper:psychometrics_mairesse_2007`, `paper:psychometrics_montag_2015`, and `paper:psychometrics_stachl_2020`.
- Context and domain-transfer failure: `paper:psychometrics_yarkoni_2010`, `paper:psychometrics_schwartz_2013`, and `paper:psychometrics_rieman_2017`.
- Cross-cultural measurement noninvariance and response artifacts: `paper:psychometrics_nye_2008`, `paper:psychometrics_rammstedt_2013`, `paper:psychometrics_laajaj_2019`, and `paper:psychometrics_mezquita_2019`.

A residual research gap remains, but is not a correction to this seat: these papers do not directly validate longitudinal calibration and update policies for an evolving digital self-model. That claim should therefore remain an open engineering question rather than be inferred from the present corpus.

## 5. Duplicate Candidates

There are no duplicate publications, duplicate canonical identifiers, or preprint/published double counts.

There are two evidence-dependency clusters that must not be counted as independent replications:

- `paper:psychometrics_mairesse_2007` reuses the essay data from `paper:psychometrics_pennebaker_king_1999` and the EAR speech data from `paper:psychometrics_mehl_2006`.
- `paper:psychometrics_kosinski_2013`, `paper:psychometrics_youyou_2015`, `paper:psychometrics_schwartz_2013`, and `paper:psychometrics_park_2015` draw from or are tightly coupled to the myPersonality/Facebook research ecosystem; `paper:psychometrics_rieman_2017` transfers a related Facebook-trained model to county-level Twitter data.

These are legitimate distinct papers with different questions or evaluations, but synthesis must preserve their shared-data lineage.

## 6. Architecture Claims to Downgrade or Reject

No record-level architecture implication requires downgrade or rejection. The implications are framed as constraints or design hypotheses rather than as demonstrated system outcomes.

The evidence supports keeping state estimates separate from trait summaries (`paper:psychometrics_fleeson_2001`, `paper:psychometrics_sherman_2015`), aggregating multiple observations and sources (`paper:psychometrics_epstein_1983`, `paper:psychometrics_borkenau_2004`), preserving context and domain calibration (`paper:psychometrics_yarkoni_2010`, `paper:psychometrics_rieman_2017`), and exposing uncertainty across populations (`paper:psychometrics_nye_2008`, `paper:psychometrics_laajaj_2019`). Privacy warnings about seemingly innocuous traces are also evidence-backed (`paper:psychometrics_kosinski_2013`, `paper:psychometrics_montag_2015`).

The corpus does **not** support a universal or context-free personality decoder. None of the reviewed records makes that architecture claim.

## 7. Final Decision

**APPROVE.** All 20 records are source-verifiable at their declared depth, use unique canonical identifiers, preserve material negative evidence, and keep architecture implications within the evidence. No factual edit to `seat-notes/psychometrics.md` is required. The only seat-file change is the approved provenance review metadata applied with the single timestamp above.
