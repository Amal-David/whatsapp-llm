# Social Relationships Cross-Review

- Subject seat: `social_relationships`
- Reviewer seat: `persona_dialogue`
- Reviewer agent ID: `019f4f07-0534-77c3-af91-f9f55fb1356f`
- Extractor: `council-social_relationships` (independent from reviewer)
- Reviewed records: 20 of 20
- Review timestamp: `2026-07-11T04:13:24Z`
- Final decision: **APPROVE after applied corrections**

## 1. Reviewed Record IDs

Every counted record was checked against a primary paper surface. The declared depth mix remains 14 abstract and 6 full text.

| # | Record ID | Depth | Review outcome |
| ---: | --- | --- | --- |
| 1 | `paper:andersen-chen-2002-relational-self` | abstract | Pass |
| 2 | `paper:postmes-spears-sakhel-degroot-2001-social-influence` | abstract | Pass |
| 3 | `paper:trefalt-2013-between-you-and-me` | abstract | Pass |
| 4 | `paper:bell-1984-audience-design` | abstract | Pass |
| 5 | `paper:youssef-1993-childrens-linguistic-choices` | abstract | Pass |
| 6 | `paper:horton-gerrig-2002-speakers-experiences` | abstract | Pass |
| 7 | `paper:brennan-clark-1996-conceptual-pacts` | abstract | Pass |
| 8 | `paper:metzing-brennan-2003-conceptual-pacts-broken` | abstract | Pass after mechanism and counterevidence correction |
| 9 | `paper:ireland-et-al-2011-language-style-matching` | abstract | Pass after metric-direction correction |
| 10 | `paper:danescu-niculescu-mizil-et-al-2011-mark-my-words` | full text | Pass after long-horizon boundary correction |
| 11 | `paper:doyle-frank-2016-sources-linguistic-alignment` | full text | Pass |
| 12 | `paper:tice-et-al-1995-when-modesty-prevails` | abstract | Pass |
| 13 | `paper:ellison-heino-gibbs-2006-managing-impressions` | full text | Pass |
| 14 | `paper:marwick-boyd-2011-context-collapse` | abstract | Pass |
| 15 | `paper:litt-hargittai-2016-imagined-audience` | full text | Pass |
| 16 | `paper:bernstein-et-al-2013-invisible-audience` | full text | Pass |
| 17 | `paper:vitak-2012-context-collapse-privacy` | abstract | Pass after audience-policy and replication correction |
| 18 | `paper:bazarova-2012-public-intimacy` | abstract | Pass |
| 19 | `paper:lampinen-et-al-2011-were-in-it-together` | full text | Pass |
| 20 | `paper:acquisti-gross-2006-imagined-communities` | abstract | Pass |

## 2. Identifier Or Metadata Corrections

No canonical identifier, primary URL, title, author list, issue year, venue, or publication-type correction was required.

- All 20 canonical DOIs resolve to the claimed publisher or proceedings record. `paper:doyle-frank-2016-sources-linguistic-alignment` correctly uses the ACL Anthology paper page as its primary HTTPS URL while retaining the DOI canonically.
- PubMed aliases `12374322`, `8921603`, and `21149854`, arXiv alias `1105.0673`, and ACL alias `P16-1050` identify the same works as their parent records.
- The 2011 issue years for `paper:ireland-et-al-2011-language-style-matching` and `paper:marwick-boyd-2011-context-collapse` are retained; their publisher records also show 2010 online-first dates.
- `paper:acquisti-gross-2006-imagined-communities` remains typed as `workshop`: Springer labels the item generically as a conference paper, but the proceedings explicitly identify PET 2006 as the 6th International Workshop on Privacy Enhancing Technologies.
- The six `full_text` declarations are defensible. Methods, data, results, and limitations were checked for `paper:danescu-niculescu-mizil-et-al-2011-mark-my-words`, `paper:doyle-frank-2016-sources-linguistic-alignment`, `paper:ellison-heino-gibbs-2006-managing-impressions`, `paper:litt-hargittai-2016-imagined-audience`, `paper:bernstein-et-al-2013-invisible-audience`, and `paper:lampinen-et-al-2011-were-in-it-together`.

## 3. Overclaims And Required Wording Changes

Four records required changes; all were applied before reviewer provenance was set.

1. `paper:metzing-brennan-2003-conceptual-pacts-broken`: changed the finding from a conclusive mechanism claim to evidence "consistent with" a partner-specific expectation. The architecture implication now prioritizes a meaning-change check without claiming it is unnecessary for a new partner.
2. `paper:ireland-et-al-2011-language-style-matching`: corrected the architecture implication because the reported LSM measure is symmetric dyadic similarity, not directional accommodation. Direction now requires temporally conditioned reply analysis, consistent with `paper:doyle-frank-2016-sources-linguistic-alignment`.
3. `paper:danescu-niculescu-mizil-et-al-2011-mark-my-words`: added the explicit boundary that the metric concerns instant turn-to-turn accommodation. The paper leaves long-term accommodation and relationship-development drift unresolved.
4. `paper:vitak-2012-context-collapse-privacy`: removed the behavioral overclaim that a mixed audience should imply less disclosure. The architecture now requests an audience check or segmentation option, and the limitation distinguishes this safety policy from the observed associations.

The other 16 core findings are restrained paraphrases at their declared inspection depth. Their architecture implications are traceable to the evidence and do not require wording changes.

## 4. Missing Counterevidence Or Foundational Work

Two missing counterevidence items materially bounded existing claims and were added to the records and synthesis:

- For `paper:metzing-brennan-2003-conceptual-pacts-broken`, Kronmüller and Barr's later experiment, ["Perspective-free pragmatics: Broken precedents and the recovery-from-preemption hypothesis"](https://doi.org/10.1016/j.jml.2006.05.002), argues that the speaker effect may arise during late recovery from partner-independent preemption rather than early partner-specific common-ground access. This changes the mechanism interpretation, not the observed same-partner delay.
- For `paper:vitak-2012-context-collapse-privacy`, Masur and Ranzini's preregistered close replication, ["Privacy calculus, privacy paradox, and context collapse: A replication of three key studies in communication privacy research"](https://doi.org/10.1093/joc/jqaf007), found the audience-composition paths replicated with small effects, while the privacy-concern-to-disclosure association did not replicate and reversed direction.

The slice already contains meaningful internal counterevidence: `paper:youssef-1993-childrens-linguistic-choices` bounds a fixed audience hierarchy; `paper:horton-gerrig-2002-speakers-experiences` bounds immediate audience adaptation; `paper:doyle-frank-2016-sources-linguistic-alignment` separates similarity from adaptation and reports lexical/discourse-act dependence; and `paper:bernstein-et-al-2013-invisible-audience` separates visible engagement from reach. No missing foundational paper blocks approval.

## 5. Duplicate Candidates

No duplicate candidate involves `social_relationships`.

- Canonical and alternate identifier comparison found 20 unique canonical papers and no collision between a subject-seat identifier and another council seat.
- `paper:danescu-niculescu-mizil-et-al-2011-mark-my-words` correctly counts the published WWW paper once and stores arXiv `1105.0673` only as an alternate ID.
- `paper:doyle-frank-2016-sources-linguistic-alignment` correctly stores DOI and ACL identifiers on one record.

## 6. Architecture Claims Downgraded Or Rejected

- **Downgraded:** `paper:metzing-brennan-2003-conceptual-pacts-broken` no longer treats same-partner wording changes as proof of early common-ground lookup or says a new partner makes checking unnecessary.
- **Corrected:** `paper:ireland-et-al-2011-language-style-matching` no longer turns symmetric aggregate similarity into a directional time-series measurement.
- **Rejected as an empirical inference:** `paper:vitak-2012-context-collapse-privacy` does not show that mixed audiences make people disclose less. A confirmation or segmentation gate remains defensible as a safety policy.
- **Bounded:** `paper:danescu-niculescu-mizil-et-al-2011-mark-my-words` does not support long-horizon accommodation, persona drift, or relationship-stage inference.

No architecture claim licenses contact-trait inference, relationship-health scoring, unconsented imitation, or reuse of third-party information. The remaining architecture claims are approved as scoped design implications rather than claims that the papers directly evaluated a digital-brain system.

## 7. Final Decision

**APPROVE.** All 20 records identify real, unique primary papers; metadata and primary URLs are accurate; declared inspection depths are honest; methods, data, modalities, findings, limitations, evidence strengths, and relevance scores are defensible after the four corrections above. The extractor and reviewer are independent. No WhatsApp content, private profiles, local databases, credentials, or keys entered the review.

Only after these corrections cleared review, every record received:

- `provenance.reviewer`: `019f4f07-0534-77c3-af91-f9f55fb1356f`
- `provenance.reviewed_at`: `2026-07-11T04:13:24Z`

## 8. Review-Enforced Validation

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -c 'from living_brain.research import load_jsonl, validate_corpus; p="research/digital-self-council/seats/social_relationships.jsonl"; print(validate_corpus(load_jsonl(p), min_count=15, max_count=25, require_reviewed=True).to_dict())'
```

Exact result:

```text
{'total_records': 20, 'counted_primary_papers': 20, 'unique_canonical_papers': 20, 'reviewed_records': 20, 'seat_counts': {'social_relationships': 20}, 'publication_type_counts': {'conference': 4, 'journal': 15, 'workshop': 1}, 'inspection_depth_counts': {'abstract': 14, 'full_text': 6}}
```

Provenance audit: 20 records, one reviewer value, one reviewed-at value, and zero null reviewer or reviewed-at fields.
