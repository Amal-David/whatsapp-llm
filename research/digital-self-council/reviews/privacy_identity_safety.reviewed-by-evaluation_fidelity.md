# Cross-review: `privacy_identity_safety`

- Subject seat: `privacy_identity_safety`
- Reviewer seat: `evaluation_fidelity`
- Reviewer agent: `019f4f07-3f74-7193-a53e-6b392adaf6f0`
- Reviewed at: `2026-07-11T08:56:25Z`
- Records reviewed: 20 counted primary papers
- Inspection-depth mix: 17 `full_text`, 3 `abstract`
- Decision: **APPROVE AFTER CORRECTION**

The review checked every record against a primary publisher page, DOI registration, official proceedings page, arXiv paper, or PubMed/PMC record at the declared inspection depth. It did not use blogs, product pages, repository documentation, private profiles, messages, local databases, or a duplicate publication as independent evidence.

## 1. Reviewed record IDs

| Record | Depth | Primary-source check | Outcome |
| --- | --- | --- | --- |
| `paper:abadi-2016-deep-learning-dp` | `full_text` | ACM DOI and arXiv paper | Approved as written. |
| `paper:shokri-2017-membership-inference` | `full_text` | IEEE DOI and arXiv paper | Approved as written. |
| `paper:carlini-2021-training-data-extraction` | `full_text` | USENIX proceedings page and paper | Approved as written. |
| `paper:kandpal-2022-dedup-privacy` | `full_text` | PMLR proceedings page and paper | Approved as written. |
| `paper:lukas-2023-pii-leakage` | `full_text` | IEEE DOI registration and arXiv paper | Approved as written; the submitted IEEE DOI is the resolving identifier. |
| `paper:bourtoule-2021-machine-unlearning` | `full_text` | IEEE DOI and arXiv paper | Approved as written. |
| `paper:mireshghallah-2024-confaide` | `full_text` | ICLR OpenReview page and arXiv paper | Approved as written. |
| `paper:jia-2018-attriguard` | `full_text` | USENIX proceedings page and paper | Approved as written. |
| `paper:kosinski-2013-private-traits` | `full_text` | PNAS DOI, PubMed, and PMC full text | Approved as written. |
| `paper:jernigan-2009-gaydar` | `full_text` | First Monday DOI page and full article | Approved as written. |
| `paper:gong-2016-attribute-inference` | `full_text` | USENIX proceedings page and paper | Approved as written. |
| `paper:buolamwini-2018-gender-shades` | `full_text` | PMLR proceedings page and paper | Approved as written. |
| `paper:scheuerman-2019-computers-see-gender` | `abstract` | ACM DOI metadata and publisher abstract | Approved at the stated abstract-only depth. |
| `paper:franz-2022-interdependent-privacy` | `full_text` | Springer DOI page and full article | Approved as written. |
| `paper:longpre-2024-consent-crisis` | `full_text` | NeurIPS proceedings page and paper | Approved as written. |
| `paper:brubaker-2016-legacy-contact` | `abstract` | ACM DOI metadata and publisher abstract | Approved at the stated abstract-only depth. |
| `paper:gach-2021-facebook-affairs` | `abstract` | ACM DOI metadata and publisher abstract | Approved at the stated abstract-only depth. |
| `paper:lu-2026-griefbot-perceptions` | `full_text` | Frontiers article, DOI, PubMed, and PMC | Approved as written. |
| `paper:san-roman-2024-audioseal` | `full_text` | PMLR proceedings page and paper | Approved after narrowing one architecture implication. |
| `paper:rossler-2019-faceforensics` | `full_text` | CVF proceedings page, paper, and IEEE DOI | Approved as written. |

## 2. Identifier or metadata corrections

- No identifier, author, title, year, venue, publication-type, or primary-URL correction was required.
- All 20 primary HTTPS URLs identify the claimed paper, and all canonical identifiers resolve to or uniquely identify that work. Alternate identifiers refer to the same work rather than independent evidence.
- In particular, `paper:lukas-2023-pii-leakage` correctly uses IEEE DOI `10.1109/SP46215.2023.10179300` and arXiv `2302.00539`.
- `paper:buolamwini-2018-gender-shades` appropriately uses a title/year fallback because the selected PMLR proceedings record does not supply a DOI or arXiv identifier.

## 3. Overclaims and required wording changes

- `paper:san-roman-2024-audioseal`: downgraded “Watermark every synthetic voice output” to a conditional implication for synthetic voice generated under the system's control and where watermark integration is compatible. AudioSeal evaluates an integrated proactive scheme; it does not establish that every external generator can be watermarked or that watermarking alone proves authorization.
- No core finding, method, population, evidence-strength label, or relevance score required correction. Quantitative claims were supported at the declared depth.
- The three abstract-depth records remain limited to claims available in publisher abstracts and bibliographic metadata; no full-text inference was introduced.

The seat note already uses the narrower phrase “where possible” for embedded watermarks and consistently distinguishes observed evidence from proposed safeguards. No factual correction to `seat-notes/privacy_identity_safety.md` was required.

## 4. Missing counterevidence or foundational work

- No blocking counterevidence omission was found. The seat contains both attack and defense evidence: extraction and inference attacks are balanced by differential privacy, deduplication, unlearning, adversarial perturbation, watermarking, and detector studies, while each defense record states its boundary conditions.
- `paper:aligned-production-extraction-2025` in the `evaluation_fidelity` seat is a useful non-blocking synthesis link because it extends the warning from `paper:carlini-2021-training-data-extraction` to an aligned production model. It reinforces that refusals and ordinary prompting behavior are not privacy guarantees. It was not added here because cross-review does not expand the submitted seat.
- The posthumous-identity cluster lacks longitudinal evidence from actual griefbot use, therapeutic outcomes, contested estates, and vulnerable populations. `paper:lu-2026-griefbot-perceptions` explicitly measures anticipated perceptions, while `paper:brubaker-2016-legacy-contact` and `paper:gach-2021-facebook-affairs` concern stewardship rather than generative replicas. This is an open evidence gap, not grounds to reject those records.
- `paper:franz-2022-interdependent-privacy` measures hypothetical disclosure intention rather than observed archive import behavior. A field test of correspondent-level consent and revocation would strengthen the architecture case.
- Watermark and manipulation-detector results remain generator-, codec-, attack-, and access-model dependent. `paper:san-roman-2024-audioseal` and `paper:rossler-2019-faceforensics` therefore support layered provenance and contestability, not universal authenticity classification.

## 5. Duplicate candidates

No duplicate canonical paper appears within `privacy_identity_safety`. A structured alias scan across the current seat files found three cross-seat collisions involving this seat:

| `privacy_identity_safety` record | Other seat record | Collision evidence |
| --- | --- | --- |
| `paper:carlini-2021-training-data-extraction` | `evaluation_fidelity` / `paper:extracting-training-data-2021` | Same arXiv ID `2012.07805` and title. |
| `paper:lukas-2023-pii-leakage` | `evaluation_fidelity` / `paper:pii-leakage-2023` | Same IEEE DOI `10.1109/SP46215.2023.10179300`, arXiv ID `2302.00539`, and title. |
| `paper:kosinski-2013-private-traits` | `psychometrics` / `paper:psychometrics_kosinski_2013` | Same DOI `10.1073/pnas.1218772110` and title. |

Per the council workflow, discovery-seat overlap is retained through cross-review and resolved in the later global deduplication phase. All 20 records therefore remain counted here; each collision must become one canonical paper in the assembled corpus.

## 6. Architecture claims to downgrade or reject

- Downgrade universal watermark deployment to generation paths under system control and compatible with the scheme. Even then, `paper:san-roman-2024-audioseal` requires an authorization record because a watermark identifies a generation path, not consent.
- Reject public visibility, crawlability, account ownership, or archive possession as blanket permission to train on another person's messages, identity, likeness, or relationship data. `paper:franz-2022-interdependent-privacy` and `paper:longpre-2024-consent-crisis` support granular source, subject, purpose, and time-bounded permission handling.
- Reject automated PII scrubbing, prompt reminders, output truncation, deduplication, or ordinary alignment refusals as standalone privacy guarantees. `paper:shokri-2017-membership-inference`, `paper:carlini-2021-training-data-extraction`, `paper:kandpal-2022-dedup-privacy`, `paper:lukas-2023-pii-leakage`, and `paper:mireshghallah-2024-confaide` require adversarial and multi-party audits.
- Reject deletion of a source row as proof of forgetting. `paper:bourtoule-2021-machine-unlearning` supports data-to-artifact lineage and post-deletion verification, with its classifier and storage limitations preserved.
- Reject a detector score or absent watermark as proof of authenticity, identity, authorization, or consent. `paper:san-roman-2024-audioseal` and `paper:rossler-2019-faceforensics` support provenance checks only as one layer.
- Reject posthumous simulation consent inferred from retention settings, kinship, or silence. The limited and qualitative evidence in `paper:brubaker-2016-legacy-contact`, `paper:gach-2021-facebook-affairs`, and `paper:lu-2026-griefbot-perceptions` supports explicit advance directives and conservative defaults, not claims of therapeutic efficacy.

## 7. Final decision

**APPROVE AFTER CORRECTION.** All 20 records were independently checked against primary sources at their declared depth. The single required architecture-wording correction has been applied; no record was rejected, removed, or marked uncounted. The corpus contains 17 full-text and 3 abstract inspections, with three cross-seat aliases reserved for global deduplication. This approval authorizes applying reviewer agent `019f4f07-3f74-7193-a53e-6b392adaf6f0` and the single shared UTC timestamp `2026-07-11T08:56:25Z` to all 20 provenance objects, followed by validation with `require_reviewed=True`.
