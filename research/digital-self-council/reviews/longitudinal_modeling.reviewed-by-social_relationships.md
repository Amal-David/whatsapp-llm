# Cross-Review: `longitudinal_modeling`

- Subject seat: `longitudinal_modeling`
- Reviewer seat: `social_relationships`
- Reviewer agent: `019f4f07-10e4-7660-9c2f-c6ea34490a72`
- Reviewed at: `2026-07-11T08:58:31Z`
- Decision: **APPROVED AFTER CORRECTIONS**
- Scope: 20 counted primary papers; 19 inspected at `full_text` depth and one at `abstract` depth

## 1. Reviewed Records

All canonical identifiers and primary HTTPS URLs were checked against publisher pages, DOI targets, arXiv, PubMed/PMC, or official proceedings. The claim review respected each record's declared inspection depth; `paper:xiang-2010-temporal-preference-fusion` remains abstract-depth and was not upgraded.

| Record | Depth | Canonical primary source | Result |
| --- | --- | --- | --- |
| `paper:koren-2009-temporal-dynamics` | `full_text` | [doi:10.1145/1557019.1557072](https://doi.org/10.1145/1557019.1557072) | Verified |
| `paper:xiang-2010-temporal-preference-fusion` | `abstract` | [doi:10.1145/1835804.1835896](https://doi.org/10.1145/1835804.1835896) | Verified at abstract depth |
| `paper:rendle-2010-fpmc` | `full_text` | [doi:10.1145/1772690.1772773](https://doi.org/10.1145/1772690.1772773) | Verified |
| `paper:wu-2017-recurrent-recommender` | `full_text` | [doi:10.1145/3018661.3018689](https://doi.org/10.1145/3018661.3018689) | Verified |
| `paper:dai-2016-deepcoevolve` | `full_text` | [doi:10.1145/2939672.2939879](https://doi.org/10.1145/2939672.2939879) | Verified |
| `paper:kumar-2019-jodie` | `full_text` | [doi:10.1145/3292500.3330895](https://doi.org/10.1145/3292500.3330895) | Verified after claim-precision correction |
| `paper:li-2020-tisasrec` | `full_text` | [doi:10.1145/3336191.3371786](https://doi.org/10.1145/3336191.3371786) | Verified |
| `paper:xu-2020-graphsail` | `full_text` | [doi:10.1145/3340531.3412754](https://doi.org/10.1145/3340531.3412754) | Verified |
| `paper:mi-2020-ader` | `full_text` | [doi:10.1145/3383313.3412218](https://doi.org/10.1145/3383313.3412218) | Verified after title, venue, and alias correction |
| `paper:mi-2020-man` | `full_text` | [doi:10.24963/ijcai.2020/300](https://doi.org/10.24963/ijcai.2020/300) | Verified |
| `paper:rashid-2002-getting-to-know-you` | `full_text` | [doi:10.1145/502716.502737](https://doi.org/10.1145/502716.502737) | Verified |
| `paper:boutilier-2003-active-cf` | `full_text` | [arXiv:1212.2442](https://arxiv.org/abs/1212.2442) | Verified after pruning-tradeoff correction |
| `paper:harpale-2008-personalized-active-learning` | `full_text` | [doi:10.1145/1390334.1390352](https://doi.org/10.1145/1390334.1390352) | Verified |
| `paper:rashid-2008-information-theoretic-elicitation` | `full_text` | [doi:10.1145/1540276.1540302](https://doi.org/10.1145/1540276.1540302) | Verified |
| `paper:ahn-2007-open-user-profiles` | `full_text` | [doi:10.1145/1242572.1242575](https://doi.org/10.1145/1242572.1242575) | Verified |
| `paper:bostandjiev-2012-tasteweights` | `full_text` | [doi:10.1145/2365952.2365964](https://doi.org/10.1145/2365952.2365964) | Verified after author-name correction |
| `paper:bakalov-2013-controllable-user-models` | `full_text` | [doi:10.1145/2449396.2449405](https://doi.org/10.1145/2449396.2449405) | Verified after author-name corrections |
| `paper:kirkpatrick-2017-ewc` | `full_text` | [doi:10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114) | Verified |
| `paper:lopez-paz-2017-gem` | `full_text` | [arXiv:1706.08840](https://arxiv.org/abs/1706.08840) | Verified |
| `paper:masson-dautume-2019-episodic-language-memory` | `full_text` | [arXiv:1906.01076](https://arxiv.org/abs/1906.01076) | Verified |

## 2. Identifier And Metadata Corrections

- `paper:mi-2020-ader`: replaced the incorrect title and CIKM venue with **“ADER: Adaptively Distilled Exemplar Replay Towards Continual Learning for Session-based Recommendation”** and **Proceedings of the 14th ACM Conference on Recommender Systems (RecSys 2020)**. The DOI `10.1145/3383313.3412218`, arXiv alias `2007.12000`, authors, year, and primary DOI URL were already correct. The `title_year` alias was updated to match the verified title.
- `paper:bostandjiev-2012-tasteweights`: corrected `Tobias Hollerer` to `Tobias Höllerer`.
- `paper:bakalov-2013-controllable-user-models`: corrected `Birgitta Konig-Ries` to `Birgitta König-Ries` and `Rene Witte` to `René Witte`.
- The remaining 17 records required no canonical-ID, alternate-ID, title, author, year, venue, publication-type, or primary-URL correction. All 20 primary URLs use HTTPS and lead to the corresponding primary record.

## 3. Overclaims And Wording Corrections

- `paper:kumar-2019-jodie`: changed state-change AUC improvement from “at least 12%” to **“12% on average,”** matching the paper's result statement. The “at least 20%” future-interaction MRR claim remains supported.
- `paper:boutilier-2003-active-cf`: replaced the unsupported inference that pruning removed 60% to 80% of online computation. The corrected finding states that prototype sets pruned 60% or 80% of candidate queries, that 60% pruning tracked the unpruned condition closely, and that 80% pruning degraded more while still outperforming the alternative query strategies.
- No other finding, method, population, architecture implication, or limitation required factual rewriting at its stated depth.

## 4. Missing Counterevidence Or Foundational Coverage

No additional foundational record is required to approve this bounded seat. The corpus already contains meaningful negative and conflicting evidence:

- Time-aware modeling is not uniformly useful: `paper:li-2020-tisasrec` reports weak visible sequential structure for Amazon Beauty, while `paper:koren-2009-temporal-dynamics` warns against indiscriminate age-based downweighting.
- Preservation can be unnecessary or harmful: `paper:mi-2020-ader` reports little forgetting from plain fine-tuning on stable YOOCHOOSE, and `paper:xu-2020-graphsail` reports a local-structure component hurting a PinSage setting.
- More retained memory carries costs and is not independently sufficient: `paper:mi-2020-man` degrades as memory shrinks, while `paper:masson-dautume-2019-episodic-language-memory` relies on stored examples without supplying consent or deletion governance.
- Owner control has conflicting evidence: editing harmed objective recommendation metrics in `paper:ahn-2007-open-user-profiles`; `paper:bostandjiev-2012-tasteweights` confounds control with additional feedback; and `paper:bakalov-2013-controllable-user-models` is a seven-person, ordered field study without a control group.
- General continual-learning evidence is indirect: `paper:kirkpatrick-2017-ewc` remained below separate Atari agents, and `paper:lopez-paz-2017-gem` assumes cleaner task boundaries than overlapping personal change.

The active-CF correction adds one previously flattened tradeoff: aggressive 80% prototype pruning loses more decision quality than 60% pruning (`paper:boutilier-2003-active-cf`). The seat note already states the key social-observability counterpoint: timing, silence, and conversational action do not identify relational meaning or owner endorsement.

## 5. Duplicate Candidates

No duplicate candidate was found. All 20 canonical identifiers are unique. Published papers with arXiv, PubMed, or PMC counterparts retain those identifiers only as aliases rather than separate counted records. `paper:mi-2020-ader` and `paper:mi-2020-man` share a lead author and year but are distinct papers with different titles, methods, venues, and DOIs.

## 6. Architecture Downgrades Or Rejections

No record-level evidence-strength score, relevance score, or architecture implication required removal. The implications are conditionally phrased and paired with specific limitations. Approval does not endorse the following stronger readings:

- Prediction gains in `paper:rendle-2010-fpmc`, `paper:wu-2017-recurrent-recommender`, `paper:dai-2016-deepcoevolve`, or `paper:kumar-2019-jodie` do not establish that latent state is an interpretable or owner-recognized personal identity.
- Anti-forgetting gains in `paper:xu-2020-graphsail`, `paper:mi-2020-ader`, `paper:kirkpatrick-2017-ewc`, `paper:lopez-paz-2017-gem`, or `paper:masson-dautume-2019-episodic-language-memory` do not establish that retained evidence is legitimate, consented, or exempt from deletion.
- Short user-control studies in `paper:ahn-2007-open-user-profiles`, `paper:bostandjiev-2012-tasteweights`, and `paper:bakalov-2013-controllable-user-models` do not establish durable correction quality at WhatsApp scale or resolve third-party information boundaries.

These are transfer constraints already represented in the records' limitations and in the seat synthesis, so no architecture field was downgraded.

## 7. Final Decision And Validation

**APPROVED AFTER CORRECTIONS.** Corrections were applied before review provenance was assigned. All 20 records now use reviewer `019f4f07-10e4-7660-9c2f-c6ea34490a72` and the single UTC timestamp `2026-07-11T08:58:31Z`; extractor provenance was preserved.

Validation was run with `require_reviewed=True`:

```text
{'total_records': 20, 'counted_primary_papers': 20, 'unique_canonical_papers': 20, 'reviewed_records': 20, 'seat_counts': {'longitudinal_modeling': 20}, 'publication_type_counts': {'conference': 18, 'journal': 2}, 'inspection_depth_counts': {'abstract': 1, 'full_text': 19}}
```
