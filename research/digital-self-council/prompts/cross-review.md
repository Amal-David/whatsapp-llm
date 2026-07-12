# Council Cross-Review Prompt v1

You are the `{reviewer_seat}` reviewer for the `{subject_seat}` evidence slice.
You did not extract these records.

Read the subject seat JSONL and synthesis, the council charter, the paper schema,
and any duplicate report available. Do not rewrite the subject synthesis merely
to match your own domain preferences.

## Review Checks

For every record selected for the final corpus, check:

- canonical identifier and primary URL identify the claimed paper
- title, authors, year, venue, and publication type are consistent
- preprint and published versions are not double-counted
- inspection depth is honest
- method, population/data, and modalities match the source
- core finding is paraphrased and does not exceed the inspected evidence
- architecture implications follow from the finding rather than from enthusiasm
- limitations and external-validity warnings are meaningful
- evidence strength and relevance score are defensible
- relevant null, negative, contradictory, or boundary-setting evidence is not omitted
- private or third-party data did not enter the artifact

## Output

Create `{review_path}` with:

1. reviewed record IDs
2. identifier or metadata corrections
3. overclaims and required wording changes
4. missing counterevidence or foundational work
5. duplicate candidates
6. architecture claims that should be downgraded or rejected
7. a final approve/revise decision

Only after required corrections are applied, set the reviewed records to:

- `provenance.reviewer`: `{reviewer_agent_id}`
- `provenance.reviewed_at`: the current UTC ISO-8601 timestamp

Run the final validator with review enforcement enabled. A reviewer must never
approve a record it originally extracted.
