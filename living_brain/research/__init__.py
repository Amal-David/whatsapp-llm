"""Research-council contracts for the evidence-grounded digital brain."""

from .corpus import (
    COUNCIL_SEATS,
    TAXONOMY_TAGS,
    CorpusReport,
    CorpusValidationError,
    load_jsonl,
    normalized_identifier,
    paper_identifier_keys,
    validate_corpus,
)
from .council import (
    CouncilMergeReport,
    CouncilRunError,
    initialize_run,
    load_manifest,
    merge_run,
    record_seat_artifact,
    summarize_run,
)
from .synthesis import (
    EVIDENCE_STATUSES,
    CorpusManifestError,
    EvidenceMapValidationError,
    build_corpus_manifest,
    summarize_corpus,
    validate_evidence_map,
)

__all__ = [
    "COUNCIL_SEATS",
    "TAXONOMY_TAGS",
    "CorpusReport",
    "CorpusValidationError",
    "CouncilMergeReport",
    "CouncilRunError",
    "CorpusManifestError",
    "EVIDENCE_STATUSES",
    "EvidenceMapValidationError",
    "build_corpus_manifest",
    "initialize_run",
    "load_jsonl",
    "load_manifest",
    "merge_run",
    "normalized_identifier",
    "paper_identifier_keys",
    "record_seat_artifact",
    "summarize_corpus",
    "summarize_run",
    "validate_corpus",
    "validate_evidence_map",
]
