import json
from datetime import datetime, timezone

import pytest
import yaml

from living_brain.identity.interview import create_interview_template, write_interview_template
from living_brain.identity.models import (
    ClaimStatus,
    DigitalSelfProfile,
    EvidenceRecord,
    IdentityClaim,
    ProvenanceType,
    RelationshipProfile,
)


def _at(day: int) -> datetime:
    return datetime(2026, 1, day, 12, 0, tzinfo=timezone.utc)


def test_profile_round_trip_preserves_typed_identity_evidence():
    evidence = EvidenceRecord.create(
        source_type="owner_interview",
        source_record_id="interview:values:1",
        observed_at=_at(1),
        content="I optimize for independence over status.",
    )
    claim = IdentityClaim.create(
        dimension="values.independence",
        statement="Prefers independence over social status.",
        status=ClaimStatus.CONFIRMED,
        confidence=1.0,
        provenance=ProvenanceType.OWNER_INTERVIEW,
        evidence_ids=[evidence.id],
        created_at=_at(2),
        valid_from=_at(1),
    )
    relationship = RelationshipProfile(
        id="relationship:friend",
        label="close friend",
        evidence_ids=[evidence.id],
        style_delta={"message_length": "shorter"},
    )
    profile = DigitalSelfProfile(
        profile_id="digital-self:amal",
        owner_id="owner:amal",
        owner_name="Amal",
        created_at=_at(2),
        updated_at=_at(2),
        evidence=[evidence],
        claims=[claim],
        relationships=[relationship],
    )

    encoded = profile.to_json()
    restored = DigitalSelfProfile.from_json(encoded)

    assert restored == profile
    assert restored.claims[0].status is ClaimStatus.CONFIRMED
    assert restored.claims[0].created_at == _at(2)
    assert encoded == restored.to_json()
    assert json.loads(encoded)["schema_version"] == "digital_self.v1"


def test_explicit_owner_claim_outranks_inference_without_erasing_history():
    inferred = IdentityClaim.create(
        dimension="preferences.work_style",
        statement="Prefers rapid experimentation.",
        status=ClaimStatus.CANDIDATE,
        confidence=0.72,
        provenance=ProvenanceType.BEHAVIORAL_INFERENCE,
        created_at=_at(1),
    )
    explicit = IdentityClaim.create(
        dimension="preferences.work_style",
        statement="Prefers careful experiments with explicit proof.",
        status=ClaimStatus.CONFIRMED,
        confidence=1.0,
        provenance=ProvenanceType.OWNER_INTERVIEW,
        created_at=_at(2),
    )
    profile = DigitalSelfProfile(
        profile_id="digital-self:amal",
        owner_id="owner:amal",
        owner_name="Amal",
        claims=[inferred, explicit],
    )

    resolved = profile.resolve_dimension("preferences.work_style", as_of=_at(3))

    assert resolved == explicit
    assert profile.claims == [inferred, explicit]


def test_superseded_and_expired_claims_are_auditable_but_not_current():
    old = IdentityClaim.create(
        dimension="preferences.location",
        statement="Prefers working from Bengaluru.",
        status=ClaimStatus.CONFIRMED,
        confidence=1.0,
        provenance=ProvenanceType.OWNER_INTERVIEW,
        created_at=_at(1),
        valid_from=_at(1),
        valid_to=_at(5),
    )
    current = IdentityClaim.create(
        dimension="preferences.location",
        statement="Prefers working from Kochi.",
        status=ClaimStatus.CONFIRMED,
        confidence=1.0,
        provenance=ProvenanceType.OWNER_INTERVIEW,
        created_at=_at(6),
        valid_from=_at(6),
        supersedes=old.id,
    )
    profile = DigitalSelfProfile(
        profile_id="digital-self:amal",
        owner_id="owner:amal",
        owner_name="Amal",
        claims=[old, current],
    )

    assert profile.resolve_dimension("preferences.location", as_of=_at(3)) == old
    assert profile.resolve_dimension("preferences.location", as_of=_at(7)) == current
    assert {claim.id for claim in profile.claims} == {old.id, current.id}


def test_protected_traits_cannot_be_created_from_inference():
    with pytest.raises(ValueError, match="protected trait"):
        IdentityClaim.create(
            dimension="protected.religion",
            statement="Inferred religious affiliation.",
            status=ClaimStatus.CANDIDATE,
            confidence=0.5,
            provenance=ProvenanceType.BEHAVIORAL_INFERENCE,
            created_at=_at(1),
        )


def test_profile_validation_rejects_missing_evidence_links():
    claim = IdentityClaim.create(
        dimension="goals.product",
        statement="Build a useful personal AI.",
        status=ClaimStatus.CONFIRMED,
        confidence=1.0,
        provenance=ProvenanceType.OWNER_INTERVIEW,
        evidence_ids=["evidence:missing"],
        created_at=_at(1),
    )
    profile = DigitalSelfProfile(
        profile_id="digital-self:amal",
        owner_id="owner:amal",
        owner_name="Amal",
        claims=[claim],
    )

    with pytest.raises(ValueError, match="unknown evidence"):
        profile.validate()


def test_interview_template_covers_general_digital_self_inputs(tmp_path):
    template = create_interview_template("Amal")
    section_ids = {section["id"] for section in template["sections"]}

    assert {
        "life_story",
        "values",
        "goals",
        "decision_rules",
        "relationships",
        "recent_changes",
        "boundaries",
    } <= section_ids

    output = tmp_path / "interview.yaml"
    write_interview_template(output, "Amal")
    stored = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert stored == template
    assert all(question["answer"] is None for section in stored["sections"] for question in section["questions"])
