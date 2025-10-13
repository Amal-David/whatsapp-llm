import argparse
import json
from collections import Counter
from typing import Dict, List

from persona import load_persona_profile


def _normalize_text(text) -> str:
    if isinstance(text, dict):
        return json.dumps(text)
    return str(text)


def load_dataset(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def evaluate_alignment(persona_path: str, dataset_path: str) -> Dict[str, object]:
    persona = load_persona_profile(persona_path).persona
    dataset_records = load_dataset(dataset_path)

    persona_summary = persona.get('persona_summary', '')
    memory_slots = persona.get('memory_slots', [])
    preferred_tones = persona.get('tone_descriptors', [])

    summary_presence = 0
    tone_matches = 0
    total_records = len(dataset_records)

    tone_counter = Counter()

    for record in dataset_records:
        text_value = _normalize_text(record.get('text', ''))
        if persona_summary and persona_summary.split('.')[0] in text_value:
            summary_presence += 1

        persona_tags = record.get('persona_tags', {})
        tone = persona_tags.get('tone')
        if tone:
            tone_counter[tone] += 1
            if tone in preferred_tones:
                tone_matches += 1

    fact_coverage = 0
    if memory_slots:
        covered = 0
        for slot in memory_slots:
            value = slot.get('value')
            if value and any(value.lower() in _normalize_text(rec.get('text', '')).lower() for rec in dataset_records):
                covered += 1
        fact_coverage = covered / len(memory_slots)

    metrics = {
        'records_evaluated': total_records,
        'persona_summary_injected_ratio': summary_presence / total_records if total_records else 0,
        'tone_alignment_ratio': tone_matches / total_records if total_records else 0,
        'fact_coverage_ratio': fact_coverage,
        'tone_distribution': dict(tone_counter)
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate persona adherence in generated datasets or responses')
    parser.add_argument('--persona', required=True, help='Path to the persona YAML/JSON file')
    parser.add_argument('--dataset', required=True, help='Path to the JSONL dataset or model responses')
    args = parser.parse_args()

    metrics = evaluate_alignment(args.persona, args.dataset)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
