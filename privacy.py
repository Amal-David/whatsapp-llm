import re
from typing import Dict, Iterable, List, Mapping


class PIIRedactor:
    """Utility for redacting personally identifiable information from text payloads."""

    def __init__(self, mask_token: str = '[REDACTED]') -> None:
        self.mask_token = mask_token
        self.patterns = [
            re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'),  # email addresses
            re.compile(r'\b\+?\d{1,3}[\s-]?\(?\d{2,3}\)?[\s-]?\d{3}[\s-]?\d{4}\b'),  # phone numbers
            re.compile(r'\b\d{1,5}\s+[A-Za-z0-9\.\s]{3,},?\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b', re.IGNORECASE),
            re.compile(r'http[s]?://\S+'),  # URLs
        ]

    def redact_text(self, text: str) -> str:
        if not text:
            return text
        redacted = text
        for pattern in self.patterns:
            redacted = pattern.sub(self.mask_token, redacted)
        return redacted

    def redact_records(self, records: Iterable[Mapping[str, object]], fields: Iterable[str]) -> List[Dict[str, object]]:
        redacted_records: List[Dict[str, object]] = []
        for record in records:
            new_record = dict(record)
            for field in fields:
                value = new_record.get(field)
                if isinstance(value, str):
                    new_record[field] = self.redact_text(value)
            redacted_records.append(new_record)
        return redacted_records
