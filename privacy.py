import re
from typing import Any, Dict, Iterable, List, Mapping


class PIIRedactor:
    """Utility for redacting personally identifiable information from text payloads."""

    def __init__(self, mask_token: str = '[REDACTED]') -> None:
        self.mask_token = mask_token
        self.patterns = [
            re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w.-]+)?'),  # email addresses
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

    def redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_value(item) for item in value)
        if isinstance(value, dict):
            return {
                self.redact_text(key) if isinstance(key, str) else key: self.redact_value(item)
                for key, item in value.items()
            }
        return value

    def redact_records(self, records: Iterable[Mapping[str, object]], fields: Iterable[str]) -> List[Dict[str, object]]:
        redacted_records: List[Dict[str, object]] = []
        for record in records:
            new_record = dict(record)
            for field in fields:
                if field in new_record:
                    new_record[field] = self.redact_value(new_record[field])
            redacted_records.append(new_record)
        return redacted_records
