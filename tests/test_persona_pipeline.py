import datetime
import json
import subprocess
import sys
import unittest

import pandas as pd

from parser import LLMFormat, jsonl_data
from persona import PersonaBuilder, persona_summary_from_file


class PersonaPipelineTest(unittest.TestCase):
    def test_plain_text_persona_summary_file_loads(self):
        summary = persona_summary_from_file('samples/persona_summary.txt')

        self.assertIn('Alex is a seasoned engineering manager', summary)

    def test_redaction_covers_nested_persona_payload(self):
        chat_records = [
            {
                'Author': 'Jamie',
                'Message': 'My email is jamie@example.com and I live in 123 Main Street',
                'Timestamp': '2024-01-01T09:00:00',
                'Tone': 'neutral',
                'Intent': 'informational',
                'Topic': 'jamie@example.com',
            }
        ]
        style_metrics = {
            'emoji_usage_rate': 0,
            'slang_usage_rate': 0,
            'capitalization_rate': 1,
            'common_phrases': {
                'email is jamie@example.com': 2,
                '123 Main Street': 1,
            },
        }

        profile = PersonaBuilder('Alex', 'Jamie', redact=True).build_profile(
            chat_records,
            style_metrics,
            [],
        )
        payload_text = json.dumps(profile.persona)

        self.assertIn('[REDACTED]', payload_text)
        self.assertNotIn('jamie@example', payload_text)
        self.assertNotIn('123 Main Street', payload_text)

    def test_contact_persona_uses_contact_style_metrics(self):
        df = pd.DataFrame([
            {
                'Author': 'Jamie',
                'Message': 'i love sunrise hikes',
                'Timestamp': datetime.datetime(2024, 1, 1, 9, 0),
            },
            {
                'Author': 'Alex',
                'Message': 'WOW!!! LETS SHIP THIS',
                'Timestamp': datetime.datetime(2024, 1, 1, 9, 1),
            },
            {
                'Author': 'Jamie',
                'Message': 'can we plan coffee tomorrow?',
                'Timestamp': datetime.datetime(2024, 1, 1, 9, 2),
            },
            {
                'Author': 'Alex',
                'Message': 'YES!!! I AM READY',
                'Timestamp': datetime.datetime(2024, 1, 1, 9, 3),
            },
        ])

        _, training_style, persona_context = jsonl_data(
            df,
            'Alex',
            LLMFormat.LLAMA2,
            contact_name='Jamie',
        )
        contact_style = persona_context['author_style_metrics']['Jamie']
        profile = PersonaBuilder('Alex', 'Jamie').build_profile(
            persona_context['tagged_messages'],
            contact_style,
            persona_context['episode_metadata'],
        )
        persona_text = json.dumps(profile.persona)

        self.assertNotIn('WOW', persona_text)
        self.assertNotIn('SHIP', persona_text)
        self.assertNotEqual(training_style, contact_style)

    def test_main_cli_import_does_not_eagerly_import_training_stack(self):
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                'import sys; import main; print("finetune" in sys.modules)',
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), 'False')


if __name__ == '__main__':
    unittest.main()
