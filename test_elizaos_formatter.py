import unittest
import pandas as pd
from datetime import datetime, timezone

# Assuming parser.py and elizaos_character.py are in the same directory or accessible in PYTHONPATH
# If they are in a package, the imports would be different (e.g., from package_name.parser import ...)
from parser import (
    format_elizaos_character, 
    ChatMessage, 
    PersonalStyleExtractor, # Used for comparison or potentially generating complex metrics if needed
    create_system_prompt, # Used for understanding system_prompt structure
    _get_elizaos_bio, # For default values
    _get_elizaos_lore,
    _get_elizaos_adjectives,
    _get_elizaos_topics,
    _get_elizaos_post_examples,
    _get_elizaos_message_examples,
    _get_elizaos_style
)
from elizaos_character import ElizaOsCharacter, MessageTurn, MessageContent, Style, KnowledgeItem

class TestElizaOsFormatter(unittest.TestCase):

    def setUp(self):
        self.your_name = "TestUser"
        self.other_user = "OtherUser"
        self.ts1 = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        self.ts2 = datetime(2023, 1, 1, 10, 0, 10, tzinfo=timezone.utc)
        self.ts3 = datetime(2023, 1, 1, 10, 0, 20, tzinfo=timezone.utc)
        self.ts4 = datetime(2023, 1, 1, 10, 0, 30, tzinfo=timezone.utc)

    def test_format_elizaos_character_basic(self):
        df_finetune = pd.DataFrame([
            {'Author': self.other_user, 'Message': 'Hello!', 'Timestamp': self.ts1},
            {'Author': self.your_name, 'Message': 'Hi there!', 'Timestamp': self.ts2},
            {'Author': self.other_user, 'Message': 'How are you?', 'Timestamp': self.ts3},
            {'Author': self.your_name, 'Message': 'Good, thanks! This is a longer message for post examples.', 'Timestamp': self.ts4},
        ])
        system_prompt_text = "This is a test bio.\nWith multiple lines."
        
        # Using fixed style_metrics for predictable results
        style_metrics = {
            'avg_message_length': 20.0,
            'emoji_usage_rate': 0.6,    # -> expressive
            'slang_usage_rate': 0.4,    # -> informal
            'capitalization_rate': 0.1, # -> casual typist
            'common_phrases': {"test phrase": 2, "another example": 1} # Already sorted by PersonalStyleExtractor
        }

        character = format_elizaos_character(
            df_finetune, self.your_name, style_metrics, system_prompt_text, context_length=2, max_examples=2
        )

        self.assertIsInstance(character, ElizaOsCharacter)
        self.assertEqual(character.name, self.your_name)

        # Test bio (derived from system_prompt)
        self.assertEqual(character.bio, ["This is a test bio.", "With multiple lines."])
        self.assertTrue(len(character.bio) >= 1)

        # Test lore (placeholder)
        self.assertEqual(character.lore, _get_elizaos_lore())
        self.assertTrue(len(character.lore) >= 1)

        # Test messageExamples
        self.assertIsInstance(character.messageExamples, list)
        self.assertTrue(len(character.messageExamples) > 0, "messageExamples should not be empty")
        self.assertTrue(len(character.messageExamples) <= 2, "messageExamples should respect max_examples")
        
        for snippet in character.messageExamples:
            self.assertIsInstance(snippet, list)
            self.assertTrue(len(snippet) >= 1, "Each snippet in messageExamples should have at least one turn")
            is_your_name_present_in_snippet = False
            for turn_idx, turn in enumerate(snippet):
                self.assertIsInstance(turn, MessageTurn)
                self.assertIsInstance(turn.user, str)
                self.assertIsInstance(turn.content, dict) # MessageContent is TypedDict
                self.assertIn('text', turn.content)
                self.assertIsInstance(turn.content['text'], str)
                self.assertIn('action', turn.content) # Action can be None or str
                if turn.user == self.your_name:
                    is_your_name_present_in_snippet = True
                    # Check if action is "CONTINUE" for the character's turn
                    self.assertEqual(turn.content.get('action'), "CONTINUE") 
            self.assertTrue(is_your_name_present_in_snippet, "Character's message should be in the snippet")

        # Test postExamples
        self.assertIsInstance(character.postExamples, list)
        self.assertTrue(len(character.postExamples) >= 1)
        # Based on the messages, the longest one from TestUser
        self.assertEqual(character.postExamples[0], 'Good, thanks! This is a longer message for post examples.')

        # Test adjectives (derived from style_metrics)
        expected_adjectives = sorted(["expressive", "informal", "casual typist"]) # Order might vary
        self.assertEqual(sorted(character.adjectives), expected_adjectives)
        self.assertTrue(len(character.adjectives) >= 1)

        # Test topics (derived from style_metrics)
        self.assertEqual(character.topics, ["test phrase", "another example"])
        self.assertTrue(len(character.topics) >= 1)

        # Test style (derived from system_prompt)
        self.assertIsInstance(character.style, Style)
        expected_style_lines = ["This is a test bio.", "With multiple lines."]
        self.assertEqual(character.style.all, expected_style_lines)
        self.assertEqual(character.style.chat, expected_style_lines)
        self.assertEqual(character.style.post, expected_style_lines)
        self.assertTrue(len(character.style.all) >= 1)

        # Test knowledge (placeholder)
        self.assertEqual(character.knowledge, []) # Optional, defaults to empty list

    def test_format_elizaos_character_empty_input(self):
        df_empty = pd.DataFrame(columns=['Author', 'Message', 'Timestamp'])
        system_prompt_empty = ""
        style_metrics_empty = {}

        character = format_elizaos_character(
            df_empty, self.your_name, style_metrics_empty, system_prompt_empty
        )

        self.assertIsInstance(character, ElizaOsCharacter)
        self.assertEqual(character.name, self.your_name)
        
        self.assertEqual(character.bio, _get_elizaos_bio(" ")) # Check against default from helper
        self.assertEqual(character.lore, _get_elizaos_lore())
        
        # Message examples should have the placeholder
        self.assertEqual(len(character.messageExamples), 1)
        self.assertEqual(len(character.messageExamples[0]), 1)
        self.assertEqual(character.messageExamples[0][0].user, "System")
        self.assertEqual(character.messageExamples[0][0].content['text'], "No message examples available.")
        
        self.assertEqual(character.postExamples, _get_elizaos_post_examples([]))
        self.assertEqual(character.adjectives, _get_elizaos_adjectives({}))
        self.assertEqual(character.topics, _get_elizaos_topics({}))
        
        default_style_dict = _get_elizaos_style(" ")
        self.assertEqual(character.style.all, default_style_dict['all'])
        self.assertEqual(character.style.chat, default_style_dict['chat'])
        self.assertEqual(character.style.post, default_style_dict['post'])


    def test_format_elizaos_character_only_your_name_messages(self):
        your_name_solo = "SoloUser"
        messages_solo = [
            {'Author': your_name_solo, 'Message': 'Just me talking.', 'Timestamp': self.ts1},
            {'Author': your_name_solo, 'Message': 'Another message from me, a bit longer for post examples.', 'Timestamp': self.ts2},
            {'Author': your_name_solo, 'Message': 'Short one.', 'Timestamp': self.ts3},
        ]
        df_solo = pd.DataFrame(messages_solo)
        
        system_prompt_solo = "This is a solo prompt."
        
        # Generate style_metrics from user's own messages
        user_message_texts = [m['Message'] for m in messages_solo if m['Author'] == your_name_solo]
        style_metrics_solo = PersonalStyleExtractor().analyze_style(user_message_texts)
        # For topics, let's ensure common_phrases is populated if PersonalStyleExtractor finds them
        # For this test, we can also use a fixed one for predictability if PersonalStyleExtractor is complex.
        # For example, if "from me" is a common phrase:
        # style_metrics_solo['common_phrases'] = {"message from me": 1} 

        character = format_elizaos_character(
            df_solo, your_name_solo, style_metrics_solo, system_prompt_solo, context_length=2, max_examples=3
        )

        self.assertIsInstance(character, ElizaOsCharacter)
        self.assertEqual(character.name, your_name_solo)
        self.assertEqual(character.bio, [system_prompt_solo])

        # Message examples: context will be empty, only character's message
        self.assertTrue(len(character.messageExamples) <= 3) # max_examples
        self.assertTrue(len(character.messageExamples) > 0)

        for snippet in character.messageExamples:
            self.assertEqual(len(snippet), 1, "Snippet should only contain the character's own message as context is empty")
            self.assertEqual(snippet[0].user, your_name_solo)
            self.assertEqual(snippet[0].content.get('action'), "CONTINUE")

        # Post examples should be the longest messages from SoloUser
        self.assertIsInstance(character.postExamples, list)
        self.assertTrue(len(character.postExamples) > 0)
        self.assertTrue(len(character.postExamples) <= 3) # _get_elizaos_post_examples returns up to 3
        self.assertEqual(character.postExamples[0], 'Another message from me, a bit longer for post examples.')

        # Adjectives and Topics will depend on PersonalStyleExtractor's output for the given messages
        self.assertTrue(len(character.adjectives) >= 1)
        self.assertTrue(len(character.topics) >= 1) # Will be placeholder if no common phrases from short texts


if __name__ == '__main__':
    unittest.main()
