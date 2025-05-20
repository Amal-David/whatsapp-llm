import unittest
import pandas as pd
import datetime
import os
import json
from unittest.mock import patch, mock_open

# Assuming parser.py is in the same directory or accessible in PYTHONPATH
from parser import (
    ChatParser,
    DataCleaner,
    PersonalStyleExtractor,
    ConversationManager, # Added as it's used by jsonl_data indirectly
    LLMFormatter,      # Added as it's used by jsonl_data
    jsonl_data,
    converter_with_debug,
    LLMFormat,
    ChatMessage,
    create_system_prompt # Added as it's used by jsonl_data
)

class TestChatParser(unittest.TestCase):
    def setUp(self):
        self.parser = ChatParser()

    def test_parse_line_standard_format(self):
        line = "10/25/23, 10:00:00 AM - John Doe: Hello there!"
        expected = ChatMessage(author="John Doe", message="Hello there!", timestamp=datetime.datetime(2023, 10, 25, 10, 0, 0))
        result = self.parser.parse_line(line)
        self.assertEqual(result, expected)

    def test_parse_line_standard_format_pm(self):
        line = "10/25/23, 10:00:00 PM - Jane Doe: Good evening!"
        expected = ChatMessage(author="Jane Doe", message="Good evening!", timestamp=datetime.datetime(2023, 10, 25, 22, 0, 0))
        result = self.parser.parse_line(line)
        self.assertEqual(result, expected)

    def test_parse_line_without_seconds(self):
        line = "11/20/23, 09:30 PM - Alice: How are you?"
        expected = ChatMessage(author="Alice", message="How are you?", timestamp=datetime.datetime(2023, 11, 20, 21, 30, 0))
        result = self.parser.parse_line(line)
        self.assertEqual(result, expected)

    def test_parse_line_bracketed_format(self):
        # This pattern is: r"\[(\d{1,2}/\d{1,2}/\d{2}),\s*(\d{1,2}:\d{2}:\d{2}\s*[APM]{2})\]\s*([^:]+):\s*(.+)"
        # It does not have brackets around the whole line, but around date and time.
        line = "[12/31/22, 01:15:45 PM] Bob: Happy New Year!"
        expected = ChatMessage(author="Bob", message="Happy New Year!", timestamp=datetime.datetime(2022, 12, 31, 13, 15, 45))
        result = self.parser.parse_line(line)
        self.assertEqual(result, expected)

    def test_parse_line_international_format_ddmmyy(self):
        # DD/MM/YY
        line = "25/12/23, 07:05:30 AM - Eve: Merry Christmas!"
        expected = ChatMessage(author="Eve", message="Merry Christmas!", timestamp=datetime.datetime(2023, 12, 25, 7, 5, 30))
        result = self.parser.parse_line(line)
        self.assertEqual(result, expected)
    
    def test_parse_line_international_format_with_seconds_optional(self):
        line = "31/12/23, 06:50 PM - Carol: Test international without seconds"
        # The regex is r"(\d{1,2}/\d{1,2}/\d{2}),\s*(\d{1,2}:\d{2}(?::\d{2})?\s*[APM]{2})\s*-\s*([^:]+):\s*(.+)"
        # So it should match with or without seconds
        expected = ChatMessage(author="Carol", message="Test international without seconds", timestamp=datetime.datetime(2023, 12, 31, 18, 50, 0))
        result = self.parser.parse_line(line)
        self.assertEqual(result, expected)

    def test_parse_line_iso_like_format_yyyymmdd(self):
        # YYYY-MM-DD
        line = "2023-01-15, 02:20:10 PM - Dave: ISO test"
        expected = ChatMessage(author="Dave", message="ISO test", timestamp=datetime.datetime(2023, 1, 15, 14, 20, 10))
        result = self.parser.parse_line(line)
        self.assertEqual(result, expected)

    def test_parse_line_iso_like_format_without_seconds(self):
        line = "2023-02-10, 03:40 AM - Frank: ISO no seconds"
        # The regex is r"(\d{4}-\d{2}-\d{2}),\s*(\d{1,2}:\d{2}(?::\d{2})?\s*[APM]{2})\s*-\s*([^:]+):\s*(.+)"
        expected = ChatMessage(author="Frank", message="ISO no seconds", timestamp=datetime.datetime(2023, 2, 10, 3, 40, 0))
        result = self.parser.parse_line(line)
        self.assertEqual(result, expected)

    def test_parse_line_invalid_format(self):
        line = "This is not a valid chat line."
        self.assertIsNone(self.parser.parse_line(line))

    def test_parse_line_timestamp_error(self):
        # Invalid date like 32nd day or 13th month
        line = "13/32/23, 10:00:00 AM - John Doe: Invalid date"
        # Expecting a warning to be logged, and parse_line to return None after trying all patterns
        with self.assertLogs(logger='parser', level='WARNING') as cm:
            result = self.parser.parse_line(line)
            self.assertIsNone(result)
        # Check that a warning was logged (the exact message format depends on the logger in parser.py)
        self.assertTrue(any("Error parsing timestamp" in log_message for log_message in cm.output))

    def test_parse_line_leading_trailing_whitespace(self):
        line = "  10/25/23, 10:00:00 AM - John Doe: Hello with spaces!   "
        expected = ChatMessage(author="John Doe", message="Hello with spaces!", timestamp=datetime.datetime(2023, 10, 25, 10, 0, 0))
        result = self.parser.parse_line(line)
        # The message part might have extra spaces if not handled by `strip()` on the message component
        # parser.py does message.strip()
        self.assertEqual(result, expected)

    def test_parse_line_empty_line(self):
        line = ""
        self.assertIsNone(self.parser.parse_line(line))

    def test_parse_line_whitespace_only_line(self):
        line = "     "
        self.assertIsNone(self.parser.parse_line(line))

    def test_parse_line_different_author_message_content(self):
        line = "01/01/24, 12:00:00 PM - Test User: This is a test message with numbers 123 and symbols !@#."
        expected = ChatMessage(author="Test User", message="This is a test message with numbers 123 and symbols !@#.", timestamp=datetime.datetime(2024, 1, 1, 12, 0, 0))
        result = self.parser.parse_line(line)
        self.assertEqual(result, expected)

class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = DataCleaner()

    def test_clean_message_basic(self):
        self.assertEqual(self.cleaner.clean_message("Hello world"), "Hello world")

    def test_clean_message_with_url(self):
        self.assertEqual(self.cleaner.clean_message("Check this: http://example.com"), "Check this: [URL]")
        self.assertEqual(self.cleaner.clean_message("Or www.example.org site"), "Or [URL] site")

    def test_clean_message_whitespace(self):
        self.assertEqual(self.cleaner.clean_message("  Extra   spaces  "), "Extra spaces")

    def test_clean_message_empty(self):
        self.assertEqual(self.cleaner.clean_message(""), "")
        self.assertEqual(self.cleaner.clean_message("   "), "")
        
    def test_clean_message_preserves_emojis(self):
        message = "Hello 👋 world 😃"
        self.assertEqual(self.cleaner.clean_message(message), message)

    def test_validate_message_valid(self):
        self.assertTrue(self.cleaner.validate_message("This is a good message."))

    def test_validate_message_too_short(self):
        self.assertFalse(self.cleaner.validate_message("Hi")) # Length 2, border case, should be True
        self.assertTrue(self.cleaner.validate_message("Hi")) # Correcting based on len < 2 is False
        self.assertFalse(self.cleaner.validate_message("H")) # Length 1
        self.assertFalse(self.cleaner.validate_message("  ")) # Empty after strip

    def test_validate_message_skip_patterns(self):
        self.assertFalse(self.cleaner.validate_message("<media omitted>"))
        self.assertFalse(self.cleaner.validate_message("This message was deleted")) # "message deleted" is a substring
        self.assertFalse(self.cleaner.validate_message("image omitted"))
        self.assertFalse(self.cleaner.validate_message("My [redacted] content"))
        self.assertFalse(self.cleaner.validate_message("video omitted by me"))
        self.assertFalse(self.cleaner.validate_message("This is an audio omitted here"))
        self.assertFalse(self.cleaner.validate_message("Please see document omitted"))


    def test_validate_message_empty(self):
        self.assertFalse(self.cleaner.validate_message(""))

class TestPersonalStyleExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = PersonalStyleExtractor()

    def test_analyze_style_empty_messages(self):
        metrics = self.extractor.analyze_style([])
        self.assertEqual(metrics['avg_message_length'], 0)
        self.assertEqual(metrics['emoji_usage_rate'], 0)
        self.assertEqual(metrics['slang_usage_rate'], 0)
        self.assertEqual(metrics['capitalization_rate'], 0)
        self.assertEqual(metrics['punctuation_patterns'], {}) # It's a defaultdict, but empty access gives 0
        self.assertEqual(metrics['common_phrases'], {})

    def test_analyze_style_sample_messages(self):
        messages = [
            "Hello world!",
            "OMG this is so cool lol 😂",
            "What's up? Btw, I'll be there asap.",
            "Another message.",
            "lol, just kidding. Or am I? 😉"
        ]
        metrics = self.extractor.analyze_style(messages)

        self.assertAlmostEqual(metrics['avg_message_length'], sum(len(m) for m in messages) / len(messages))
        self.assertAlmostEqual(metrics['emoji_usage_rate'], 2 / 5) # 😂 and 😉
        # Slang: OMG, lol, Btw, asap, lol
        self.assertAlmostEqual(metrics['slang_usage_rate'], 5 / 5) 
        # Capitalized: Hello, OMG, What's, Another
        self.assertAlmostEqual(metrics['capitalization_rate'], 4 / 5)

        self.assertEqual(metrics['punctuation_patterns']['!'], 1)
        self.assertEqual(metrics['punctuation_patterns']['?'], 2) # "What's up?" and "Or am I?"
        self.assertEqual(metrics['punctuation_patterns']['.'], 1) # "Another message."

        # Common phrases (3-grams) - depends on exact logic, let's check for presence of some
        # "lol, just kidding." -> "lol, just kidding"
        # "just kidding. Or" -> "just kidding. Or" (punctuation part of word)
        # "kidding. Or am" -> "kidding. Or am"
        # "Or am I?" -> "Or am I?"
        # "OMG this is"
        # "this is so"
        # "is so cool"
        # "so cool lol"
        # "What's up? Btw," -> "What's up? Btw,"
        # "up? Btw, I'll" -> "up? Btw, I'll"
        # "Btw, I'll be" -> "Btw, I'll be"
        # "I'll be there" -> "I'll be there"
        # "be there asap." -> "be there asap."
        # "lol, just kidding"
        # "just kidding. Or"
        # "kidding. Or am"
        # "Or am I?"
        # Needs more specific checks if required, but the structure is a dict.
        # For example, check if a known phrase exists
        self.assertIn("OMG this is", metrics['common_phrases'])
        self.assertIn("lol, just kidding", metrics['common_phrases'])
        self.assertEqual(metrics['common_phrases']["OMG this is"], 1)

    def test_analyze_style_no_slang_no_emoji(self):
        messages = [
            "This is a formal sentence.",
            "Another formal statement follows.",
            "Indeed, this is quite proper."
        ]
        metrics = self.extractor.analyze_style(messages)
        self.assertAlmostEqual(metrics['emoji_usage_rate'], 0)
        self.assertAlmostEqual(metrics['slang_usage_rate'], 0)
        self.assertAlmostEqual(metrics['capitalization_rate'], 3/3) # All start with capital
        self.assertEqual(metrics['punctuation_patterns']['.'], 3)

class TestJsonlData(unittest.TestCase):
    def setUp(self):
        self.sample_data = {
            'Author': ['Alice', 'Bob', 'Alice', 'Bob', 'Alice'],
            'Message': [
                "Hello Bob!", 
                "Hi Alice! How are you? lol", 
                "I'm good, thanks! And you? 😊", 
                "Doing great! Btw, did you see the news?",
                "No, what happened? OMG"
            ],
            'Timestamp': [
                datetime.datetime(2023, 1, 1, 10, 0, 0),
                datetime.datetime(2023, 1, 1, 10, 1, 0),
                datetime.datetime(2023, 1, 1, 10, 2, 0),
                datetime.datetime(2023, 1, 1, 10, 3, 0),
                datetime.datetime(2023, 1, 1, 10, 4, 0)
            ]
        }
        self.df = pd.DataFrame(self.sample_data)
        self.your_name_alice = "Alice"
        self.your_name_bob = "Bob"
        self.style_extractor = PersonalStyleExtractor() # For system prompt generation comparison

    def test_jsonl_data_empty_df(self):
        empty_df = pd.DataFrame(columns=['Author', 'Message', 'Timestamp'])
        formatted_data, style_metrics = jsonl_data(empty_df, "NonExistentUser")
        self.assertEqual(len(formatted_data), 0)
        # Check default style metrics (all should be zero or empty)
        self.assertEqual(style_metrics['avg_message_length'], 0)
        self.assertEqual(style_metrics['emoji_usage_rate'], 0)

    def test_jsonl_data_alice_is_you(self):
        # Alice's messages: "Hello Bob!", "I'm good, thanks! And you? 😊", "No, what happened? OMG"
        # Bob's messages (context): "Hi Alice! How are you? lol", "Doing great! Btw, did you see the news?"
        # Expected pairs (Context by Bob, Response by Alice):
        # 1. Context: "Bob: Hi Alice! How are you? lol"
        #    Response: "Alice: I'm good, thanks! And you? 😊"
        # 2. Context: "Bob: Doing great! Btw, did you see the news?"
        #    Response: "Alice: No, what happened? OMG" (assuming context_length=1 for simplicity in manual trace)
        #    If context_length=3, context for 2nd response:
        #    "Alice: Hello Bob!\nBob: Hi Alice! How are you? lol\nAlice: I'm good, thanks! And you? 😊\nBob: Doing great! Btw, did you see the news?"
        #    No, this is wrong. The context is always from the *other* person(s) leading up to *your* message.
        #    The function `jsonl_data` works by iterating through df.
        #    When message.author == your_name AND len(conversation_manager.current_conversation) > 0:
        #       context = conversation_manager.format_context (current_conversation[-context_length:])
        #       response = message.message
        #    Then it adds the message to current_conversation.

        # Let's trace for Alice, context_length=3 (default)
        # 1. Alice: "Hello Bob!" -> added to conv. conv = [A1]
        # 2. Bob: "Hi Alice! How are you? lol" -> added to conv. conv = [A1, B1]
        # 3. Alice: "I'm good, thanks! And you? 😊" (your_name == Alice, len(conv) > 0)
        #    Context: last 3 from [A1, B1] -> "Alice: Hello Bob!\nBob: Hi Alice! How are you? lol"
        #    Response: "I'm good, thanks! And you? 😊"
        #    -> Output 1. Then add A2 to conv. conv = [A1, B1, A2]
        # 4. Bob: "Doing great! Btw, did you see the news?" -> added to conv. conv = [A1, B1, A2, B2]
        # 5. Alice: "No, what happened? OMG" (your_name == Alice, len(conv) > 0)
        #    Context: last 3 from [A1, B1, A2, B2] -> "Bob: Hi Alice! How are you? lol\nAlice: I'm good, thanks! And you? 😊\nBob: Doing great! Btw, did you see the news?"
        #    Response: "No, what happened? OMG"
        #    -> Output 2. Then add A3 to conv. conv = [A1, B1, A2, B2, A3]

        formatted_data, style_metrics = jsonl_data(self.df, self.your_name_alice, context_length=3)
        
        self.assertEqual(len(formatted_data), 2) # Two messages from Alice are responses

        # Check style_metrics for Alice
        alice_messages = ["Hello Bob!", "I'm good, thanks! And you? 😊", "No, what happened? OMG"]
        expected_style_metrics = self.style_extractor.analyze_style(alice_messages)
        self.assertEqual(style_metrics['avg_message_length'], expected_style_metrics['avg_message_length'])
        self.assertEqual(style_metrics['emoji_usage_rate'], expected_style_metrics['emoji_usage_rate']) # 1 emoji / 3 msgs
        self.assertEqual(style_metrics['slang_usage_rate'], expected_style_metrics['slang_usage_rate'])   # 1 slang (OMG) / 3 msgs
        
        system_prompt = create_system_prompt(expected_style_metrics)

        # Entry 1 (Alice's response: "I'm good, thanks! And you? 😊")
        context1 = "Alice: Hello Bob!\nBob: Hi Alice! How are you? lol" # context_length=3, but only 2 available
        response1 = "I'm good, thanks! And you? 😊"
        expected_text1_llama2 = LLMFormatter.format_llama2(context1, response1, system_prompt)
        self.assertEqual(formatted_data[0]['text'], expected_text1_llama2)
        self.assertEqual(formatted_data[0]['timestamp'], self.sample_data['Timestamp'][2].isoformat())

        # Entry 2 (Alice's response: "No, what happened? OMG")
        context2 = "Bob: Hi Alice! How are you? lol\nAlice: I'm good, thanks! And you? 😊\nBob: Doing great! Btw, did you see the news?" # context_length=3 from [A1,B1,A2,B2] -> B1,A2,B2
        response2 = "No, what happened? OMG"
        expected_text2_llama2 = LLMFormatter.format_llama2(context2, response2, system_prompt)
        self.assertEqual(formatted_data[1]['text'], expected_text2_llama2)
        self.assertEqual(formatted_data[1]['timestamp'], self.sample_data['Timestamp'][4].isoformat())

    def test_jsonl_data_bob_is_you_context_length_1(self):
        # Bob's messages: "Hi Alice! How are you? lol", "Doing great! Btw, did you see the news?"
        # Alice's messages (context): "Hello Bob!", "I'm good, thanks! And you? 😊", "No, what happened? OMG"
        # Expected pairs (Context by Alice, Response by Bob):
        # 1. Context: "Alice: Hello Bob!" (conv=[A1])
        #    Response: "Bob: Hi Alice! How are you? lol"
        #    -> Output 1. conv=[A1,B1]
        # 2. Context: "Alice: I'm good, thanks! And you? 😊" (conv=[A1,B1,A2], context_length=1 from [B1,A2] -> A2)
        #    Response: "Bob: Doing great! Btw, did you see the news?"
        #    -> Output 2. conv=[A1,B1,A2,B2]
        
        formatted_data, style_metrics = jsonl_data(self.df, self.your_name_bob, context_length=1)
        self.assertEqual(len(formatted_data), 2)

        bob_messages = ["Hi Alice! How are you? lol", "Doing great! Btw, did you see the news?"]
        expected_style_metrics = self.style_extractor.analyze_style(bob_messages)
        self.assertEqual(style_metrics['slang_usage_rate'], expected_style_metrics['slang_usage_rate']) # lol, Btw = 2 slang / 2 msgs
        system_prompt = create_system_prompt(expected_style_metrics)

        # Entry 1 (Bob's response: "Hi Alice! How are you? lol")
        context1 = "Alice: Hello Bob!"
        response1 = "Hi Alice! How are you? lol"
        expected_text1_llama2 = LLMFormatter.format_llama2(context1, response1, system_prompt)
        self.assertEqual(formatted_data[0]['text'], expected_text1_llama2)

        # Entry 2 (Bob's response: "Doing great! Btw, did you see the news?")
        # Conversation history before this: [A1, B1, A2]
        # Context for B2 (context_length=1): A2 ("Alice: I'm good, thanks! And you? 😊")
        context2 = "Alice: I'm good, thanks! And you? 😊"
        response2 = "Doing great! Btw, did you see the news?"
        expected_text2_llama2 = LLMFormatter.format_llama2(context2, response2, system_prompt)
        self.assertEqual(formatted_data[1]['text'], expected_text2_llama2)

    def test_jsonl_data_all_formats(self):
        # Simplified test for formats, using Alice as 'you' and one entry
        df_short = pd.DataFrame({
            'Author': ['Bob', 'Alice'],
            'Message': ["Hey Alice", "Hey Bob, sup?"],
            'Timestamp': [datetime.datetime.now(), datetime.datetime.now() + datetime.timedelta(seconds=1)]
        })
        alice_messages_short = ["Hey Bob, sup?"]
        style_metrics = self.style_extractor.analyze_style(alice_messages_short)
        system_prompt = create_system_prompt(style_metrics)
        context = "Bob: Hey Alice"
        response = "Hey Bob, sup?"

        for llm_format_enum in LLMFormat:
            with self.subTest(format=llm_format_enum.value):
                formatted_data, _ = jsonl_data(df_short, "Alice", llm_format=llm_format_enum, context_length=1)
                self.assertEqual(len(formatted_data), 1)
                entry = formatted_data[0]
                
                if llm_format_enum == LLMFormat.LLAMA2:
                    expected_text = LLMFormatter.format_llama2(context, response, system_prompt)
                    self.assertEqual(entry['text'], expected_text)
                elif llm_format_enum == LLMFormat.LLAMA3:
                    expected_text = LLMFormatter.format_llama3(context, response, system_prompt)
                    self.assertEqual(entry['text'], expected_text)
                elif llm_format_enum == LLMFormat.MISTRAL:
                    expected_text = LLMFormatter.format_mistral(context, response, system_prompt)
                    self.assertEqual(entry['text'], expected_text)
                elif llm_format_enum == LLMFormat.FALCON:
                    expected_text = LLMFormatter.format_falcon(context, response, system_prompt)
                    self.assertEqual(entry['text'], expected_text)
                elif llm_format_enum == LLMFormat.QWEN:
                    expected_text = LLMFormatter.format_qwen(context, response, system_prompt)
                    self.assertEqual(entry['text'], expected_text)
                elif llm_format_enum == LLMFormat.GPT:
                    expected_gpt_format = LLMFormatter.format_gpt(context, response, system_prompt)
                    self.assertEqual(entry['text'], expected_gpt_format) # GPT format returns a dict
                    self.assertIsInstance(entry['text'], dict)
                else:
                    self.fail(f"Unhandled LLMFormat: {llm_format_enum}")

    def test_jsonl_data_your_name_no_messages(self):
        formatted_data, style_metrics = jsonl_data(self.df, "NotInChat", context_length=3)
        self.assertEqual(len(formatted_data), 0) # No messages from "NotInChat" to be a response
        # Style metrics should be for an empty list of messages
        empty_style_metrics = self.style_extractor.analyze_style([])
        self.assertEqual(style_metrics, empty_style_metrics)

    def test_jsonl_data_message_cleaning_and_validation(self):
        df_dirty = pd.DataFrame({
            'Author': ['Alice', 'Bob', 'Alice', 'Bob', 'Alice'],
            'Message': [
                "Hello Bob!", 
                "<media omitted>",  # Bob's message to be skipped
                "I'm good, http://example.com thanks! 😊", # Alice's message, URL to be cleaned
                "  ", # Bob's message, too short/empty, should be skipped by add_message if not by validate
                "  H " # Alice's message, too short
            ],
            'Timestamp': [
                datetime.datetime(2023, 1, 1, 10, 0, 0),
                datetime.datetime(2023, 1, 1, 10, 1, 0),
                datetime.datetime(2023, 1, 1, 10, 2, 0),
                datetime.datetime(2023, 1, 1, 10, 3, 0),
                datetime.datetime(2023, 1, 1, 10, 4, 0)
            ]
        })
        # Trace:
        # 1. A1: "Hello Bob!" -> conv=[A1]
        # 2. B1: "<media omitted>" -> validate=False. Not added to conv. conv=[A1]
        # 3. A2: "I'm good, http://example.com thanks! 😊" (your_name=Alice, len(conv)=1 > 0)
        #    Context: "Alice: Hello Bob!"
        #    Response: "I'm good, [URL] thanks! 😊" (cleaned)
        #    -> Output 1. Add cleaned A2 to conv. conv=[A1, A2_cleaned]
        # 4. B2: "  " -> validate=False (empty after strip). Not added. conv=[A1, A2_cleaned]
        # 5. A3: "  H " -> validate=False (too short). Not added. conv=[A1, A2_cleaned]

        formatted_data, style_metrics = jsonl_data(df_dirty, "Alice", context_length=1)
        
        self.assertEqual(len(formatted_data), 1)
        
        alice_messages_for_style = ["Hello Bob!", "I'm good, http://example.com thanks! 😊", "  H "] # Raw messages for style
        # However, DataCleaner.validate_message is applied *before* message is used for style in the current jsonl_data logic
        # This is a bit inconsistent: style is based on *all* your_name messages from input df,
        # but formatting is only for *valid* messages.
        # Let's assume style_metrics are based on *all* messages by your_name from the initial DF.
        # The problem statement says "Extract personal style, your_messages = df[df['Author'] == your_name]['Message'].tolist()"
        # This happens BEFORE the loop where validation and cleaning occur for context/response.
        
        expected_style_metrics = self.style_extractor.analyze_style(alice_messages_for_style)
        self.assertEqual(style_metrics['avg_message_length'], expected_style_metrics['avg_message_length'])

        system_prompt = create_system_prompt(style_metrics) # Based on possibly 'dirty' messages
        
        context1 = "Alice: Hello Bob!" # A1
        response1_cleaned = "I'm good, [URL] thanks! 😊" # A2 cleaned
        expected_text1_llama2 = LLMFormatter.format_llama2(context1, response1_cleaned, system_prompt)
        self.assertEqual(formatted_data[0]['text'], expected_text1_llama2)

class TestConverterWithDebug(unittest.TestCase):
    def setUp(self):
        self.test_chat_file = "test_chat.txt"
        self.prompter = "Alice"
        self.responder = "Bob"
        self.your_name = "Alice" # For style extraction and system prompt

        # Sample chat content
        self.sample_chat_content_valid = (
            "10/25/23, 10:00:00 AM - Alice: Hello Bob!\n"
            "10/25/23, 10:01:00 AM - Bob: Hi Alice! How are you? lol\n"
            "10/25/23, 10:02:00 AM - Alice: I'm good, thanks! 😊 And you? omg\n"
            "10/25/23, 10:03:00 AM - Bob: Doing great! Btw, did you see the news?\n"
            "10/25/23, 10:04:00 AM - System: System message, to be ignored.\n" # This won't be parsed by current patterns
            "10/25/23, 10:05:00 AM - Alice: No, what happened?\n"
        )
        self.sample_chat_content_malformed = (
            "This is a completely malformed line.\n"
            "10/25/23, 10:00:00 AM - Alice: Valid line 1.\n"
            "Another malformed line.\n"
            "10/25/23, 10:01:00 AM - Bob: Valid line 2.\n"
            "10/25/23, 10:02:00 AM - Alice: Valid line 3, with some slang omg.\n"
            "Invalid Date, 10:00 AM - User: Message with bad date.\n" # This will log a timestamp error
        )

    def tearDown(self):
        # Clean up any files created during tests if necessary
        if os.path.exists(self.test_chat_file):
            os.remove(self.test_chat_file)
        output_files = [f for f in os.listdir('.') if f.startswith('output_') or f.startswith('formatted_') or f.startswith('style_metrics_')]
        for f in output_files:
            os.remove(f)

    @patch('os.path.exists')
    def test_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            converter_with_debug("non_existent_file.txt", self.prompter, self.responder, self.your_name)

    @patch('os.path.getsize')
    @patch('os.path.exists')
    def test_empty_file(self, mock_exists, mock_getsize):
        mock_exists.return_value = True
        mock_getsize.return_value = 0
        # Mock open to simulate an empty file read if getsize wasn't enough
        m_open = mock_open(read_data="")
        with patch('builtins.open', m_open):
            with self.assertLogs(logger='parser', level='WARNING') as cm: # Expect ValueError + warning
                # The function raises ValueError for empty file, then logs warning for no pairs
                with self.assertRaises(ValueError) as ve:
                    converter_with_debug(self.test_chat_file, self.prompter, self.responder, self.your_name)
                self.assertIn("Chat file is empty", str(ve.exception))
            
            # If the ValueError is caught and processing continues to attempt to find pairs:
            # This part of the test might not be reached if ValueError is raised and not handled to continue
            # The current converter_with_debug raises ValueError if empty, so it won't proceed to log "No valid conversation pairs"
            # However, if it *did* proceed, we'd check for the warning.
            # For now, the ValueError check is primary for "empty file" condition.

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_malformed_lines(self, mock_getsize, mock_exists, mock_file_open):
        mock_exists.return_value = True
        mock_getsize.return_value = len(self.sample_chat_content_malformed.encode('utf-8'))
        mock_file_open.return_value.read.return_value = self.sample_chat_content_malformed
        # If your converter_with_debug uses readline or iterates, adjust mock:
        mock_file_open.return_value.__enter__.return_value.readlines.return_value = self.sample_chat_content_malformed.splitlines(True)
        mock_file_open.return_value.__iter__.return_value = self.sample_chat_content_malformed.splitlines(True)


        with self.assertLogs(logger='parser', level='WARNING') as cm:
            df, formatted_data, style_metrics = converter_with_debug(
                self.test_chat_file, self.prompter, self.responder, self.your_name
            )
        
        # Check logs for malformed lines and timestamp errors
        self.assertTrue(any("did not match any pattern" in log_msg for log_msg in cm.output))
        self.assertTrue(any("Error parsing timestamp" in log_msg for log_msg in cm.output)) # For "Invalid Date..."

        # Check that valid lines were processed
        # Prompter: Alice, Responder: Bob
        # Valid pairs:
        # 1. Alice: Valid line 1. -> Bob: Valid line 2.
        # result_dict will store {0: {'prompt': 'Valid line 1.', 'completion': 'Valid line 2.'}}
        # df_finetune will have: A:VL1, B:VL2, A:VL3
        self.assertEqual(len(df), 1) # One prompter/responder pair
        self.assertEqual(df.iloc[0]['prompt'], "Valid line 1.")
        self.assertEqual(df.iloc[0]['completion'], "Valid line 2.")

        # formatted_data (JSONL) for 'your_name' (Alice)
        # Alice's messages: "Valid line 1.", "Valid line 3, with some slang omg."
        # 1. Response: "Valid line 3, with some slang omg."
        #    Context: "Bob: Valid line 2." (assuming context_length=1 for easier check here, default is 3)
        # Let's re-run with context_length=1 for this check for simplicity
        _, formatted_data_ctx1, _ = converter_with_debug(
                self.test_chat_file, self.prompter, self.responder, self.your_name, llm_format=LLMFormat.LLAMA2 # need to provide mock_file_open again
            )

        # Need to re-patch open for the second call
        mock_file_open.return_value.__enter__.return_value.readlines.return_value = self.sample_chat_content_malformed.splitlines(True)
        mock_file_open.return_value.__iter__.return_value = self.sample_chat_content_malformed.splitlines(True)
        _, formatted_data_ctx1, _ = converter_with_debug(
                self.test_chat_file, self.prompter, self.responder, self.your_name, llm_format=LLMFormat.LLAMA2, context_length=1
            )


        self.assertEqual(len(formatted_data_ctx1), 1) # Alice's "Valid line 3..." is a response
        
        alice_messages_for_style = ["Valid line 1.", "Valid line 3, with some slang omg."]
        expected_style = PersonalStyleExtractor().analyze_style(alice_messages_for_style)
        system_prompt = create_system_prompt(expected_style)
        
        context = "Bob: Valid line 2."
        response = "Valid line 3, with some slang omg."
        expected_text = LLMFormatter.format_llama2(context, response, system_prompt)
        self.assertEqual(formatted_data_ctx1[0]['text'], expected_text)


    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_valid_chat_processing(self, mock_getsize, mock_exists, mock_file_open):
        mock_exists.return_value = True
        mock_getsize.return_value = len(self.sample_chat_content_valid.encode('utf-8'))
        mock_file_open.return_value.read.return_value = self.sample_chat_content_valid
        mock_file_open.return_value.__enter__.return_value.readlines.return_value = self.sample_chat_content_valid.splitlines(True)
        mock_file_open.return_value.__iter__.return_value = self.sample_chat_content_valid.splitlines(True)

        df, formatted_data, style_metrics = converter_with_debug(
            self.test_chat_file, self.prompter, self.responder, self.your_name, LLMFormat.LLAMA2, context_length=2
        )

        # Verify df (prompter/responder pairs)
        # A: Hello Bob! -> B: Hi Alice! How are you? lol  (Pair 1)
        # A: I'm good, thanks! 😊 And you? omg -> B: Doing great! Btw, did you see the news? (Pair 2)
        # System message is ignored.
        # A: No, what happened? (No response from Bob for this one in the sample)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['prompt'], "Hello Bob!")
        self.assertEqual(df.iloc[0]['completion'], "Hi Alice! How are you? lol")
        self.assertEqual(df.iloc[1]['prompt'], "I'm good, thanks! 😊 And you? omg")
        self.assertEqual(df.iloc[1]['completion'], "Doing great! Btw, did you see the news?")

        # Verify style_metrics for 'Alice'
        alice_messages = ["Hello Bob!", "I'm good, thanks! 😊 And you? omg", "No, what happened?"]
        expected_style = PersonalStyleExtractor().analyze_style(alice_messages)
        self.assertEqual(style_metrics['avg_message_length'], expected_style['avg_message_length'])
        self.assertEqual(style_metrics['emoji_usage_rate'], expected_style['emoji_usage_rate']) # 1/3
        self.assertEqual(style_metrics['slang_usage_rate'], expected_style['slang_usage_rate']) # omg = 1/3

        system_prompt = create_system_prompt(expected_style)

        # Verify formatted_data for 'Alice' (LLAMA2 format, context_length=2)
        # Alice's messages: A1, A2, A3
        # Responses from Alice: A2, A3
        # 1. Response A2: "I'm good, thanks! 😊 And you? omg"
        #    Conv before A2: [A1, B1]
        #    Context (len 2): "Alice: Hello Bob!\nBob: Hi Alice! How are you? lol"
        # 2. Response A3: "No, what happened?"
        #    Conv before A3: [A1, B1, A2, B2] (System msg ignored)
        #    Context (len 2): "Alice: I'm good, thanks! 😊 And you? omg\nBob: Doing great! Btw, did you see the news?"
        
        self.assertEqual(len(formatted_data), 2)

        context1 = "Alice: Hello Bob!\nBob: Hi Alice! How are you? lol"
        response1 = "I'm good, thanks! 😊 And you? omg"
        expected_text1 = LLMFormatter.format_llama2(context1, response1, system_prompt)
        self.assertEqual(formatted_data[0]['text'], expected_text1)

        context2 = "Alice: I'm good, thanks! 😊 And you? omg\nBob: Doing great! Btw, did you see the news?"
        response2 = "No, what happened?" # Cleaned, no change
        expected_text2 = LLMFormatter.format_llama2(context2, response2, system_prompt)
        self.assertEqual(formatted_data[1]['text'], expected_text2)

    def test_type_error_for_path(self):
        with self.assertRaises(TypeError) as cm:
            converter_with_debug(123, self.prompter, self.responder, self.your_name)
        self.assertIn("Filepath must be a string", str(cm.exception))


if __name__ == '__main__':
    unittest.main()
