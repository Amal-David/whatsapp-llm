# Persona Replica Enhancement Plan

This document outlines concrete improvements that would help the toolkit export WhatsApp chats with a specific person and generate a high-quality persona profile alongside the existing fine-tuning workflow.

## 1. Data Extraction Enhancements

### 1.1 Contact-Scoped Export Helpers
- Add a helper function (e.g., `parser.extract_contact_chat(chat_path, contact_name)`) that filters raw exports down to a single interlocutor before running the main parser.
- Provide CLI flags such as `--contact` and `--include-self` to let users focus on a dyadic conversation without manual preprocessing.
- Implement validation that warns when multi-person (group) messages or media placeholders remain after filtering.

### 1.2 Conversation Episode Detection
- Detect long pauses (configurable threshold, e.g., 6 hours) and label conversation episodes to organize persona traits by context (work, family, hobbies).
- Store metadata about episode boundaries in the generated CSV/JSONL to support downstream analysis and persona summarization.

## 2. Persona Profiling Pipeline

### 2.1 Character Sheet Generator
- Introduce a `persona.py` module that consumes the style metrics JSON plus contact-scoped messages and outputs a structured persona file (YAML/JSON).
- Capture key attributes: tone descriptors, recurring topics, sentiment distribution, conversation role (initiator/responder), and notable quirks (emoji, slang, signature phrases).
- Provide sample prompt snippets that downstream LLMs can ingest to stay in-character.

### 2.2 Topic and Intent Mining
- Apply simple NLP techniques (e.g., keyword extraction, TF-IDF clustering, sentence transformers) to surface top conversation themes and intents.
- Map intents (informational, emotional support, planning) into the persona file to guide generated responses.

### 2.3 Memory Slots and Canonical Facts
- Automatically extract factual statements ("I live in…", "My favorite…") and store them as "memory slots" with source message references.
- Deduplicate contradictory facts by timestamp precedence and provide a confidence score based on message frequency.

## 3. Fine-Tuning Integration

### 3.1 Persona-Aware Training Records
- Extend `parser.py` to append persona tags (tone, topic, intent) to each message pair record, enabling conditional training.
- Update `finetune.py` to optionally prepend the generated persona summary into each training sample for few-shot alignment.

### 3.2 Evaluation Hooks
- Add a lightweight evaluation script that prompts the fine-tuned model with canonical facts and checks for consistency using the persona file as ground truth.
- Track divergence metrics (e.g., persona trait adherence) to alert users when the model drifts from the intended character.

## 4. User Experience Improvements

### 4.1 CLI Workflow
- Add a top-level CLI (e.g., `python main.py persona --chat chat.txt --contact "Jamie"`) orchestrating parse → persona → finetune steps with sensible defaults.
- Include progress logging, persona preview outputs, and human-readable summaries (Markdown/HTML) for quick review.

### 4.2 Documentation and Templates
- Provide a persona template file demonstrating the schema and how it feeds into fine-tuning prompts.
- Update `README.md` with a dedicated guide for contact-specific persona cloning, including privacy considerations and example outputs.

## 5. Privacy and Safety Considerations
- Implement redaction utilities for personally identifiable information (addresses, phone numbers) before exporting persona files.
- Offer configurable anonymization (hash contact names, mask URLs) to protect the other party’s identity while preserving behavioral signals.

By prioritizing these enhancements, the project will better support users who want to export conversations with a specific person and craft a faithful persona profile for character-driven applications.
