"""
Main entry point for Living Brain CLI.
"""

import argparse
import json
import logging
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_WHATSAPP_MAC_DATABASE = Path.home() / (
    "Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite"
)
DEFAULT_PSEUDONYM_KEY = Path.home() / ".config/living-brain/pseudonym.key"


def cmd_parse(args):
    """Parse a WhatsApp export file."""
    from .ingest.style_analyzer import StyleAnalyzer
    from .ingest.whatsapp_parser import WhatsAppParser

    parser = WhatsAppParser(your_name=args.your_name, date_order=args.date_order)

    # Get participants
    participants = parser.get_participants(args.input)
    print(f"Participants found: {', '.join(participants)}")

    if args.your_name not in participants:
        print(f"Warning: '{args.your_name}' not found in participants")
        print("Available names:", participants)
        return

    # Parse to conversations
    conversations = parser.parse_to_conversations(
        args.input,
        gap_minutes=args.gap_minutes,
    )
    print(f"Found {len(conversations)} conversations")

    # Analyze style
    your_messages = parser.get_your_messages(args.input)
    analyzer = StyleAnalyzer()
    metrics = analyzer.analyze(your_messages)

    print(f"\nStyle Analysis for {args.your_name}:")
    print(f"  Messages analyzed: {metrics.message_count}")
    print(f"  Avg message length: {metrics.avg_message_length:.1f} chars")
    print(f"  Emoji usage: {metrics.emoji_usage_rate:.1%}")
    print(f"  Capitalization: {metrics.capitalization_rate:.1%}")

    # Save metrics
    if args.output:
        output_path = Path(args.output)
        metrics.save(output_path.with_suffix('.json'))
        print(f"\nStyle metrics saved to: {output_path.with_suffix('.json')}")

        # Export JSONL for training
        parser.export_jsonl(args.input, output_path.with_suffix('.jsonl'))
        print(f"Messages exported to: {output_path.with_suffix('.jsonl')}")


def cmd_train(args):
    """Train a style adapter."""
    from .style.data_formatter import DataFormatter
    from .style.trainer import StyleTrainer, TrainingConfig

    # Create training data if input is a chat file
    data_path = Path(args.data)
    if data_path.suffix == '.txt':
        print("Creating training data from chat file...")
        formatter = DataFormatter(
            your_name=args.your_name,
            context_turns=args.context_turns,
        )
        training_data_path = data_path.with_name(f"{data_path.stem}_training.jsonl")
        count = formatter.create_dataset_from_file(
            data_path,
            training_data_path,
            format_type="alpaca",
        )
        print(f"Created {count} training examples")
        data_path = training_data_path

    # Training config
    config = TrainingConfig(
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
    )

    # Train
    trainer = StyleTrainer(
        model_name=args.model,
        config=config,
        load_in_4bit=not args.no_4bit,
    )

    print(f"Training with model: {args.model}")
    adapter_path = trainer.train(data_path)
    print(f"\nAdapter saved to: {adapter_path}")

    # Export to GGUF if requested
    if args.export_gguf:
        gguf_path = Path(adapter_path).parent / "model.gguf"
        trainer.export_gguf(adapter_path, gguf_path, args.quantization)
        print(f"GGUF exported to: {gguf_path}")


def cmd_chat(args):
    """Launch the chat interface."""
    from .inference.chat import launch_chat

    launch_chat(
        config_path=args.config,
        adapter_name=args.adapter,
        use_gguf=args.gguf is not None,
        gguf_path=args.gguf,
        share=args.share,
        port=args.port,
    )


def cmd_workbench(args):
    """Launch the persona dataset workbench."""
    from .inference.dataset_ui import launch_dataset_workbench

    launch_dataset_workbench(
        share=args.share,
        port=args.port,
        server_name=args.server_name,
    )


def cmd_ingest(args):
    """Ingest conversations into memory."""
    from .core.config import load_config
    from .ingest.whatsapp_parser import WhatsAppParser
    from .memory.fact_store import FactStore
    from .memory.vector_store import VectorStore

    config = load_config(args.config)

    # Initialize stores
    vector_store = VectorStore(
        persist_directory=config.memory.chroma_persist_dir,
        collection_name=config.memory.collection_name,
    )
    fact_store = FactStore(facts_path=config.facts.facts_path)

    # Parse and ingest
    parser = WhatsAppParser(your_name=args.your_name)
    conversations = parser.parse_to_conversations(
        args.input,
        gap_minutes=args.gap_minutes,
    )

    print(f"Found {len(conversations)} conversations")

    # Add to memory
    memories_added = 0
    for conv in conversations:
        memory_ids = vector_store.add_chunked(
            content=conv.to_text(),
            timestamp=conv.start_time,
            metadata={
                "source": str(args.input),
                "message_count": len(conv.messages),
            },
            chunk_size=config.memory.chunk_size,
            chunk_overlap=config.memory.chunk_overlap,
        )
        memories_added += len(memory_ids)

    print(f"Added {memories_added} memories")

    # Extract facts if requested
    if args.extract_facts:
        your_messages = parser.get_your_messages(args.input)
        facts_extracted = 0
        for msg in your_messages:
            facts = fact_store.ingest_and_extract(msg.message)
            facts_extracted += len(facts)
        print(f"Extracted {facts_extracted} facts")

    print(f"\nTotal memories: {vector_store.count()}")
    print(f"Total facts: {fact_store.count()}")


def cmd_watch(args):
    """Watch for new exports."""
    from .core.config import load_config
    from .ingest.watcher import ExportWatcher
    from .memory.fact_store import FactStore
    from .memory.vector_store import VectorStore

    config = load_config(args.config)

    vector_store = VectorStore(
        persist_directory=config.memory.chroma_persist_dir,
        collection_name=config.memory.collection_name,
    )
    fact_store = FactStore(facts_path=config.facts.facts_path)

    watcher = ExportWatcher(
        watch_dir=args.watch_dir,
        your_name=args.your_name,
        vector_store=vector_store,
        fact_store=fact_store,
        auto_extract_facts=args.extract_facts,
        chunk_size=config.memory.chunk_size,
        chunk_overlap=config.memory.chunk_overlap,
    )

    # Process existing files first
    if args.process_existing:
        count = watcher.process_existing()
        print(f"Processed {count} existing files")

    print(f"Watching for new exports in: {args.watch_dir}")
    print("Press Ctrl+C to stop")

    watcher.start(blocking=True)


def cmd_stats(args):
    """Show system statistics."""
    from .core.config import load_config
    from .memory.fact_store import FactStore
    from .memory.vector_store import VectorStore
    from .style.adapter_manager import AdapterManager

    config = load_config(args.config)

    print("Living Brain Statistics")
    print("=" * 40)

    # Memory stats
    try:
        vector_store = VectorStore(
            persist_directory=config.memory.chroma_persist_dir,
            collection_name=config.memory.collection_name,
        )
        print(f"\nMemories: {vector_store.count()}")

        recent = vector_store.get_recent(3)
        if recent:
            print("  Recent memories:")
            for m in recent:
                print(f"    - [{m.timestamp.date()}] {m.content[:50]}...")
    except Exception as e:
        print(f"\nMemories: Error - {e}")

    # Fact stats
    try:
        fact_store = FactStore(facts_path=config.facts.facts_path)
        print(f"\nFacts: {fact_store.count()}")

        facts = fact_store.get_all_active()[:5]
        if facts:
            print("  Sample facts:")
            for f in facts:
                print(f"    - {f.to_natural()}")
    except Exception as e:
        print(f"\nFacts: Error - {e}")

    # Adapter stats
    try:
        adapter_manager = AdapterManager()
        adapters = adapter_manager.list()
        print(f"\nAdapters: {len(adapters)}")
        for a in adapters:
            print(f"    - {a.name} ({a.base_model})")
    except Exception as e:
        print(f"\nAdapters: Error - {e}")


def _load_pseudonym_key(path):
    """Load or create a stable local pseudonym key with owner-only permissions."""
    key_path = Path(path).expanduser()
    if key_path.exists():
        key = key_path.read_bytes()
    else:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_bytes(32)
        descriptor = os.open(
            key_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            os.write(descriptor, key)
        finally:
            os.close(descriptor)
    if len(key) < 16:
        raise ValueError("pseudonym key must contain at least 16 bytes")
    key_path.chmod(0o600)
    return key


def _identity_source(args):
    """Create one local message source without opening it for writes."""
    from .identity.sources import JsonMessageSource
    from .identity.wacli import WacliSource
    from .identity.whatsapp_mac import WhatsAppMacSource

    source_path = Path(args.path).expanduser() if args.path else None
    key = _load_pseudonym_key(args.key_file)
    if args.source == "whatsapp-mac":
        return WhatsAppMacSource(
            source_path or DEFAULT_WHATSAPP_MAC_DATABASE,
            pseudonym_key=key,
        )
    if source_path is None:
        raise ValueError(f"--path is required for the {args.source} source")
    if args.source == "wacli":
        return WacliSource(source_path, pseudonym_key=key)
    if args.source == "json":
        return JsonMessageSource(source_path, pseudonym_key=key)
    raise ValueError(f"unsupported message source: {args.source}")


def _identity_build_result(args):
    from .identity.builder import DigitalSelfBuilder
    from .identity.interview import load_interview

    source = _identity_source(args)
    messages = source.read_messages(
        chat_ids=args.chat,
        all_chats=args.all_chats,
    )
    interview = load_interview(args.interview) if args.interview else None
    result = DigitalSelfBuilder(
        include_third_party_context=args.include_third_party_context,
    ).build(
        messages,
        owner_name=args.owner_name,
        interview=interview,
    )
    return result, interview


def cmd_self_interview(args):
    """Create an editable owner interview."""
    from .identity.interview import write_interview_template

    output_path = write_interview_template(args.output, args.owner_name)
    print(f"Interview template written to: {output_path}")


def cmd_self_chats(args):
    """Inspect locally available chats without reading message bodies."""
    chats = _identity_source(args).list_chats()
    print(f"Found {len(chats)} chats")
    for chat in chats:
        last_message = chat.last_message_at.isoformat() if chat.last_message_at else "unknown"
        print(
            json.dumps(
                {
                    "source_chat_id": chat.source_chat_id,
                    "display_name": chat.display_name,
                    "kind": chat.kind,
                    "message_count": chat.message_count,
                    "last_message_at": last_message,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )


def cmd_self_build(args):
    """Build a general digital-self profile from selected local chats."""
    result, _interview = _identity_build_result(args)
    result.profile.save(args.output)
    split_counts = {
        split: len(messages) for split, messages in result.messages_by_split.items()
    }
    print(f"Digital-self profile written to: {args.output}")
    print(
        f"Selected {result.profile.source_summary['chat_count']} chats and "
        f"{result.profile.source_summary['owner_message_count']} owner messages"
    )
    print(f"Message split counts: {split_counts}")


def cmd_self_validate(args):
    """Validate a serialized digital-self profile."""
    from .identity.models import DigitalSelfProfile

    profile = DigitalSelfProfile.load(args.profile)
    profile.validate()
    print(
        f"Profile is valid: {profile.owner_name}; "
        f"{len(profile.claims)} claims; {len(profile.relationships)} relationships"
    )


def cmd_self_evaluate(args):
    """Export private held-out and interview-retest evaluation rows."""
    from .identity.evaluation import EvaluationSuiteBuilder
    from .identity.models import DigitalSelfProfile

    result, interview = _identity_build_result(args)
    if args.profile:
        profile = DigitalSelfProfile.load(args.profile)
        if profile.owner_id != result.profile.owner_id:
            raise ValueError("evaluation source owner does not match --profile")
        if profile.source_summary != result.profile.source_summary:
            raise ValueError("evaluation source selection does not match --profile")

    suite = EvaluationSuiteBuilder().build(result, interview=interview)
    suite.save(args.output)
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else Path(args.output).with_name(f"{Path(args.output).stem}-summary.json")
    )
    suite.save_summary(summary_path)
    missing = suite.summary()["missing_required_tags"]
    print(f"Private evaluation suite written to: {args.output}")
    print(f"Text-free evaluation summary written to: {summary_path}")
    print(
        "Required coverage: complete"
        if not missing
        else f"Required coverage still missing: {', '.join(missing)}"
    )


def cmd_self(args):
    """Dispatch digital-self subcommands."""
    commands = {
        "interview": cmd_self_interview,
        "chats": cmd_self_chats,
        "build": cmd_self_build,
        "validate": cmd_self_validate,
        "evaluate": cmd_self_evaluate,
    }
    commands[args.self_command](args)


def _add_self_source_arguments(parser, *, selection=False):
    parser.add_argument(
        "--source",
        choices=["whatsapp-mac", "wacli", "json"],
        default="whatsapp-mac",
        help="Local read-only message source",
    )
    parser.add_argument(
        "--path",
        help=(
            "Source database or JSON path; WhatsApp Mac uses its standard "
            "ChatStorage.sqlite location by default"
        ),
    )
    parser.add_argument(
        "--key-file",
        default=str(DEFAULT_PSEUDONYM_KEY),
        help="Local secret used to create stable pseudonymous identifiers",
    )
    if selection:
        parser.add_argument(
            "--chat",
            action="append",
            default=[],
            help="Source chat ID to include; repeat for multiple chats",
        )
        parser.add_argument(
            "--all-chats",
            action="store_true",
            help="Include every chat from the selected source",
        )


def _add_self_build_arguments(parser):
    _add_self_source_arguments(parser, selection=True)
    parser.add_argument("--owner-name", required=True, help="Name of the profile owner")
    parser.add_argument("--interview", help="Completed owner interview YAML")
    parser.add_argument(
        "--include-third-party-context",
        action="store_true",
        help="Store third-party text as context, never as identity evidence",
    )


def cmd_research_init(args):
    """Initialize or resume a research-council run."""
    from .research.council import initialize_run

    manifest = initialize_run(
        args.run_dir,
        run_id=args.run_id,
        council_path=args.council,
        schema_path=args.schema,
    )
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "run_dir": str(Path(args.run_dir).resolve()),
                "seat_count": len(manifest["seats"]),
                "status": "initialized",
            },
            sort_keys=True,
        )
    )


def cmd_research_validate_seat(args):
    """Validate and register one research-seat artifact."""
    from .research.council import record_seat_artifact

    manifest = record_seat_artifact(
        args.run_dir,
        seat_id=args.seat,
        artifact_path=args.input,
        agent_id=args.agent_id,
        query_strategy=args.query,
    )
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "seat": args.seat,
                **manifest["seats"][args.seat],
            },
            sort_keys=True,
        )
    )


def cmd_research_merge(args):
    """Merge every validated council seat into one deduplicated draft corpus."""
    from .research.council import merge_run

    report = merge_run(args.run_dir, output_path=args.output)
    print(json.dumps(report.to_dict(), sort_keys=True))


def cmd_research_status(args):
    """Report text-free council run progress."""
    from .research.council import summarize_run

    print(json.dumps(summarize_run(args.run_dir), sort_keys=True))


def cmd_research(args):
    """Dispatch research-council subcommands."""
    commands = {
        "init": cmd_research_init,
        "validate-seat": cmd_research_validate_seat,
        "merge": cmd_research_merge,
        "status": cmd_research_status,
    }
    commands[args.research_command](args)


def cmd_brain_guide(args):
    """Run the guided digital-brain workflow in an explicit safe source mode."""
    from .brain.demo import run_guided_demo

    if not args.demo:
        raise ValueError("the guided workflow requires an explicit source mode")
    summary = run_guided_demo(args.workspace, as_of=args.as_of)
    print(json.dumps(summary.to_dict(), ensure_ascii=True, sort_keys=True))


def cmd_brain_coverage(args):
    """Report evidence strength and the next owner questions without state content."""
    from .brain.coverage import analyze_coverage
    from .brain.models import DigitalBrain

    brain = DigitalBrain.load(args.brain)
    report = analyze_coverage(
        brain,
        as_of=args.as_of,
        stale_after_days=args.stale_after_days,
        max_questions=args.max_questions,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=True, sort_keys=True))


def cmd_brain_inspect(args):
    """Inspect layered state while permanently redacting third-party content."""
    from .brain.inspection import inspect_brain
    from .brain.models import BrainLayer, DigitalBrain

    brain = DigitalBrain.load(args.brain)
    result = inspect_brain(
        brain,
        as_of=args.as_of,
        layer=BrainLayer(args.layer) if args.layer else None,
        relationship_id=args.relationship_id,
        include_history=args.history,
        include_payload=args.include_payload,
        include_sensitive=args.include_sensitive,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


def cmd_brain_migrate(args):
    """Migrate one private digital-self v1 profile into brain v2."""
    from .brain.migration import migrate_v1_profile
    from .identity.models import DigitalSelfProfile

    brain = migrate_v1_profile(DigitalSelfProfile.load(args.profile))
    brain.save(args.output)
    print(
        json.dumps(
            {
                "schema_version": brain.schema_version,
                "brain_id": brain.brain_id,
                "brain_version": brain.version,
                "output": str(Path(args.output)),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def cmd_brain_correct(args):
    """Apply one owner-authored correction from a private JSON file."""
    from .brain.models import DigitalBrain

    correction = json.loads(Path(args.correction).read_text(encoding="utf-8"))
    if not isinstance(correction, dict):
        raise ValueError("correction file must contain one JSON object")
    required = {"summary", "payload", "reason", "corrected_at"}
    missing = sorted(required - set(correction))
    unknown = sorted(set(correction) - required)
    if missing:
        raise ValueError(f"correction file is missing fields: {missing}")
    if unknown:
        raise ValueError(f"correction file has unknown fields: {unknown}")
    if not isinstance(correction["payload"], dict):
        raise ValueError("correction payload must be a JSON object")

    brain = DigitalBrain.load(args.brain)
    corrected = brain.apply_owner_correction(
        args.item_id,
        summary=correction["summary"],
        payload=correction["payload"],
        corrected_at=_aware_datetime(correction["corrected_at"]),
        reason=correction["reason"],
    )
    brain.save(args.output)
    print(
        json.dumps(
            {
                "brain_id": brain.brain_id,
                "brain_version": brain.version,
                "original_item_id": args.item_id,
                "replacement_item_id": corrected.id,
                "output": str(Path(args.output)),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def cmd_brain(args):
    """Dispatch digital-brain subcommands."""
    commands = {
        "guide": cmd_brain_guide,
        "coverage": cmd_brain_coverage,
        "inspect": cmd_brain_inspect,
        "migrate": cmd_brain_migrate,
        "correct": cmd_brain_correct,
    }
    commands[args.brain_command](args)


def _aware_datetime(value):
    """Parse one ISO timestamp for deterministic brain operations."""
    try:
        normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timestamp must be ISO 8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Living Brain: A Continuous Personal AI Clone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to config file",
        default=None,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse a WhatsApp export")
    parse_parser.add_argument("input", help="Path to WhatsApp export file")
    parse_parser.add_argument("--your-name", "-n", required=True, help="Your name in the chat")
    parse_parser.add_argument("--output", "-o", help="Output path for processed data")
    parse_parser.add_argument("--gap-minutes", type=int, default=60, help="Gap to split conversations")
    parse_parser.add_argument(
        "--date-order",
        choices=["mdy", "dmy"],
        default="mdy",
        help="Order for ambiguous slash dates",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a style adapter")
    train_parser.add_argument("data", help="Path to training data (JSONL or chat file)")
    train_parser.add_argument("--your-name", "-n", help="Your name (required if data is chat file)")
    train_parser.add_argument("--model", "-m", default="llama-3.2-3b", help="Base model")
    train_parser.add_argument("--output", "-o", default="./models/adapter", help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    train_parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    train_parser.add_argument("--context-turns", type=int, default=3, help="Context turns")
    train_parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    train_parser.add_argument("--export-gguf", action="store_true", help="Export to GGUF")
    train_parser.add_argument("--quantization", default="q4_k_m", help="GGUF quantization")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Launch chat interface")
    chat_parser.add_argument("--adapter", "-a", help="Adapter name to use")
    chat_parser.add_argument("--gguf", help="Path to GGUF model")
    chat_parser.add_argument("--share", action="store_true", help="Create public link")
    chat_parser.add_argument("--port", type=int, default=7860, help="Server port")

    # Dataset workbench command
    workbench_parser = subparsers.add_parser(
        "workbench",
        help="Launch persona dataset workbench",
    )
    workbench_parser.add_argument("--share", action="store_true", help="Create public link")
    workbench_parser.add_argument("--port", type=int, default=7861, help="Server port")
    workbench_parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Server host",
    )

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest conversations into memory")
    ingest_parser.add_argument("input", help="Path to WhatsApp export file")
    ingest_parser.add_argument("--your-name", "-n", required=True, help="Your name")
    ingest_parser.add_argument("--gap-minutes", type=int, default=60, help="Gap to split conversations")
    ingest_parser.add_argument("--extract-facts", action="store_true", help="Extract facts")

    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch for new exports")
    watch_parser.add_argument("watch_dir", help="Directory to watch")
    watch_parser.add_argument("--your-name", "-n", required=True, help="Your name")
    watch_parser.add_argument("--process-existing", action="store_true", help="Process existing files")
    watch_parser.add_argument("--extract-facts", action="store_true", help="Extract facts")

    # Stats command
    subparsers.add_parser("stats", help="Show system statistics")

    # General digital-self commands
    self_parser = subparsers.add_parser(
        "self",
        help="Build and evaluate a general digital self",
    )
    self_subparsers = self_parser.add_subparsers(
        dest="self_command",
        required=True,
        help="Digital-self commands",
    )

    interview_parser = self_subparsers.add_parser(
        "interview",
        help="Create a guided owner interview",
    )
    interview_parser.add_argument("--owner-name", required=True, help="Profile owner")
    interview_parser.add_argument("--output", required=True, help="Interview YAML path")

    chats_parser = self_subparsers.add_parser(
        "chats",
        help="List chats from a local read-only source",
    )
    _add_self_source_arguments(chats_parser)

    build_parser = self_subparsers.add_parser(
        "build",
        help="Build a versioned digital-self profile",
    )
    _add_self_build_arguments(build_parser)
    build_parser.add_argument("--output", required=True, help="Profile JSON path")

    validate_parser = self_subparsers.add_parser(
        "validate",
        help="Validate a digital-self profile",
    )
    validate_parser.add_argument("profile", help="Profile JSON path")

    evaluate_parser = self_subparsers.add_parser(
        "evaluate",
        help="Export held-out and interview-retest evaluation artifacts",
    )
    _add_self_build_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--profile",
        help="Existing profile whose owner and source summary must match",
    )
    evaluate_parser.add_argument("--output", required=True, help="Private suite JSON path")
    evaluate_parser.add_argument(
        "--summary-output",
        help="Text-free summary path (defaults beside the private suite)",
    )

    # Public research-council commands
    research_parser = subparsers.add_parser(
        "research",
        help="Run and validate the digital-self research council",
    )
    research_subparsers = research_parser.add_subparsers(
        dest="research_command",
        required=True,
        help="Research council commands",
    )

    research_init_parser = research_subparsers.add_parser(
        "init",
        help="Initialize or resume a council run",
    )
    research_init_parser.add_argument("--run-dir", required=True, help="Run state directory")
    research_init_parser.add_argument("--run-id", required=True, help="Stable run identifier")
    research_init_parser.add_argument(
        "--council",
        required=True,
        help="Council YAML contract",
    )
    research_init_parser.add_argument(
        "--schema",
        required=True,
        help="Paper record JSON Schema",
    )

    research_validate_parser = research_subparsers.add_parser(
        "validate-seat",
        help="Validate and register one seat JSONL artifact",
    )
    research_validate_parser.add_argument(
        "--run-dir",
        required=True,
        help="Run state directory",
    )
    research_validate_parser.add_argument("--seat", required=True, help="Council seat id")
    research_validate_parser.add_argument("--input", required=True, help="Seat JSONL path")
    research_validate_parser.add_argument(
        "--agent-id",
        required=True,
        help="Research agent provenance id",
    )
    research_validate_parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Search strategy query; repeat to record multiple queries",
    )

    research_merge_parser = research_subparsers.add_parser(
        "merge",
        help="Merge all complete seats into a deduplicated draft corpus",
    )
    research_merge_parser.add_argument("--run-dir", required=True, help="Run state directory")
    research_merge_parser.add_argument("--output", required=True, help="Draft corpus JSONL")

    research_status_parser = research_subparsers.add_parser(
        "status",
        help="Report council run progress",
    )
    research_status_parser.add_argument("--run-dir", required=True, help="Run state directory")

    # Evidence-grounded digital-brain commands
    brain_parser = subparsers.add_parser(
        "brain",
        help="Inspect and exercise the evidence-grounded digital brain",
    )
    brain_subparsers = brain_parser.add_subparsers(
        dest="brain_command",
        required=True,
        help="Digital-brain commands",
    )
    brain_guide_parser = brain_subparsers.add_parser(
        "guide",
        help="Run source selection through evaluation in one local workflow",
    )
    brain_guide_parser.add_argument(
        "--demo",
        action="store_true",
        required=True,
        help="Use the bundled synthetic fixture without reading private data",
    )
    brain_guide_parser.add_argument(
        "--workspace",
        required=True,
        help="Private directory for versioned workflow artifacts",
    )
    brain_guide_parser.add_argument(
        "--as-of",
        type=_aware_datetime,
        default=datetime.now(timezone.utc),
        help="Evaluation time as an offset-aware ISO 8601 timestamp",
    )

    brain_migrate_parser = brain_subparsers.add_parser(
        "migrate",
        help="Migrate a private digital-self v1 profile into brain v2",
    )
    brain_migrate_parser.add_argument("profile", help="Digital-self v1 JSON path")
    brain_migrate_parser.add_argument(
        "--output",
        required=True,
        help="Owner-only digital-brain v2 JSON path",
    )

    brain_correct_parser = brain_subparsers.add_parser(
        "correct",
        help="Apply an owner correction without placing private text in shell history",
    )
    brain_correct_parser.add_argument("brain", help="Digital-brain v2 JSON path")
    brain_correct_parser.add_argument(
        "--item-id",
        required=True,
        help="Current state item to supersede",
    )
    brain_correct_parser.add_argument(
        "--correction",
        required=True,
        help="Private JSON with summary, payload, reason, and corrected_at",
    )
    brain_correct_parser.add_argument(
        "--output",
        required=True,
        help="Owner-only corrected digital-brain JSON path",
    )

    brain_coverage_parser = brain_subparsers.add_parser(
        "coverage",
        help="Report strong, weak, stale, unknown, and conflicting state",
    )
    brain_coverage_parser.add_argument("brain", help="Digital-brain v2 JSON path")
    brain_coverage_parser.add_argument(
        "--as-of",
        type=_aware_datetime,
        default=datetime.now(timezone.utc),
        help="Coverage time as an offset-aware ISO 8601 timestamp",
    )
    brain_coverage_parser.add_argument(
        "--stale-after-days",
        type=int,
        default=180,
        help="Days after which persistent state needs owner reconfirmation",
    )
    brain_coverage_parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        help="Maximum highest-value owner questions to return",
    )

    brain_inspect_parser = brain_subparsers.add_parser(
        "inspect",
        help="Inspect state by layer, relationship scope, and provenance",
    )
    brain_inspect_parser.add_argument("brain", help="Digital-brain v2 JSON path")
    brain_inspect_parser.add_argument(
        "--as-of",
        type=_aware_datetime,
        default=datetime.now(timezone.utc),
        help="Inspection time as an offset-aware ISO 8601 timestamp",
    )
    brain_inspect_parser.add_argument(
        "--layer",
        choices=[
            "event",
            "episode",
            "semantic",
            "procedural",
            "self_schema",
            "values_goals",
            "affect",
            "social",
            "narrative",
            "communication",
            "uncertainty",
            "reflection",
        ],
        help="Restrict inspection to one brain layer",
    )
    brain_inspect_parser.add_argument(
        "--relationship-id",
        help="Include global state and exactly one relationship scope",
    )
    brain_inspect_parser.add_argument(
        "--history",
        action="store_true",
        help="Include historical and rejected versions",
    )
    brain_inspect_parser.add_argument(
        "--include-payload",
        action="store_true",
        help="Include owner payloads; third-party payloads remain redacted",
    )
    brain_inspect_parser.add_argument(
        "--include-sensitive",
        action="store_true",
        help="Reveal owner-sensitive summaries when explicitly requested",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Route to command
    commands = {
        "parse": cmd_parse,
        "train": cmd_train,
        "chat": cmd_chat,
        "workbench": cmd_workbench,
        "ingest": cmd_ingest,
        "watch": cmd_watch,
        "stats": cmd_stats,
        "self": cmd_self,
        "research": cmd_research,
        "brain": cmd_brain,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
