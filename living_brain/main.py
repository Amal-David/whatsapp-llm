"""
Main entry point for Living Brain CLI.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_parse(args):
    """Parse a WhatsApp export file."""
    from .ingest.style_analyzer import StyleAnalyzer
    from .ingest.whatsapp_parser import WhatsAppParser

    parser = WhatsAppParser(your_name=args.your_name)

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
        vector_store.add(
            content=conv.to_text(),
            timestamp=conv.start_time,
            metadata={
                "source": str(args.input),
                "message_count": len(conv.messages),
            },
        )
        memories_added += 1

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
