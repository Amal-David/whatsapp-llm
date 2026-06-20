import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from parser import (
    LLMFormat,
    converter_with_debug
)
from persona import PersonaBuilder, save_persona_profile
from privacy import PIIRedactor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def write_jsonl(records, path: Path) -> None:
    with path.open('w') as f:
        for item in records:
            f.write(json.dumps(item) + '\n')


def style_metrics_for_author(author_style_metrics, author_name: str, fallback):
    for candidate, metrics in author_style_metrics.items():
        if str(candidate).casefold() == author_name.casefold():
            return metrics
    return fallback


def run_finetune_pipeline(
    data_path: Path,
    model_name: str,
    output_dir: Path,
    tokenizer_name: Optional[str],
    persona_summary_path: Optional[Path],
    style_metrics_path: Optional[Path],
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    max_length: int,
    gradient_accumulation_steps: int,
    use_8bit: bool,
    use_peft: bool,
    use_flash_attention: bool
) -> None:
    from finetune import (
        DataCollatorForLanguageModeling,
        TrainingArguments,
        load_and_process_data,
        setup_model_and_tokenizer,
        setup_peft_config,
        train
    )
    from peft import get_peft_model

    persona_summary = None
    if persona_summary_path and persona_summary_path.exists():
        persona_summary = str(persona_summary_path)

    style_metrics = None
    if style_metrics_path and style_metrics_path.exists():
        with style_metrics_path.open() as f:
            style_metrics = json.load(f)

    model, tokenizer = setup_model_and_tokenizer(
        model_name,
        tokenizer_name,
        load_in_8bit=use_8bit,
        style_metrics=style_metrics
    )

    dataset = load_and_process_data(
        str(data_path),
        tokenizer,
        max_length=max_length,
        persona_summary_path=persona_summary
    )

    if use_peft:
        logger.info("Applying LoRA configuration for fine-tuning")
        peft_config = setup_peft_config(model_name)
        model = get_peft_model(model, peft_config)

    if use_flash_attention:
        is_llama3 = "llama-3" in model_name.lower() or "llama3" in model_name.lower()
        is_qwen = "qwen" in model_name.lower()
        if is_llama3 or is_qwen:
            logger.info("Enabling Flash Attention for %s", model_name)
            model.config.use_flash_attn = True
        else:
            logger.warning("Flash Attention is not supported for model %s", model_name)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    train(model, dataset, training_args, data_collator, style_metrics)


def run_persona_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir or f"persona_artifacts_{args.contact.lower().replace(' ', '_')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    redactor = PIIRedactor() if args.redact else None

    llm_format = LLMFormat[args.llm_format.upper()]
    parsed_pairs, formatted_records, style_metrics, persona_context = converter_with_debug(
        args.chat,
        args.prompter,
        args.responder,
        args.your_name,
        llm_format,
        args.context_length,
        contact_name=args.contact,
        include_self=args.include_self,
        episode_gap_hours=args.episode_gap_hours
    )

    pairs_path = output_dir / 'conversation_pairs.csv'
    parsed_pairs.to_csv(pairs_path)
    logger.info("Saved conversation pairs to %s", pairs_path)

    formatted_path = output_dir / 'formatted_dataset.jsonl'
    write_jsonl(formatted_records, formatted_path)
    logger.info("Saved formatted dataset to %s", formatted_path)

    style_metrics_path = output_dir / f'style_metrics_{args.your_name}.json'
    style_metrics_output = redactor.redact_value(style_metrics) if redactor else style_metrics
    with style_metrics_path.open('w') as f:
        json.dump(style_metrics_output, f, indent=2)
    logger.info("Saved style metrics to %s", style_metrics_path)

    persona_metadata_path = output_dir / 'persona_metadata.json'
    persona_metadata_output = redactor.redact_value(persona_context) if redactor else persona_context
    with persona_metadata_path.open('w') as f:
        json.dump(persona_metadata_output, f, indent=2, default=str)
    logger.info("Saved persona metadata to %s", persona_metadata_path)

    builder = PersonaBuilder(
        your_name=args.your_name,
        contact_name=args.contact,
        redactor=redactor,
        redact=args.redact
    )
    persona_style_metrics = style_metrics_for_author(
        persona_context.get('author_style_metrics', {}),
        args.contact,
        style_metrics
    )
    persona_profile = builder.build_profile(
        persona_context.get('tagged_messages', []),
        persona_style_metrics,
        persona_context.get('episode_metadata')
    )

    persona_path = Path(args.persona_path) if args.persona_path else output_dir / f"persona_{args.contact.lower().replace(' ', '_')}.yaml"
    save_persona_profile(persona_profile, str(persona_path))
    logger.info("Saved persona profile to %s", persona_path)

    if args.run_finetune:
        if not args.model_name or not args.output_model_dir:
            raise ValueError('Model name and output model directory are required when --run-finetune is set.')

        run_finetune_pipeline(
            formatted_path,
            args.model_name,
            Path(args.output_model_dir),
            args.tokenizer_name,
            persona_path,
            style_metrics_path,
            args.batch_size,
            args.learning_rate,
            args.num_epochs,
            args.max_length,
            args.gradient_accumulation_steps,
            args.use_8bit,
            args.use_peft,
            args.use_flash_attention
        )


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='WhatsApp persona export orchestration CLI')
    subparsers = parser.add_subparsers(dest='command')

    persona_parser = subparsers.add_parser('persona', help='Generate persona artifacts from a chat log')
    persona_parser.add_argument('--chat', required=True, help='Path to the exported WhatsApp chat file')
    persona_parser.add_argument('--contact', required=True, help='Name of the contact to model')
    persona_parser.add_argument('--your-name', required=True, help='Your name as it appears in the chat')
    persona_parser.add_argument('--prompter', required=True, help='Name considered as the conversation starter')
    persona_parser.add_argument('--responder', required=True, help='Name considered as the responder (typically you)')
    persona_parser.add_argument('--llm-format', default='llama2', help='Target LLM format for dataset generation')
    persona_parser.add_argument('--context-length', type=int, default=3, help='Context window size for dataset creation')
    persona_parser.add_argument('--include-self', action='store_true', help='Include your own messages when filtering by contact')
    persona_parser.add_argument('--episode-gap-hours', type=float, default=6.0, help='Episode gap threshold in hours')
    persona_parser.add_argument('--output-dir', type=str, default=None, help='Directory to store generated artifacts')
    persona_parser.add_argument('--persona-path', type=str, default=None, help='Optional custom path for the persona file')
    persona_parser.add_argument('--redact', action='store_true', help='Apply PII redaction to persona outputs')
    persona_parser.add_argument('--run-finetune', action='store_true', help='Run fine-tuning after persona generation')
    persona_parser.add_argument('--model-name', type=str, default=None, help='Base model name or path for fine-tuning')
    persona_parser.add_argument('--tokenizer-name', type=str, default=None, help='Tokenizer name if different from model')
    persona_parser.add_argument('--output-model-dir', type=str, default=None, help='Directory for the fine-tuned model')
    persona_parser.add_argument('--batch-size', type=int, default=4)
    persona_parser.add_argument('--learning-rate', type=float, default=2e-5)
    persona_parser.add_argument('--num-epochs', type=int, default=3)
    persona_parser.add_argument('--max-length', type=int, default=2048)
    persona_parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    persona_parser.add_argument('--use-8bit', action='store_true')
    persona_parser.add_argument('--use-peft', action='store_true')
    persona_parser.add_argument('--use-flash-attention', action='store_true')
    persona_parser.set_defaults(func=run_persona_command)

    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    if not getattr(args, 'command', None):
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()
