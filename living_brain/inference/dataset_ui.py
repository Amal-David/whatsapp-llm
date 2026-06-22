"""
Gradio workbench for persona dataset and character-file exports.
"""

from __future__ import annotations

import inspect
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from ..ingest.persona_dataset import PersonaBuildResult, PersonaDatasetBuilder
from ..ingest.sample_recommender import SampleTextRecommender
from ..ingest.whatsapp_parser import WhatsAppParser

logger = logging.getLogger(__name__)


class DatasetWorkbench:
    """Drop-in Gradio UI for WhatsApp persona dataset preparation."""

    def __init__(self, title: str = "Persona Dataset Workbench"):
        self.title = title

    def _upload_path(self, upload: Any) -> Path:
        if upload is None:
            raise ValueError("Upload a WhatsApp .txt export first.")
        if isinstance(upload, (str, Path)):
            return Path(upload)
        name = getattr(upload, "name", None)
        if name:
            return Path(name)
        raise ValueError("Could not read the uploaded file path.")

    def _analyze_upload(self, upload: Any):
        try:
            import gradio as gr

            path = self._upload_path(upload)
            parser = WhatsAppParser()
            participants = sorted(parser.get_participants(path))
            messages = list(parser.parse_file(path, skip_system=False))
            system_count = sum(1 for message in messages if message.is_system)
            analysis = {
                "file": path.name,
                "participants": participants,
                "message_count": len(messages),
                "system_or_media_count": system_count,
                "next_step": "Choose the participant whose authorized style profile you want to build.",
            }
            selected = participants[0] if participants else None
            return gr.update(choices=participants, value=selected), analysis, ""
        except Exception as exc:
            logger.exception("Dataset analysis failed")
            return None, {"error": str(exc)}, ""

    def _build_artifacts(
        self,
        upload: Any,
        participant: str,
        owner_type: str,
        consent_confirmed: bool,
        include_third_party_context: bool,
        context_turns: int,
        gap_minutes: int,
    ):
        try:
            if not consent_confirmed:
                raise ValueError(
                    "Confirm that the target messages are yours or explicitly consented."
                )
            if not participant:
                raise ValueError("Choose a participant first.")

            path = self._upload_path(upload)
            builder = PersonaDatasetBuilder(
                context_turns=int(context_turns),
                gap_minutes=int(gap_minutes),
                include_third_party_context=include_third_party_context,
            )
            result = builder.build_from_file(path, participant, owner_type=owner_type)
            zip_path = self._zip_artifacts(builder, result)
            summary = json.dumps(result.summary, ensure_ascii=False, indent=2)
            return summary, result.recommendation_markdown, result.persona_markdown, zip_path, ""
        except Exception as exc:
            logger.exception("Dataset build failed")
            return "{}", "", "", None, str(exc)

    def _analyze_sample(
        self,
        sample_text: str,
        sample_context: str,
        sample_persona: str,
        sample_consent: bool,
        target_message_count: int,
        desired_outputs: int,
        unary_feedback_labels: int,
    ):
        try:
            if not sample_consent:
                raise ValueError("Confirm that this sample is yours or explicitly consented.")
            if not sample_text.strip():
                raise ValueError("Paste sample text first.")

            recommendation = SampleTextRecommender().recommend(
                sample_text=sample_text,
                target_message_count=int(target_message_count),
                desired_outputs=int(desired_outputs),
                persona_name=sample_persona.strip() or "Target",
                context=sample_context.strip(),
                unary_feedback_labels=int(unary_feedback_labels),
            )
            export_path = self._sample_export_file(recommendation)
            return (
                recommendation.to_json(),
                recommendation.to_markdown(),
                json.dumps(recommendation.style_card, ensure_ascii=False, indent=2),
                recommendation.prompt_snippet,
                export_path,
                "",
            )
        except Exception as exc:
            logger.exception("Sample recommendation failed")
            return "{}", "", "{}", "", None, str(exc)

    def _zip_artifacts(self, builder: PersonaDatasetBuilder, result: PersonaBuildResult) -> str:
        temp = tempfile.NamedTemporaryFile(
            prefix=f"{result.participant.replace(' ', '_')}_persona_",
            suffix=".zip",
            delete=False,
        )
        temp.close()
        with zipfile.ZipFile(temp.name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for filename, content in builder.artifact_strings(result).items():
                archive.writestr(filename, content)
        return temp.name

    def _sample_export_file(self, recommendation) -> str:
        temp = tempfile.NamedTemporaryFile(
            prefix="sample_text_recommendation_",
            suffix=".json",
            delete=False,
            mode="w",
            encoding="utf-8",
        )
        with temp:
            temp.write(recommendation.to_json())
        return temp.name

    def create_interface(self):
        """Create and return the Gradio interface."""
        try:
            import gradio as gr
        except ImportError as exc:
            raise ImportError(
                "gradio is required for DatasetWorkbench. Install with: pip install gradio"
            ) from exc

        block_kwargs = {"title": self.title}
        if "theme" in inspect.signature(gr.Blocks).parameters:
            block_kwargs["theme"] = gr.themes.Soft()

        with gr.Blocks(**block_kwargs) as interface:
            gr.Markdown(f"# {self.title}")

            with gr.Tabs():
                with gr.Tab("WhatsApp Export"):
                    with gr.Row():
                        upload = gr.File(
                            label="WhatsApp export",
                            file_types=[".txt"],
                            type="filepath",
                        )
                        with gr.Column():
                            owner_type = gr.Dropdown(
                                label="Profile owner",
                                choices=["self", "organization", "consented_person"],
                                value="self",
                            )
                            consent_confirmed = gr.Checkbox(
                                label="I wrote these target messages or have explicit permission.",
                                value=False,
                            )

                    with gr.Row():
                        context_turns = gr.Slider(
                            label="Context turns",
                            minimum=1,
                            maximum=12,
                            value=6,
                            step=1,
                        )
                        gap_minutes = gr.Slider(
                            label="Conversation gap minutes",
                            minimum=15,
                            maximum=360,
                            value=60,
                            step=15,
                        )
                        include_third_party_context = gr.Checkbox(
                            label="Include other participants as context",
                            value=True,
                        )

                    with gr.Row():
                        analyze = gr.Button("Analyze", variant="secondary")
                        participant = gr.Dropdown(label="Participant", choices=[], value=None)
                        build = gr.Button("Build Dataset", variant="primary")

                    status = gr.Textbox(label="Status", interactive=False)
                    analysis = gr.JSON(label="Analysis")

                    with gr.Tabs():
                        with gr.Tab("Summary"):
                            summary = gr.Code(label="summary.json", language="json")
                        with gr.Tab("Recommendation"):
                            recommendation = gr.Markdown()
                        with gr.Tab("Persona"):
                            persona = gr.Code(label="persona.md", language="markdown")
                        with gr.Tab("Artifacts"):
                            artifacts = gr.File(label="Download ZIP")

                    analyze.click(
                        self._analyze_upload,
                        inputs=[upload],
                        outputs=[participant, analysis, status],
                    )
                    build.click(
                        self._build_artifacts,
                        inputs=[
                            upload,
                            participant,
                            owner_type,
                            consent_confirmed,
                            include_third_party_context,
                            context_turns,
                            gap_minutes,
                        ],
                        outputs=[summary, recommendation, persona, artifacts, status],
                    )

                with gr.Tab("Sample Text"):
                    with gr.Row():
                        sample_text = gr.Textbox(
                            label="Sample text",
                            lines=12,
                            placeholder="Paste messages, one per line. Speaker prefixes are okay.",
                        )
                        with gr.Column():
                            sample_persona = gr.Textbox(label="Persona name", value="Target")
                            sample_context = gr.Textbox(
                                label="Optional context",
                                lines=4,
                                placeholder="Relationship, language mix, chat setting, or target use case.",
                            )
                            sample_consent = gr.Checkbox(
                                label="I wrote this sample or have explicit permission.",
                                value=False,
                            )

                    with gr.Row():
                        target_message_count = gr.Number(
                            label="Available target messages",
                            value=0,
                            precision=0,
                        )
                        desired_outputs = gr.Number(
                            label="Desired generated examples",
                            value=100,
                            precision=0,
                        )
                        unary_feedback_labels = gr.Number(
                            label="Unary feedback labels",
                            value=0,
                            precision=0,
                        )

                    analyze_sample = gr.Button("Analyze Sample", variant="primary")
                    sample_status = gr.Textbox(label="Sample status", interactive=False)

                    with gr.Tabs():
                        with gr.Tab("Recommendation JSON"):
                            sample_json = gr.Code(label="sample_recommendation.json", language="json")
                        with gr.Tab("Readable Plan"):
                            sample_markdown = gr.Markdown()
                        with gr.Tab("Style Card"):
                            sample_style_card = gr.Code(label="style_card.json", language="json")
                        with gr.Tab("Prompt"):
                            sample_prompt = gr.Code(label="prompt_snippet.md", language="markdown")
                        with gr.Tab("Export"):
                            sample_export = gr.File(label="Download JSON")

                    analyze_sample.click(
                        self._analyze_sample,
                        inputs=[
                            sample_text,
                            sample_context,
                            sample_persona,
                            sample_consent,
                            target_message_count,
                            desired_outputs,
                            unary_feedback_labels,
                        ],
                        outputs=[
                            sample_json,
                            sample_markdown,
                            sample_style_card,
                            sample_prompt,
                            sample_export,
                            sample_status,
                        ],
                    )

        return interface

    def launch(
        self,
        share: bool = False,
        server_name: str = "127.0.0.1",
        server_port: int = 7861,
        **kwargs,
    ):
        try:
            import gradio as gr
        except ImportError as exc:
            raise ImportError(
                "gradio is required for DatasetWorkbench. Install with: pip install gradio"
            ) from exc

        interface = self.create_interface()
        launch_kwargs = dict(kwargs)
        if "theme" in inspect.signature(interface.launch).parameters:
            launch_kwargs.setdefault("theme", gr.themes.Soft())

        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            **launch_kwargs,
        )


def launch_dataset_workbench(
    share: bool = False,
    port: int = 7861,
    server_name: str = "127.0.0.1",
):
    """Launch the dataset workbench without loading a model."""
    DatasetWorkbench().launch(
        share=share,
        server_name=server_name,
        server_port=port,
    )
