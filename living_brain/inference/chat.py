"""
Gradio-based chat interface for Living Brain.
"""

import logging
from pathlib import Path
from typing import Optional

from .orchestrator import Orchestrator
from ..core.config import Config

logger = logging.getLogger(__name__)


class ChatInterface:
    """
    Gradio-based chat interface for interacting with the Living Brain.
    """

    def __init__(
        self,
        orchestrator: Optional[Orchestrator] = None,
        config: Optional[Config] = None,
        title: str = "Living Brain Chat",
    ):
        """
        Initialize the chat interface.

        Args:
            orchestrator: Pre-configured orchestrator (will create one if not provided)
            config: Configuration (used if orchestrator not provided)
            title: Title for the chat interface
        """
        self.orchestrator = orchestrator
        self.config = config or Config()
        self.title = title

        if self.orchestrator is None:
            self.orchestrator = Orchestrator(config=self.config)

    def _respond(
        self,
        message: str,
        history: list,
        use_memory: bool,
        use_facts: bool,
        temperature: float,
    ):
        """Generate a response for Gradio."""
        if not message.strip():
            return "", history

        try:
            result = self.orchestrator.generate(
                message=message,
                use_memory=use_memory,
                use_facts=use_facts,
                temperature=temperature,
            )

            # Update history for Gradio
            history.append((message, result.response))

            return "", history

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "", history + [(message, f"Error: {str(e)}")]

    def _clear(self):
        """Clear chat history."""
        self.orchestrator.clear_history()
        return None, []

    def _get_stats(self):
        """Get system statistics."""
        stats = self.orchestrator.get_stats()
        lines = [
            f"**Model**: {stats.get('model', 'N/A')}",
            f"**Adapter**: {stats.get('adapter', 'None')}",
            f"**Memories**: {stats.get('total_memories', 0)}",
            f"**Facts**: {stats.get('total_facts', 0)}",
            f"**History**: {stats.get('history_length', 0)} turns",
        ]
        return "\n".join(lines)

    def _add_fact(self, subject: str, predicate: str, obj: str):
        """Add a fact to the knowledge base."""
        if not all([subject.strip(), predicate.strip(), obj.strip()]):
            return "Please fill in all fields"

        self.orchestrator.fact_store.add(
            subject=subject.strip(),
            predicate=predicate.strip(),
            obj=obj.strip(),
            source="manual",
        )
        return f"Added: {subject} {predicate} {obj}"

    def create_interface(self):
        """Create and return the Gradio interface."""
        try:
            import gradio as gr
        except ImportError:
            raise ImportError(
                "gradio is required for ChatInterface. "
                "Install with: pip install gradio"
            )

        with gr.Blocks(title=self.title, theme=gr.themes.Soft()) as interface:
            gr.Markdown(f"# {self.title}")
            gr.Markdown(
                f"Chat with your personal AI clone: **{self.config.persona_name}**"
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        show_copy_button=True,
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here...",
                            scale=4,
                            show_label=False,
                        )
                        submit = gr.Button("Send", variant="primary", scale=1)

                    with gr.Row():
                        clear = gr.Button("Clear Chat")
                        save_memory = gr.Button("Save to Memory")

                with gr.Column(scale=1):
                    gr.Markdown("### Settings")

                    use_memory = gr.Checkbox(
                        label="Use Memory (RAG)",
                        value=True,
                    )
                    use_facts = gr.Checkbox(
                        label="Use Facts",
                        value=True,
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                    )

                    gr.Markdown("### System Stats")
                    stats_display = gr.Markdown(self._get_stats())
                    refresh_stats = gr.Button("Refresh Stats", size="sm")

                    gr.Markdown("### Add Fact")
                    fact_subject = gr.Textbox(label="Subject", placeholder="I")
                    fact_predicate = gr.Textbox(label="Predicate", placeholder="live in")
                    fact_object = gr.Textbox(label="Object", placeholder="NYC")
                    add_fact_btn = gr.Button("Add Fact", size="sm")
                    fact_result = gr.Textbox(label="Result", interactive=False)

            # Event handlers
            submit.click(
                self._respond,
                inputs=[msg, chatbot, use_memory, use_facts, temperature],
                outputs=[msg, chatbot],
            )
            msg.submit(
                self._respond,
                inputs=[msg, chatbot, use_memory, use_facts, temperature],
                outputs=[msg, chatbot],
            )
            clear.click(
                self._clear,
                outputs=[msg, chatbot],
            )
            refresh_stats.click(
                self._get_stats,
                outputs=stats_display,
            )
            add_fact_btn.click(
                self._add_fact,
                inputs=[fact_subject, fact_predicate, fact_object],
                outputs=fact_result,
            )

            def save_conversation(history):
                if not history:
                    return "No conversation to save"
                conversation = "\n".join([
                    f"User: {h[0]}\nAssistant: {h[1]}"
                    for h in history
                ])
                memory_id, facts = self.orchestrator.add_to_memory(conversation)
                return f"Saved to memory (ID: {memory_id[:8]}..., {len(facts)} facts extracted)"

            save_memory.click(
                save_conversation,
                inputs=[chatbot],
                outputs=fact_result,
            )

        return interface

    def launch(
        self,
        share: bool = False,
        server_name: str = "127.0.0.1",
        server_port: int = 7860,
        **kwargs,
    ):
        """
        Launch the Gradio interface.

        Args:
            share: Whether to create a public link
            server_name: Server host
            server_port: Server port
            **kwargs: Additional arguments to pass to Gradio launch
        """
        interface = self.create_interface()

        # Load model before launching
        logger.info("Loading model...")
        self.orchestrator.load_model()
        logger.info("Model loaded, launching interface...")

        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            **kwargs,
        )


def launch_chat(
    config_path: Optional[str] = None,
    adapter_name: Optional[str] = None,
    use_gguf: bool = False,
    gguf_path: Optional[str] = None,
    share: bool = False,
    port: int = 7860,
):
    """
    Convenience function to launch the chat interface.

    Args:
        config_path: Path to config YAML file
        adapter_name: Name of LoRA adapter to use
        use_gguf: Whether to use GGUF model
        gguf_path: Path to GGUF model file
        share: Whether to create public link
        port: Server port
    """
    from ..core.config import load_config

    config = load_config(config_path)

    orchestrator = Orchestrator(
        config=config,
        adapter_name=adapter_name,
        use_gguf=use_gguf,
        gguf_path=gguf_path,
    )

    interface = ChatInterface(
        orchestrator=orchestrator,
        config=config,
        title=f"Living Brain: {config.persona_name}",
    )

    interface.launch(share=share, server_port=port)
