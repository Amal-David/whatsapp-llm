"""
File watcher for continuous ingestion of new WhatsApp exports.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


class WhatsAppExportHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """Handler for new WhatsApp export files."""

    def __init__(
        self,
        your_name: str,
        on_new_file: Callable[[Path], None],
        file_patterns: tuple[str, ...] = ("*.txt", "*.zip"),
    ):
        """
        Initialize the handler.

        Args:
            your_name: Your name for parsing
            on_new_file: Callback when a new file is detected
            file_patterns: File patterns to watch for
        """
        if WATCHDOG_AVAILABLE:
            super().__init__()
        self.your_name = your_name
        self.on_new_file = on_new_file
        self.file_patterns = file_patterns
        self._processed_files: set[str] = set()

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return

        filepath = Path(event.src_path)

        # Check if file matches patterns
        if not any(filepath.match(pattern) for pattern in self.file_patterns):
            return

        # Avoid processing the same file twice
        if str(filepath) in self._processed_files:
            return

        self._processed_files.add(str(filepath))
        logger.info(f"New file detected: {filepath}")

        # Wait a moment for the file to be fully written
        time.sleep(1)

        try:
            self.on_new_file(filepath)
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")


class ExportWatcher:
    """
    Watches a directory for new WhatsApp exports and processes them.
    """

    def __init__(
        self,
        watch_dir: str | Path,
        your_name: str,
        vector_store=None,
        fact_store=None,
        auto_extract_facts: bool = True,
    ):
        """
        Initialize the watcher.

        Args:
            watch_dir: Directory to watch for new exports
            your_name: Your name for parsing
            vector_store: VectorStore instance for storing memories
            fact_store: FactStore instance for extracting facts
            auto_extract_facts: Whether to automatically extract facts
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError(
                "watchdog is required for ExportWatcher. "
                "Install with: pip install watchdog"
            )

        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        self.your_name = your_name
        self.vector_store = vector_store
        self.fact_store = fact_store
        self.auto_extract_facts = auto_extract_facts

        self._observer = None
        self._handler = None
        self._is_running = False

        # Stats
        self._files_processed = 0
        self._memories_added = 0
        self._facts_extracted = 0

    def _process_file(self, filepath: Path) -> None:
        """Process a new WhatsApp export file."""
        from .whatsapp_parser import WhatsAppParser
        from .style_analyzer import StyleAnalyzer

        logger.info(f"Processing: {filepath}")

        # Handle zip files
        if filepath.suffix == ".zip":
            filepath = self._extract_zip(filepath)
            if filepath is None:
                return

        # Parse the file
        parser = WhatsAppParser(your_name=self.your_name)
        conversations = parser.parse_to_conversations(filepath)

        if not conversations:
            logger.warning(f"No conversations found in {filepath}")
            return

        # Get your messages for style analysis
        your_messages = parser.get_your_messages(filepath)

        # Store conversations in memory
        if self.vector_store:
            for conv in conversations:
                conv_text = conv.to_text()
                metadata = {
                    "source_file": str(filepath),
                    "participants": ",".join(set(m.author for m in conv.messages)),
                    "message_count": len(conv.messages),
                }

                self.vector_store.add(
                    content=conv_text,
                    timestamp=conv.start_time,
                    metadata=metadata,
                )
                self._memories_added += 1

        # Extract facts
        if self.fact_store and self.auto_extract_facts:
            for msg in your_messages:
                facts = self.fact_store.ingest_and_extract(msg.message)
                self._facts_extracted += len(facts)

        self._files_processed += 1
        logger.info(
            f"Processed {filepath}: "
            f"{len(conversations)} conversations, "
            f"{len(your_messages)} messages"
        )

    def _extract_zip(self, zip_path: Path) -> Optional[Path]:
        """Extract a zip file and return the txt file path."""
        import zipfile

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find txt files
                txt_files = [f for f in zf.namelist() if f.endswith('.txt')]
                if not txt_files:
                    logger.warning(f"No txt files found in {zip_path}")
                    return None

                # Extract to same directory
                txt_file = txt_files[0]
                extract_path = zip_path.parent / txt_file
                zf.extract(txt_file, zip_path.parent)

                return extract_path

        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return None

    def start(self, blocking: bool = True) -> None:
        """
        Start watching for new files.

        Args:
            blocking: Whether to block the main thread
        """
        if self._is_running:
            logger.warning("Watcher is already running")
            return

        self._handler = WhatsAppExportHandler(
            your_name=self.your_name,
            on_new_file=self._process_file,
        )

        self._observer = Observer()
        self._observer.schedule(self._handler, str(self.watch_dir), recursive=False)
        self._observer.start()
        self._is_running = True

        logger.info(f"Watching for new exports in: {self.watch_dir}")

        if blocking:
            try:
                while self._is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()

    def stop(self) -> None:
        """Stop watching for new files."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._is_running = False
            logger.info("Watcher stopped")

    def process_existing(self) -> int:
        """
        Process any existing files in the watch directory.

        Returns:
            Number of files processed
        """
        count = 0
        for filepath in self.watch_dir.glob("*.txt"):
            try:
                self._process_file(filepath)
                count += 1
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")

        for filepath in self.watch_dir.glob("*.zip"):
            try:
                self._process_file(filepath)
                count += 1
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")

        return count

    def get_stats(self) -> dict:
        """Get watcher statistics."""
        return {
            "files_processed": self._files_processed,
            "memories_added": self._memories_added,
            "facts_extracted": self._facts_extracted,
            "is_running": self._is_running,
            "watch_dir": str(self.watch_dir),
        }
