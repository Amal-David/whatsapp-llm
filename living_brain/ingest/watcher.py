"""
File watcher for continuous ingestion of new WhatsApp exports.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


class WhatsAppExportHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):  # type: ignore[misc]
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

        logger.info(f"New file detected: {filepath}")

        # Wait a moment for the file to be fully written
        time.sleep(1)

        try:
            self.on_new_file(filepath)
            self._processed_files.add(str(filepath))
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
        chunk_size: int = 512,
        chunk_overlap: int = 50,
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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._observer: Any = None
        self._handler: WhatsAppExportHandler | None = None
        self._is_running = False

        # Stats
        self._files_processed = 0
        self._memories_added = 0
        self._facts_extracted = 0

    def _process_file(
        self,
        filepath: Path,
        *,
        source_path: Path | None = None,
    ) -> None:
        """Process a new WhatsApp export file."""
        from .whatsapp_parser import WhatsAppParser

        source_path = source_path or filepath
        logger.info(f"Processing: {filepath}")

        # Handle zip files
        if filepath.suffix == ".zip":
            extracted_path = self._extract_zip(filepath)
            if extracted_path is None:
                return
            try:
                self._process_file(extracted_path, source_path=source_path)
            finally:
                extracted_path.unlink(missing_ok=True)
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
                    "source_file": str(source_path),
                    "participants": ",".join(set(m.author for m in conv.messages)),
                    "message_count": len(conv.messages),
                }

                chunk_ids = self.vector_store.add_chunked(
                    content=conv_text,
                    timestamp=conv.start_time,
                    metadata=metadata,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                self._memories_added += len(chunk_ids)

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

    def _extract_zip(self, zip_path: Path) -> Path | None:
        """Extract a zip file and return the txt file path."""
        import shutil
        import tempfile
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            txt_files = [name for name in zf.namelist() if name.endswith(".txt")]
            if not txt_files:
                raise ValueError(f"No txt files found in {zip_path}")

            txt_file = txt_files[0]
            member_path = Path(txt_file)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe archive member: {txt_file}")

            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix="living-brain-export-",
                    suffix=".txt",
                    delete=False,
                ) as target:
                    temp_path = Path(target.name)
                    with zf.open(txt_file) as source:
                        shutil.copyfileobj(source, target)
                return temp_path
            except Exception:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
                raise

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

        observer = Observer()
        observer.schedule(self._handler, str(self.watch_dir), recursive=False)
        observer.start()
        self._observer = observer
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
