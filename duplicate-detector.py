#!/usr/bin/env -S uv run
"""Unified Duplicate Detector - Metadata + Neural Features with Integrated Review UI."""

import atexit
import base64
import contextlib
import gc
import hashlib
import multiprocessing as mp
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import warnings
import webbrowser
from collections import OrderedDict, defaultdict
from collections.abc import Generator
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from io import BytesIO
from pathlib import Path
from queue import Queue
from types import FrameType, TracebackType
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

mp.set_start_method("spawn", force=True)

import builtins

import cv2
import exifread
import humanize
import numpy as np
import pillow_heif
import psutil
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
from PIL.ExifTags import TAGS
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from sklearn.preprocessing import normalize  # type: ignore[reportUnknownVariableType]
from torch.utils.data import DataLoader, Dataset
from waitress import serve

pillow_heif.register_heif_opener()  # type: ignore[reportUnknownMemberType]

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

console = Console()

# Global flag for cleanup
_cleanup_in_progress = False
_active_dataloaders: list[Any] = []

# Flask app - templates in current directory
app = Flask(__name__, template_folder=".")

# ==============================================================================
# Constants
# ==============================================================================

# Minimum number of ORB descriptors required for geometric verification
MIN_DESCRIPTORS = 10

# Threshold for "large feature set" - affects inlier ratio requirements
LARGE_FEATURE_THRESHOLD = 1000

# Minimum number of images to form a valid duplicate group
MIN_GROUP_SIZE = 2

# Maximum entries in route cache before cleanup
MAX_ROUTE_CACHE_ENTRIES = 100

# kNN matching returns k=2 neighbors for Lowe's ratio test
KNN_MATCH_COUNT = 2

# String split for key=value pairs produces 2 parts
KEY_VALUE_SPLIT_PARTS = 2

# Datetime string needs at least date and time parts
MIN_DATETIME_PARTS = 2


# LRU caches with proper size limits
class LRUCache(Generic[K, V]):
    """Thread-safe LRU cache with configurable capacity."""

    def __init__(self, capacity: int) -> None:
        """Initialize the cache with given capacity.

        Args:
            capacity: Maximum number of items to store.

        """
        self.cache: OrderedDict[K, V] = OrderedDict()
        self.capacity = capacity
        # Thread safety lock for concurrent access
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        """Retrieve an item from the cache.

        Args:
            key: The key to look up.

        Returns:
            The cached value or None if not found.

        """
        with self._lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: K, value: V) -> None:
        """Store an item in the cache.

        Args:
            key: The key to store under.
            value: The value to store.

        """
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: The key to check.

        Returns:
            True if the key exists, False otherwise.

        """
        with self._lock:
            return key in self.cache

    def clear(self) -> None:
        """Remove all items from the cache."""
        with self._lock:
            self.cache.clear()

    def remove(self, key: K) -> None:
        """Remove a specific key from the cache.

        Args:
            key: The key to remove.

        """
        with self._lock:
            self.cache.pop(key, None)


# State management
class AppState:
    """Global application state for the web UI."""

    db_path: Path | None
    current_group_id: str | None
    initial_active_groups: list[str]
    active_groups_set: set[str]
    thumbnail_cache: LRUCache[str, str]
    group_data_cache: LRUCache[str, dict[str, Any]]
    total_groups_cache: int | None
    connection_pool: "ConnectionPool | None"

    def __init__(self) -> None:
        """Initialize application state with default values."""
        self.db_path = None
        self.current_group_id = None
        self.initial_active_groups = []
        self.active_groups_set = set()
        self.thumbnail_cache = LRUCache(100)
        self.group_data_cache = LRUCache(50)
        self.total_groups_cache = None
        self.connection_pool = None
        # Thread safety lock for concurrent access
        self._lock = threading.Lock()

    def reset_groups(self) -> None:
        """Reset group-related state to initial values."""
        with self._lock:
            self.current_group_id = None
            self.initial_active_groups = []
            self.active_groups_set = set()
            self.total_groups_cache = None

    @contextmanager
    def locked(self) -> Generator[None, None, None]:
        """Acquire the state lock for thread-safe operations.

        Yields:
            None after acquiring the lock.

        """
        with self._lock:
            yield


app_state = AppState()

# ==============================================================================
# Database Schema
# ==============================================================================


def init_database(db_path: Path) -> None:
    """Initialize SQLite database with schema.

    Args:
        db_path: Path to the SQLite database file.

    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            group_id TEXT,
            is_representative BOOLEAN DEFAULT 0,
            detection_method TEXT,
            sscd_score REAL DEFAULT 0,
            geometric_inliers INTEGER DEFAULT 0,
            metadata_bonus BOOLEAN DEFAULT 0,
            resolution TEXT,
            status TEXT DEFAULT 'active',
            datetime_original TEXT,
            camera_make TEXT,
            camera_model TEXT,
            width INTEGER,
            height INTEGER,
            processing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_group_id ON images(group_id);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_status ON images(status);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_path ON images(path);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_path_status ON images(path, status);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_status_group ON images(status, group_id);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_group_status_rep ON images(group_id, status, is_representative);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_status_active ON images(status) WHERE status = 'active';
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deletion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            group_id TEXT,
            deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ==============================================================================
# Signal Handler for Graceful Shutdown
# ==============================================================================


def signal_handler(_signum: int, _frame: FrameType | None) -> None:
    """Handle interrupt signals gracefully.

    Args:
        _signum: Signal number received (unused).
        _frame: Current stack frame (unused).

    """
    global _cleanup_in_progress  # noqa: PLW0603
    if not _cleanup_in_progress:
        _cleanup_in_progress = True
        console.print("\n[yellow]Received interrupt signal. Shutting down...[/yellow]")
        # Don't call cleanup_resources here - let atexit handle it
        # Just exit cleanly
        os._exit(0)


def cleanup_resources() -> None:
    """Clean up all resources including dataloaders and GPU cache."""
    # Only clean up if we have active resources
    if not _active_dataloaders and not torch.cuda.is_available():
        return

    # Clean up dataloaders
    for dataloader in _active_dataloaders:
        try:
            if hasattr(dataloader, "_iterator"):
                del dataloader._iterator  # noqa: SLF001
            del dataloader
        except (AttributeError, TypeError):
            pass
    _active_dataloaders.clear()

    # Force garbage collection
    gc.collect()

    # Clear PyTorch cache if available
    try:
        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()
    except (AttributeError, RuntimeError):
        pass


# Only register SIGINT handler, not SIGTERM
signal.signal(signal.SIGINT, signal_handler)
# Don't register SIGTERM - let Waitress handle it

# Register cleanup on exit
atexit.register(cleanup_resources)

# ==============================================================================
# Configuration
# ==============================================================================


@dataclass
class Config:
    """Configuration for enhanced duplicate detection."""

    image_folder: Path = Path()
    base_dir: Path = Path(".duplicate-detector")

    mode: str = "integrated"

    batch_size: int = 64

    sscd_threshold: float = 0.86
    sscd_threshold_with_metadata: float = 0.90

    sscd_weight: float = 0.85
    metadata_weight: float = 0.15
    integrated_threshold: float = 0.88

    min_inlier_ratio: float = 0.3
    min_inlier_ratio_metadata: float = 0.5
    min_absolute_inliers: int = 100
    min_good_matches: int = 30
    orb_nfeatures: int = 6000
    orb_scale_factor: float = 1.2
    orb_nlevels: int = 8
    max_homography_det: float = 2.0
    min_homography_det: float = 0.5
    lowe_ratio: float = 0.7

    num_workers: int = 11
    dataloader_workers: int = 8
    device: str = "mps"
    use_mixed_precision: bool = False

    image_extensions: list[str] = field(
        default_factory=lambda: [".heic", ".HEIC", ".jpg", ".jpeg", ".JPG", ".JPEG"]
    )

    # Computed paths (set in __post_init__)
    db_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    model_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Set up computed paths and create directories."""
        # Set up directory structure under base_dir
        self.db_dir = self.base_dir / "db"
        self.model_dir = self.base_dir / "models"
        self.cache_dir = self.base_dir / "cache"

        # Create all directories
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set database path
        self.db_path = self.db_dir / "detector.db"

    def get_cache_key(self) -> str:
        """Generate a cache key based on configuration parameters."""
        key_parts = [
            str(self.image_folder.absolute()),
            str(self.sscd_threshold),
            str(self.orb_nfeatures),
            str(self.min_inlier_ratio),
            "padded_enhanced",
            "_".join(sorted(self.image_extensions)),
        ]

        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode(), usedforsecurity=False).hexdigest()[:16]


# ==============================================================================
# Metadata Extraction
# ==============================================================================


class MetadataExtractor:
    """Extract and compare image metadata."""

    def __init__(self) -> None:
        """Initialize the metadata extractor with an LRU cache."""
        # Use bounded LRUCache instead of unbounded dictionary to prevent memory leaks
        self._cache: LRUCache[tuple[str, float], dict[str, Any]] = LRUCache(capacity=10000)

    def _extract_image_dimensions(
        self, image_path: Path, metadata: dict[str, Any]
    ) -> None:
        """Extract image dimensions using PIL.

        Args:
            image_path: Path to the image file.
            metadata: Metadata dict to update in place.

        """
        try:
            with Image.open(image_path) as img:
                metadata["width"] = img.width
                metadata["height"] = img.height
        except (OSError, Image.DecompressionBombError) as e:
            print(f"Error getting dimensions for {image_path}: {e}")

    def _extract_exifread_metadata(
        self, image_path: Path, metadata: dict[str, Any]
    ) -> None:
        """Extract metadata using exifread library.

        Args:
            image_path: Path to the image file.
            metadata: Metadata dict to update in place.

        """
        try:
            with image_path.open("rb") as f:
                tags = exifread.process_file(f, details=False, strict=False)

                if "EXIF DateTimeOriginal" in tags:
                    metadata["datetime_original"] = str(tags["EXIF DateTimeOriginal"])
                elif "Image DateTime" in tags:
                    metadata["datetime_original"] = str(tags["Image DateTime"])

                if "Image Make" in tags:
                    metadata["camera_make"] = str(tags["Image Make"]).strip()

                if "Image Model" in tags:
                    metadata["camera_model"] = str(tags["Image Model"]).strip()

        # exifread's HEIC parser may raise AssertionError
        except (OSError, KeyError, AttributeError, AssertionError, exifread.core.heic.BadSize) as e:  # type: ignore[attr-defined]
            if isinstance(e, exifread.core.heic.BadSize):  # type: ignore[attr-defined]
                print(f"Warning: Corrupted HEIC file detected: {image_path}")

    def _extract_pil_exif_metadata(
        self, image_path: Path, metadata: dict[str, Any]
    ) -> None:
        """Extract EXIF metadata using PIL as fallback.

        Args:
            image_path: Path to the image file.
            metadata: Metadata dict to update in place.

        """
        if metadata["datetime_original"] and metadata["camera_make"]:
            return

        try:
            img = Image.open(image_path)
            try:
                exif_data: dict[int, Any] | None = img._getexif()  # type: ignore[attr-defined]  # noqa: SLF001
                if exif_data:
                    self._parse_pil_exif_tags(exif_data, metadata)  # type: ignore[reportUnknownArgumentType]
            except AttributeError:
                pass
            finally:
                img.close()
        except (OSError, ValueError, Image.DecompressionBombError):
            pass

    def _parse_pil_exif_tags(
        self, exif_data: dict[int, Any], metadata: dict[str, Any]
    ) -> None:
        """Parse EXIF tags from PIL exif data.

        Args:
            exif_data: Raw EXIF data from PIL.
            metadata: Metadata dict to update in place.

        """
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)

            if tag_name == "DateTimeOriginal" and not metadata["datetime_original"]:
                metadata["datetime_original"] = str(value)
            elif tag_name == "Make" and not metadata["camera_make"]:
                metadata["camera_make"] = str(value).strip()
            elif tag_name == "Model" and not metadata["camera_model"]:
                metadata["camera_model"] = str(value).strip()

    def _extract_macos_metadata(
        self, image_path: Path, metadata: dict[str, Any]
    ) -> None:
        """Extract metadata using macOS mdls command.

        Args:
            image_path: Path to the image file.
            metadata: Metadata dict to update in place.

        """
        if metadata["datetime_original"] and metadata["camera_make"]:
            return

        try:
            result = subprocess.run(  # noqa: S603
                [  # noqa: S607 - mdls is a macOS system utility
                    "mdls",
                    "-name", "kMDItemContentCreationDate",
                    "-name", "kMDItemAcquisitionMake",
                    "-name", "kMDItemAcquisitionModel",
                    str(image_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                self._parse_mdls_output(result.stdout, metadata)

        except (subprocess.SubprocessError, OSError):
            pass

    def _parse_mdls_output(self, output: str, metadata: dict[str, Any]) -> None:
        """Parse mdls command output.

        Args:
            output: Raw output from mdls command.
            metadata: Metadata dict to update in place.

        """
        for line in output.strip().split("\n"):
            if "kMDItemContentCreationDate" in line and not metadata["datetime_original"]:
                self._parse_mdls_datetime(line, metadata)
            elif "kMDItemAcquisitionMake" in line and not metadata["camera_make"]:
                self._parse_mdls_field(line, "camera_make", metadata)
            elif "kMDItemAcquisitionModel" in line and not metadata["camera_model"]:
                self._parse_mdls_field(line, "camera_model", metadata)

    def _parse_mdls_datetime(self, line: str, metadata: dict[str, Any]) -> None:
        """Parse datetime from mdls output line.

        Args:
            line: Single line from mdls output.
            metadata: Metadata dict to update in place.

        """
        parts = line.split("=", 1)
        if len(parts) != KEY_VALUE_SPLIT_PARTS:
            return
        date_str = parts[1].strip()
        if not date_str or date_str == "(null)":
            return
        date_str = date_str.strip('"')
        date_parts = date_str.split(" ")
        if len(date_parts) >= MIN_DATETIME_PARTS:
            date = date_parts[0].replace("-", ":")
            time_val = date_parts[1]
            metadata["datetime_original"] = f"{date} {time_val}"

    def _parse_mdls_field(
        self, line: str, field: str, metadata: dict[str, Any]
    ) -> None:
        """Parse a simple field from mdls output line.

        Args:
            line: Single line from mdls output.
            field: Metadata field name to update.
            metadata: Metadata dict to update in place.

        """
        parts = line.split("=", 1)
        if len(parts) == KEY_VALUE_SPLIT_PARTS:
            value = parts[1].strip().strip('"')
            if value and value != "(null)":
                metadata[field] = value

    def extract_metadata(self, image_path: Path) -> dict[str, Any]:
        """Extract relevant metadata from image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary containing metadata fields.

        """
        cache_key = (str(image_path), image_path.stat().st_mtime)
        cached_metadata = self._cache.get(cache_key)
        if cached_metadata is not None:
            return cached_metadata.copy()

        metadata: dict[str, Any] = {
            "datetime_original": None,
            "camera_make": None,
            "camera_model": None,
            "width": None,
            "height": None,
            "valid": False,
        }

        self._extract_image_dimensions(image_path, metadata)
        self._extract_exifread_metadata(image_path, metadata)
        self._extract_pil_exif_metadata(image_path, metadata)
        self._extract_macos_metadata(image_path, metadata)

        if metadata["datetime_original"] and metadata["camera_make"]:
            metadata["valid"] = True

        self._cache.put(cache_key, metadata.copy())
        return metadata

    @staticmethod
    def create_metadata_key(metadata: dict[str, Any]) -> str | None:
        """Create a unique key from metadata for comparison.

        Args:
            metadata: Metadata dictionary to create key from.

        Returns:
            String key or None if metadata is invalid.

        """
        if not metadata["valid"]:
            return None

        key_parts = [
            metadata["datetime_original"] or "",
            metadata["camera_make"] or "",
            metadata["camera_model"] or "",
        ]

        return "|".join(key_parts)


# ==============================================================================
# System Monitor
# ==============================================================================


class SystemMonitor:
    """Background system monitoring with proper cleanup."""

    def __init__(self) -> None:
        """Initialize the system monitor and start background thread."""
        self.cpu_percent = 0.0
        self.memory_used = 0
        self.memory_total = psutil.virtual_memory().total
        self.running = True
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self) -> None:
        """Background monitoring loop."""
        while self.running and not self._stop_event.is_set():
            try:
                self.cpu_percent = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                self.memory_used = mem.used
                self._stop_event.wait(0.5)
            except Exception:  # noqa: BLE001
                break

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self.running = False
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def get_status(self) -> str:
        """Get formatted status string.

        Returns:
            Formatted string with CPU and memory usage.

        """
        return (
            f"CPU: {self.cpu_percent:5.1f}% | "
            f"RAM: {humanize.naturalsize(self.memory_used)}/{humanize.naturalsize(self.memory_total)}"
        )


# ==============================================================================
# Progress Manager
# ==============================================================================


class ProgressManager:
    """Centralized progress management with proper cleanup."""

    def __init__(self) -> None:
        """Initialize the progress manager with Rich progress bars."""
        self.progress: Any = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True,
        )
        self.tasks: dict[str, int] = {}
        self.system_monitor = SystemMonitor()
        self.start_time = time.time()

    def __enter__(self) -> "ProgressManager":
        """Enter context manager."""
        self.progress.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager and clean up resources."""
        with contextlib.suppress(Exception):
            self.system_monitor.stop()
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    def add_task(self, description: str, total: int) -> int:
        """Add a new progress task.

        Args:
            description: Task description to display.
            total: Total number of items.

        Returns:
            Task ID for updates.

        """
        task_id: int = self.progress.add_task(description, total=total)
        self.tasks[description] = task_id
        return task_id

    def update(self, description: str, advance: int = 1) -> None:
        """Update task progress.

        Args:
            description: Task description to update.
            advance: Amount to advance by.

        """
        if description in self.tasks:
            self.progress.advance(self.tasks[description], advance)

    def log(self, message: str, style: str = "") -> None:
        """Log a message to the console.

        Args:
            message: Message to display.
            style: Rich style to apply.

        """
        self.progress.console.print(message, style=style)

    def status_panel(self) -> Any:  # noqa: ANN401 - Rich Panel type
        """Create a status panel with system info.

        Returns:
            Rich Panel with system status.

        """
        elapsed = time.time() - self.start_time
        return Panel(
            f"{self.system_monitor.get_status()}\nElapsed: {humanize.naturaldelta(elapsed)}",
            title="System Status",
            border_style="cyan",
        )


# ==============================================================================
# Color Normalization and Dataset
# ==============================================================================


def apply_color_normalization(image_array: np.ndarray) -> np.ndarray:
    """Apply YUV histogram equalization and CLAHE for photographed prints.

    Args:
        image_array: RGB image as numpy array.

    Returns:
        Normalized RGB image as numpy array.

    """
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    img_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_normalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    lab = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img_normalized = cv2.GaussianBlur(img_normalized, (3, 3), 0)

    return cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)


class FastSSCDDataset(Dataset):  # type: ignore[type-arg]
    """Fast dataset for SSCD with color normalization and black padding."""

    def __init__(self, paths: list[Path], *, apply_color_norm: bool = True) -> None:
        """Initialize the dataset.

        Args:
            paths: List of image paths to process.
            apply_color_norm: Whether to apply color normalization.

        """
        self.paths = paths
        self.apply_color_norm = apply_color_norm

        self.transform: Any = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.paths)

    def resize_with_padding(self, img: Any, target_size: int = 288) -> Any:  # noqa: ANN401
        """Resize image to fit within target_size with black padding.

        Args:
            img: PIL Image to resize.
            target_size: Target dimension for both width and height.

        Returns:
            Resized and padded PIL Image.

        """
        orig_width, orig_height = img.size

        scale = min(target_size / orig_width, target_size / orig_height)

        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        img_resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

        canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))

        left = (target_size - new_width) // 2
        top = (target_size - new_height) // 2

        canvas.paste(img_resized, (left, top))

        return canvas

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Tuple of (transformed tensor, index).

        """
        try:
            img = Image.open(self.paths[idx]).convert("RGB")

            if self.apply_color_norm:
                img_array = np.array(img)
                img_array = apply_color_normalization(img_array)
                img = Image.fromarray(img_array)

            img = self.resize_with_padding(img, target_size=288)
            return self.transform(img), idx

        except Exception:  # noqa: BLE001
            return torch.zeros(3, 288, 288), idx


# ==============================================================================
# Model Manager
# ==============================================================================


class FastModelManager:
    """Optimized model manager with single-pass extraction and cleanup."""

    def __init__(self, config: Config, pm: ProgressManager) -> None:
        """Initialize the model manager.

        Args:
            config: Detection configuration.
            pm: Progress manager for logging.

        """
        self.config = config
        self.pm = pm
        self.device: Any = torch.device(config.device)
        self.models: dict[str, Any] = {}
        self.dims: dict[str, int] = {}

    def load_models(self) -> None:
        """Load SSCD disc_large model."""
        self.pm.log("[bold]Loading copy detection model...[/bold]")

        self.pm.log("Loading SSCD disc_large (1024-dim)...")
        self._load_sscd()

        self.pm.log("[green]✓ Model loaded successfully[/green]")

    def _load_sscd(self) -> None:
        """Load SSCD disc_large model from cache or download."""
        model_url = (
            "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt"
        )
        cache_path = self.config.model_dir / "sscd_disc_large.torchscript.pt"

        if not cache_path.exists():
            self.pm.log("Downloading SSCD model...")

            download_task = self.pm.progress.add_task(
                "[bold blue]Downloading SSCD model", total=None, visible=True
            )

            def download_hook(block_num: int, block_size: int, total_size: int) -> None:
                if self.pm.progress.tasks[download_task].total is None and total_size > 0:
                    self.pm.progress.update(download_task, total=total_size)
                downloaded = block_num * block_size
                self.pm.progress.update(download_task, completed=downloaded)

            try:
                urllib.request.urlretrieve(model_url, cache_path, reporthook=download_hook)  # noqa: S310
                self.pm.progress.update(download_task, visible=False)
            except Exception:
                self.pm.progress.update(download_task, visible=False)
                raise

        model = torch.jit.load(cache_path, map_location="cpu")  # type: ignore[reportUnknownMemberType]
        self.models["sscd"] = model.to(self.device).eval()  # type: ignore[reportUnknownMemberType]
        self.dims["sscd"] = 1024
        self.pm.log("[green]✓ SSCD loaded[/green]")

    def cleanup(self) -> None:
        """Clean up models and free memory."""
        for model in self.models.values():
            del model
        self.models.clear()
        gc.collect()
        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()

    @torch.no_grad()  # type: ignore[reportUntypedFunctionDecorator]
    def extract_features_fast(self, image_paths: list[Path]) -> np.ndarray:
        """Extract features using optimized pipeline with proper cleanup.

        Args:
            image_paths: List of paths to images.

        Returns:
            Normalized feature array.

        """
        global _active_dataloaders  # noqa: PLW0602 - modifying global list

        n = len(image_paths)
        features = np.zeros((n, self.dims["sscd"]), dtype=np.float32)

        start_time = time.time()

        self.pm.log("\n[cyan]SSCD feature extraction (multi-worker)[/cyan]")
        self.pm.log(f"Using {self.config.dataloader_workers} dataloader workers")
        self.pm.log(
            "[yellow]Note: Images will be padded with black to preserve aspect ratio[/yellow]"
        )

        dataset = FastSSCDDataset(image_paths, apply_color_norm=True)

        dataloader: DataLoader[tuple[Any, int]] = DataLoader(  # type: ignore[reportUnknownVariableType]
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.dataloader_workers,
            prefetch_factor=2 if self.config.dataloader_workers > 0 else None,
            pin_memory=False,
            persistent_workers=False,
        )

        _active_dataloaders.append(dataloader)

        self.pm.add_task("SSCD features", total=n)

        try:
            for batch, indices in dataloader:
                if _cleanup_in_progress:
                    break

                batch_tensor = batch.to(self.device)
                batch_features = self.models["sscd"](batch_tensor)
                batch_features = F.normalize(batch_features, dim=1)

                for i, idx in enumerate(indices):
                    features[idx] = batch_features[i].cpu().numpy()

                self.pm.update("SSCD features", advance=len(indices))

                if indices[0] % (self.config.batch_size * 20) == 0:
                    if hasattr(torch, "mps") and torch.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
        finally:
            try:
                if hasattr(dataloader, "_iterator"):
                    del dataloader._iterator  # type: ignore[reportPrivateUsage]  # noqa: SLF001
                _active_dataloaders.remove(dataloader)
                del dataloader
            except (RuntimeError, ValueError, AttributeError):
                pass
            gc.collect()

        features = normalize(features, axis=1)  # type: ignore[assignment]

        elapsed = time.time() - start_time
        self.pm.log(
            f"[green]✓ Feature extraction complete in {elapsed:.1f}s ({n / elapsed:.1f} img/s)[/green]"
        )

        return features  # type: ignore[return-value]


# ==============================================================================
# Geometric Verification Worker (for ProcessPoolExecutor)
# ==============================================================================

# Maximum dimension for image resizing in geometric verification
GV_MAX_IMAGE_DIM = 1500


def _load_grayscale_images(
    path1: Path, path2: Path
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and convert two images to grayscale arrays.

    Args:
        path1: Path to first image.
        path2: Path to second image.

    Returns:
        Tuple of (img1, img2) as numpy arrays, or None on error.

    """
    try:
        with Image.open(path1) as pil_img1, Image.open(path2) as pil_img2:
            gray1 = pil_img1.convert("L")
            gray2 = pil_img2.convert("L")
            return np.array(gray1), np.array(gray2)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def _resize_if_needed(img: np.ndarray, max_dim: int = GV_MAX_IMAGE_DIM) -> np.ndarray:
    """Resize image if it exceeds maximum dimension.

    Args:
        img: Input grayscale image array.
        max_dim: Maximum allowed dimension.

    Returns:
        Resized image array.

    """
    h, w = img.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))  # type: ignore[return-value]
    return img


def _compute_good_matches(
    des1: Any, des2: Any, lowe_ratio: float  # noqa: ANN401 - OpenCV types
) -> list[Any]:
    """Compute good matches using Lowe's ratio test.

    Args:
        des1: Descriptors from first image.
        des2: Descriptors from second image.
        lowe_ratio: Lowe's ratio threshold.

    Returns:
        List of good matches passing the ratio test.

    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches: list[Any] = []
    for match_pair in matches:
        if len(match_pair) == KNN_MATCH_COUNT:
            m, n = match_pair
            if m.distance < lowe_ratio * n.distance:
                good_matches.append(m)
    return good_matches


def _verify_homography(
    good_matches: list[Any],
    kp1: Any,  # noqa: ANN401 - OpenCV types
    kp2: Any,  # noqa: ANN401 - OpenCV types
    candidate: dict[str, Any],
    config_params: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Verify geometric consistency using homography.

    Args:
        good_matches: List of good feature matches.
        kp1: Keypoints from first image.
        kp2: Keypoints from second image.
        candidate: Candidate duplicate dict.
        config_params: Configuration parameters.

    Returns:
        Tuple of (is_valid, updated_candidate).

    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # type: ignore[arg-type]
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # type: ignore[arg-type]

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if homography is None:  # type: ignore[reportUnnecessaryComparison]
        return False, candidate

    inliers = np.sum(mask)
    min_features = min(len(kp1), len(kp2))
    inlier_ratio = inliers / len(good_matches)
    feature_coverage = inliers / min_features

    if inliers < config_params["min_absolute_inliers"]:
        return False, candidate

    has_metadata_bonus = candidate.get("has_metadata_bonus", False)
    required_ratio = (
        config_params["min_inlier_ratio_metadata"]
        if has_metadata_bonus
        else config_params["min_inlier_ratio"]
    )

    if min_features > LARGE_FEATURE_THRESHOLD:
        if feature_coverage < (required_ratio * 0.5):
            return False, candidate
    elif inlier_ratio < required_ratio:
        return False, candidate

    det = np.linalg.det(homography[:2, :2])
    if not (config_params["min_homography_det"] <= det <= config_params["max_homography_det"]):
        return False, candidate

    candidate["geometric_inliers"] = int(inliers)
    candidate["inlier_ratio"] = float(inlier_ratio)
    candidate["feature_coverage"] = float(feature_coverage)
    candidate["good_matches"] = len(good_matches)
    candidate["min_features"] = int(min_features)

    return True, candidate


def geometric_verification_worker(
    candidate_data: tuple[dict[str, Any], list[Path], dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Perform geometric verification in a separate process.

    Args:
        candidate_data: Tuple of (candidate dict, image paths, config params).

    Returns:
        Tuple of (verification result, updated candidate dict).

    """
    candidate, image_paths, config_params = candidate_data

    path1: Path = image_paths[candidate["idx1"]]  # type: ignore[reportUnknownVariableType]
    path2: Path = image_paths[candidate["idx2"]]  # type: ignore[reportUnknownVariableType]

    images = _load_grayscale_images(path1, path2)  # type: ignore[reportUnknownArgumentType]
    if images is None:
        return False, candidate
    img1, img2 = images

    img1 = _resize_if_needed(img1)
    img2 = _resize_if_needed(img2)

    orb = cv2.ORB_create(  # type: ignore[attr-defined]
        nfeatures=config_params["orb_nfeatures"],
        scaleFactor=config_params["orb_scale_factor"],
        nlevels=config_params["orb_nlevels"],
    )

    kp1, des1 = orb.detectAndCompute(img1, None)  # type: ignore[reportUnknownMemberType]
    kp2, des2 = orb.detectAndCompute(img2, None)  # type: ignore[reportUnknownMemberType]

    if des1 is None or des2 is None or len(des1) < MIN_DESCRIPTORS or len(des2) < MIN_DESCRIPTORS:  # type: ignore[reportUnknownArgumentType]
        return False, candidate

    good_matches = _compute_good_matches(des1, des2, config_params["lowe_ratio"])

    if len(good_matches) < config_params["min_good_matches"]:
        return False, candidate

    return _verify_homography(good_matches, kp1, kp2, candidate, config_params)


# ==============================================================================
# Duplicate Detection
# ==============================================================================


class DuplicateFinder:
    """Find duplicates with metadata and/or neural features."""

    def __init__(self, config: Config, pm: ProgressManager) -> None:
        """Initialize the duplicate finder.

        Args:
            config: Detection configuration.
            pm: Progress manager for logging.

        """
        self.config = config
        self.pm = pm
        self.metadata_extractor = MetadataExtractor()
        self.all_metadata: dict[int, dict[str, Any]] = {}
        self.metadata_keys: dict[int, str | None] = {}
        self.detailed_results: list[dict[str, Any]] = []

    def extract_all_metadata(self, image_paths: list[Path]) -> None:
        """Extract metadata for all images upfront.

        Args:
            image_paths: List of paths to images.

        """
        self.pm.log("\n[bold cyan]Extracting metadata for all images[/bold cyan]")
        self.pm.log("[yellow]Using exifread + macOS mdls for better HEIC compatibility[/yellow]")

        self.pm.add_task("Extracting metadata", total=len(image_paths))

        # Parallel metadata extraction
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures: dict[Future[dict[str, Any]], int] = {}
            for idx, path in enumerate(image_paths):
                if _cleanup_in_progress:
                    break
                future = executor.submit(self.metadata_extractor.extract_metadata, path)
                futures[future] = idx

            valid_metadata_count = 0
            for future in as_completed(futures):
                if _cleanup_in_progress:
                    break
                idx = futures[future]
                metadata = future.result()
                metadata_key = self.metadata_extractor.create_metadata_key(metadata)

                self.all_metadata[idx] = metadata
                self.metadata_keys[idx] = metadata_key

                if metadata_key:
                    valid_metadata_count += 1

                self.pm.update("Extracting metadata")

        self.pm.log(
            f"[dim]Found valid metadata in {valid_metadata_count}/{len(image_paths)} images[/dim]"
        )

    def find_metadata_duplicates(self, image_paths: list[Path]) -> tuple[list[list[int]], set[int]]:
        """Find duplicates based on metadata.

        Args:
            image_paths: List of paths to images.

        Returns:
            Tuple of (duplicate groups, set of all duplicate indices).

        """
        self.pm.log("\n[bold cyan]Finding metadata-based duplicates[/bold cyan]")

        if not self.all_metadata:
            self.extract_all_metadata(image_paths)

        metadata_groups: defaultdict[str, list[int]] = defaultdict(list)
        for idx, metadata_key in self.metadata_keys.items():
            if metadata_key:
                metadata_groups[metadata_key].append(idx)

        duplicate_groups: list[list[int]] = []
        all_duplicates: set[int] = set()

        for indices in metadata_groups.values():
            if len(indices) > 1:
                duplicate_groups.append(indices)
                all_duplicates.update(indices)

        self.pm.log(
            f"[green]✓ Found {len(duplicate_groups)} metadata duplicate groups "
            f"({len(all_duplicates)} total images)[/green]"
        )

        if duplicate_groups and len(metadata_groups) > 0:
            sample_key = next(iter(metadata_groups.keys()))
            self.pm.log(f"[dim]Sample metadata key: {sample_key}[/dim]")

        return duplicate_groups, all_duplicates

    def find_neural_candidates(
        self, features: np.ndarray, image_paths: list[Path]
    ) -> list[dict[str, Any]]:
        """Find candidates using neural features directly.

        Args:
            features: Feature array from SSCD model.
            image_paths: List of paths to images.

        Returns:
            List of candidate dictionaries.

        """
        self.pm.log("\n[bold cyan]Finding neural candidates[/bold cyan]")

        n = len(image_paths)
        candidates: list[dict[str, Any]] = []

        # Batch similarity computation
        batch_size = 1000
        self.pm.add_task("Computing similarities", total=n)

        for i in range(0, n, batch_size):
            if _cleanup_in_progress:
                break

            end_i = min(i + batch_size, n)
            batch_features = features[i:end_i]

            # Compute similarities for this batch against all features
            similarities = np.dot(batch_features, features.T)

            # Find pairs above threshold
            for bi in range(end_i - i):
                global_i = i + bi
                for j in range(global_i + 1, n):
                    if similarities[bi, j] >= self.config.sscd_threshold:
                        candidates.append(  # noqa: PERF401 - nested loop, comprehension hurts readability
                            {
                                "idx1": global_i,
                                "idx2": j,
                                "sscd_similarity": float(similarities[bi, j]),
                                "metadata_match": 0.0,
                                "integrated_score": float(similarities[bi, j]),
                                "has_metadata_bonus": False,
                            }
                        )

            self.pm.update("Computing similarities", advance=end_i - i)

        self.pm.log(f"[green]✓ Found {len(candidates)} neural candidates[/green]")
        return candidates

    def find_integrated_candidates(
        self, features: np.ndarray, image_paths: list[Path]
    ) -> list[dict[str, Any]]:
        """Find candidates using integrated scoring (neural + metadata bonus).

        Args:
            features: Feature array from SSCD model.
            image_paths: List of paths to images.

        Returns:
            List of candidate dictionaries with integrated scores.

        """
        self.pm.log("\n[bold cyan]Finding candidates with integrated scoring[/bold cyan]")

        n = len(image_paths)
        candidates: list[dict[str, Any]] = []

        if not self.all_metadata:
            self.extract_all_metadata(image_paths)

        has_any_metadata = any(key is not None for key in self.metadata_keys.values())

        if not has_any_metadata:
            self.pm.log("[yellow]No valid metadata found - using neural features only[/yellow]")
            self.pm.log(f"[dim]SSCD threshold: {self.config.sscd_threshold}[/dim]")

            return self.find_neural_candidates(features, image_paths)

        self.pm.log(
            f"[dim]Weights: SSCD={self.config.sscd_weight}, Metadata={self.config.metadata_weight}[/dim]"
        )

        self.pm.add_task("Computing integrated scores", total=n)

        # Efficient matrix-based approach (replaces O(n²) nested loops)
        batch_size = 1000
        for i in range(0, n, batch_size):
            if _cleanup_in_progress:
                break

            end_i = min(i + batch_size, n)
            batch_features = features[i:end_i]

            # Compute similarities for this batch against all features - EFFICIENT!
            similarities = np.dot(batch_features, features.T)

            # Find pairs above threshold with metadata bonus logic
            for bi in range(end_i - i):
                global_i = i + bi
                for j in range(global_i + 1, n):
                    sscd_sim = float(similarities[bi, j])

                    # Calculate metadata match
                    metadata_match = 0.0
                    if (
                        self.metadata_keys[global_i]
                        and self.metadata_keys[j]
                        and self.metadata_keys[global_i] == self.metadata_keys[j]
                    ):
                        metadata_match = 1.0

                    # Calculate integrated score
                    integrated_score = (
                        self.config.sscd_weight * sscd_sim
                        + self.config.metadata_weight * metadata_match
                    )

                    # Check if candidate meets any threshold
                    if (
                        (metadata_match and sscd_sim >= self.config.sscd_threshold_with_metadata)
                        or (not metadata_match and sscd_sim >= self.config.sscd_threshold)
                        or (integrated_score >= self.config.integrated_threshold)
                    ):
                        candidates.append(
                            {
                                "idx1": global_i,
                                "idx2": j,
                                "sscd_similarity": sscd_sim,
                                "metadata_match": metadata_match,
                                "integrated_score": integrated_score,
                                "has_metadata_bonus": metadata_match > 0,
                            }
                        )

            self.pm.update("Computing integrated scores", advance=end_i - i)

        self.pm.log(f"[green]✓ Found {len(candidates)} integrated candidates[/green]")

        with_metadata_bonus = sum(1 for c in candidates if c["has_metadata_bonus"])
        self.pm.log(f"[dim]  - {with_metadata_bonus} pairs had metadata bonus[/dim]")

        return candidates

    def _resolve_verification_paths(
        self, candidate: dict[str, Any], image_paths: list[Path]
    ) -> tuple[Path, Path] | None:
        """Resolve paths for geometric verification.

        Args:
            candidate: Candidate duplicate pair.
            image_paths: List of paths to images.

        Returns:
            Tuple of (path1, path2) or None if paths cannot be resolved.

        """
        if "idx1" in candidate and "idx2" in candidate:
            idx1: int = candidate["idx1"]
            idx2: int = candidate["idx2"]
            if idx1 < len(image_paths) and idx2 < len(image_paths):
                return image_paths[idx1], image_paths[idx2]
            if len(image_paths) >= MIN_GROUP_SIZE:
                return image_paths[0], image_paths[1]
        elif len(image_paths) >= MIN_GROUP_SIZE:
            return image_paths[0], image_paths[1]
        return None

    def _compute_orb_features(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> tuple[Any, Any, Any, Any] | None:
        """Compute ORB features for two images.

        Args:
            img1: First grayscale image.
            img2: Second grayscale image.

        Returns:
            Tuple of (kp1, des1, kp2, des2) or None if insufficient features.

        """
        orb = cv2.ORB_create(  # type: ignore[attr-defined]
            nfeatures=self.config.orb_nfeatures,
            scaleFactor=self.config.orb_scale_factor,
            nlevels=self.config.orb_nlevels,
        )
        kp1, des1 = orb.detectAndCompute(img1, None)  # type: ignore[reportUnknownMemberType]
        kp2, des2 = orb.detectAndCompute(img2, None)  # type: ignore[reportUnknownMemberType]

        if des1 is None or des2 is None or len(des1) < MIN_DESCRIPTORS or len(des2) < MIN_DESCRIPTORS:  # type: ignore[reportUnknownArgumentType]
            return None
        return kp1, des1, kp2, des2  # type: ignore[reportUnknownVariableType]

    def _verify_homography_internal(
        self, good_matches: list[Any], kp1: Any, kp2: Any, candidate: dict[str, Any]  # noqa: ANN401
    ) -> bool:
        """Verify geometric consistency and update candidate.

        Args:
            good_matches: List of good feature matches.
            kp1: Keypoints from first image.
            kp2: Keypoints from second image.
            candidate: Candidate dict to update.

        Returns:
            True if verification passes.

        """
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # type: ignore[arg-type]
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # type: ignore[arg-type]

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is None:  # type: ignore[reportUnnecessaryComparison]
            return False

        inliers = np.sum(mask)
        min_features = min(len(kp1), len(kp2))
        inlier_ratio = inliers / len(good_matches)
        feature_coverage = inliers / min_features

        if inliers < self.config.min_absolute_inliers:
            return False

        has_metadata_bonus = candidate.get("has_metadata_bonus", False)
        required_ratio = (
            self.config.min_inlier_ratio_metadata
            if has_metadata_bonus
            else self.config.min_inlier_ratio
        )

        if min_features > LARGE_FEATURE_THRESHOLD:
            if feature_coverage < (required_ratio * 0.5):
                return False
        elif inlier_ratio < required_ratio:
            return False

        det = np.linalg.det(homography[:2, :2])
        if not (self.config.min_homography_det <= det <= self.config.max_homography_det):
            return False

        candidate["geometric_inliers"] = int(inliers)
        candidate["inlier_ratio"] = float(inlier_ratio)
        candidate["feature_coverage"] = float(feature_coverage)
        candidate["good_matches"] = len(good_matches)
        candidate["min_features"] = int(min_features)
        return True

    def geometric_verification(
        self, candidate: dict[str, Any], image_paths: list[Path]
    ) -> bool:
        """Verify with ORB+RANSAC using ratio-based thresholds.

        Args:
            candidate: Candidate duplicate pair to verify.
            image_paths: List of paths to images.

        Returns:
            True if verification passes, False otherwise.

        """
        paths = self._resolve_verification_paths(candidate, image_paths)
        if paths is None:
            return False

        images = _load_grayscale_images(paths[0], paths[1])
        if images is None:
            return False

        img1 = _resize_if_needed(images[0])
        img2 = _resize_if_needed(images[1])

        features = self._compute_orb_features(img1, img2)
        if features is None:
            return False
        kp1, des1, kp2, des2 = features

        good_matches = _compute_good_matches(des1, des2, self.config.lowe_ratio)
        if len(good_matches) < self.config.min_good_matches:
            return False

        return self._verify_homography_internal(good_matches, kp1, kp2, candidate)

    def _get_config_params(self) -> dict[str, Any]:
        """Get configuration parameters for worker processes.

        Returns:
            Dictionary of configuration parameters.

        """
        return {
            "orb_nfeatures": self.config.orb_nfeatures,
            "orb_scale_factor": self.config.orb_scale_factor,
            "orb_nlevels": self.config.orb_nlevels,
            "lowe_ratio": self.config.lowe_ratio,
            "min_good_matches": self.config.min_good_matches,
            "min_inlier_ratio": self.config.min_inlier_ratio,
            "min_inlier_ratio_metadata": self.config.min_inlier_ratio_metadata,
            "min_absolute_inliers": self.config.min_absolute_inliers,
            "min_homography_det": self.config.min_homography_det,
            "max_homography_det": self.config.max_homography_det,
        }

    def _run_parallel_verification(
        self,
        neural_candidates: list[dict[str, Any]],
        image_paths: list[Path],
        config_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run parallel geometric verification on candidates.

        Args:
            neural_candidates: List of candidate pairs.
            image_paths: List of image paths.
            config_params: Configuration parameters.

        Returns:
            List of verified candidates.

        """
        import multiprocessing as mp

        worker_count = max(1, int(mp.cpu_count() * 0.75))
        final_verified: list[dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures: dict[Future[tuple[bool, dict[str, Any]]], dict[str, Any]] = {}
            for candidate in neural_candidates:
                if _cleanup_in_progress:
                    break
                temp_candidate = {
                    "idx1": candidate["orig_idx1"],
                    "idx2": candidate["orig_idx2"],
                    "sscd_similarity": candidate["sscd_similarity"],
                    "has_metadata_bonus": candidate.get("has_metadata_bonus", False),
                }
                future = executor.submit(
                    geometric_verification_worker, (temp_candidate, image_paths, config_params)
                )
                futures[future] = candidate

            for future in as_completed(futures):
                if _cleanup_in_progress:
                    break
                candidate = futures[future]
                success, updated = future.result()
                if success:
                    candidate["geometric_inliers"] = updated["geometric_inliers"]
                    candidate["inlier_ratio"] = updated.get("inlier_ratio", 0)
                    candidate["feature_coverage"] = updated.get("feature_coverage", 0)
                    candidate["good_matches"] = updated.get("good_matches", 0)
                    candidate["min_features"] = updated.get("min_features", 0)
                    final_verified.append(candidate)
                self.pm.update("Geometric verification")

        return final_verified

    def _form_duplicate_groups(
        self, verified: list[dict[str, Any]], valid_indices: list[int], n: int
    ) -> list[list[int]]:
        """Form duplicate groups using union-find.

        Args:
            verified: List of verified candidate pairs.
            valid_indices: List of valid image indices.
            n: Total number of images.

        Returns:
            List of duplicate groups.

        """
        parent = list(range(n))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for candidate in verified:
            union(candidate["orig_idx1"], candidate["orig_idx2"])

        groups_dict: dict[int, list[int]] = defaultdict(list)
        for i in valid_indices:
            groups_dict[find(i)].append(i)

        return [members for members in groups_dict.values() if len(members) > 1]

    def find_ml_duplicates(
        self,
        features: np.ndarray,
        image_paths: list[Path],
        exclude_indices: set[int] | None = None,
        mode: str = "ml",
    ) -> list[list[int]]:
        """Complete ML duplicate detection pipeline.

        Args:
            features: Feature array from SSCD model.
            image_paths: List of paths to images.
            exclude_indices: Set of indices to exclude from detection.
            mode: Detection mode ('ml' or 'integrated').

        Returns:
            List of duplicate groups (each group is a list of indices).

        """
        if exclude_indices is None:
            exclude_indices = set()

        max_valid_index = min(len(image_paths), len(features)) - 1
        valid_indices = [
            i for i in range(len(image_paths)) if i not in exclude_indices and i <= max_valid_index
        ]
        if not valid_indices:
            return []

        idx_map = dict(enumerate(valid_indices))
        filtered_features = features[valid_indices]
        filtered_paths = [image_paths[i] for i in valid_indices]

        if mode == "integrated":
            neural_candidates = self.find_integrated_candidates(filtered_features, filtered_paths)
        else:
            neural_candidates = self.find_neural_candidates(filtered_features, filtered_paths)

        for candidate in neural_candidates:
            candidate["orig_idx1"] = idx_map[candidate["idx1"]]
            candidate["orig_idx2"] = idx_map[candidate["idx2"]]

        self.pm.log("\n[bold cyan]Geometric verification[/bold cyan]")
        self.pm.add_task("Geometric verification", total=len(neural_candidates))

        config_params = self._get_config_params()
        final_verified = self._run_parallel_verification(
            neural_candidates, image_paths, config_params
        )

        self.pm.log(
            f"[green]✓ Geometrically verified {len(final_verified)}/{len(neural_candidates)} pairs[/green]"
        )

        self.pm.log("\n[bold cyan]Forming duplicate groups...[/bold cyan]")
        groups = self._form_duplicate_groups(final_verified, valid_indices, len(image_paths))

        self.detailed_results = final_verified
        self.pm.log(f"[green]✓ Found {len(groups)} {mode.upper()} duplicate groups[/green]")

        return groups


# ==============================================================================
# Database Operations
# ==============================================================================


class ConnectionPool:
    """SQLite connection pool for concurrent database access."""

    def __init__(self, db_path: Path, pool_size: int = 5) -> None:
        """Initialize the connection pool.

        Args:
            db_path: Path to the SQLite database.
            pool_size: Number of connections to maintain.

        """
        self.db_path = db_path
        self.pool: Queue[sqlite3.Connection] = Queue(maxsize=pool_size)
        self._lock = threading.Lock()

        # Initialize pool
        for _ in range(pool_size):
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            self.pool.put(conn)

    @contextmanager
    def get_connection(self) -> Any:  # noqa: ANN401 - Generator type
        """Get a connection from the pool.

        Yields:
            SQLite connection.

        """
        conn = self.pool.get()
        try:
            yield conn
        finally:
            self.pool.put(conn)


@contextmanager
def get_db_connection(db_path: Path) -> Any:  # noqa: ANN401 - Generator type
    """Get a database connection from the pool.

    Args:
        db_path: Path to the SQLite database.

    Yields:
        SQLite connection.

    """
    if app_state.connection_pool is None:
        app_state.connection_pool = ConnectionPool(db_path)

    with app_state.connection_pool.get_connection() as conn:
        yield conn


def _prepare_metadata_batch(
    metadata_groups: list[list[int]],
    image_paths: list[Path],
    finder: DuplicateFinder,
) -> list[tuple[Any, ...]]:
    """Prepare batch data for metadata groups.

    Args:
        metadata_groups: List of metadata duplicate groups.
        image_paths: List of image paths.
        finder: DuplicateFinder with metadata.

    Returns:
        List of tuples for database insertion.

    """
    batch_data: list[tuple[Any, ...]] = []
    for group_id, group_members in enumerate(metadata_groups, 1):
        for member in group_members:
            metadata = finder.all_metadata.get(member, {})
            width = metadata.get("width", 0)
            height = metadata.get("height", 0)
            resolution = f"{width}x{height}" if width and height else "unknown"
            batch_data.append((
                str(image_paths[member].absolute()),
                image_paths[member].name,
                f"META_{group_id}",
                member == group_members[0],
                "metadata",
                1.0, 0, False,
                resolution, "active",
                metadata.get("datetime_original"),
                metadata.get("camera_make"),
                metadata.get("camera_model"),
                width, height,
            ))
    return batch_data


def _get_member_details(member: int, finder: DuplicateFinder) -> dict[str, Any] | None:
    """Get best matching details for a member.

    Args:
        member: Member index.
        finder: DuplicateFinder instance.

    Returns:
        Best matching result details or None.

    """
    member_details = None
    for result in finder.detailed_results:
        if result["orig_idx1"] != member and result["orig_idx2"] != member:
            continue
        if member_details is None or result.get(
            "integrated_score", result["sscd_similarity"]
        ) > member_details.get("integrated_score", member_details["sscd_similarity"]):
            member_details = result
    return member_details


def _prepare_ml_batch(
    ml_groups: list[list[int]],
    image_paths: list[Path],
    finder: DuplicateFinder,
    mode: str,
) -> list[tuple[Any, ...]]:
    """Prepare batch data for ML groups.

    Args:
        ml_groups: List of ML duplicate groups.
        image_paths: List of image paths.
        finder: DuplicateFinder with metadata.
        mode: Detection mode.

    Returns:
        List of tuples for database insertion.

    """
    detection_mode = "integrated" if mode == "integrated" else "ml"
    ml_batch_data: list[tuple[Any, ...]] = []

    for group_id, group_members in enumerate(ml_groups, 1):
        member_scores: dict[int, float] = {}
        for member in group_members:
            scores = [
                result.get("integrated_score", result["sscd_similarity"])
                for result in finder.detailed_results
                if result["orig_idx1"] == member or result["orig_idx2"] == member
            ]
            if scores:
                member_scores[member] = float(np.mean(scores))

        best_rep = max(member_scores, key=lambda k: member_scores[k]) if member_scores else group_members[0]

        for member in group_members:
            member_details = _get_member_details(member, finder)
            detection_method = "ml"
            if mode == "integrated" and member_details and member_details.get("has_metadata_bonus", False):
                detection_method = "ml+metadata"

            metadata = finder.all_metadata.get(member, {})
            width = metadata.get("width", 0)
            height = metadata.get("height", 0)
            resolution = f"{width}x{height}" if width and height else "unknown"

            ml_batch_data.append((
                str(image_paths[member].absolute()),
                image_paths[member].name,
                f"{detection_mode.upper()}_{group_id}",
                member == best_rep,
                detection_method,
                member_details["sscd_similarity"] if member_details else 0,
                member_details.get("geometric_inliers", 0) if member_details else 0,
                member_details.get("has_metadata_bonus", False) if member_details else False,
                resolution, "active",
                metadata.get("datetime_original"),
                metadata.get("camera_make"),
                metadata.get("camera_model"),
                width, height,
            ))

    return ml_batch_data


def save_results_to_db(  # noqa: PLR0913 - database save with all results
    config: Config,
    image_paths: list[Path],
    finder: DuplicateFinder,
    metadata_groups: list[list[int]],
    ml_groups: list[list[int]],
    exclude_indices: set[int],  # noqa: ARG001
    mode: str,
    features: np.ndarray | None = None,  # noqa: ARG001
) -> None:
    """Save detection results to database.

    Args:
        config: Detection configuration.
        image_paths: List of paths to images.
        finder: DuplicateFinder instance with metadata.
        metadata_groups: List of metadata duplicate groups.
        ml_groups: List of ML duplicate groups.
        exclude_indices: Set of indices to exclude (unused, kept for API compatibility).
        mode: Detection mode used.
        features: Optional feature array (unused, kept for API compatibility).

    """
    with get_db_connection(config.db_path) as conn:
        cursor = conn.cursor()

        folder_path = str(config.image_folder.absolute())
        folder_path_with_sep = folder_path if folder_path.endswith(os.sep) else folder_path + os.sep
        cursor.execute(
            "DELETE FROM images WHERE path = ? OR path LIKE ?",
            (folder_path, f"{folder_path_with_sep}%"),
        )

        batch_data = _prepare_metadata_batch(metadata_groups, image_paths, finder)
        cursor.executemany(
            """INSERT INTO images (path, name, group_id, is_representative,
                detection_method, sscd_score, geometric_inliers, metadata_bonus,
                resolution, status, datetime_original, camera_make, camera_model,
                width, height) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            batch_data,
        )

        ml_batch_data = _prepare_ml_batch(ml_groups, image_paths, finder, mode)
        if ml_batch_data:
            cursor.executemany(
                """INSERT INTO images (path, name, group_id, is_representative,
                    detection_method, sscd_score, geometric_inliers, metadata_bonus,
                    resolution, status, datetime_original, camera_make, camera_model,
                    width, height) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ml_batch_data,
            )

        conn.commit()


def load_active_groups(*, check_files: bool = False) -> list[str]:
    """Load active groups from database.

    Args:
        check_files: Whether to check file existence.

    Returns:
        List of active group IDs.

    """
    if app_state.db_path is None:
        msg = "db_path must be set before loading groups"
        raise ValueError(msg)
    with get_db_connection(app_state.db_path) as conn:
        cursor = conn.cursor()

        # Only check file existence if requested (expensive operation)
        if check_files:
            cursor.execute("SELECT id, path FROM images WHERE status = 'active'")
            rows = cursor.fetchall()

            # Batch check file existence
            missing_ids: list[int] = []
            with ThreadPoolExecutor(max_workers=10) as executor:

                def check_file(row: Any) -> tuple[int, bool]:  # noqa: ANN401
                    return (row["id"], Path(row["path"]).exists())

                results = executor.map(check_file, rows)
                missing_ids = [id_ for id_, exists in results if not exists]

            # Batch update missing files
            if missing_ids:
                cursor.executemany(
                    "UPDATE images SET status = 'missing' WHERE id = ?",
                    [(id_,) for id_ in missing_ids],
                )
                conn.commit()

        # Get groups with 2+ active images
        cursor.execute("""
            SELECT group_id, COUNT(*) as active_count
            FROM images
            WHERE status = 'active'
            GROUP BY group_id
            HAVING active_count >= 2
            ORDER BY group_id
        """)

        app_state.initial_active_groups = [row["group_id"] for row in cursor.fetchall()]
        app_state.active_groups_set = set(app_state.initial_active_groups)

        if app_state.initial_active_groups and (
            app_state.current_group_id is None
            or app_state.current_group_id not in app_state.initial_active_groups
        ):
            app_state.current_group_id = app_state.initial_active_groups[0]

    return app_state.initial_active_groups


# ==============================================================================
# Flask Routes
# ==============================================================================


def create_thumbnail(
    image_path: Path, max_size: tuple[int, int] = (800, 800)
) -> str | None:
    """Create base64 encoded thumbnail with caching.

    Args:
        image_path: Path to the image file.
        max_size: Maximum thumbnail dimensions.

    Returns:
        Base64 encoded thumbnail string or None on error.

    """
    # Check cache first
    path_str = str(image_path)
    cached = app_state.thumbnail_cache.get(path_str)
    if cached:
        return cached

    try:
        img = Image.open(image_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        if img.mode in ("RGBA", "P"):
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = rgb_img

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        thumbnail = base64.b64encode(buffer.getvalue()).decode()

        # Cache the result
        app_state.thumbnail_cache.put(path_str, thumbnail)
    except (OSError, Image.DecompressionBombError) as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    else:
        return thumbnail


def get_current_position() -> int:
    """Get the current position in the navigation list.

    Returns:
        Index of current group in the active groups list.

    """
    if app_state.current_group_id in app_state.initial_active_groups:
        return app_state.initial_active_groups.index(app_state.current_group_id)
    return 0


def is_group_still_active(group_id: str) -> bool:
    """Check if a group still has enough active images.

    Args:
        group_id: The group ID to check.

    Returns:
        True if group is still active.

    """
    return group_id in app_state.active_groups_set


def clear_caches() -> None:
    """Clear all caches."""
    app_state.thumbnail_cache.clear()
    app_state.group_data_cache.clear()


def cache_route(timeout: int = 5) -> Any:  # noqa: ANN401 - decorator type
    """Create a caching decorator for routes.

    Args:
        timeout: Cache timeout in seconds.

    Returns:
        Decorator function.

    """

    def decorator(f: Any) -> Any:  # noqa: ANN401 - callable type
        cache: dict[tuple[Any, ...], tuple[Any, float]] = {}

        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            cache_key = (args, frozenset(kwargs.items()))

            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < timeout:
                    return result

            # Generate result
            result = f(*args, **kwargs)
            cache[cache_key] = (result, time.time())

            # Clean old entries
            if len(cache) > MAX_ROUTE_CACHE_ENTRIES:
                oldest = min(cache.items(), key=lambda x: x[1][1])
                del cache[oldest[0]]

            return result

        return decorated_function

    return decorator


def _render_all_done() -> Any:  # noqa: ANN401 - Flask Response type
    """Render the all-done template.

    Returns:
        Rendered template for completed state.

    """
    if app_state.db_path is None:
        return "Database not initialized"
    with get_db_connection(app_state.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM images WHERE status = 'deleted'")
        total_deleted = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM images WHERE status = 'active'")
        total_remaining = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT group_id) FROM images")
        total_groups = cursor.fetchone()[0]

    return render_template(
        "duplicate-detector-ui.html",
        all_done=True,
        total_groups=total_groups,
        total_deleted=total_deleted,
        total_remaining=total_remaining,
    )


def _prepare_image_list(group_data: list[Any]) -> tuple[list[dict[str, Any]], int, str]:
    """Prepare image data list from group data.

    Args:
        group_data: Raw database rows for the group.

    Returns:
        Tuple of (images list, active count, detection method).

    """
    images: list[dict[str, Any]] = []
    active_count = 0
    detection_method = "unknown"

    for row in group_data:
        image_path = Path(row["path"])
        image_info = {
            "id": row["id"],
            "name": row["name"],
            "path": row["path"],
            "exists": image_path.exists() and row["status"] != "missing",
            "is_representative": row["is_representative"],
            "sscd_score": row["sscd_score"],
            "geometric_inliers": row["geometric_inliers"],
            "status": row["status"],
            "resolution": row["resolution"],
            "datetime_original": row["datetime_original"],
            "file_size": 0,
        }
        if image_info["exists"]:
            with contextlib.suppress(builtins.BaseException):
                image_info["file_size"] = image_path.stat().st_size
            if image_info["status"] == "active":
                active_count += 1
        if detection_method == "unknown":
            detection_method = row["detection_method"]
        images.append(image_info)

    return images, active_count, detection_method


def _ensure_valid_current_group() -> None:
    """Ensure current_group_id is valid and in active set."""
    if app_state.current_group_id not in app_state.active_groups_set:
        for group_id in app_state.initial_active_groups:
            if group_id in app_state.active_groups_set:
                app_state.current_group_id = group_id
                break


def _load_group_data_if_needed() -> tuple[Any, int]:
    """Load current group data and total groups from cache or database.

    Returns:
        Tuple of (group data rows, total groups count).

    """
    group_id = app_state.current_group_id
    if group_id is None:
        return [], 0

    current_group_data = app_state.group_data_cache.get(group_id)
    total_groups = app_state.total_groups_cache

    if current_group_data is None or total_groups is None:
        if app_state.db_path is None:
            return [], 0
        with get_db_connection(app_state.db_path) as conn:
            cursor = conn.cursor()
            if current_group_data is None:
                cursor.execute(
                    "SELECT * FROM images WHERE group_id = ? ORDER BY is_representative DESC, id LIMIT 100",
                    (group_id,),
                )
                current_group_data = cursor.fetchall()
                app_state.group_data_cache.put(group_id, current_group_data)
            if total_groups is None:
                cursor.execute("SELECT COUNT(DISTINCT group_id) FROM images")
                total_groups = cursor.fetchone()[0]
                app_state.total_groups_cache = total_groups

    return current_group_data, total_groups or 0


@app.route("/")
def index() -> Any:  # noqa: ANN401 - Flask Response typing
    """Show current group."""
    if not app_state.db_path or not Path(app_state.db_path).exists():
        return "No database found"

    if not app_state.initial_active_groups:
        load_active_groups(check_files=True)

    if not app_state.active_groups_set:
        return _render_all_done()

    _ensure_valid_current_group()
    current_group_data, total_groups = _load_group_data_if_needed()

    if not current_group_data:
        return _render_all_done()

    images, active_count, detection_method = _prepare_image_list(current_group_data)
    current_pos = get_current_position()
    completed_groups = len(app_state.initial_active_groups) - len(app_state.active_groups_set)
    progress_percent = int((completed_groups / total_groups) * 100) if total_groups else 0

    has_next_active = any(
        app_state.initial_active_groups[i] in app_state.active_groups_set
        for i in range(current_pos + 1, len(app_state.initial_active_groups))
    )

    return render_template(
        "duplicate-detector-ui.html",
        all_done=False,
        current_display_group=current_pos + 1,
        active_remaining=len(app_state.active_groups_set),
        total_groups=total_groups,
        completed_groups=completed_groups,
        progress_percent=progress_percent,
        group_id=app_state.current_group_id,
        detection_method=detection_method,
        active_images_count=active_count,
        images=images,
        is_first_group=current_pos == 0,
        is_last_group=current_pos >= len(app_state.initial_active_groups) - 1 and not has_next_active,
    )


def _move_to_trash(image_path: Path) -> bool:
    """Move file to trash or delete it.

    Args:
        image_path: Path to the file to delete.

    Returns:
        True if successful.

    """
    trash_dir = Path.home() / ".Trash"
    if trash_dir.exists():
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        trash_name = f"{timestamp}_{image_path.name}"
        shutil.move(str(image_path), str(trash_dir / trash_name))
    else:
        image_path.unlink()
    return True


def _process_file_deletions(
    files_to_trash: list[tuple[Path, Any]], group_id: str | None
) -> tuple[int, list[str], list[Any]]:
    """Process file deletions.

    Args:
        files_to_trash: List of (path, row) tuples.
        group_id: Current group ID.

    Returns:
        Tuple of (deleted count, results list, successfully deleted rows).

    """
    deleted_count = 0
    deletion_results: list[str] = []
    successfully_deleted: list[Any] = []

    for image_path, row in files_to_trash:
        if not image_path.exists():
            deletion_results.append(f"✗ Not found: {row['name']}")
            continue

        try:
            _move_to_trash(image_path)
            deleted_count += 1
            deletion_results.append(f"✓ Deleted: {row['name']}")
            successfully_deleted.append(row)

            with Path("deleted_files_log.txt").open("a") as f:
                f.write(f"{datetime.now(tz=UTC)}: Deleted {image_path} (Group: {group_id})\n")

        except (OSError, shutil.Error) as e:
            deletion_results.append(f"✗ Failed: {row['name']} - {e!s}")

    return deleted_count, deletion_results, successfully_deleted


@app.route("/delete", methods=["POST"])
def delete_images() -> Any:  # noqa: ANN401 - Flask Response typing
    """Delete selected images and update database."""
    data = request.get_json()
    indices = data.get("indices", [])

    if app_state.current_group_id is None:
        return jsonify({"success": False, "error": "No active group"})

    if app_state.db_path is None:
        return jsonify({"success": False, "error": "Database not initialized"})
    with get_db_connection(app_state.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM images WHERE group_id = ? ORDER BY is_representative DESC, id LIMIT 100",
            (app_state.current_group_id,),
        )
        current_group_data = list(cursor.fetchall())

        ids_to_delete: list[int] = []
        files_to_trash: list[tuple[Path, Any]] = []
        for idx in indices:
            if idx < len(current_group_data):
                row = current_group_data[idx]
                ids_to_delete.append(row["id"])
                files_to_trash.append((Path(row["path"]), row))

        if ids_to_delete:
            placeholders = ",".join("?" * len(ids_to_delete))
            cursor.execute(
                f"UPDATE images SET status = 'deleted' WHERE id IN ({placeholders})",  # noqa: S608
                ids_to_delete,
            )
            conn.commit()

    deleted_count, deletion_results, successfully_deleted = _process_file_deletions(
        files_to_trash, app_state.current_group_id
    )

    group_complete = False
    if successfully_deleted:
        with get_db_connection(app_state.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO deletion_log (image_path, group_id) VALUES (?, ?)",
                [(row["path"], app_state.current_group_id) for row in successfully_deleted],
            )
            conn.commit()

        with app_state.locked():
            app_state.group_data_cache.remove(app_state.current_group_id)
            app_state.total_groups_cache = None
            cursor.execute(
                "SELECT COUNT(*) as active_count FROM images WHERE group_id = ? AND status = 'active'",
                (app_state.current_group_id,),
            )
            active_count = cursor.fetchone()["active_count"]
            group_complete = active_count < MIN_GROUP_SIZE
            if group_complete:
                app_state.active_groups_set.discard(app_state.current_group_id)
                app_state.group_data_cache.remove(app_state.current_group_id)

    return jsonify({
        "success": True,
        "deleted_count": deleted_count,
        "requested_count": len(indices),
        "group_complete": group_complete,
        "deletion_results": deletion_results,
    })


@app.route("/next")
def next_group() -> Any:  # noqa: ANN401 - Flask Response typing
    """Go to next active group."""
    with app_state.locked():
        current_pos = get_current_position()
        found_next = False

        # Look for next active group in our initial list
        for i in range(current_pos + 1, len(app_state.initial_active_groups)):
            group_id = app_state.initial_active_groups[i]
            if group_id in app_state.active_groups_set:
                app_state.current_group_id = group_id
                found_next = True
                break

        # If not found, wrap around to beginning
        if not found_next:
            for i in range(current_pos):
                group_id = app_state.initial_active_groups[i]
                if group_id in app_state.active_groups_set:
                    app_state.current_group_id = group_id
                    break

    return index()


@app.route("/previous")
def previous_group() -> Any:  # noqa: ANN401 - Flask Response typing
    """Go to previous active group."""
    with app_state.locked():
        current_pos = get_current_position()

        # Look for previous active group
        for i in range(current_pos - 1, -1, -1):
            group_id = app_state.initial_active_groups[i]
            if group_id in app_state.active_groups_set:
                app_state.current_group_id = group_id
                break

    return index()


@app.route("/image/<int:image_id>")
def serve_image(image_id: int) -> Any:  # noqa: ANN401 - Flask Response typing
    """Securely serve an image using its database ID.

    Args:
        image_id: Database ID of the image.

    Returns:
        Image response or error tuple.

    """
    try:
        if app_state.db_path is None:
            return "Database not initialized", 500
        with get_db_connection(app_state.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT path FROM images WHERE id = ?", (image_id,))
            row = cursor.fetchone()

            if not row:
                return "Image not found", 404

            path = Path(row["path"])
            if not path.exists():
                return "Image file not found", 404

            # Generate thumbnail on-demand using existing function
            thumb_data = create_thumbnail(path)
            if not thumb_data:
                return "Failed to generate thumbnail", 500

            # Convert base64 back to bytes for proper HTTP response
            image_bytes = base64.b64decode(thumb_data)

            # Create response with proper headers
            return Response(
                image_bytes,
                mimetype="image/jpeg",
                headers={
                    "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                    "ETag": f'"{hash(str(path) + str(path.stat().st_mtime))}"',
                },
            )

    except (OSError, sqlite3.Error, ValueError) as e:
        return f"Error serving image: {e!s}", 500


# ==============================================================================
# Main Pipeline
# ==============================================================================

# Detection mode descriptions
MODE_DESCRIPTIONS = {
    "metadata": "Metadata only (EXIF DateTime + Camera)",
    "ml": "ML only (Neural + Geometric)",
    "both": "Sequential (Metadata → ML on remainder)",
    "integrated": "Integrated (ML with metadata bonus)",
}


def _scan_image_files(config: Config, pm: ProgressManager) -> list[Path]:
    """Scan folder for image files.

    Args:
        config: Detection configuration.
        pm: Progress manager for logging.

    Returns:
        Sorted list of absolute image paths.

    """
    pm.log("\n[bold]Scanning for images...[/bold]")
    image_paths: list[Path] = []
    for ext in config.image_extensions:
        image_paths.extend(config.image_folder.glob(f"*{ext}"))
    return sorted({path.absolute() for path in image_paths})


def _log_image_types(image_paths: list[Path], pm: ProgressManager) -> None:
    """Log image type statistics.

    Args:
        image_paths: List of image paths.
        pm: Progress manager for logging.

    """
    type_counts: dict[str, int] = defaultdict(int)
    for path in image_paths:
        type_counts[path.suffix.lower()] += 1

    pm.log("Image types:")
    for ext, count in sorted(type_counts.items()):
        pm.log(f"  {ext}: {count}")


def _extract_features_with_cache(
    config: Config, pm: ProgressManager, image_paths: list[Path]
) -> np.ndarray:
    """Extract features with caching support.

    Args:
        config: Detection configuration.
        pm: Progress manager for logging.
        image_paths: List of image paths.

    Returns:
        Feature array for all images.

    """
    cache_key = config.get_cache_key()
    feature_cache_file = config.cache_dir / f"features_enhanced_{cache_key}.npz"

    if feature_cache_file.exists():
        pm.log("Loading cached features...")
        cache = np.load(feature_cache_file, mmap_mode="r")
        features = np.array(cache["sscd"])
        pm.log("[green]✓ Loaded features from cache[/green]")
        return features

    model_manager = FastModelManager(config, pm)
    model_manager.load_models()
    features = model_manager.extract_features_fast(image_paths)

    pm.log("Saving feature cache...")
    np.savez_compressed(feature_cache_file, sscd=features)
    model_manager.cleanup()

    return features


def _display_detection_summary(  # noqa: PLR0913 - display function with related params
    config: Config,
    image_paths: list[Path],
    metadata_groups: list[list[int]],
    ml_groups: list[list[int]],
    exclude_indices: set[int],
    detection_mode: str,
    finder: DuplicateFinder,
    elapsed: float,
) -> None:
    """Display the detection summary table.

    Args:
        config: Detection configuration.
        image_paths: List of all image paths.
        metadata_groups: Metadata duplicate groups.
        ml_groups: ML duplicate groups.
        exclude_indices: Indices excluded by metadata detection.
        detection_mode: The detection mode used.
        finder: DuplicateFinder instance.
        elapsed: Total processing time.

    """
    summary_table = Table(title="Detection Summary", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total images", str(len(image_paths)))
    summary_table.add_row("Detection mode", f"{config.mode} - {MODE_DESCRIPTIONS[config.mode]}")

    if config.mode in ["metadata", "both"]:
        summary_table.add_row("Metadata duplicate groups", str(len(metadata_groups)))
        summary_table.add_row("Metadata duplicate images", str(len(exclude_indices)))

    if config.mode in ["ml", "both", "integrated"]:
        summary_table.add_row(f"{detection_mode.upper()} duplicate groups", str(len(ml_groups)))
        summary_table.add_row(
            f"{detection_mode.upper()} duplicate images", str(sum(len(g) for g in ml_groups))
        )

        if config.mode == "integrated":
            groups_with_metadata = _count_groups_with_metadata_bonus(ml_groups, finder)
            summary_table.add_row("Groups with metadata bonus", str(len(groups_with_metadata)))

    total_dups = len(exclude_indices) + sum(len(g) for g in ml_groups)
    unique_images = len(image_paths) - total_dups
    summary_table.add_row("Total duplicate images", str(total_dups))
    summary_table.add_row("Unique images", str(unique_images))
    summary_table.add_row("Processing time", humanize.naturaldelta(elapsed))
    summary_table.add_row("Speed", f"{len(image_paths) / elapsed:.1f} images/second")

    console.print("\n")
    console.print(summary_table)


def _count_groups_with_metadata_bonus(
    ml_groups: list[list[int]], finder: DuplicateFinder
) -> set[tuple[int, ...]]:
    """Count groups that have metadata bonus applied.

    Args:
        ml_groups: ML duplicate groups.
        finder: DuplicateFinder instance.

    Returns:
        Set of group tuples with metadata bonus.

    """
    groups_with_metadata: set[tuple[int, ...]] = set()
    for group in ml_groups:
        for member in group:
            for result in finder.detailed_results:
                if (
                    result["orig_idx1"] == member or result["orig_idx2"] == member
                ) and result.get("has_metadata_bonus", False):
                    groups_with_metadata.add(tuple(group))
                    break
    return groups_with_metadata


def run_detection(config: Config, pm: ProgressManager) -> bool:
    """Run the duplicate detection pipeline.

    Args:
        config: Detection configuration.
        pm: Progress manager for logging.

    Returns:
        True if detection completed successfully.

    """
    start_time = time.time()
    pm.log(f"\nMode: {config.mode} - {MODE_DESCRIPTIONS[config.mode]}")

    image_paths = _scan_image_files(config, pm)

    if not image_paths:
        pm.log("[red]No images found![/red]")
        return False

    pm.log(f"[green]✓ Found {len(image_paths)} images[/green]")
    _log_image_types(image_paths, pm)

    init_database(config.db_path)
    finder = DuplicateFinder(config, pm)
    finder.extract_all_metadata(image_paths)

    metadata_groups: list[list[int]] = []
    ml_groups: list[list[int]] = []
    exclude_indices: set[int] = set()
    features: np.ndarray | None = None
    detection_mode = ""

    if config.mode in ["metadata", "both"]:
        pm.log("\n[bold cyan]Phase 1: Metadata Detection[/bold cyan]")
        metadata_groups, exclude_indices = finder.find_metadata_duplicates(image_paths)

    if config.mode in ["ml", "both", "integrated"]:
        if config.mode == "integrated":
            pm.log("\n[bold cyan]Integrated Detection (ML + Metadata Bonus)[/bold cyan]")
        else:
            pm.log("\n[bold cyan]Phase 2: ML Detection (Neural + Geometric)[/bold cyan]")

        features = _extract_features_with_cache(config, pm, image_paths)

        if config.mode == "both":
            pm.log(f"[yellow]Excluding {len(exclude_indices)} images already found by metadata[/yellow]")

        detection_mode = "integrated" if config.mode == "integrated" else "ml"
        ml_groups = finder.find_ml_duplicates(
            features,
            image_paths,
            exclude_indices if config.mode == "both" else set(),
            mode=detection_mode,
        )

    pm.log("\n[bold]Saving results to database...[/bold]")
    save_results_to_db(
        config, image_paths, finder, metadata_groups, ml_groups,
        exclude_indices, config.mode, features,
    )
    pm.log("[green]✓ Results saved to database[/green]")

    elapsed = time.time() - start_time
    _display_detection_summary(
        config, image_paths, metadata_groups, ml_groups,
        exclude_indices, detection_mode, finder, elapsed,
    )

    return True


def _parse_arguments() -> Any:  # noqa: ANN401 - argparse Namespace
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.

    """
    import argparse

    parser = argparse.ArgumentParser(description="Unified Duplicate Detector - Detection + Review")
    parser.add_argument("image_folder", help="Folder containing images")
    parser.add_argument(
        "--mode", "-m", choices=["metadata", "ml", "both", "integrated"],
        default="integrated", help="Detection mode (default: integrated)",
    )
    parser.add_argument("--threshold", "-t", type=float, help="SSCD threshold")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--workers", "-w", type=int, default=11, help="Number of workers")
    parser.add_argument("--dataloader-workers", "-dw", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cache")
    parser.add_argument("--port", "-p", type=int, default=5555, help="Web UI port")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    return parser.parse_args()


def _configure_from_args(args: Any) -> Config:  # noqa: ANN401 - argparse Namespace
    """Create Config from parsed arguments.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Configured Config instance.

    """
    config = Config()
    config.image_folder = Path(args.image_folder).absolute()
    config.mode = args.mode
    config.batch_size = args.batch_size
    config.num_workers = args.workers
    config.dataloader_workers = args.dataloader_workers

    if args.threshold:
        config.sscd_threshold = args.threshold

    if args.no_cache and config.cache_dir.exists():
        for cache_file in config.cache_dir.glob("features_*.npz"):
            cache_file.unlink()
        console.print("[yellow]Cleared feature cache[/yellow]")

    return config


def _start_web_server(port: int, *, auto_open: bool) -> None:
    """Start the web UI server.

    Args:
        port: Port number for the server.
        auto_open: Whether to auto-open browser.

    """
    console.print("\n[bold cyan]Launching review interface...[/bold cyan]")
    console.print(f"Starting server on http://localhost:{port}")
    console.print("Press Ctrl+C to stop\n")

    if auto_open:
        def open_browser() -> None:
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

    try:
        serve(app, host="0.0.0.0", port=port, threads=4)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
        sys.exit(0)


def main() -> None:
    """Execute the main entry point."""
    args = _parse_arguments()

    console.print(
        Panel.fit(
            "[bold cyan]Unified Duplicate Detector[/bold cyan]\n"
            "Detection + Review in one tool\n"
            "Supports: HEIC/JPEG | Methods: Metadata + ML",
            border_style="cyan",
        )
    )

    config = _configure_from_args(args)

    try:
        with ProgressManager() as pm:
            start_time = time.time()
            pm.log("\n[bold]System Information:[/bold]")
            pm.log(f"Device: {config.device.upper()}")
            pm.log(f"CPU cores: {mp.cpu_count()}")
            pm.log(f"Workers: {config.num_workers}")
            pm.log(f"DataLoader workers: {config.dataloader_workers}")
            pm.log(f"Memory: {humanize.naturalsize(psutil.virtual_memory().total)}")

            success = run_detection(config, pm)
            if success:
                elapsed = time.time() - start_time
                pm.log(f"\n[green]✓ Detection completed in {humanize.naturaldelta(elapsed)}[/green]")
            else:
                sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Detection interrupted[/yellow]")
        sys.exit(0)

    app_state.db_path = config.db_path
    if app_state.db_path.exists():
        load_active_groups(check_files=True)
        console.print(f"[dim]Loaded {len(app_state.initial_active_groups)} active groups from database[/dim]")

    _start_web_server(args.port, auto_open=not args.no_browser)


if __name__ == "__main__":
    main()
