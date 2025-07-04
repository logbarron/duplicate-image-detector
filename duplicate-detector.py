#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "torchvision",
#   "numpy",
#   "pillow",
#   "pillow-heif",
#   "pandas",
#   "tqdm",
#   "psutil",
#   "scipy",
#   "scikit-learn",
#   "opencv-python",
#   "rich",
#   "humanize",
#   "exifread",
#   "flask",
#   "waitress",
# ]
# ///
"""
Unified Duplicate Detector - Metadata + Neural Features with Integrated Review UI
"""

import os
import sys
import time
import logging
import warnings
import hashlib
import signal
import threading
import subprocess
import atexit
import gc
import sqlite3
import json
import shutil
import base64
import webbrowser
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from datetime import datetime
from io import BytesIO
import urllib.request

mp.set_start_method('spawn', force=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import psutil
import humanize
import exifread

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn,
    TimeElapsedColumn, MofNCompleteColumn, TaskProgressColumn, DownloadColumn,
    TransferSpeedColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text

from flask import Flask, render_template, request, jsonify, send_file
from waitress import serve

import pillow_heif
pillow_heif.register_heif_opener()

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

console = Console()

# Global flag for cleanup
_cleanup_in_progress = False
_active_dataloaders = []

# Flask app - templates in current directory
app = Flask(__name__, template_folder='.')

# LRU caches with proper size limits
from functools import lru_cache
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
        # Thread safety lock for concurrent access
        self._lock = threading.Lock()
    
    def get(self, key):
        with self._lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key, value):
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def __contains__(self, key):
        with self._lock:
            return key in self.cache
    
    def clear(self):
        with self._lock:
            self.cache.clear()
    
    def remove(self, key):
        with self._lock:
            self.cache.pop(key, None)

# State management
class AppState:
    def __init__(self):
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
    
    def reset_groups(self):
        with self._lock:
            self.current_group_id = None
            self.initial_active_groups = []
            self.active_groups_set = set()
            self.total_groups_cache = None

app_state = AppState()

# ==============================================================================
# Database Schema
# ==============================================================================

def init_database(db_path: Path):
    """Initialize SQLite database with schema"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
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
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_group_id ON images(group_id);
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_status ON images(status);
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_path ON images(path);
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_path_status ON images(path, status);
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_status_group ON images(status, group_id);
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_group_status_rep ON images(group_id, status, is_representative);
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_status_active ON images(status) WHERE status = 'active';
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS deletion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            group_id TEXT,
            deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# ==============================================================================
# Signal Handler for Graceful Shutdown
# ==============================================================================

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global _cleanup_in_progress
    if not _cleanup_in_progress:
        _cleanup_in_progress = True
        console.print("\n[yellow]Received interrupt signal. Shutting down...[/yellow]")
        # Don't call cleanup_resources here - let atexit handle it
        # Just exit cleanly
        os._exit(0)

def cleanup_resources():
    """Clean up all resources"""
    global _active_dataloaders
    
    # Only clean up if we have active resources
    if not _active_dataloaders and not torch.cuda.is_available():
        return
    
    # Clean up dataloaders
    for dataloader in _active_dataloaders:
        try:
            if hasattr(dataloader, '_iterator'):
                del dataloader._iterator
            del dataloader
        except (AttributeError, TypeError):
            pass
    _active_dataloaders.clear()
    
    # Force garbage collection
    gc.collect()
    
    # Clear PyTorch cache if available
    try:
        if hasattr(torch, 'mps') and torch.mps.is_available():
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
    """Configuration for enhanced duplicate detection"""
    image_folder: Path = Path(".")
    db_path: Path = Path(".duplicate_detector.db")
    cache_dir: Path = Path(".duplicate_cache")
    
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
    
    image_extensions: List[str] = field(default_factory=lambda: [
        ".heic", ".HEIC", ".jpg", ".jpeg", ".JPG", ".JPEG"
    ])
    
    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.image_folder / self.db_path.name
    
    def get_cache_key(self) -> str:
        """Generate a cache key based on configuration parameters"""
        key_parts = [
            str(self.image_folder.absolute()),
            str(self.sscd_threshold),
            str(self.orb_nfeatures),
            str(self.min_inlier_ratio),
            "padded_enhanced",
            "_".join(sorted(self.image_extensions))
        ]
        
        key_string = "_".join(key_parts)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()[:16]
        
        return cache_key

# ==============================================================================
# Metadata Extraction
# ==============================================================================

class MetadataExtractor:
    """Extract and compare image metadata"""
    
    def __init__(self):
        # Use bounded LRUCache instead of unbounded dictionary to prevent memory leaks
        self._cache = LRUCache(capacity=10000)
    
    def extract_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract relevant metadata from image"""
        # Check cache first
        cache_key = (str(image_path), image_path.stat().st_mtime)
        cached_metadata = self._cache.get(cache_key)
        if cached_metadata is not None:
            return cached_metadata.copy()
        
        metadata = {
            'datetime_original': None,
            'camera_make': None,
            'camera_model': None,
            'width': None,
            'height': None,
            'valid': False
        }
        
        try:
            # Fast image info without loading full image
            with Image.open(image_path) as img:
                metadata['width'] = img.width
                metadata['height'] = img.height
        except (IOError, OSError, Image.DecompressionBombError) as e:
            print(f"Error getting dimensions for {image_path}: {e}")
        
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False, strict=False)
                
                if 'EXIF DateTimeOriginal' in tags:
                    metadata['datetime_original'] = str(tags['EXIF DateTimeOriginal'])
                elif 'Image DateTime' in tags:
                    metadata['datetime_original'] = str(tags['Image DateTime'])
                
                if 'Image Make' in tags:
                    metadata['camera_make'] = str(tags['Image Make']).strip()
                
                if 'Image Model' in tags:
                    metadata['camera_model'] = str(tags['Image Model']).strip()

        # exifread’s HEIC parser may raise a plain AssertionError – catch it as well
        except (IOError, OSError, KeyError, AttributeError, AssertionError):
            pass
        
        if not metadata['datetime_original'] or not metadata['camera_make']:
            try:
                img = Image.open(image_path)
                
                try:
                    exif_data = img._getexif()
                    if exif_data:
                        for tag, value in exif_data.items():
                            tag_name = TAGS.get(tag, tag)
                            
                            if tag_name == 'DateTimeOriginal' and not metadata['datetime_original']:
                                metadata['datetime_original'] = str(value)
                            elif tag_name == 'Make' and not metadata['camera_make']:
                                metadata['camera_make'] = str(value).strip()
                            elif tag_name == 'Model' and not metadata['camera_model']:
                                metadata['camera_model'] = str(value).strip()
                except AttributeError:
                    pass
                finally:
                    img.close()
                    
            except Exception as e:
                pass
        
        if not metadata['datetime_original'] or not metadata['camera_make']:
            try:
                result = subprocess.run(
                    ['mdls', '-name', 'kMDItemContentCreationDate', 
                     '-name', 'kMDItemAcquisitionMake',
                     '-name', 'kMDItemAcquisitionModel', str(image_path)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'kMDItemContentCreationDate' in line and not metadata['datetime_original']:
                            parts = line.split('=', 1)
                            if len(parts) == 2:
                                date_str = parts[1].strip()
                                if date_str and date_str != '(null)':
                                    date_str = date_str.strip('"')
                                    date_parts = date_str.split(' ')
                                    if len(date_parts) >= 2:
                                        date = date_parts[0].replace('-', ':')
                                        time = date_parts[1]
                                        metadata['datetime_original'] = f"{date} {time}"
                        
                        elif 'kMDItemAcquisitionMake' in line and not metadata['camera_make']:
                            parts = line.split('=', 1)
                            if len(parts) == 2:
                                make = parts[1].strip().strip('"')
                                if make and make != '(null)':
                                    metadata['camera_make'] = make
                        
                        elif 'kMDItemAcquisitionModel' in line and not metadata['camera_model']:
                            parts = line.split('=', 1)
                            if len(parts) == 2:
                                model = parts[1].strip().strip('"')
                                if model and model != '(null)':
                                    metadata['camera_model'] = model
                                    
            except Exception as e:
                pass
        
        if metadata['datetime_original'] and metadata['camera_make']:
            metadata['valid'] = True
        
        # Store in cache for future use
        self._cache.put(cache_key, metadata.copy())
        
        return metadata
    
    @staticmethod
    def create_metadata_key(metadata: Dict[str, Any]) -> Optional[str]:
        """Create a unique key from metadata for comparison"""
        if not metadata['valid']:
            return None
        
        key_parts = [
            metadata['datetime_original'] or '',
            metadata['camera_make'] or '',
            metadata['camera_model'] or ''
        ]
        
        return '|'.join(key_parts)

# ==============================================================================
# System Monitor
# ==============================================================================

class SystemMonitor:
    """Background system monitoring with proper cleanup"""
    def __init__(self):
        self.cpu_percent = 0.0
        self.memory_used = 0
        self.memory_total = psutil.virtual_memory().total
        self.running = True
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
    
    def _monitor(self):
        while self.running and not self._stop_event.is_set():
            try:
                self.cpu_percent = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                self.memory_used = mem.used
                self._stop_event.wait(0.5)
            except:
                break
    
    def stop(self):
        self.running = False
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
    
    def get_status(self) -> str:
        return (f"CPU: {self.cpu_percent:5.1f}% | "
                f"RAM: {humanize.naturalsize(self.memory_used)}/{humanize.naturalsize(self.memory_total)}")

# ==============================================================================
# Progress Manager
# ==============================================================================

class ProgressManager:
    """Centralized progress management with proper cleanup"""
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True
        )
        self.tasks = {}
        self.system_monitor = SystemMonitor()
        self.start_time = time.time()
    
    def __enter__(self):
        self.progress.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.system_monitor.stop()
        except:
            pass
        self.progress.__exit__(exc_type, exc_val, exc_tb)
    
    def add_task(self, description: str, total: int) -> int:
        task_id = self.progress.add_task(description, total=total)
        self.tasks[description] = task_id
        return task_id
    
    def update(self, description: str, advance: int = 1):
        if description in self.tasks:
            self.progress.advance(self.tasks[description], advance)
    
    def log(self, message: str, style: str = ""):
        self.progress.console.print(message, style=style)
    
    def status_panel(self):
        elapsed = time.time() - self.start_time
        return Panel(
            f"{self.system_monitor.get_status()}\n"
            f"Elapsed: {humanize.naturaldelta(elapsed)}",
            title="System Status",
            border_style="cyan"
        )

# ==============================================================================
# Color Normalization and Dataset
# ==============================================================================

def apply_color_normalization(image_array: np.ndarray) -> np.ndarray:
    """YUV histogram equalization + CLAHE for photographed prints"""
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

class FastSSCDDataset(Dataset):
    """Fast dataset for SSCD with color normalization and black padding"""
    def __init__(self, paths: List[Path], apply_color_norm: bool = True):
        self.paths = paths
        self.apply_color_norm = apply_color_norm
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def resize_with_padding(self, img: Image.Image, target_size: int = 288) -> Image.Image:
        """Resize image to fit within target_size x target_size with black padding"""
        orig_width, orig_height = img.size
        
        scale = min(target_size / orig_width, target_size / orig_height)
        
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        img_resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        
        left = (target_size - new_width) // 2
        top = (target_size - new_height) // 2
        
        canvas.paste(img_resized, (left, top))
        
        return canvas
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            
            if self.apply_color_norm:
                img_array = np.array(img)
                img_array = apply_color_normalization(img_array)
                img = Image.fromarray(img_array)
            
            img = self.resize_with_padding(img, target_size=288)
            return self.transform(img), idx
            
        except Exception as e:
            return torch.zeros(3, 288, 288), idx

# ==============================================================================
# Model Manager
# ==============================================================================

class FastModelManager:
    """Optimized model manager with single-pass extraction and cleanup"""
    def __init__(self, config: Config, pm: ProgressManager):
        self.config = config
        self.pm = pm
        self.device = torch.device(config.device)
        self.models = {}
        self.dims = {}
    
    def load_models(self):
        """Load SSCD disc_large"""
        self.pm.log("[bold]Loading copy detection model...[/bold]")
        
        self.pm.log("Loading SSCD disc_large (1024-dim)...")
        self._load_sscd()
        
        self.pm.log("[green]✓ Model loaded successfully[/green]")
    
    def _load_sscd(self):
        """Load SSCD disc_large"""
        model_url = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt"
        cache_path = self.config.cache_dir / "sscd_disc_large.torchscript.pt"
        
        if not cache_path.exists():
            self.pm.log("Downloading SSCD model...")
            
            download_task = self.pm.progress.add_task(
                "[bold blue]Downloading SSCD model",
                total=None,
                visible=True
            )
            
            def download_hook(block_num, block_size, total_size):
                if self.pm.progress.tasks[download_task].total is None and total_size > 0:
                    self.pm.progress.update(download_task, total=total_size)
                downloaded = block_num * block_size
                self.pm.progress.update(download_task, completed=downloaded)
            
            try:
                urllib.request.urlretrieve(model_url, cache_path, reporthook=download_hook)
                self.pm.progress.update(download_task, visible=False)
            except Exception as e:
                self.pm.progress.update(download_task, visible=False)
                raise e
        
        model = torch.jit.load(cache_path, map_location='cpu')
        self.models['sscd'] = model.to(self.device).eval()
        self.dims['sscd'] = 1024
        self.pm.log("[green]✓ SSCD loaded[/green]")
    
    def cleanup(self):
        """Clean up models and free memory"""
        for name, model in self.models.items():
            del model
        self.models.clear()
        gc.collect()
        if hasattr(torch, 'mps') and torch.mps.is_available():
            torch.mps.empty_cache()
    
    @torch.no_grad()
    def extract_features_fast(self, image_paths: List[Path]) -> np.ndarray:
        """Extract features using optimized pipeline with proper cleanup"""
        global _active_dataloaders
        
        n = len(image_paths)
        features = np.zeros((n, self.dims['sscd']), dtype=np.float32)
        
        start_time = time.time()
        
        self.pm.log("\n[cyan]SSCD feature extraction (multi-worker)[/cyan]")
        self.pm.log(f"Using {self.config.dataloader_workers} dataloader workers")
        self.pm.log("[yellow]Note: Images will be padded with black to preserve aspect ratio[/yellow]")
        
        dataset = FastSSCDDataset(image_paths, apply_color_norm=True)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.dataloader_workers,
            prefetch_factor=2 if self.config.dataloader_workers > 0 else None,
            pin_memory=False,
            persistent_workers=False
        )
        
        _active_dataloaders.append(dataloader)
        
        task_id = self.pm.add_task("SSCD features", total=n)
        
        try:
            for batch, indices in dataloader:
                if _cleanup_in_progress:
                    break
                    
                batch = batch.to(self.device)
                batch_features = self.models['sscd'](batch)
                batch_features = F.normalize(batch_features, dim=1)
                
                for i, idx in enumerate(indices):
                    features[idx] = batch_features[i].cpu().numpy()
                
                self.pm.update("SSCD features", advance=len(indices))
                
                if indices[0] % (self.config.batch_size * 20) == 0:
                    if hasattr(torch, 'mps') and torch.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
        finally:
            try:
                if hasattr(dataloader, '_iterator'):
                    del dataloader._iterator
                _active_dataloaders.remove(dataloader)
                del dataloader
            except:
                pass
            gc.collect()
        
        features = normalize(features, axis=1)
        
        elapsed = time.time() - start_time
        self.pm.log(f"[green]✓ Feature extraction complete in {elapsed:.1f}s ({n/elapsed:.1f} img/s)[/green]")
        
        return features

# ==============================================================================
# Geometric Verification Worker (for ProcessPoolExecutor)
# ==============================================================================

def geometric_verification_worker(candidate_data):
    """Standalone worker function for geometric verification in separate processes"""
    candidate, image_paths, config_params = candidate_data
    
    path1 = image_paths[candidate['idx1']]
    path2 = image_paths[candidate['idx2']]
    
    try:
        # Use context managers to ensure images are closed even if errors occur
        with Image.open(path1) as pil_img1, Image.open(path2) as pil_img2:
            # Convert to grayscale
            pil_img1 = pil_img1.convert('L')
            pil_img2 = pil_img2.convert('L')
            
            # Convert to numpy arrays
            img1 = np.array(pil_img1)
            img2 = np.array(pil_img2)
    except Exception as e:
        return False, candidate
    
    max_dim = 1500
    h1, w1 = img1.shape
    if max(h1, w1) > max_dim:
        scale = max_dim / max(h1, w1)
        img1 = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
    
    h2, w2 = img2.shape
    if max(h2, w2) > max_dim:
        scale = max_dim / max(h2, w2)
        img2 = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))
    
    orb = cv2.ORB_create(
        nfeatures=config_params['orb_nfeatures'],
        scaleFactor=config_params['orb_scale_factor'],
        nlevels=config_params['orb_nlevels']
    )
    
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return False, candidate
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < config_params['lowe_ratio'] * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < config_params['min_good_matches']:
        return False, candidate
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if homography is None:
        return False, candidate
    
    inliers = np.sum(mask)
    
    has_metadata_bonus = candidate.get('has_metadata_bonus', False)
    required_ratio = config_params['min_inlier_ratio_metadata'] if has_metadata_bonus else config_params['min_inlier_ratio']
    
    min_features = min(len(kp1), len(kp2))
    inlier_ratio = inliers / len(good_matches)
    feature_coverage = inliers / min_features
    
    if inliers < config_params['min_absolute_inliers']:
        return False, candidate
        
    if min_features > 1000:
        if feature_coverage < (required_ratio * 0.5):
            return False, candidate
    else:
        if inlier_ratio < required_ratio:
            return False, candidate
    
    det = np.linalg.det(homography[:2, :2])
    if not (config_params['min_homography_det'] <= det <= config_params['max_homography_det']):
        return False, candidate
    
    # Update candidate with results
    candidate['geometric_inliers'] = int(inliers)
    candidate['inlier_ratio'] = float(inlier_ratio)
    candidate['feature_coverage'] = float(feature_coverage)
    candidate['good_matches'] = len(good_matches)
    candidate['min_features'] = int(min_features)
    
    return True, candidate

# ==============================================================================
# Duplicate Detection
# ==============================================================================

class DuplicateFinder:
    """Find duplicates with metadata and/or neural features"""
    def __init__(self, config: Config, pm: ProgressManager):
        self.config = config
        self.pm = pm
        self.metadata_extractor = MetadataExtractor()
        self.all_metadata = {}
        self.metadata_keys = {}
        self.detailed_results = []  # Store detailed results
    
    def extract_all_metadata(self, image_paths: List[Path]):
        """Extract metadata for all images upfront"""
        self.pm.log("\n[bold cyan]Extracting metadata for all images[/bold cyan]")
        self.pm.log("[yellow]Using exifread + macOS mdls for better HEIC compatibility[/yellow]")
        
        task_id = self.pm.add_task("Extracting metadata", total=len(image_paths))
        
        # Parallel metadata extraction
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {}
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
        
        self.pm.log(f"[dim]Found valid metadata in {valid_metadata_count}/{len(image_paths)} images[/dim]")
    
    def find_metadata_duplicates(self, image_paths: List[Path]) -> Tuple[List[List[int]], Set[int]]:
        """Find duplicates based on metadata"""
        self.pm.log("\n[bold cyan]Finding metadata-based duplicates[/bold cyan]")
        
        if not self.all_metadata:
            self.extract_all_metadata(image_paths)
        
        metadata_groups = defaultdict(list)
        for idx, metadata_key in self.metadata_keys.items():
            if metadata_key:
                metadata_groups[metadata_key].append(idx)
        
        duplicate_groups = []
        all_duplicates = set()
        
        for key, indices in metadata_groups.items():
            if len(indices) > 1:
                duplicate_groups.append(indices)
                all_duplicates.update(indices)
        
        self.pm.log(f"[green]✓ Found {len(duplicate_groups)} metadata duplicate groups "
                   f"({len(all_duplicates)} total images)[/green]")
        
        if duplicate_groups and len(metadata_groups) > 0:
            sample_key = list(metadata_groups.keys())[0]
            self.pm.log(f"[dim]Sample metadata key: {sample_key}[/dim]")
        
        return duplicate_groups, all_duplicates
    
    def find_neural_candidates(self, features: np.ndarray, 
                             image_paths: List[Path]) -> List[Dict]:
        """Find candidates using neural features directly"""
        self.pm.log("\n[bold cyan]Finding neural candidates[/bold cyan]")
        
        n = len(image_paths)
        candidates = []
        
        # Batch similarity computation
        batch_size = 1000
        task_id = self.pm.add_task("Computing similarities", total=n)
        
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
                        candidates.append({
                            'idx1': global_i,
                            'idx2': j,
                            'sscd_similarity': float(similarities[bi, j]),
                            'metadata_match': 0.0,
                            'integrated_score': float(similarities[bi, j]),
                            'has_metadata_bonus': False
                        })
            
            self.pm.update("Computing similarities", advance=end_i - i)
        
        self.pm.log(f"[green]✓ Found {len(candidates)} neural candidates[/green]")
        return candidates
    
    def find_integrated_candidates(self, features: np.ndarray, 
                                 image_paths: List[Path]) -> List[Dict]:
        """Find candidates using integrated scoring (neural + metadata bonus)"""
        self.pm.log("\n[bold cyan]Finding candidates with integrated scoring[/bold cyan]")
        
        n = len(image_paths)
        candidates = []
        
        if not self.all_metadata:
            self.extract_all_metadata(image_paths)
        
        has_any_metadata = any(key is not None for key in self.metadata_keys.values())
        
        if not has_any_metadata:
            self.pm.log("[yellow]No valid metadata found - using neural features only[/yellow]")
            self.pm.log(f"[dim]SSCD threshold: {self.config.sscd_threshold}[/dim]")
            
            return self.find_neural_candidates(features, image_paths)
        
        self.pm.log(f"[dim]Weights: SSCD={self.config.sscd_weight}, Metadata={self.config.metadata_weight}[/dim]")
        
        task_id = self.pm.add_task("Computing integrated scores", total=n)
        
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
                    if self.metadata_keys[global_i] and self.metadata_keys[j]:
                        if self.metadata_keys[global_i] == self.metadata_keys[j]:
                            metadata_match = 1.0
                    
                    # Calculate integrated score
                    integrated_score = (self.config.sscd_weight * sscd_sim + 
                                      self.config.metadata_weight * metadata_match)
                    
                    # Check if candidate meets any threshold
                    if (
                        (metadata_match and sscd_sim >= self.config.sscd_threshold_with_metadata)
                        or (not metadata_match and sscd_sim >= self.config.sscd_threshold)
                        or (integrated_score >= self.config.integrated_threshold)
                    ):
                        candidates.append({
                            'idx1': global_i,
                            'idx2': j,
                            'sscd_similarity': sscd_sim,
                            'metadata_match': metadata_match,
                            'integrated_score': integrated_score,
                            'has_metadata_bonus': metadata_match > 0,
                        })
            
            self.pm.update("Computing integrated scores", advance=end_i - i)
        
        self.pm.log(f"[green]✓ Found {len(candidates)} integrated candidates[/green]")
        
        with_metadata_bonus = sum(1 for c in candidates if c['has_metadata_bonus'])
        self.pm.log(f"[dim]  - {with_metadata_bonus} pairs had metadata bonus[/dim]")
        
        return candidates
    
    def geometric_verification(self, candidate: Dict, image_paths: List[Path]) -> bool:
        """Verify with ORB+RANSAC using ratio-based thresholds"""
        # Handle both index-based and direct path access
        if 'idx1' in candidate and 'idx2' in candidate:
            # Normal case: indices into image_paths
            idx1, idx2 = candidate['idx1'], candidate['idx2']
            if idx1 >= len(image_paths) or idx2 >= len(image_paths):
                # For cross-matches, paths are provided directly
                path1 = image_paths[0] if len(image_paths) > 0 else None
                path2 = image_paths[1] if len(image_paths) > 1 else None
            else:
                path1 = image_paths[idx1]
                path2 = image_paths[idx2]
        else:
            # Direct path case
            path1 = image_paths[0] if len(image_paths) > 0 else None
            path2 = image_paths[1] if len(image_paths) > 1 else None
        
        if path1 is None or path2 is None:
            return False
        
        try:
            # Use context managers to ensure images are closed even if errors occur
            with Image.open(path1) as pil_img1, Image.open(path2) as pil_img2:
                # Convert to grayscale
                pil_img1 = pil_img1.convert('L')
                pil_img2 = pil_img2.convert('L')
                
                # Convert to numpy arrays
                img1 = np.array(pil_img1)
                img2 = np.array(pil_img2)
        except Exception as e:
            return False
        
        max_dim = 1500
        h1, w1 = img1.shape
        if max(h1, w1) > max_dim:
            scale = max_dim / max(h1, w1)
            img1 = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
        
        h2, w2 = img2.shape
        if max(h2, w2) > max_dim:
            scale = max_dim / max(h2, w2)
            img2 = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))
        
        orb = cv2.ORB_create(
            nfeatures=self.config.orb_nfeatures,
            scaleFactor=self.config.orb_scale_factor,
            nlevels=self.config.orb_nlevels
        )
        
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return False
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.config.lowe_ratio * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.config.min_good_matches:
            return False
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if homography is None:
            return False
        
        inliers = np.sum(mask)
        
        has_metadata_bonus = candidate.get('has_metadata_bonus', False)
        required_ratio = self.config.min_inlier_ratio_metadata if has_metadata_bonus else self.config.min_inlier_ratio
        
        min_features = min(len(kp1), len(kp2))
        inlier_ratio = inliers / len(good_matches)
        feature_coverage = inliers / min_features
        
        if inliers < self.config.min_absolute_inliers:
            return False
            
        if min_features > 1000:
            if feature_coverage < (required_ratio * 0.5):
                return False
        else:
            if inlier_ratio < required_ratio:
                return False
        
        det = np.linalg.det(homography[:2, :2])
        if not (self.config.min_homography_det <= det <= self.config.max_homography_det):
            return False
        
        candidate['geometric_inliers'] = int(inliers)
        candidate['inlier_ratio'] = float(inlier_ratio)
        candidate['feature_coverage'] = float(feature_coverage)
        candidate['good_matches'] = len(good_matches)
        candidate['min_features'] = int(min_features)
        
        return True
    
    def find_ml_duplicates(self, features: np.ndarray,
                          image_paths: List[Path],
                          exclude_indices: Set[int] = None,
                          mode: str = "ml") -> List[List[int]]:
        """Complete ML duplicate detection pipeline"""
        if exclude_indices is None:
            exclude_indices = set()
        
        # Debug logging for incremental detection
        self.pm.log(f"[dim]Debug: features.shape={features.shape}, len(image_paths)={len(image_paths)}, exclude_indices={exclude_indices}[/dim]")
        
        # Ensure indices are within bounds of both features and image_paths
        max_valid_index = min(len(image_paths), len(features)) - 1
        valid_indices = [i for i in range(len(image_paths)) if i not in exclude_indices and i <= max_valid_index]
        
        if not valid_indices:
            return []
        
        idx_map = {new_idx: old_idx for new_idx, old_idx in enumerate(valid_indices)}
        reverse_map = {old_idx: new_idx for new_idx, old_idx in idx_map.items()}
        
        filtered_features = features[valid_indices]
        filtered_paths = [image_paths[i] for i in valid_indices]
        
        if mode == "integrated":
            neural_candidates = self.find_integrated_candidates(filtered_features, filtered_paths)
            # Map filtered indices back to original indices
            for candidate in neural_candidates:
                candidate['orig_idx1'] = idx_map[candidate['idx1']]
                candidate['orig_idx2'] = idx_map[candidate['idx2']]
        else:
            neural_candidates = self.find_neural_candidates(filtered_features, filtered_paths)
            for candidate in neural_candidates:
                candidate['orig_idx1'] = idx_map[candidate['idx1']]
                candidate['orig_idx2'] = idx_map[candidate['idx2']]
        
        self.pm.log("\n[bold cyan]Geometric verification[/bold cyan]")
        self.pm.log(f"[dim]Using ratio-based thresholds: {self.config.min_inlier_ratio:.0%} (normal), "
                   f"{self.config.min_inlier_ratio_metadata:.0%} (metadata match)[/dim]")
        self.pm.log(f"[dim]Minimum absolute inliers: {self.config.min_absolute_inliers}[/dim]")
        
        final_verified = []
        
        task_id = self.pm.add_task("Geometric verification", total=len(neural_candidates))
        
        # Extract config parameters for worker processes
        config_params = {
            'orb_nfeatures': self.config.orb_nfeatures,
            'orb_scale_factor': self.config.orb_scale_factor,
            'orb_nlevels': self.config.orb_nlevels,
            'lowe_ratio': self.config.lowe_ratio,
            'min_good_matches': self.config.min_good_matches,
            'min_inlier_ratio': self.config.min_inlier_ratio,
            'min_inlier_ratio_metadata': self.config.min_inlier_ratio_metadata,
            'min_absolute_inliers': self.config.min_absolute_inliers,
            'min_homography_det': self.config.min_homography_det,
            'max_homography_det': self.config.max_homography_det
        }
        
        # Calculate worker count (75% of CPU cores)
        import multiprocessing as mp
        worker_count = max(1, int(mp.cpu_count() * 0.75))
        
        # Parallel geometric verification
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {}
            
            for candidate in neural_candidates:
                if _cleanup_in_progress:
                    break
                
                temp_candidate = {
                    'idx1': candidate['orig_idx1'],
                    'idx2': candidate['orig_idx2'],
                    'sscd_similarity': candidate['sscd_similarity'],
                    'has_metadata_bonus': candidate.get('has_metadata_bonus', False)
                }
                
                candidate_data = (temp_candidate, image_paths, config_params)
                future = executor.submit(geometric_verification_worker, candidate_data)
                futures[future] = candidate
            
            for future in as_completed(futures):
                if _cleanup_in_progress:
                    break
                
                candidate = futures[future]
                success, updated_candidate = future.result()
                if success:
                    candidate['geometric_inliers'] = updated_candidate['geometric_inliers']
                    candidate['inlier_ratio'] = updated_candidate.get('inlier_ratio', 0)
                    candidate['feature_coverage'] = updated_candidate.get('feature_coverage', 0)
                    candidate['good_matches'] = updated_candidate.get('good_matches', 0)
                    candidate['min_features'] = updated_candidate.get('min_features', 0)
                    final_verified.append(candidate)
                self.pm.update("Geometric verification")
        
        self.pm.log(f"[green]✓ Geometrically verified {len(final_verified)}/{len(neural_candidates)} pairs[/green]")
        
        if final_verified:
            avg_inliers = np.mean([c['geometric_inliers'] for c in final_verified])
            avg_ratio = np.mean([c['inlier_ratio'] for c in final_verified])
            avg_coverage = np.mean([c['feature_coverage'] for c in final_verified])
            self.pm.log(f"[dim]Average inliers: {avg_inliers:.0f}, "
                       f"inlier ratio: {avg_ratio:.1%}, "
                       f"feature coverage: {avg_coverage:.1%}[/dim]")
        
        self.pm.log("\n[bold cyan]Forming duplicate groups...[/bold cyan]")
        n = len(image_paths)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for candidate in final_verified:
            union(candidate['orig_idx1'], candidate['orig_idx2'])
        
        groups_dict = defaultdict(list)
        for i in valid_indices:
            root = find(i)
            groups_dict[root].append(i)
        
        groups = [members for members in groups_dict.values() if len(members) > 1]
        
        self.detailed_results = final_verified
        
        self.pm.log(f"[green]✓ Found {len(groups)} {mode.upper()} duplicate groups[/green]")
        
        return groups

# ==============================================================================
# Database Operations
# ==============================================================================

from contextlib import contextmanager

from queue import Queue
import threading

class ConnectionPool:
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool = Queue(maxsize=pool_size)
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
    def get_connection(self):
        conn = self.pool.get()
        try:
            yield conn
        finally:
            self.pool.put(conn)

@contextmanager
def get_db_connection(db_path: Path):
    """Context manager for database connections"""
    if app_state.connection_pool is None:
        app_state.connection_pool = ConnectionPool(db_path)
    
    with app_state.connection_pool.get_connection() as conn:
        yield conn

def save_results_to_db(config: Config, image_paths: List[Path], finder: DuplicateFinder,
                      metadata_groups: List[List[int]], ml_groups: List[List[int]],
                      exclude_indices: Set[int], mode: str, features: np.ndarray = None):
    """Save detection results to database - simpler approach"""
    with get_db_connection(config.db_path) as conn:
        cursor = conn.cursor()
        
        # Clear existing data for this folder - use exact path matching to avoid deleting unrelated data
        folder_path = str(config.image_folder.absolute())
        # Ensure path ends with separator for safe matching
        if not folder_path.endswith(os.sep):
            folder_path_with_sep = folder_path + os.sep
        else:
            folder_path_with_sep = folder_path
        
        # Delete only files in this exact folder and its subdirectories
        # This prevents deleting from folders like '/photos2' when scanning '/photos'
        cursor.execute("""
            DELETE FROM images 
            WHERE path = ? OR path LIKE ?
        """, (folder_path, f"{folder_path_with_sep}%"))
        
        # Prepare batch data
        batch_data = []
        
        # Process metadata groups
        for group_id, group_members in enumerate(metadata_groups, 1):
            for member in group_members:
                metadata = finder.all_metadata.get(member, {})
                width = metadata.get('width', 0)
                height = metadata.get('height', 0)
                resolution = f"{width}x{height}" if width and height else "unknown"
                
                batch_data.append((
                    str(image_paths[member].absolute()),
                    image_paths[member].name,
                    f"META_{group_id}",
                    member == group_members[0],
                    'metadata',
                    1.0,
                    0,
                    False,
                    resolution,
                    'active',
                    metadata.get('datetime_original'),
                    metadata.get('camera_make'),
                    metadata.get('camera_model'),
                    width,
                    height
                ))
        
        # Batch insert
        cursor.executemany('''
            INSERT INTO images (path, name, group_id, is_representative, 
                              detection_method, sscd_score, geometric_inliers, 
                              metadata_bonus, resolution, status,
                              datetime_original, camera_make, camera_model,
                              width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', batch_data)
        
        # Process ML groups
        detection_mode = "integrated" if mode == "integrated" else "ml"
        ml_batch_data = []
        
        for group_id, group_members in enumerate(ml_groups, 1):
            best_rep = group_members[0]
            member_scores = {}
            
            for member in group_members:
                scores = []
                for result in finder.detailed_results:
                    if result['orig_idx1'] == member or result['orig_idx2'] == member:
                        scores.append(result.get('integrated_score', result['sscd_similarity']))
                if scores:
                    member_scores[member] = np.mean(scores)
            
            if member_scores:
                best_rep = max(member_scores, key=member_scores.get)
            
            for member in group_members:
                member_details = None
                for result in finder.detailed_results:
                    if result['orig_idx1'] == member or result['orig_idx2'] == member:
                        if member_details is None or result.get('integrated_score', result['sscd_similarity']) > member_details.get('integrated_score', member_details['sscd_similarity']):
                            member_details = result
                
                if mode == "integrated" and member_details:
                    if member_details.get('has_metadata_bonus', False):
                        detection_method = 'ml+metadata'
                    else:
                        detection_method = 'ml'
                else:
                    detection_method = 'ml'
                
                metadata = finder.all_metadata.get(member, {})
                width = metadata.get('width', 0)
                height = metadata.get('height', 0)
                resolution = f"{width}x{height}" if width and height else "unknown"
                
                ml_batch_data.append((
                    str(image_paths[member].absolute()),
                    image_paths[member].name,
                    f"{detection_mode.upper()}_{group_id}",
                    member == best_rep,
                    detection_method,
                    member_details['sscd_similarity'] if member_details else 0,
                    member_details.get('geometric_inliers', 0) if member_details else 0,
                    member_details.get('has_metadata_bonus', False) if member_details else False,
                    resolution,
                    'active',
                    metadata.get('datetime_original'),
                    metadata.get('camera_make'),
                    metadata.get('camera_model'),
                    width,
                    height
                ))
        
        # Batch insert ML groups
        if ml_batch_data:
            cursor.executemany('''
                INSERT INTO images (path, name, group_id, is_representative, 
                                  detection_method, sscd_score, geometric_inliers, 
                                  metadata_bonus, resolution, status,
                                  datetime_original, camera_make, camera_model,
                                  width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', ml_batch_data)
        
        conn.commit()  # Final commit

def load_active_groups(check_files=False):
    """Load active groups from database"""
    with get_db_connection(app_state.db_path) as conn:
        cursor = conn.cursor()
        
        # Only check file existence if requested (expensive operation)
        if check_files:
            cursor.execute("SELECT id, path FROM images WHERE status = 'active'")
            rows = cursor.fetchall()
            
            # Batch check file existence
            missing_ids = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                def check_file(row):
                    return (row['id'], Path(row['path']).exists())
                
                results = executor.map(check_file, rows)
                missing_ids = [id_ for id_, exists in results if not exists]
            
            # Batch update missing files
            if missing_ids:
                cursor.executemany(
                    "UPDATE images SET status = 'missing' WHERE id = ?",
                    [(id_,) for id_ in missing_ids]
                )
                conn.commit()
        
        # Get groups with 2+ active images
        cursor.execute('''
            SELECT group_id, COUNT(*) as active_count
            FROM images
            WHERE status = 'active'
            GROUP BY group_id
            HAVING active_count >= 2
            ORDER BY group_id
        ''')
        
        app_state.initial_active_groups = [row['group_id'] for row in cursor.fetchall()]
        app_state.active_groups_set = set(app_state.initial_active_groups)
        
        if app_state.initial_active_groups:
            if app_state.current_group_id is None or app_state.current_group_id not in app_state.initial_active_groups:
                app_state.current_group_id = app_state.initial_active_groups[0]
    
    return app_state.initial_active_groups

# ==============================================================================
# Flask Routes
# ==============================================================================

def create_thumbnail(image_path, max_size=(800, 800)):
    """Create base64 encoded thumbnail with caching"""
    # Check cache first
    path_str = str(image_path)
    cached = app_state.thumbnail_cache.get(path_str)
    if cached:
        return cached
    
    try:
        img = Image.open(image_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        if img.mode in ('RGBA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = rgb_img
        
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        thumbnail = base64.b64encode(buffer.getvalue()).decode()
        
        # Cache the result
        app_state.thumbnail_cache.put(path_str, thumbnail)
        return thumbnail
    except (IOError, OSError, Image.DecompressionBombError) as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_current_position():
    """Get the current position in the navigation list"""
    if app_state.current_group_id in app_state.initial_active_groups:
        return app_state.initial_active_groups.index(app_state.current_group_id)
    return 0

def is_group_still_active(group_id):
    """Check if a group still has enough active images"""
    return group_id in app_state.active_groups_set

def clear_caches():
    """Clear all caches"""
    app_state.thumbnail_cache.clear()
    app_state.group_data_cache.clear()

from functools import wraps
import time

def cache_route(timeout=5):
    def decorator(f):
        cache = {}
        
        @wraps(f)
        def decorated_function(*args, **kwargs):
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
            if len(cache) > 100:
                oldest = min(cache.items(), key=lambda x: x[1][1])
                del cache[oldest[0]]
            
            return result
        
        return decorated_function
    return decorator

@app.route('/')
def index():
    """Show current group"""
    if not app_state.db_path or not Path(app_state.db_path).exists():
        return "No database found"
    
    # Only load active groups if not already loaded
    if not app_state.initial_active_groups:
        load_active_groups(check_files=True)  # Check files on initial load
    
    # Check if all groups are done
    if not app_state.active_groups_set:
        with get_db_connection(app_state.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM images WHERE status = 'deleted'")
            total_deleted = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM images WHERE status = 'active'")
            total_remaining = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT group_id) FROM images")
            total_groups = cursor.fetchone()[0]
        
        return render_template('duplicate-detector-ui.html',
            all_done=True,
            total_groups=total_groups,
            total_deleted=total_deleted,
            total_remaining=total_remaining
        )
    
    # Ensure current group is valid
    if app_state.current_group_id not in app_state.active_groups_set:
        # Find first active group
        for group_id in app_state.initial_active_groups:
            if group_id in app_state.active_groups_set:
                app_state.current_group_id = group_id
                break

    # Get current group data
    # Get current group data and total groups in one connection
    current_group_data = app_state.group_data_cache.get(app_state.current_group_id)
    total_groups = app_state.total_groups_cache
    
    if current_group_data is None or total_groups is None:
        with get_db_connection(app_state.db_path) as conn:
            cursor = conn.cursor()
            if current_group_data is None:
                cursor.execute(
                    "SELECT * FROM images WHERE group_id = ? ORDER BY is_representative DESC, id LIMIT 100",
                    (app_state.current_group_id,),
                )
                current_group_data = cursor.fetchall()
                app_state.group_data_cache.put(app_state.current_group_id, current_group_data)
            
            if total_groups is None:
                cursor.execute("SELECT COUNT(DISTINCT group_id) FROM images")
                total_groups = cursor.fetchone()[0]
                app_state.total_groups_cache = total_groups
    
    # Prepare image data
    images = []
    active_count = 0
    detection_method = 'unknown'
    
    for row in current_group_data:
        image_path = Path(row['path'])
        
        image_info = {
            'name': row['name'],
            'path': row['path'],
            'data': None,
            'exists': image_path.exists() and row['status'] != 'missing',
            'is_representative': row['is_representative'],
            'sscd_score': row['sscd_score'],
            'geometric_inliers': row['geometric_inliers'],
            'status': row['status'],
            'resolution': row['resolution'],
            'datetime_original': row['datetime_original'],
            'file_size': 0  # Will calculate if file exists
        }
        
        if image_info['exists']:
            try:
                image_info['file_size'] = image_path.stat().st_size
            except:
                pass
            
            if image_info['status'] == 'active':
                thumb = create_thumbnail(image_path)
                if thumb:
                    image_info['data'] = thumb
                    active_count += 1
        
        if detection_method == 'unknown':
            detection_method = row['detection_method']
        
        images.append(image_info)
    
    # Calculate progress
    current_pos = get_current_position()
    completed_groups = len(app_state.initial_active_groups) - len(app_state.active_groups_set)
    
    progress_percent = int((completed_groups / total_groups) * 100) if total_groups else 0
    
    # Check navigation boundaries
    is_first = current_pos == 0
    has_next_active = False
    for i in range(current_pos + 1, len(app_state.initial_active_groups)):
        if app_state.initial_active_groups[i] in app_state.active_groups_set:
            has_next_active = True
            break
    
    return render_template('duplicate-detector-ui.html',
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
        is_first_group=is_first,
        is_last_group=current_pos >= len(app_state.initial_active_groups) - 1 and not has_next_active
    )

@app.route('/delete', methods=['POST'])
def delete_images():
    """Delete selected images and update database"""
    data = request.get_json()
    indices = data.get('indices', [])
    
    if app_state.current_group_id is None:
        return jsonify({'success': False, 'error': 'No active group'})
    
    with get_db_connection(app_state.db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM images WHERE group_id = ? ORDER BY is_representative DESC, id LIMIT 100", (app_state.current_group_id,))
        current_group_data = list(cursor.fetchall())
        
        deleted_count = 0
        deletion_results = []
        
        # Collect all updates first
        ids_to_delete = []
        files_to_trash = []
        
        for idx in indices:
            if idx < len(current_group_data):
                row = current_group_data[idx]
                ids_to_delete.append(row['id'])
                files_to_trash.append((Path(row['path']), row))
        
        # Batch update database status first (fast operation)
        if ids_to_delete:
            placeholders = ','.join('?' * len(ids_to_delete))
            cursor.execute(f"UPDATE images SET status = 'deleted' WHERE id IN ({placeholders})", ids_to_delete)
            conn.commit()  # Commit database changes immediately
    
    # Process file deletions outside of database transaction
    successfully_deleted = []
    for image_path, row in files_to_trash:
        if image_path.exists():
            try:
                # Move to trash
                trash_dir = Path.home() / '.Trash'
                if trash_dir.exists():
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    trash_name = f"{timestamp}_{image_path.name}"
                    shutil.move(str(image_path), str(trash_dir / trash_name))
                else:
                    image_path.unlink()
                
                deleted_count += 1
                deletion_results.append(f"✓ Deleted: {row['name']}")
                successfully_deleted.append(row)
                
                # Also log to text file for compatibility
                with open('deleted_files_log.txt', 'a') as f:
                    f.write(f"{datetime.now()}: Deleted {image_path} (Group: {app_state.current_group_id})\n")
                
            except Exception as e:
                deletion_results.append(f"✗ Failed: {row['name']} - {str(e)}")
        else:
            deletion_results.append(f"✗ Not found: {row['name']}")
    
    # Log successful deletions in separate transaction
    if successfully_deleted:
        with get_db_connection(app_state.db_path) as conn:
            cursor = conn.cursor()
            deletion_log_data = [(row['path'], app_state.current_group_id) for row in successfully_deleted]
            cursor.executemany('''
                INSERT INTO deletion_log (image_path, group_id)
                VALUES (?, ?)
            ''', deletion_log_data)
            conn.commit()

        # Thread-safe state modifications
        with app_state._lock:
            # Invalidate caches for this group and the total‑groups count
            app_state.group_data_cache.remove(app_state.current_group_id)
            app_state.total_groups_cache = None
            
            # Check if group is still active
            cursor.execute('''
                SELECT COUNT(*) as active_count
                FROM images
                WHERE group_id = ? AND status = 'active'
            ''', (app_state.current_group_id,))
            
            active_count = cursor.fetchone()['active_count']
            group_complete = active_count < 2
            
            if group_complete:
                # Remove from active groups set
                app_state.active_groups_set.discard(app_state.current_group_id)
                # Clear cache for this group (avoid double removal)
                app_state.group_data_cache.remove(app_state.current_group_id)
    
    return jsonify({
        'success': True,
        'deleted_count': deleted_count,
        'requested_count': len(indices),
        'group_complete': group_complete,
        'deletion_results': deletion_results
    })

@app.route('/next')
def next_group():
    """Go to next active group"""
    with app_state._lock:
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
            for i in range(0, current_pos):
                group_id = app_state.initial_active_groups[i]
                if group_id in app_state.active_groups_set:
                    app_state.current_group_id = group_id
                    break
    
    return index()


@app.route('/previous')
def previous_group():
    """Go to previous active group"""
    with app_state._lock:
        current_pos = get_current_position()
        found_prev = False
        
        # Look for previous active group
        for i in range(current_pos - 1, -1, -1):
            group_id = app_state.initial_active_groups[i]
            if group_id in app_state.active_groups_set:
                app_state.current_group_id = group_id
                found_prev = True
                break
    
    return index()

# ==============================================================================
# Main Pipeline
# ==============================================================================















def run_detection(config: Config, pm: ProgressManager):
    """
    Run the duplicate detection pipeline
    
    Processes all images in the specified folder to detect duplicates
    using the configured detection method (metadata, ML, or integrated).
    
    Args:
        config: Detection configuration
        pm: Progress manager for logging
    """
    start_time = time.time()
    
    mode_descriptions = {
        "metadata": "Metadata only (EXIF DateTime + Camera)",
        "ml": "ML only (Neural + Geometric)",
        "both": "Sequential (Metadata → ML on remainder)",
        "integrated": "Integrated (ML with metadata bonus)"
    }
    
    pm.log(f"\nMode: {config.mode} - {mode_descriptions[config.mode]}")
    
    # Note: duplicate-detector-ui.html template should be in same directory as this script
    
    # Simpler approach: Always scan all images, but optimize feature extraction
    pm.log("\n[bold]Scanning for images...[/bold]")
    image_paths = []
    for ext in config.image_extensions:
        image_paths.extend(list(config.image_folder.glob(f"*{ext}")))
    # Ensure absolute paths and unique
    image_paths = sorted(list(set(path.absolute() for path in image_paths)))
    
    # Check if we have existing results
    has_existing_data = False
    if config.db_path.exists():
        with get_db_connection(config.db_path) as conn:
            cursor = conn.cursor()
            folder_path = str(config.image_folder.absolute())
            cursor.execute("SELECT COUNT(*) FROM images WHERE path LIKE ?", (f"{folder_path}%",))
            count = cursor.fetchone()[0]
            has_existing_data = count > 0
    
    if has_existing_data and len(image_paths) == 0:
        pm.log("\n[yellow]No images found in folder.[/yellow]")
        return False
        
    pm.log(f"[green]✓ Found {len(image_paths)} images[/green]")
    
    # Count by type
    type_counts = defaultdict(int)
    for path in image_paths:
        type_counts[path.suffix.lower()] += 1
    
    pm.log("Image types:")
    for ext, count in sorted(type_counts.items()):
        pm.log(f"  {ext}: {count}")
    
    if not image_paths:
        pm.log("[red]No images found![/red]")
        return False
    
    # Initialize database
    init_database(config.db_path)
    
    # Initialize finder
    finder = DuplicateFinder(config, pm)
    finder.extract_all_metadata(image_paths)
    
    metadata_groups = []
    ml_groups = []
    exclude_indices = set()
    model_manager = None
    features = None
    
    # Metadata detection
    if config.mode in ["metadata", "both"]:
        pm.log("\n[bold cyan]Phase 1: Metadata Detection[/bold cyan]")
        metadata_groups, exclude_indices = finder.find_metadata_duplicates(image_paths)
    
    # ML detection
    if config.mode in ["ml", "both", "integrated"]:
        if config.mode == "integrated":
            pm.log("\n[bold cyan]Integrated Detection (ML + Metadata Bonus)[/bold cyan]")
        else:
            pm.log("\n[bold cyan]Phase 2: ML Detection (Neural + Geometric)[/bold cyan]")
        
        # Extract features with smart caching
        cache_key = config.get_cache_key()
        feature_cache_file = config.cache_dir / f"features_enhanced_{cache_key}.npz"
        
        # Always extract features for all images (simpler approach)
        if feature_cache_file.exists():
            # Cache exists - use cached features
            pm.log("Loading cached features...")
            cache = np.load(feature_cache_file, mmap_mode='r')
            features = np.array(cache['sscd'])
            pm.log("[green]✓ Loaded features from cache[/green]")
        else:
            # Extract features for all images
            model_manager = FastModelManager(config, pm)
            model_manager.load_models()
            features = model_manager.extract_features_fast(image_paths)
            
            # Save updated cache
            pm.log("Saving feature cache...")
            np.savez_compressed(feature_cache_file, sscd=features)
            
            if model_manager:
                model_manager.cleanup()
        
        # Find ML duplicates
        if config.mode == "both":
            pm.log(f"[yellow]Excluding {len(exclude_indices)} images already found by metadata[/yellow]")
        
        detection_mode = "integrated" if config.mode == "integrated" else "ml"
        
        ml_groups = finder.find_ml_duplicates(
            features,
            image_paths,
            exclude_indices if config.mode == "both" else set(),
            mode=detection_mode
        )
        
        if model_manager:
            model_manager.cleanup()
    
    # Save results to database
    pm.log("\n[bold]Saving results to database...[/bold]")
    save_results_to_db(config, image_paths, finder, metadata_groups, ml_groups, 
                      exclude_indices, config.mode, features)
    pm.log(f"[green]✓ Results saved to database[/green]")
    
    # Summary
    elapsed = time.time() - start_time
    
    summary_table = Table(title="Detection Summary", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total images", str(len(image_paths)))
    summary_table.add_row("Detection mode", f"{config.mode} - {mode_descriptions[config.mode]}")
    
    if config.mode in ["metadata", "both"]:
        summary_table.add_row("Metadata duplicate groups", str(len(metadata_groups)))
        summary_table.add_row("Metadata duplicate images", str(len(exclude_indices)))
    
    if config.mode in ["ml", "both", "integrated"]:
        summary_table.add_row(f"{detection_mode.upper()} duplicate groups", str(len(ml_groups)))
        summary_table.add_row(f"{detection_mode.upper()} duplicate images", str(sum(len(g) for g in ml_groups)))
        
        if config.mode == "integrated":
            # Count groups with metadata bonus
            groups_with_metadata = set()
            for group in ml_groups:
                for member in group:
                    for result in finder.detailed_results:
                        if (result['orig_idx1'] == member or result['orig_idx2'] == member) and result.get('has_metadata_bonus', False):
                            groups_with_metadata.add(tuple(group))
                            break
            summary_table.add_row("Groups with metadata bonus", str(len(groups_with_metadata)))
    
    total_dups = len(exclude_indices) + sum(len(g) for g in ml_groups)
    unique_images = len(image_paths) - total_dups
    summary_table.add_row("Total duplicate images", str(total_dups))
    summary_table.add_row("Unique images", str(unique_images))
    summary_table.add_row("Processing time", humanize.naturaldelta(elapsed))
    summary_table.add_row("Speed", f"{len(image_paths) / elapsed:.1f} images/second")
    
    console.print("\n")
    console.print(summary_table)
    
    return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Duplicate Detector - Detection + Review"
    )
    parser.add_argument("image_folder", help="Folder containing images")
    parser.add_argument("--mode", "-m", choices=["metadata", "ml", "both", "integrated"],
                       default="integrated",
                       help="Detection mode (default: integrated)")
    parser.add_argument("--threshold", "-t", type=float, help="SSCD threshold")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--workers", "-w", type=int, default=11, help="Number of workers")
    parser.add_argument("--dataloader-workers", "-dw", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cache")
    parser.add_argument("--port", "-p", type=int, default=5555, help="Web UI port")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]Unified Duplicate Detector[/bold cyan]\n"
        "Detection + Review in one tool\n"
        "Supports: HEIC/JPEG | Methods: Metadata + ML",
        border_style="cyan"
    ))
    
    config = Config()
    config.image_folder = Path(args.image_folder).absolute()
    config.mode = args.mode
    config.batch_size = args.batch_size
    config.num_workers = args.workers
    config.dataloader_workers = args.dataloader_workers
    
    if args.threshold:
        config.sscd_threshold = args.threshold
    
    if args.no_cache and config.cache_dir.exists():
        import shutil
        shutil.rmtree(config.cache_dir)
        console.print("[yellow]Cleared cache directory[/yellow]")
    
    # Run detection
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
    
    # Set database path and initialize groups
    app_state.db_path = config.db_path
    
    # Load initial groups before starting server
    if app_state.db_path.exists():
        load_active_groups(check_files=True)  # Check files on initial load
        console.print(f"[dim]Loaded {len(app_state.initial_active_groups)} active groups from database[/dim]")
    
    # Launch web UI
    console.print(f"\n[bold cyan]Launching review interface...[/bold cyan]")
    console.print(f"Starting server on http://localhost:{args.port}")
    console.print("Press Ctrl+C to stop\n")
    
    # Auto-open browser after a short delay
    if not args.no_browser:
        def open_browser():
            time.sleep(1.5)  # Give server time to start
            webbrowser.open(f'http://localhost:{args.port}')
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
    
    # Use waitress instead of Flask dev server
    try:
        serve(app, host='0.0.0.0', port=args.port, threads=4)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
        sys.exit(0)

if __name__ == '__main__':
    main()