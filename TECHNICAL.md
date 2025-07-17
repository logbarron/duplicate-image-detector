# Technical Documentation

## Architecture Overview

The duplicate image detector is a self-contained Python script that operates in two main stages: detection and review.

### Detection Pipeline

1. **Image Scanning:** Finds all images in the specified directory (non-recursive)
2. **Metadata Extraction:** Extracts EXIF data (timestamp, camera make/model) in parallel
3. **Feature Extraction (ML Mode):** A PyTorch-based SSCD model (`sscd_disc_large`) generates a 1024-dimension feature vector for each image. Features are cached on disk for subsequent runs
4. **Candidate Pairing:** A fast similarity search (dot product on normalized vectors) identifies potential duplicate pairs based on a configurable threshold
5. **Geometric Verification:** OpenCV is used to perform a robust check on candidate pairs. It extracts ORB features and uses a RANSAC-based homography check to confirm the images are geometrically consistent, filtering out false positives
6. **Grouping:** A Disjoint Set Union (DSU) data structure is used to efficiently group images into sets of duplicates based on the verified pairs
7. **Database Storage:** All identified duplicate groups and their members are stored in an SQLite database (`.duplicate_detector.db`)

### Review UI

- **Web Server:** A lightweight Flask/Waitress server is launched to serve the UI
- **Frontend:** A single-page HTML interface with vanilla JavaScript fetches data from the backend API
- **Backend API:** Provides endpoints to navigate between duplicate groups, view thumbnails, and delete selected images

## Detection Algorithms

### 1. Metadata-Based Detection

- Extracts EXIF metadata focusing on `DateTimeOriginal` and camera model information
- Groups images with identical metadata keys (timestamp + camera)
- Very fast but only finds exact duplicates from the same camera
- Used for fast, exact-duplicate detection

### 2. Machine Learning Detection

- **SSCD Model:** Uses the Self-Supervised Copy Detection neural network (`sscd_disc_large`)
- **Feature Vectors:** Generates 1024-dimensional feature vectors (embeddings) for each image
- **GPU Acceleration:** Process is accelerated using PyTorch DataLoader with multiple workers and GPU support (Apple Silicon's MPS or CUDA)
- **Similarity Search:** Efficient all-pairs similarity search using dot products on normalized vectors
- **Threshold-based:** Pairs with similarity score above configured threshold are considered candidates

### 3. Geometric Verification

- **ORB Feature Detector:** Extracts keypoints and descriptors from candidate image pairs
- **RANSAC Algorithm:** Finds homography matrix between images to confirm geometric consistency
- **Inlier Matching:** A match is confirmed if sufficient number of inliers are found
- **False Positive Reduction:** Eliminates images with similar objects but different compositions
- **Parallelization:** Uses ProcessPoolExecutor for maximum throughput

### 4. Integrated Mode (Default)

- Combines ML detection with metadata bonus scoring
- Images with matching metadata receive scoring bonus
- Allows system to identify near-duplicates more reliably
- Recommended for most use cases

## Command-Line Options Reference

### Basic Arguments

- `image_folder`: Directory to scan for images (required)
- `--mode`: Detection mode (`integrated`, `ml`, `metadata`, `both`)
- `--no-cache`: Forces rebuild ignoring cached results

### Advanced Configuration

- `--sscd-threshold`: Similarity threshold for SSCD model (default: 0.86)
- `--orb-features`: Maximum number of features for ORB geometric verification (default: 6000)
- `--inlier-ratio`: Minimum inlier ratio for geometric match (default: 0.3)
- `--workers`: Number of parallel workers for processing tasks (default: 11)
- `--dataloader-workers`: Number of workers for PyTorch DataLoader (default: 8)
- `--device`: PyTorch device (`cpu`, `cuda`, `mps`)

## Database Schema

The SQLite database (`.duplicate_detector.db`) stores:

- **Images table:** Image paths, metadata, group IDs, processing status
- **Groups table:** Duplicate group information, scores, verification status
- **Relationships:** Verified pairs and their similarity scores

## Performance Characteristics

### Speed Optimizations

- **Feature Caching:** Neural features cached in `.duplicate_cache/` directory
- **Parallel Processing:** Multi-core processing for metadata extraction and geometric verification
- **GPU Acceleration:** MPS (Apple Silicon) or CUDA support for neural network inference
- **Batched Operations:** Efficient batch processing for large image collections

### Memory Management

- **Streaming Processing:** Images processed in batches to manage memory usage
- **Cache Management:** Intelligent caching of computed features
- **Resource Cleanup:** Proper cleanup of neural network resources

## File Structure

```
project/
├── duplicate-detector.py      # Main application
├── duplicate-detector-ui.html # Web UI template
├── .duplicate_detector.db     # SQLite database (generated)
├── .duplicate_cache/          # Feature cache directory (generated)
├── deleted_files_log.txt      # Deletion log (generated)
└── README.md                  # This file
```

## Web API Endpoints

- `GET /`: Serves the main UI
- `POST /delete`: Handles image deletion
- `GET /next`: Navigate to next duplicate group
- `GET /previous`: Navigate to previous duplicate group
- `GET /thumbnails/<path>`: Serves image thumbnails

## Troubleshooting

### Common Issues

1. **Memory Issues:** Reduce batch size or number of workers
2. **GPU Issues:** Fall back to CPU mode with `--device cpu`
3. **Permission Issues:** Ensure read/write access to image directory
4. **Port Conflicts:** Web UI uses port 5555 by default

### Performance Tuning

- **Large Collections:** Increase `--workers` for more parallelism
- **GPU Systems:** Use `--device cuda` or `--device mps`
- **Storage:** Place cache on fast storage (SSD)
- **Memory-Limited:** Reduce `--dataloader-workers` and batch sizes

### Debug Information

The application provides detailed logging including:
- Processing progress with Rich terminal output
- Error handling for corrupted images
- Performance timing information
- Cache hit/miss statistics

## Dependencies

The script uses PEP 723 inline dependencies and automatically installs:

- `torch` & `torchvision`: Neural network inference
- `opencv-python`: Computer vision algorithms
- `pillow` & `pillow-heif`: Image processing
- `flask` & `waitress`: Web server
- `rich`: Terminal output formatting
- `numpy`, `scipy`, `scikit-learn`: Numerical computing
- `pandas`: Data manipulation
- `tqdm`: Progress bars
- `psutil`: System resource monitoring
- `humanize`: Human-readable formatting
- `exifread`: EXIF metadata extraction