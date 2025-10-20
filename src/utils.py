"""
Utility Functions Module for Person Re-Identification and Scene Classification

This module provides common utility functions used throughout the pipeline:
- Reproducibility: Random seed setting for consistent results
- File I/O: JSON reading/writing with error handling
- Video Processing: Frame extraction and timestamp conversion
- Metrics: Similarity and distance computations for Re-ID
- Geometry: Bounding box operations (IoU calculation)

These utilities are designed to be reusable, robust, and well-tested.
All functions include type hints and comprehensive error handling.
"""

import os
import random
import numpy as np
import torch
import json
import cv2
from typing import List, Tuple, Dict, Optional


# ========================================
# Reproducibility Utilities
# ========================================

# noinspection SpellCheckingInspection
def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    This function ensures consistent results across multiple runs by seeding
    all random number generators used in the pipeline. Critical for:
    - Debugging (reproduce exact behavior)
    - Scientific validation (consistent experiments)
    - Clustering algorithms (deterministic results)

    Args:
        seed (int): Random seed value. Common choices:
                   - 42: The answer to everything (default)
                   - 0: Simple baseline
                   - Any integer for different random sequences

    Side Effects:
        Sets seeds for:
        - Python's random module
        - NumPy's random number generator
        - PyTorch CPU operations
        - PyTorch CUDA (GPU) operations
        - Python's hash seed (for dictionary ordering)

    Notes:
        - Does NOT guarantee 100% reproducibility across different:
          * Hardware (CPU vs GPU differences)
          * PyTorch versions
          * CUDA versions
        - For strict reproducibility, also set:
          torch.backends.cudnn.deterministic = True
          torch.backends.cudnn.benchmark = False
          (Warning: May reduce performance by ~20-30%)

    Example:
        >>> set_seed(42)
        >>> x = torch.randn(3, 3)  # Always generates same tensor
        >>> y = np.random.rand(5)  # Always generates same array
    """
    # Seed Python's built-in random module
    # Affects: random.choice(), random.shuffle(), etc.
    random.seed(seed)

    # Seed NumPy's random number generator
    # Affects: np.random.rand(), np.random.choice(), etc.
    np.random.seed(seed)

    # Seed PyTorch's random number generator (CPU)
    # Affects: torch.randn(), torch.rand(), dropout layers, etc.
    torch.manual_seed(seed)

    # Seed PyTorch's random number generator (all GPUs)
    # Ensures consistent results across multi-GPU setups
    torch.cuda.manual_seed_all(seed)

    # Set Python hash seed for dictionary ordering consistency
    # Important for Python 3.6+ where dicts maintain insertion order
    os.environ['PYTHONHASHSEED'] = str(seed)

    # ========================================
    # Optional: Strict Determinism (Commented Out)
    # ========================================
    # Uncomment these lines for guaranteed reproducibility at cost of performance

    # Force CUDA operations to use deterministic algorithms
    # torch.backends.cudnn.deterministic = True

    # Disable CUDA benchmarking (auto-tuning of conv algorithms)
    # torch.backends.cudnn.benchmark = False

    # Note: These settings can reduce training/inference speed by 20-30%


# ========================================
# File I/O Utilities
# ========================================

def save_json(filepath: str, data: Dict | List) -> None:
    """
    Save data to JSON file with automatic directory creation.

    This function provides robust JSON serialization with:
    - Automatic parent directory creation
    - Pretty printing (4-space indentation)
    - UTF-8 encoding for international characters
    - Error handling for edge cases

    Args:
        filepath (str): Destination file path (absolute or relative)
                       Example: "../outputs/results.json"
        data (Dict | List): Python dictionary or list to serialize
                           Must be JSON-serializable (no custom objects)

    Raises:
        ValueError: If filepath is empty string
        IOError: If file cannot be written (permissions, disk space)
        TypeError: If data contains non-serializable objects

    JSON Format:
        - Indentation: 4 spaces (readable format)
        - ensure_ascii=False: Allows UTF-8 characters (émojis, 中文, etc.)
        - No trailing whitespace

    Examples:
        >>> save_json("results.json", {"accuracy": 0.95})
        >>> save_json("../data/config.json", [1, 2, 3])

    Side Effects:
        - Creates parent directories if they don't exist
        - Overwrites existing file without warning
    """
    # ----------------------------------------
    # Validation: Check for empty filepath
    # ----------------------------------------
    if not filepath:
        raise ValueError("Filepath cannot be empty")

    # ----------------------------------------
    # Create Parent Directory If Needed
    # ----------------------------------------
    # Extract directory path from filepath
    dir_path = os.path.dirname(filepath)

    # Only create directory if filepath contains directory component
    # Example: "file.json" has no dir_path, "../data/file.json" does
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)  # exist_ok prevents error if exists

    # ----------------------------------------
    # Write JSON to File
    # ----------------------------------------
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(
            data,
            f,         # type: ignore
            indent=4,  # Pretty print with 4-space indentation
            ensure_ascii=False  # Allow non-ASCII characters (UTF-8)
        )  # type: ignore

    # File is automatically closed when exiting 'with' block
    # Even if an exception occurs, file handle is properly released


def load_json(filepath: str) -> Dict:
    """
    Load data from JSON file with error handling.

    This function reads and parses JSON files, providing:
    - UTF-8 encoding support
    - Automatic type conversion (str -> int, float, bool, etc.)
    - Clear error messages for debugging

    Args:
        filepath (str): Source file path (must exist)
                       Example: "../outputs/identity_catalogue.json"

    Returns:
        Dict: Parsed JSON data as Python dictionary
              (or list if JSON root is array)

    Raises:
        FileNotFoundError: If file doesn't exist at specified path
        json.JSONDecodeError: If file contains invalid JSON
        PermissionError: If file cannot be read (permissions)

    Type Conversions:
        JSON -> Python:
        - object -> dict
        - array -> list
        - string -> str
        - number (int) -> int
        - number (real) -> float
        - true/false -> True/False
        - null -> None

    Examples:
        >>> data = load_json("config.json")
        >>> print(data["model_name"])

        >>> catalogue = load_json("../outputs/identity_catalogue.json")
        >>> print(f"Found {len(catalogue)} persons")

    Notes:
        - Large files (>100MB) may consume significant memory
        - For streaming large JSON files, consider ijson library
        - Always validate loaded data structure before use
    """
    # Open file with UTF-8 encoding (supports international characters)
    with open(filepath, 'r', encoding='utf-8') as f:
        # Parse JSON and return as Python data structure
        return json.load(f)

    # File is automatically closed when exiting 'with' block
    # json.load() raises JSONDecodeError if file contains invalid JSON


# ========================================
# Video Processing Utilities
# ========================================

# noinspection SpellCheckingInspection
def frame_to_timestamp(frame_idx: int, fps: float) -> str:
    """
    Convert frame index to human-readable timestamp format.

    This function converts frame numbers to MM:SS.SS format for:
    - User-friendly time references in logs
    - Video annotation and labeling
    - Crime event reporting with timestamps
    - Debugging and verification

    Args:
        frame_idx (int): Frame number (0-indexed)
                        Example: 150 (the 151st frame)
        fps (float): Video frames per second
                    Example: 30.0 (30 FPS video)

    Returns:
        str: Formatted timestamp as "MM:SS.SS m:s"
             Format breakdown:
             - MM: Minutes (zero-padded, 2 digits)
             - SS.SS: Seconds with 2 decimal places (5 chars total)
             - "m:s": Literal suffix indicating minutes:seconds

    Examples:
        >>> frame_to_timestamp(0, 30.0)
        '00:00.00 m:s'  # First frame

        >>> frame_to_timestamp(150, 30.0)
        '00:05.00 m:s'  # 5 seconds in (150 frames / 30 fps)

        >>> frame_to_timestamp(1800, 30.0)
        '01:00.00 m:s'  # 1 minute mark

        >>> frame_to_timestamp(1875, 30.0)
        '01:02.50 m:s'  # 1 minute, 2.5 seconds

    Notes:
        - Precision: 2 decimal places (~0.01 second accuracy)
        - Works for any FPS (handles fractional FPS like 29.97)
        - No upper limit on video length
        - Format: Fixed width for alignment in logs

    Use Cases:
        - Logging: "Crime detected at 01:23.45 m:s"
        - Annotations: "Person enters frame at 00:15.30 m:s"
        - Reports: "Weapon visible from 02:10.00 to 02:15.50 m:s"
    """
    # Calculate total seconds from frame index and FPS
    seconds = frame_idx / fps

    # Divmod: Simultaneously get quotient and remainder
    # mins = seconds // 60 (integer division)
    # secs = seconds % 60 (remainder)
    mins, secs = divmod(seconds, 60)

    # Format string breakdown:
    # {int(mins):02d} - Minutes as 2-digit integer (zero-padded)
    # {secs:05.2f} - Seconds as 5-char float with 2 decimals (e.g., "05.00")
    # "m:s" - Literal suffix for clarity
    return f"{int(mins):02d}:{secs:05.2f} m:s"


# ========================================
# Similarity and Distance Metrics
# ========================================
# noinspection PyShadowingNames
def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Compute cosine similarity between two feature vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    providing a similarity score independent of vector magnitude. It's ideal for:
    - Person re-identification (comparing appearance embeddings)
    - Document similarity (comparing text embeddings)
    - Image similarity (comparing CNN features)

    Formula:
        cos(θ) = (A · B) / (||A|| × ||B||)

    Where:
        - A · B: Dot product of vectors
        - ||A||, ||B||: Euclidean norms (magnitudes)
        - θ: Angle between vectors

    Args:
        feat1 (np.ndarray): First feature vector, shape (N,)
                           Example: ResNet50 embedding [2048]
        feat2 (np.ndarray): Second feature vector, shape (N,)
                           Must have same dimensionality as feat1

    Returns:
        float: Cosine similarity score
               Range: [-1.0, 1.0]
               - 1.0: Identical direction (perfect match)
               - 0.0: Orthogonal (completely different)
               - -1.0: Opposite direction (polar opposite)

               For Re-ID applications:
               - > 0.7: Likely same person
               - 0.4-0.7: Uncertain (needs more evidence)
               - < 0.4: Likely different person

    Edge Cases:
        - Zero vectors: Returns 0.0 (undefined mathematically, but safe default)
        - NaN values: Will propagate (consider validating inputs)
        - Different shapes: Will raise ValueError from numpy

    Examples:
        >>> feat1 = np.array([1, 0, 0])
        >>> feat2 = np.array([1, 0, 0])
        >>> cosine_similarity(feat1, feat2)
        1.0  # Identical

        >>> feat1 = np.array([1, 0, 0])
        >>> feat2 = np.array([0, 1, 0])
        >>> cosine_similarity(feat1, feat2)
        0.0  # Orthogonal

        >>> feat1 = np.array([1, 1])
        >>> feat2 = np.array([2, 2])
        >>> cosine_similarity(feat1, feat2)
        1.0  # Same direction, different magnitude

    Comparison with Euclidean Distance:
        - Cosine: Measures angle (direction similarity)
        - Euclidean: Measures straight-line distance (magnitude matters)
        - For normalized vectors: 1 - cos(θ) = euclidean²/2

    Performance:
        - O(N) time complexity for N-dimensional vectors
        - Efficient for high-dimensional embeddings (e.g., 2048-D)
        - Consider batch operations for many comparisons
    """
    # ----------------------------------------
    # Calculate Vector Norms (Magnitudes)
    # ----------------------------------------
    # L2 norm: sqrt(sum of squared elements)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)

    # ----------------------------------------
    # Handle Edge Case: Zero Vectors
    # ----------------------------------------
    # If either vector has zero magnitude, similarity is undefined
    # Return 0.0 as safe default (no similarity)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # ----------------------------------------
    # Compute Cosine Similarity
    # ----------------------------------------
    # Numerator: Dot product (measures alignment)
    # Denominator: Product of magnitudes (normalization)
    # Cast to float for type consistency
    return float(np.dot(feat1, feat2) / (norm1 * norm2))


# ========================================
# Bounding Box Utilities
# ========================================
# noinspection PyShadowingNames
def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes.

    IoU is a standard metric for measuring bounding box overlap, used in:
    - Object detection evaluation (mAP calculation)
    - Non-Maximum Suppression (NMS)
    - Tracking (matching detections across frames)
    - Person re-identification (spatial verification)

    Formula:
        IoU = Area(Intersection) / Area(Union)
        IoU = Area(Intersection) / (Area(Box1) + Area(Box2) - Area(Intersection))

    Args:
        box1 (List[float]): First bounding box in format [x1, y1, x2, y2]
                           - (x1, y1): Top-left corner coordinates
                           - (x2, y2): Bottom-right corner coordinates
                           - Coordinates can be absolute pixels or normalized [0,1]
        box2 (List[float]): Second bounding box in same format as box1

    Returns:
        float: IoU score
               Range: [0.0, 1.0]
               - 1.0: Perfect overlap (identical boxes)
               - 0.7+: Strong overlap (typically same object)
               - 0.5: Moderate overlap (common detection threshold)
               - 0.3: Weak overlap (may be different objects)
               - 0.0: No overlap (completely separate boxes)

    Common IoU Thresholds in Practice:
        - Object Detection (COCO): 0.5 - 0.95 (strict evaluation)
        - NMS: 0.3 - 0.5 (suppress duplicate detections)
        - Tracking: 0.3 - 0.5 (match detections across frames)
        - Re-ID Verification: 0.5+ (confirm same person spatially)

    Examples:
        >>> box1 = [0, 0, 10, 10]  # 10x10 box at origin
        >>> box2 = [0, 0, 10, 10]  # Identical box
        >>> compute_iou(box1, box2)
        1.0  # Perfect overlap

        >>> box1 = [0, 0, 10, 10]
        >>> box2 = [5, 5, 15, 15]  # Overlapping box
        >>> compute_iou(box1, box2)
        0.142857...  # (25 intersection / 175 union)

        >>> box1 = [0, 0, 10, 10]
        >>> box2 = [20, 20, 30, 30]  # No overlap
        >>> compute_iou(box1, box2)
        0.0

    Edge Cases:
        - Zero-area boxes: Returns 0.0 (handled by max() operations)
        - Invalid boxes (x1 > x2): Returns 0.0 (negative dimensions)
        - Identical boxes: Returns 1.0 (perfect match)
        - Partially overlapping: Returns appropriate fraction

    Algorithm Complexity:
        - Time: O(1) - constant time operations
        - Space: O(1) - no additional memory
        - Vectorization: Can batch process with numpy for efficiency

    Visualization:
        Box1:     Box2:         Intersection:    Union:
        ┌───┐     ┌───┐         ┌─┐             ┌───┐
        │   │     │   │    →    │█│        /    │░░░│
        └───┘     └───┘         └─┘             └───┘

    Alternative Metrics:
        - GIoU (Generalized IoU): Handles non-overlapping boxes better
        - DIoU (Distance IoU): Considers box center distances
        - CIoU (Complete IoU): Includes aspect ratio matching
    """
    # ----------------------------------------
    # Compute Intersection Rectangle
    # ----------------------------------------
    # Find the coordinates of the intersection rectangle
    # Intersection top-left: (max(x1_a, x1_b), max(y1_a, y1_b))
    # Intersection bottom-right: (min(x2_a, x2_b), min(y2_a, y2_b))

    x1 = max(box1[0], box2[0])  # Leftmost point of intersection
    y1 = max(box1[1], box2[1])  # Topmost point of intersection
    x2 = min(box1[2], box2[2])  # Rightmost point of intersection
    y2 = min(box1[3], box2[3])  # Bottommost point of intersection

    # ----------------------------------------
    # Compute Intersection Area
    # ----------------------------------------
    # Width and height must be non-negative (max ensures 0 for no overlap)
    # Area = width × height
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    # ----------------------------------------
    # Compute Individual Box Areas
    # ----------------------------------------
    # Area = (x2 - x1) × (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # ----------------------------------------
    # Compute Union Area
    # ----------------------------------------
    # Union = Area1 + Area2 - Intersection
    # Subtract intersection to avoid double-counting overlapping region
    union = area1 + area2 - intersection

    # ----------------------------------------
    # Compute IoU Score
    # ----------------------------------------
    # Handle division by zero (occurs with zero-area boxes)
    # Return 0.0 if union is zero or negative (invalid boxes)
    return float(intersection / union) if union > 0 else 0.0


# ========================================
# Video Frame Extraction Utilities
# ========================================

# noinspection PyShadowingNames
def extract_frames(video_path: str, sample_rate: int = 5, max_frames: Optional[int] = None) -> Tuple[
    List[np.ndarray], float, int]:
    """
    Extract frames from video at specified sampling rate.

    This function provides efficient video frame extraction with:
    - Configurable sampling rate (process every Nth frame)
    - Optional frame limit (memory management)
    - Metadata extraction (FPS, total frame count)
    - Robust error handling

    Use Cases:
        - Dataset creation (extract keyframes for training)
        - Video preview (sample frames for thumbnails)
        - Preprocessing (reduce video size before analysis)
        - Testing (quick evaluation on subset of frames)

    Args:
        video_path (str): Path to video file
                         Supported formats: .mp4, .avi, .mov, .mkv, etc.
                         Example: "../data/Videos/clip1.mp4"

        sample_rate (int, optional): Extract every Nth frame
                                     - 1: Extract all frames (no sampling)
                                     - 5: Extract every 5th frame (default)
                                     - 10: Extract every 10th frame (faster, less data)
                                     Higher values = faster but may miss events

        max_frames (Optional[int], optional): Maximum number of frames to extract
                                             Useful for:
                                             - Memory management (limit RAM usage)
                                             - Quick testing (process only first N frames)
                                             - Uniform sampling across videos
                                             None = extract all sampled frames (default)

    Returns:
        Tuple[List[np.ndarray], float, int]: Three-element tuple containing:

            1. frames (List[np.ndarray]): Extracted frames as numpy arrays
               - Each frame shape: (height, width, 3) in BGR format
               - Data type: uint8 (0-255)
               - Color order: Blue, Green, Red (OpenCV convention)

            2. fps (float): Video frames per second
               - Example: 30.0, 29.97, 60.0
               - Used for timestamp calculations
               - Important for synchronization

            3. total_frames (int): Total number of frames in video
               - Count before sampling
               - Useful for progress tracking
               - Helps estimate processing time

    Raises:
        FileNotFoundError: If video file doesn't exist at specified path
                          Check: Path typos, relative vs absolute paths

        ValueError: If video cannot be opened
                   Possible causes:
                   - Corrupted video file
                   - Unsupported codec
                   - Missing codec libraries
                   - File is not a video

    Examples:
        >>> # Extract every 10th frame, unlimited
        >>> frames, fps, total = extract_frames("video.mp4", sample_rate=10)
        >>> print(f"Extracted {len(frames)} frames from {total} total")

        >>> # Extract first 100 frames only
        >>> frames, fps, total = extract_frames("video.mp4", max_frames=100)
        >>> print(f"FPS: {fps}, Shape: {frames[0].shape}")

        >>> # Extract all frames (memory intensive!)
        >>> frames, fps, total = extract_frames("short_video.mp4", sample_rate=1)

    Memory Considerations:
        - Each frame: ~height × width × 3 bytes
        - 1080p frame: 1920 × 1080 × 3 ≈ 6.2 MB
        - 1000 frames @ 1080p: ~6.2 GB RAM
        - Use max_frames or higher sample_rate for long videos

    Performance Tips:
        - sample_rate=30: ~1 frame/second @ 30fps (good for preview)
        - sample_rate=5-10: Good balance for analysis
        - sample_rate=1: Only for short videos or high-end systems
        - Consider processing video in chunks for very long videos

    Alternative Approaches:
        - For very large videos: Process frames one-at-a-time (streaming)
        - For temporal analysis: Extract keyframes based on scene changes
        - For training data: Use video data loaders (PyTorch VideoDataset)
    """
    # ----------------------------------------
    # Validate Video Path
    # ----------------------------------------
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # ----------------------------------------
    # Open Video Capture
    # ----------------------------------------
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    # Common failures: corrupted file, unsupported codec, wrong path
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # ----------------------------------------
    # Extract Video Metadata
    # ----------------------------------------
    # Get frames per second (FPS)
    # Important for: timestamp conversion, playback speed, synchronization
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get total frame count
    # Note: May be inaccurate for some video formats (VBR encoding)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize frame storage
    frames = []
    frame_idx = 0

    # ----------------------------------------
    # Extract Frames with Sampling
    # ----------------------------------------
    while cap.isOpened():
        # Read next frame
        # ret: boolean indicating success
        # frame: numpy array of frame data (BGR format)
        ret, frame = cap.read()

        # Break if end of video or read error
        if not ret:
            break

        # ----------------------------------------
        # Sample Frame Based on Rate
        # ----------------------------------------
        # Only process frames at sampling intervals
        # Example: sample_rate=5 processes frames 0, 5, 10, 15, ...
        if frame_idx % sample_rate == 0:
            frames.append(frame)

            # ----------------------------------------
            # Check Maximum Frame Limit
            # ----------------------------------------
            # Stop extraction if max_frames limit reached
            # Useful for memory management and quick testing
            if max_frames and len(frames) >= max_frames:
                break

        frame_idx += 1

    # ----------------------------------------
    # Cleanup Resources
    # ----------------------------------------
    # Release video capture object
    # Frees file handle and associated resources
    cap.release()

    # ----------------------------------------
    # Return Results
    # ----------------------------------------
    return frames, fps, total_frames

# ========================================
# Additional Utility Functions (Optional)
# ========================================
# These could be added for future enhancements:

# def normalize_bbox(bbox: List[float], frame_width: int, frame_height: int) -> List[float]:
#     """Normalize bounding box coordinates to [0, 1] range."""
#     x1, y1, x2, y2 = bbox
#     return [x1/frame_width, y1/frame_height, x2/frame_width, y2/frame_height]

# def denormalize_bbox(norm_bbox: List[float], frame_width: int, frame_height: int) -> List[int]:
#     """Convert normalized bbox [0,1] back to pixel coordinates."""
#     x1, y1, x2, y2 = norm_bbox
#     return [int(x1*frame_width), int(y1*frame_height), int(x2*frame_width), int(y2*frame_height)]

# def bbox_area(bbox: List[float]) -> float:
#     """Calculate area of bounding box."""
#     x1, y1, x2, y2 = bbox
#     return (x2 - x1) * (y2 - y1)

# def bbox_center(bbox: List[float]) -> Tuple[float, float]:
#     """Calculate center point of bounding box."""
#     x1, y1, x2, y2 = bbox
#     return ((x1 + x2) / 2, (y1 + y2) / 2)

# def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
#     """Calculate Euclidean distance between two points."""
#     return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)