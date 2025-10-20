"""
Video Processor Module for Person Detection, Tracking, and Re-Identification

This module handles the core video processing pipeline:
1. Person Detection: Uses YOLO11n to detect persons in video frames
2. Person Tracking: Maintains consistent IDs for persons across frames
3. Feature Extraction: Extracts Re-ID embeddings using ResNet50
4. Data Collection: Stores detection metadata and embeddings for downstream processing

The output is structured detection data that enables person re-identification
across different video clips.
"""

import cv2
import os
import torch
from ultralytics import YOLO
from torchvision.models.resnet import resnet50, ResNet50_Weights
from utils import  frame_to_timestamp
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm

# ========================================
# Environment Configuration
# ========================================
# Set torch home directory for caching pre-trained models
# This prevents re-downloading models and allows offline usage
os.environ["TORCH_HOME"] = "..\\models\\resnet"


class VideoProcessor:
    """
    Processes video files to detect, track, and extract features from persons.

    This class encapsulates the entire detection pipeline for a single video:
    - Frame-by-frame processing with optional sampling
    - YOLO-based person detection and tracking
    - ResNet50-based feature embedding extraction
    - Optional visualization with bounding boxes
    - Structured output data for re-identification

    Attributes:
        video_path (str): Path to input video file
        clip_id (str): Unique identifier for this video clip
        device (str): Compute device ('cuda' or 'cpu')
        sample_rate (int): Process every Nth frame
        yolo_model (YOLO): Person detection and tracking model
        reid_model (nn.Module): Feature extraction model for Re-ID
        person_data (list): Collected detection data with embeddings
    """

    def __init__(self, video_path, clip_id, model_device=None, sample_rate=1,
                 output_vis_dir=Path("..", "outputs", "visualizations"),
                 show_detections=0, save_det_video=1):
        """
        Initializes the VideoProcessor with video and model settings.

        Args:
            video_path (str or Path): Path to the input video file (.mp4)
            clip_id (str): Unique identifier for this video clip (used in output naming)
            model_device (str, optional): Device for model inference ('cuda' or 'cpu'). Auto-detects if None.
            sample_rate (int, optional): Frame sampling rate.
                                        1 = process every frame
                                        2 = process every 2nd frame
                                        Higher values = faster but may miss short appearances
            output_vis_dir (Path, optional): Directory for saving visualization videos
            show_detections (bool, optional): If True, display real-time detection window
                                             (useful for debugging, slow for production)
            save_det_video (bool, optional): If True, save annotated video with detections

        Notes:
            - YOLO model is downloaded on first use to ../models/yolo/
            - ResNet weights are cached in TORCH_HOME directory
            - Default settings prioritize accuracy over speed
        """
        # Auto-detect device if not specified (prefer GPU if available)
        if model_device is None:
            model_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Store initialization parameters
        self.video_path = video_path
        self.clip_id = clip_id
        self.device = model_device
        self.sample_rate = sample_rate
        self.output_dir = output_vis_dir
        self.show_detections = show_detections
        self.save_det_video = save_det_video

        # Video properties (populated during processing)
        self.total_frames = None
        self.fps = None
        self.frame_width = None
        self.frame_height = None

        # ========================================
        # Initialize Detection Model (YOLO11n)
        # ========================================
        # YOLO11n: Nano version optimized for speed/accuracy tradeoff
        # First run downloads model to ../models/yolo/
        # Configuration settings stored in: %APPDATA%\Roaming\Ultralytics\settings.yaml
        # noinspection SpellCheckingInspection
        self.yolo_model = YOLO("..\\models\\yolo\\yolo11n.pt")

        # ========================================
        # Initialize Re-ID Feature Extractor (ResNet50)
        # ========================================
        # Use ResNet50 pre-trained on ImageNet (80.858% top-1 accuracy)
        # This provides strong visual features for person re-identification
        self.reid_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(self.device).eval()

        # Remove final classification layer (FC layer)
        # We only need the feature embeddings, not ImageNet class predictions
        # Output becomes 2048-dimensional feature vector instead of 1000 class scores
        self.reid_model = torch.nn.Sequential(*(list(self.reid_model.children())[:-1]))

        # ========================================
        # Define Image Preprocessing Pipeline
        # ========================================
        # Transform person crops to match ResNet training conditions
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((256, 128)),  # Standard aspect ratio for person Re-ID
            # Height=256, Width=128 (2:1 ratio)
            transforms.ToTensor(),  # Convert to tensor, normalize to [0,1]
            transforms.Normalize(  # Normalize using ImageNet statistics
                mean=[0.485, 0.456, 0.406],  # Per-channel means (RGB)
                std=[0.229, 0.224, 0.225]  # Per-channel std deviations
            )
        ])

        # Storage for all detection data from this video
        self.person_data = []

    def process_video(self):
        """
        Processes the entire video, extracting person detections and embeddings.

        Pipeline:
        1. Open video file and read properties (FPS, resolution, frame count)
        2. Loop through frames (with optional sampling)
        3. For each sampled frame:
           a. Run YOLO detection and tracking
           b. For each detected person:
              - Crop person region
              - Extract Re-ID embedding
              - Store metadata and embedding
        4. Optionally save annotated video
        5. Clean up resources

        Side Effects:
            - Populates self.person_data with detection records
            - Creates annotated video if save_det_video=True
            - Shows real-time window if show_detections=True

        Returns:
            None (data accessible via get_person_data())

        Raises:
            Prints error message if video file cannot be opened
        """
        # ========================================
        # Step 1: Open Video and Read Properties
        # ========================================
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return

        # Extract video metadata
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0  # Current frame index (0-based)

        # ========================================
        # Setup Video Writer (if saving annotated video)
        # ========================================
        # noinspection GrazieInspection
        if self.save_det_video:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            output_video_path = os.path.join(self.output_dir, f"annotated_detections_{self.clip_id}.mp4")

            # Define codec and create VideoWriter
            # 'mp4v' = MPEG-4 codec (widely compatible)
            # noinspection PyUnresolvedReferences
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))

        # ========================================
        # Step 2: Process Each Frame
        # ========================================
        # Progress bar for user feedback
        pbar = tqdm(total=self.total_frames, desc=f"Processing {self.video_path}")

        while cap.isOpened():
            # Read next frame
            success, frame = cap.read()
            if not success:
                break  # End of video

            # ----------------------------------------
            # Frame Sampling Logic
            # ----------------------------------------
            # Skip frames based on sample_rate to reduce computation
            # Example: sample_rate=5 means process frames 0, 5, 10, 15, ...
            if frame_idx % self.sample_rate != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            # ----------------------------------------
            # Step 3: Run YOLO Detection and Tracking
            # ----------------------------------------
            # track() maintains consistent IDs across frames (vs. detect())
            # persist=True: Keep track IDs consistent across frames
            # classes=[0]: Only detect "person" class (COCO class 0)
            # verbose=False: Suppress per-frame output
            # iou=0.3: IoU threshold for matching detections to tracks
            # conf=0.3: Minimum confidence threshold for detections
            results = self.yolo_model.track(
                frame,
                persist=True,  # Enable tracking across frames
                classes=[0],  # Person class only
                verbose=False,  # Suppress console output
                iou=0.3,  # Lower = more strict matching
                conf=0.3  # Confidence threshold (0-1)
            )[0]

            # Convert frame index to timestamp (seconds)
            timestamp = frame_to_timestamp(frame_idx, self.fps)

            # ----------------------------------------
            # Optional: Display Detections in Real-Time
            # ----------------------------------------
            if self.show_detections:
                # Draw bounding boxes and IDs on frame
                annotated_frame = results.plot()
                # Display in window (waitKey(1) = 1ms delay)
                cv2.imshow('Object Detection', annotated_frame)
                cv2.waitKey(1)

            # ----------------------------------------
            # Step 4: Process Each Detection
            # ----------------------------------------
            # Check if any persons were detected and tracked
            if results and results.boxes.id is not None:
                # Extract detection information
                boxes = results.boxes.xyxy.cpu()  # Absolute coordinates (x1,y1,x2,y2)
                nboxes = results.boxes.xyxyn.cpu()  # Normalized coordinates [0,1]
                # noinspection SpellCheckingInspection
                cboxes = results.boxes.xywh.cpu()  # Center format (cx,cy,w,h)
                track_ids = results.boxes.id.int().cpu().tolist()  # Tracking IDs
                confidences = results.boxes.conf.cpu().numpy()  # Detection confidences

                # Process each detected person
                for box, track_id, conf, nbox, cbox in zip(boxes, track_ids, confidences, nboxes, cboxes):
                    # ----------------------------------------
                    # Extract Bounding Box Coordinates
                    # ----------------------------------------
                    x1, y1, x2, y2 = map(int, box)

                    # Clamp coordinates to frame boundaries
                    # Prevents errors from boxes extending outside frame
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    # Crop person region from frame
                    person_crop = frame[y1:y2, x1:x2]

                    # Skip if crop is empty (edge case)
                    if person_crop.size > 0:
                        # ----------------------------------------
                        # Extract Re-ID Feature Embedding
                        # ----------------------------------------
                        embedding = self.get_embedding(person_crop)

                        # Calculate normalized bounding box area
                        # Used to weight embeddings (larger/closer persons = more reliable)
                        area = (x2 - x1) * (y2 - y1) / (self.frame_width * self.frame_height)

                        # ----------------------------------------
                        # Store Detection Record
                        # ----------------------------------------
                        self.person_data.append({
                            "clip_id": self.clip_id,  # Video identifier
                            "frame_idx": frame_idx,  # Frame number
                            "timestamp": timestamp,  # Time in seconds
                            "track_id": track_id,  # Local tracking ID
                            "bbox": (x1, y1, x2, y2),  # Absolute coordinates
                            "norm_bbox": nbox.tolist(),  # Normalized [0,1]
                            "norm_area": area,  # Normalized area
                            "cntr_box": cbox.tolist(),  # Center format
                            "confidence": float(conf),  # Detection confidence
                            "embedding": embedding.tolist(),  # Re-ID feature vector
                        })

            # ----------------------------------------
            # Save Annotated Frame to Output Video
            # ----------------------------------------
            if self.save_det_video:
                annotated_frame = results.plot()
                # noinspection PyUnboundLocalVariable
                out.write(annotated_frame)

            # Update counters
            frame_idx += 1
            pbar.update(1)

        # ========================================
        # Cleanup Resources
        # ========================================
        if self.show_detections:
            cv2.destroyAllWindows()  # Close display window
        if self.save_det_video:
            out.release()  # Finalize output video

        cap.release()  # Release video file
        pbar.close()  # Close progress bar

    def get_embedding(self, image):
        """
        Extracts a Re-ID feature embedding from a person crop image.

        This method processes a cropped person image through ResNet50 to obtain
        a 2048-dimensional feature vector that captures appearance characteristics.
        These embeddings enable person re-identification by comparing similarity.

        Args:
            image (np.ndarray): BGR image crop of a person (H x W x 3)

        Returns:
            np.ndarray: Feature embedding vector of shape (2048,)

        Pipeline:
            1. Resize to 256x128 (standard Re-ID input size)
            2. Convert to tensor and normalize using ImageNet stats
            3. Forward pass through ResNet50 (minus final FC layer)
            4. Flatten global average pooling output to 1D vector

        Notes:
            - Runs in inference mode (no gradients computed)
            - Embedding is L2-normalized for cosine similarity comparison
            - Same preprocessing must be used for all embeddings to enable comparison
        """
        # Apply preprocessing transformations
        # Converts BGR -> RGB, resizes, normalizes
        processed_image = self.preprocess(image)

        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        # Move to appropriate device (CPU or GPU)
        img_tensor = processed_image.unsqueeze(0).to(self.device)

        # Extract features without computing gradients (faster, less memory)
        with torch.no_grad():
            # Forward pass through ResNet50
            # Output shape: (1, 2048, 1, 1) after global average pooling
            embedding = self.reid_model(img_tensor).detach().numpy().flatten()
            # Flatten to 1D: (2048,)

        return embedding

    def get_person_data(self):
        """
        Returns the collected person detection data.

        Returns:
            list: List of detection dictionaries, each containing:
                - clip_id: Video identifier
                - frame_idx: Frame number
                - timestamp: Time in seconds
                - track_id: Local tracking ID
                - bbox: Bounding box (x1, y1, x2, y2)
                - norm_bbox: Normalized bbox [0,1]
                - norm_area: Normalized area
                - cntr_box: Center format (cx, cy, w, h)
                - confidence: Detection confidence
                - embedding: 2048-dim feature vector

        Notes:
            - Data is only available after calling process_video()
            - Each entry represents one person detection in one frame
            - Multiple entries can have the same track_id (one per frame)
        """
        return self.person_data


# ========================================
# Standalone Execution Example
# ========================================
if __name__ == "__main__":
    """
        Example usage of VideoProcessor for testing and demonstration.
    
        This standalone script:
        1. Loads a video from the data directory
        2. Processes it with detection and tracking
        3. Shows real-time visualization
        4. Prints summary statistics
    
    Usage:
        python video_processor.py
    
    Notes:
        - Set show_detections=True to see real-time bounding boxes
        - Set save_det_video=True to save annotated video
        - Adjust sample_rate to control speed vs. accuracy tradeoff
    """
    # ========================================
    # Setup Paths
    # ========================================
    videos_dir = Path("../data/Videos")
    video_files = sorted(videos_dir.glob("*.mp4"))
    output_dir = Path("../outputs")

    # Check if videos exist
    if not video_files:
        print(f"No video files found in {videos_dir}")
        exit(1)

    print(f"Found {len(video_files)} video file(s)")
    print(f"Processing: {video_files[0]}")

    # ========================================
    # Initialize and Run Processor
    # ========================================
    # Example configuration:
    # - Show detections in real-time (good for debugging)
    # - Don't save annotated video (faster for testing)
    # - Process all frames (sample_rate=1)
    vp = VideoProcessor(
        video_files[0],  # First video in directory
        clip_id='1',  # Identifier for this clip
        show_detections=True,  # Display real-time window
        save_det_video=False  # Skip saving annotated video
    )

    # Process the video
    print("\nStarting video processing...")
    vp.process_video()

    # ========================================
    # Display Results Summary
    # ========================================
    data = vp.get_person_data()

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total person detections: {len(data)}")

    if data:
        # Calculate statistics
        unique_tracks = len(set([d['track_id'] for d in data]))
        avg_confidence = sum([d['confidence'] for d in data]) / len(data)
        avg_area = sum([d['norm_area'] for d in data]) / len(data)

        print(f"Unique track IDs: {unique_tracks}")
        print(f"Average detection confidence: {avg_confidence:.3f}")
        print(f"Average normalized area: {avg_area:.4f}")

        # Show sample detection
        print("\nSample detection record:")
        sample = data[0]
        print(f"  Clip ID: {sample['clip_id']}")
        print(f"  Frame: {sample['frame_idx']}")
        print(f"  Timestamp: {sample['timestamp']:.2f}s")
        print(f"  Track ID: {sample['track_id']}")
        print(f"  Bbox: {sample['bbox']}")
        print(f"  Confidence: {sample['confidence']:.3f}")
        print(f"  Embedding shape: {len(sample['embedding'])} dimensions")
    else:
        print("No person detections found in video!")

    print("=" * 60)