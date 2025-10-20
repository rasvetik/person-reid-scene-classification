"""
Video Annotator Module for Person Re-Identification Visualization

This module creates annotated videos that visualize person re-identification results.
Unlike VideoProcessor which runs detection models, this module:
1. Uses pre-computed detection data (no model inference)
2. Maps local track IDs to global person IDs from the catalogue
3. Renders bounding boxes with consistent colors per person
4. Displays both global IDs (across clips) and local track IDs (within clip)

The output videos help verify re-identification quality and debug tracking issues.
"""

import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors

from utils import load_json

# noinspection SpellCheckingInspection
class VideoAnnotator:
    """
    Annotates videos with person re-identification results using pre-computed data.

    This class takes existing detection data and the identity catalogue to create
    visualization videos. It does NOT re-run detection models, making it much faster
    than the original detection process. Useful for:
    - Verifying re-identification accuracy
    - Debugging tracking issues
    - Creating demo videos
    - Quality assurance of the pipeline

    Attributes:
        video_path (str): Path to input video file
        clip_id (str): Unique identifier for this video clip
        catalogue (dict): Global identity catalogue mapping persons across clips
        person_data (list): Pre-computed detection data with embeddings
        frames_data (defaultdict): Frame-indexed lookup for fast annotation
        track_to_global_map (dict): Maps local track IDs to global person IDs
    """

    def __init__(self, video_path, clip_id, catalogue, person_data, show_detections=0,
                 output_vis_dir=Path("../outputs/visualizations"), save_video=True):
        """
        Initializes the VideoAnnotator with pre-computed detection data.

        Args:
            video_path (str or Path): Path to the input video file (.mp4)
            clip_id (str): Identifier for the video clip (must match catalogue entries)
            catalogue (dict): Global identity catalogue from Re-ID pipeline
                Format: {
                    "person_0": [
                        {
                            "clip_id": "clip1",
                            "local_id": "track_5",
                            "frame_range": (10, 150),
                            ...
                        }
                    ]
                }
            person_data (list): Raw detection records from VideoProcessor
                Each record contains: clip_id, frame_idx, track_id, bbox, confidence, etc.
            show_detections (bool, optional): If True, display real-time annotation window
                (useful for debugging, slower for batch processing)
            output_vis_dir (Path, optional): Directory for saving annotated videos
            save_video (bool, optional): If True, save the annotated video to disk

        Notes:
            - Pre-processes person_data into frame-indexed structure for O(1) lookup
            - No model inference occurs (unlike VideoProcessor)
            - Colors are consistent per global person ID for easy visual tracking
        """
        # Store initialization parameters
        self.video_path = video_path
        self.clip_id = clip_id
        self.catalogue = catalogue
        self.person_data = person_data
        self.output_dir = output_vis_dir
        self.save_video = save_video
        self.show_detections = show_detections

        # ========================================
        # Pre-process Detection Data for Fast Lookup
        # ========================================
        # Convert list of detections into frame-indexed dictionary
        # This enables O(1) lookup instead of O(n) search for each frame
        # Structure: {frame_idx: [detection1, detection2, ...]}
        self.frames_data = defaultdict(list)

        for record in self.person_data:
            # Only include detections from this specific clip
            if record["clip_id"] == self.clip_id:
                self.frames_data[record["frame_idx"]].append(record)

        # Note: defaultdict(list) automatically returns empty list for missing frames
        # This prevents KeyError when accessing frames with no detections

    def annotate_video(self):
        """
        Generates and saves an annotated video with person IDs and bounding boxes.

        This method overlays the following information on each frame:
        - Bounding boxes around detected persons
        - Global person ID (consistent across all clips)
        - Local track ID (unique within this clip)
        - Detection confidence score
        - Color-coded boxes (same person = same color across frames)

        Pipeline:
        1. Open video and read properties
        2. Build track-to-global ID mapping from catalogue
        3. For each frame:
           a. Retrieve pre-computed detections
           b. Map local track IDs to global person IDs
           c. Draw annotated bounding boxes
           d. Write to output video
        4. Clean up resources

        Side Effects:
            - Creates annotated video file if save_video=True
            - Shows real-time window if show_detections=True

        Returns:
            None

        Output Format:
            Annotated video saved to: {output_dir}/annotated_reid_results{clip_id}.mp4
        """
        # ========================================
        # Step 1: Open Video and Read Properties
        # ========================================
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return

        # Extract video metadata for output video creation
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ========================================
        # Setup Video Writer (if saving)
        # ========================================
        if self.save_video:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            output_video_path = os.path.join(self.output_dir, f"annotated_reid_results{self.clip_id}.mp4")

            # Define codec and create VideoWriter
            # 'mp4v' = MPEG-4 codec (widely compatible)
            # noinspection PyUnresolvedReferences
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # ========================================
        # Step 2: Build Track-to-Global ID Mapping
        # ========================================
        # Maps local track IDs (within this clip) to global person IDs (across all clips)
        # This is the core of re-identification visualization
        # Structure: {local_track_id: "person_X"}
        track_to_global_map = {}

        if self.catalogue:
            # Iterate through all identified persons
            for global_id, appearances in self.catalogue.items():
                # Check each appearance of this person
                for appearance in appearances:
                    # Only map appearances in this specific clip
                    if appearance["clip_id"] == self.clip_id:
                        # Extract track ID from local_id string
                        # Format: "track_5" -> 5
                        track_id = int(appearance["local_id"].split("_")[-1])

                        # Create mapping: local track ID -> global person ID
                        track_to_global_map[track_id] = global_id

        # If track_to_global_map is empty, all detections will be marked "unknown"

        # ========================================
        # Step 3: Process Each Frame
        # ========================================
        # Progress bar for user feedback
        pbar = tqdm(total=total_frames, desc=f"Annotating {self.video_path}")

        frame_idx = 0
        while cap.isOpened():
            # Read next frame
            success, frame = cap.read()
            if not success:
                break  # End of video

            # ----------------------------------------
            # Initialize Ultralytics Annotator
            # ----------------------------------------
            # Annotator provides professional-looking visualization utilities
            # line_width=2: Thickness of bounding box lines
            # example: Used internally by Annotator for text size calculation
            # noinspection PyShadowingNames
            annotator = Annotator(
                frame,
                line_width=2,
                example=str(self.frames_data.get(frame_idx, ""))
            )

            # ----------------------------------------
            # Get Detections for Current Frame
            # ----------------------------------------
            # Retrieve pre-computed detections (O(1) lookup thanks to preprocessing)
            detections_for_frame = self.frames_data.get(frame_idx, [])

            # Process each detection in this frame
            for det in detections_for_frame:
                # ----------------------------------------
                # Extract Detection Information
                # ----------------------------------------
                bbox = det["bbox"]  # (x1, y1, x2, y2)
                track_id = det["track_id"]  # Local track ID
                confidence = det["confidence"]  # Detection confidence [0,1]

                # ----------------------------------------
                # Map Local Track ID to Global Person ID
                # ----------------------------------------
                # Look up global ID from the catalogue mapping
                global_id = track_to_global_map.get(track_id, "unknown")

                # Parse global ID string: "person_5" -> ["person", "5"]
                global_id_parts = global_id.split("_")

                # Skip detections that couldn't be matched to a global identity
                # This happens for:
                # - Tracklets filtered out during aggregation (too short/small)
                # - Detections outside the Re-ID pipeline
                if global_id_parts[0] == "unknown":
                    continue

                # Round confidence to 2 decimal places for cleaner display
                confidence = np.ceil(confidence * 100) / 100

                # ----------------------------------------
                # Format Label Text
                # ----------------------------------------
                # Display format: "ID:5 person (T12, 0.87)"
                # - ID:5 = Global person ID (consistent across clips)
                # - person = Label type
                # - T12 = Local track ID (within this clip)
                # - 0.87 = Detection confidence
                label = f"ID:{global_id_parts[-1]} {global_id_parts[0]} (T{track_id}, {confidence:.2f})"

                # ----------------------------------------
                # Choose Consistent Color
                # ----------------------------------------
                # Use the same color for the same global person ID
                # This enables visual tracking of individuals across frames
                # colors() is a function from Ultralytics that provides 18 distinct colors
                # Modulo ensures we cycle through colors if more than 18 persons
                color = colors(int(global_id_parts[-1]) % 18, True)

                # ----------------------------------------
                # Draw Bounding Box and Label
                # ----------------------------------------
                # box_label() draws:
                # - Rectangle around person
                # - Filled background for text
                # - Label text above box
                annotator.box_label(bbox, label=label, color=color)

            # ----------------------------------------
            # Finalize Annotated Frame
            # ----------------------------------------
            # Get the frame with all annotations drawn
            annotated_frame = annotator.result()

            # Optional: Display in real-time window
            if self.show_detections:
                cv2.imshow('Global Object Detection', annotated_frame)
                cv2.waitKey(1)  # 1ms delay for responsiveness

            # Write frame to output video
            if self.save_video:
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
        if self.save_video:
            out.release()  # Finalize output video file
        cap.release()  # Release input video file
        pbar.close()  # Close progress bar


# ========================================
# Standalone Execution Example
# ========================================
if __name__ == "__main__":
    """
    Example usage of VideoAnnotator for testing and demonstration.

    This standalone script:
    1. Loads pre-computed detection data and identity catalogue
    2. Annotates the first video with person IDs
    3. Shows real-time visualization (optional)
    4. Saves annotated video (optional)

    Prerequisites:
        - Run main.py first to generate:
            * identity_catalogue.json
            * detections{clip_id}.json files

    Usage:
        python video_annotator.py

    Notes:
        - Much faster than VideoProcessor (no model inference)
        - Useful for visualizing Re-ID results
        - Good for creating demo videos
    """
    # ========================================
    # Setup Paths
    # ========================================
    videos_dir = Path("../data/Videos")
    video_files = sorted(videos_dir.glob("*.mp4"))
    output_dir = Path("../outputs")

    # Paths to pre-computed data
    catalogue_path = os.path.join(output_dir, "identity_catalogue.json")
    detection_path = os.path.join(output_dir, "detections1.json")

    # ========================================
    # Load Pre-computed Data
    # ========================================
    print("Loading identity catalogue...")
    identity_catalogue = load_json(catalogue_path)
    print(f"  Found {len(identity_catalogue)} unique persons")

    print("Loading detection data...")
    detections = load_json(detection_path)
    print(f"  Found {len(detections)} detections")

    # ========================================
    # Annotate Video
    # ========================================
    if video_files:
        print(f"\nAnnotating video: {video_files[0]}")

        # Create annotator instance
        # Note: clip_id must match the clip_id used during detection
        annotator = VideoAnnotator(
            video_files[0],  # First video file
            clip_id="1",  # Must match detection file naming
            catalogue=identity_catalogue,  # Global person catalogue
            person_data=detections,  # Pre-computed detections
            show_detections=True,  # Display real-time window
            output_vis_dir=output_dir,  # Save to outputs directory
            save_video=False  # Skip saving for quick testing
        )

        # Run annotation process
        annotator.annotate_video()

        print("\nAnnotation complete!")
        if annotator.save_video:
            print(f"Annotated video saved to: {output_dir}/annotated_reid_results1.mp4")
    else:
        print("No video files found to annotate.")
        print(f"Please place video files in: {videos_dir}")

# "ID:5 person (T12, 0.87)"
#  │   │    │     │    └─ Detection confidence
#  │   │    │     └────── Local track ID (this clip only)
#  │   │    └──────────── Person label type
#  │   └───────────────── Global person number
#  └───────────────────── Global person ID prefix