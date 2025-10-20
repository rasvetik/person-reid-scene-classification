"""
Main script for Person Re-Identification and Scene Classification

This pipeline performs two main tasks:
1. Person Re-ID: Detects, tracks, and identifies unique persons across video clips
2. Scene Classification: Analyzes scenes to detect normal vs. crime activities

The pipeline processes multiple video files and generates:
- Detection data with embeddings
- Identity catalogue mapping persons across clips
- Annotated videos with person IDs
- Scene classification results
"""

import os
import argparse
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from pathlib import Path
from video_processor import VideoProcessor
from video_annotator import VideoAnnotator
from scene_classifier import SceneClassifier
from utils import load_json, save_json, set_seed

# noinspection SpellCheckingInspection
def aggregate_embeddings(person_data):
    """
    Aggregates person embeddings from tracking data into tracklet-level representations.

    This function processes raw detection data and creates a single representative embedding
    for each tracklet (continuous sequence of detections for one person in one clip).
    The aggregation uses quality-weighted averaging based on detection confidence and bbox area.

    Args:
        person_data (list): List of detection records, each containing:
            - clip_id: Video clip identifier
            - track_id: Local tracking ID within the clip
            - embedding: Feature vector for the person
            - frame_idx: Frame number
            - timestamp: Time in video
            - bbox: Bounding box coordinates
            - norm_area: Normalized bbox area
            - confidence: Detection confidence score
            - norm_bbox: Normalized bbox coordinates
            - cntr_box: Center box coordinates

    Returns:
        tuple: (aggregated_embeddings, tracklet_info)
            - aggregated_embeddings (np.array): Array of averaged embeddings, shape (N, embedding_dim)
            - tracklet_info (list): Metadata for each tracklet including temporal info and quality metrics

    Notes:
        - Filters out short tracklets (<= 5 frames) and small detections (area <= 0.01)
        - Quality score combines normalized confidence (50%) and normalized area (50%)
        - Embeddings are normalized before averaging to ensure unit length
    """
    # Initialize dictionaries to collect data for each tracklet
    # Key format: (clip_id, track_id) uniquely identifies each tracklet
    tracklet_embeddings = defaultdict(list)  # Feature embeddings
    tracklet_frames = defaultdict(list)  # Frame indices
    tracklet_times = defaultdict(list)  # Timestamps
    tracklet_bbox = defaultdict(list)  # Bounding boxes
    tracklet_area = defaultdict(list)  # Normalized areas
    tracklet_conf = defaultdict(list)  # Detection confidences
    tracklet_nbox = defaultdict(list)  # Normalized bboxes
    tracklet_cbox = defaultdict(list)  # Center boxes

    # Group all detections by tracklet
    for record in person_data:
        key = (record["clip_id"], record["track_id"])
        tracklet_embeddings[key].append(record["embedding"])
        tracklet_frames[key].append(record["frame_idx"])
        tracklet_times[key].append(record["timestamp"])
        tracklet_bbox[key].append(record["bbox"])
        tracklet_area[key].append(record["norm_area"])
        tracklet_conf[key].append(record["confidence"])
        tracklet_nbox[key].append(record["norm_bbox"])
        tracklet_cbox[key].append(record["cntr_box"])

    # Prepare output containers
    aggregated_embeddings = []
    tracklet_info = []

    # Process each tracklet
    for key, embeddings in tracklet_embeddings.items():
        # Calculate tracklet statistics
        number_frames = len(tracklet_frames[key])
        avg_area = np.mean(tracklet_area[key])

        # Filter: Keep only tracklets with sufficient length and size
        # This removes noise from brief/distant detections
        if number_frames > 5 and avg_area > 0.01:
            # Compute quality weights for each detection in the tracklet
            # Higher quality detections (larger, more confident) get more weight

            # Normalize areas within this tracklet (range 0-1)
            areas = np.array(tracklet_area[key]).reshape(1, -1)
            norm_areas = areas / (np.max(areas) + 1e-6)  # Add epsilon to avoid division by zero

            # Normalize confidences within this tracklet (range 0-1)
            confidences = np.array(tracklet_conf[key]).reshape(1, -1)
            norm_conf = confidences / (np.max(confidences) + 1e-6)

            # Combine into overall quality score (50% confidence, 50% area)
            quality_scores = 0.5 * norm_conf[0] + 0.5 * norm_areas[0]

            # Compute quality-weighted average embedding
            # Step 1: Normalize all embeddings to unit length (for cosine similarity)
            # Step 2: Take weighted average using quality scores
            avg_embedding = np.average(normalize(embeddings), axis=0, weights=quality_scores)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-6)  # Normalize after averaging
            # Store the aggregated embedding
            aggregated_embeddings.append(avg_embedding)

            # Store metadata about this tracklet for later reference
            tracklet_info.append({
                "clip_id": key[0],
                "track_id": key[1],
                "frame_range": (min(tracklet_frames[key]), max(tracklet_frames[key])),
                "time_range": (min(tracklet_times[key]), max(tracklet_times[key])),
                "number_frames": number_frames,
                "avg_area": avg_area,
                "avg_conf": np.mean(tracklet_conf[key])
            })

    return np.array(aggregated_embeddings), tracklet_info

# noinspection SpellCheckingInspection
def generate_catalogue(aggregated_embeddings, tracklet_info):
    """
    Clusters tracklets to identify unique persons across all video clips.

    Uses Agglomerative Clustering with cosine metric to group similar embeddings,
    effectively solving the person re-identification problem across clips.

    Args:
        aggregated_embeddings (np.array): Array of tracklet embeddings, shape (N, embedding_dim)
        tracklet_info (list): Metadata for each tracklet (must align with embeddings)

    Returns:
        dict: Identity catalogue mapping global person IDs to their appearances
            Format: {
                "person_0": [
                    {
                        "clip_id": "clip1",
                        "frame_range": (start, end),
                        "time_range": (start_time, end_time),
                        "local_id": "track_5",
                        "number_frames": 120,
                        "avg_area": 0.045,
                        "avg_confidence": 0.89
                    },
                    ...
                ],
                "person_1": [...],
                ...
            }

    Notes:
        - Distance threshold of 0.2 is hardcoded for cosine distance
        - Uses "average" linkage (UPGMA) which works well with cosine metric
        - Tracklets in the same cluster are considered the same person
    """
    # Handle edge case: no valid tracklets
    if not aggregated_embeddings.size:
        return {}

    # Perform hierarchical clustering
    # - n_clusters=None: Let distance_threshold determine number of clusters
    # - distance_threshold=0.2: Cosine distance threshold (0 = identical, 2 = opposite)
    # - metric="cosine": Cosine distance (1 - cosine_similarity)
    # - linkage="average": Average distance between all pairs (UPGMA)
    agg_clust = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.2,
        metric="cosine",
        linkage="average"
    )
    clusters = agg_clust.fit_predict(aggregated_embeddings)

    # Build the identity catalogue
    catalogue = defaultdict(list)

    for i, cluster_id in enumerate(clusters):
        # Skip noise points (shouldn't occur with Agglomerative, but safe check)
        if cluster_id == -1:
            continue

        # Create global person ID
        global_id = f"person_{cluster_id}"

        # Get tracklet metadata
        record = tracklet_info[i]

        # Add this tracklet appearance to the person's catalogue entry
        catalogue[global_id].append({
            "clip_id": record["clip_id"],
            "frame_range": record["frame_range"],
            "time_range": record["time_range"],
            "local_id": f"track_{record['track_id']}",
            "number_frames": record["number_frames"],
            "avg_area": record["avg_area"],
            "avg_confidence": record["avg_conf"]
        })

    return catalogue

# noinspection SpellCheckingInspection
def run_person_reid(video_dir, output_dir, device='cpu', sample_rate=1, load_data=0):
    """
    Main pipeline for Person Re-Identification across multiple video clips.

    This function orchestrates the complete Re-ID workflow:
    1. Detection: Detect and track persons in each video clip
    2. Aggregation: Create representative embeddings for each tracklet
    3. Clustering: Group tracklets across clips to identify unique persons
    4. Visualization: Generate annotated videos with global person IDs

    Args:
        video_dir (Path): Directory containing input video files (.mp4)
        output_dir (Path): Directory for saving all outputs
        device (str): Device for inference ('cuda' or 'cpu')
        sample_rate (int): Process every Nth frame (1 = all frames)
        load_data (int): If 1, load cached intermediate results instead of reprocessing

    Returns:
        dict: Identity catalogue mapping person IDs to their appearances across clips

    Outputs Created:
        - detections_{clip_id}.json: Raw detection data for each clip
        - tracklet_info.json: Aggregated tracklet metadata
        - aggregated_embeddings.npy: Numpy array of tracklet embeddings
        - identity_catalogue.json: Final person identity mapping
        - visualizations/: Annotated videos with person IDs
    """
    # Container for all detection data across clips
    all_person_data = []

    # Get all video files sorted by name for consistent ordering
    video_files = sorted(video_dir.glob('*.mp4'))

    # Print pipeline header
    print("=" * 60)
    print("PERSON RE-IDENTIFICATION & SCENE CLASSIFICATION")
    print("=" * 60)
    print(f"\nFound {len(video_files)} video files")
    print(f"\nOutput directory: {output_dir}")
    print(f"Sample rate: every {sample_rate} frames")
    print(f"Device: {device}")
    print("=" * 60)

    # ========================================
    # PART A: Person Re-Identification
    # ========================================
    print("\n" + "=" * 60)
    print("PART A: PERSON RE-IDENTIFICATION")
    print("=" * 60)

    # ----------------------------------------
    # Step 1: Detection and Tracking
    # ----------------------------------------
    print("\n" + "-" * 30)
    print(f"1. Detection Step")
    print("-" * 30)

    for video_file in video_files:
        clip_id = video_file.stem  # Extract filename without extension
        output_det_path = os.path.join(output_dir, f"detections_{clip_id}.json")

        # Check if cached detections exist
        if os.path.exists(output_det_path) and load_data:
            print(f"Loading detections{clip_id}.json for Re-ID...")
            person_data = load_json(output_det_path)
            all_person_data.extend(person_data)
        else:
            # Process video: detect persons, track them, extract embeddings
            print(f"Processing {video_file} for Re-ID...")
            processor = VideoProcessor(
                video_file,
                clip_id,
                device,
                sample_rate,
                save_det_video=1,  # Save annotated detection video
                show_detections=0  # Don't display real-time
            )
            processor.process_video()

            # Get detection data with embeddings
            person_data = processor.get_person_data()
            all_person_data.extend(person_data)

            # Cache results
            save_json(output_det_path, person_data)

    # ----------------------------------------
    # Step 2: Embedding Aggregation
    # ----------------------------------------
    print("\n" + "-" * 30)
    print(f"2. Aggregating embeddings...")
    print("-" * 30)

    output_track_path = os.path.join(output_dir, "tracklet_info.json")
    output_emb_path = os.path.join(output_dir, "aggregated_embeddings.npy")

    # Check if cached aggregated data exists
    if os.path.exists(output_track_path) and os.path.exists(output_emb_path) and load_data:
        print(f"Loading aggregated_embeddings and tracklet_info.")
        tracklet_info = load_json(output_track_path)
        aggregated_embeddings = np.load(output_emb_path)
    else:
        # Aggregate multiple detections per tracklet into single embedding
        aggregated_embeddings, tracklet_info = aggregate_embeddings(all_person_data)

        # Cache results
        save_json(output_track_path, tracklet_info)
        np.save(output_emb_path, aggregated_embeddings)

    # ----------------------------------------
    # Step 3: Clustering and Catalogue Generation
    # ----------------------------------------
    print("\n" + "-" * 40)
    print(f"3. Clustering and Catalog generating...")
    print("-" * 40)

    output_catalogue_path = os.path.join(output_dir, "identity_catalogue.json")

    # Check if cached catalogue exists
    if os.path.exists(output_catalogue_path) and load_data:
        print(f"Loading identity catalogue.")
        catalogue = load_json(output_catalogue_path)
    else:
        # Cluster tracklets to identify unique persons across clips
        catalogue = generate_catalogue(aggregated_embeddings, tracklet_info)

        # Cache results
        save_json(output_catalogue_path, catalogue)
        print(f"Catalogue generated at {output_catalogue_path}")

    print(f"  Total unique persons: {len(catalogue)}")

    # ----------------------------------------
    # Step 4: Generate Annotated Videos
    # ----------------------------------------
    print("\n" + "-" * 30)
    print(f"4. Generate visualizations...")
    print("-" * 30)
    generate_visualizations(video_dir, catalogue, all_person_data, output_dir)

    return catalogue


def generate_visualizations(video_dir, catalogue, person_data, output_dir):
    """
    Creates annotated videos showing global person IDs on each detection.

    Overlays bounding boxes and person IDs on the original videos to visualize
    the re-identification results. This helps verify that the same person is
    consistently labeled across different clips.

    Args:
        video_dir (Path): Directory containing original video files
        catalogue (dict): Identity catalogue mapping person IDs to appearances
        person_data (list): Raw detection data with embeddings
        output_dir (Path): Directory for saving annotated videos

    Outputs:
        Creates annotated videos in: {output_dir}/visualizations/{clip_id}_annotated.mp4
    """
    # Get all video files in sorted order
    video_files = sorted(video_dir.glob('*.mp4'))

    # Create subdirectory for visualization outputs
    output_vis_dir = Path(output_dir, 'visualizations')

    # Process each video
    for video_file in video_files:
        clip_id = video_file.stem
        print(f"Generating visualizations for {video_file}...")

        # Create annotator and generate video with person ID labels
        annotator = VideoAnnotator(
            video_file,
            clip_id,
            catalogue,
            person_data,
            show_detections=0,  # Don't display real-time
            output_vis_dir=output_vis_dir
        )
        annotator.annotate_video()


def run_scene_classification(video_dir, catalogue, output_dir, device='cpu', sample_rate=1):
    """
    Classifies each video clip as 'normal' or 'crime' scene.

    Uses the person identity catalogue along with video analysis to determine
    if suspicious activities are occurring. The classifier considers:
    - Person movements and interactions
    - Scene context and environment
    - Behavioral patterns

    Args:
        video_dir (Path): Directory containing video files
        catalogue (dict): Person identity catalogue from Re-ID pipeline
        output_dir (Path): Directory for saving classification results
        device (str): Device for inference ('cuda' or 'cpu')
        sample_rate (int): Process every Nth frame

    Returns:
        None (saves results to JSON file)

    Outputs:
        Creates scene_labels.json with structure:
        {
            "dataset_summary": {
                "total_clips": N,
                "normal_count": X,
                "crime_count": Y
            },
            "clips": [
                {
                    "clip_id": "clip1",
                    "label": "crime",
                    "confidence_score": 0.85,
                    "justification": "Explanation of classification"
                },
                ...
            ]
        }
    """
    # Containers for results
    scene_labels = []
    video_files = sorted(video_dir.glob('*.mp4'))

    # ========================================
    # PART B: Scene Classification
    # ========================================
    print("\n" + "=" * 60)
    print("PART B: SCENE CLASSIFICATION")
    print("=" * 60)

    # Initialize summary structure
    summary = {
        'dataset_summary': {
            'total_clips': len(video_files),
            'normal_count': 0,
            'crime_count': 0
        },
        'clips': []
    }

    # Process each video clip
    for video_file in video_files:
        clip_id = video_file.stem

        # Load detection data for this clip
        detection_path = os.path.join(outputs_dir, f"detections_{clip_id}.json")
        detections = load_json(detection_path)

        print(f"Classifying scene for {video_file}...")

        # Initialize scene classifier
        classifier = SceneClassifier(
            video_file,
            clip_id,
            catalogue,
            model_device=device,
            sample_rate=sample_rate,
            show_detections=0,  # Don't display real-time
            person_data=detections
        )

        # Analyze the scene and get classification
        label_info = classifier.analyze_scene()

        # Store results
        scene_labels.append(label_info)
        summary['clips'].append(label_info)

        # Update counters based on classification
        if label_info['label'] == 'crime':
            summary['dataset_summary']['crime_count'] += 1
        else:
            summary['dataset_summary']['normal_count'] += 1

    # Save classification results to JSON
    output_labels_path = os.path.join(output_dir, "scene_labels.json")
    save_json(output_labels_path, summary)
    print(f"Scene labels generated at {output_labels_path}")

    # ========================================
    # Print Summary Report
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nDataset Statistics:")
    print(f"  Total clips: {summary['dataset_summary']['total_clips']}")
    print(f"  Normal scenes: {summary['dataset_summary']['normal_count']}")
    print(f"  Crime scenes: {summary['dataset_summary']['crime_count']}")

    print(f"\nPer-Clip Results:")
    for clip in summary['clips']:
        print(f"\n  Clip {clip['clip_id']}: {clip['label'].upper()}")
        print(f"    Confidence: {clip['confidence_score']:.2f}")
        print(f"    Justification: {clip['justification']}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()


if __name__ == "__main__":
    """
    Main entry point for the Person Re-ID and Scene Classification pipeline.

    Command-line usage examples:
        # Basic usage with defaults
        python main.py

        # Custom video directory and output location
        python main.py --video_dir ./videos --output_dir ./results

        # Use GPU and process every 5th frame
        python main.py --device cuda --sample_rate 5

        # Load cached data to skip reprocessing
        python main.py --load_data True

        # Set random seed for reproducibility
        python main.py --seed 123
    """

    # ========================================
    # Parse Command-Line Arguments
    # ========================================
    parser = argparse.ArgumentParser(
        description='Person Re-ID and Scene Classification Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show defaults in help
    )

    # Input/Output paths
    parser.add_argument(
        '--video_dir',
        type=str,
        default='..\\data\\Videos',
        help='Directory containing video files (.mp4)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='..\\outputs',
        help='Directory for output files (detections, catalogue, visualizations)'
    )

    # Processing parameters
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=1,
        help='Process every Nth frame (1=all frames, 5=every 5th frame). Higher values = faster but less accurate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cuda', 'cpu'],
        help='Device to use for model inference (cuda requires GPU)'
    )

    # Reproducibility and caching
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (affects clustering and sampling)'
    )
    parser.add_argument(
        '--load_data',
        type=bool,
        default=0,
        help='Load previously generated cached data instead of reprocessing (0=False, 1=True)'
    )

    # Parse arguments
    args = parser.parse_args()

    # ========================================
    # Setup and Initialization
    # ========================================

    # Set random seed for reproducibility
    # Affects: numpy operations, clustering, any random sampling
    set_seed(args.seed)

    # Create output directory if it doesn't exist
    outputs_dir = Path(args.output_dir)
    outputs_dir.mkdir(exist_ok=True)

    # Locate input videos
    videos_dir = Path(args.video_dir)

    # ========================================
    # PART A: Person Re-Identification
    # ========================================

    # Check if we can load cached person catalogue
    person_catalogue_path = os.path.join(outputs_dir, "identity_catalogue.json")

    if os.path.exists(person_catalogue_path) and args.load_data:
        # Load pre-computed catalogue (skips entire Re-ID pipeline)
        print(f"Loading person identity catalogue...")
        person_catalogue = load_json(person_catalogue_path)
    else:
        # Run full Re-ID pipeline:
        # 1. Detect and track persons in each video
        # 2. Aggregate embeddings per tracklet
        # 3. Cluster tracklets to identify unique persons
        # 4. Generate annotated visualization videos
        person_catalogue = run_person_reid(
            videos_dir,
            outputs_dir,
            args.device,
            args.sample_rate,
            args.load_data
        )

    # ========================================
    # PART B: Scene Classification
    # ========================================

    # Use the person catalogue to help classify scenes
    # The catalogue provides context about who appears where and when,
    # which can inform whether a scene contains suspicious activity
    run_scene_classification(
        videos_dir,
        person_catalogue,
        outputs_dir,
        args.device,
        args.sample_rate
    )
