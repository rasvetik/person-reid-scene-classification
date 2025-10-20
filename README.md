# Person Re-Identification & Scene Classification

This repository contains the solution for the take-home task involving person re-identification and scene classification across a set of video clips.

## Approach

### Part A: Person Identity Catalogue (Cross-Clip)

1.  **Person Detection and Tracking**: We use a pre-trained YOLO11n model (`yolov11n.pt`) from the `ultralytics` library. It detects and assigns temporary track IDs to persons within each video clip.
2.  **Feature Extraction**: For each tracked person instance (cropped from the frame), we extract a feature embedding. This is done using a pre-trained ResNet50 model, which is fine-tuned for image classification but provides robust feature representations suitable for re-identification. The final fully connected layer is removed to get a feature vector instead of a class label.
3.  **Embedding Aggregation**: For each tracklet (a continuous track of a person within a single clip), we average the feature embeddings across all its frames to create a single, more stable representation of that person.
4.  **Cross-Clip Clustering**: We perform unsupervised clustering (using `Agglomerative clustering` with cosine similarity) on all aggregated tracklet embeddings from all videos. This groups tracklets belonging to the same individual, regardless of the clip they appear in.
5.  **Catalogue Generation**: A unique global ID is assigned to each cluster. The output is a JSON file mapping each global ID to its appearances across different clips, including frame ranges.

### Part B: Scene Labelling Per Clip

1.  **Multi-Model Detection**: There are three detectors run in parallel and their outputs are fused by timestamp and bounding‑box overlap.
- General detector: identifies people and common objects such as bags and loose.
- Weapon detector: specifically flags knives, guns, and other potential weapons.
- Violence detector: detects aggressive interactions, fighting gestures, and rapid contact.
2.  **Behavioral analysis (state machine)**:  
- A deterministic state machine tracks each person and their interactions with objects through clear states: Idle → Picked Item → Bagged Item → Exit Without Checkout.
- Transitions are driven by detector events and tracked object associations; a transition from Bagged Item to Exit Without Checkout is scored as a suspicious event and triggers further verification
3. **Heuristic classification rule**:
- Object‑based rule: label as suspicious if a weapon is detected in a person’s hands or immediate vicinity.
- Behavior‑based rule: label as suspicious if a person performs a theft pattern such as picking an item, placing it into a bag, and exiting without passing a checkout zone.
- Spatial reasoning rule: use defined zones (checkout, entrance/exit) to determine intent and confirm suspicious exits.
- Final scene score combines object, behavior, and spatial signals with predefined weights to produce a binary label.
4.  **Justification and reportin**: 
- A “crime” label includes concise evidence: timestamps, relevant frame ranges, detector types, and the global person ID(s) from Part A.
- A “normal” label states the absence of weapon detections, theft patterns, or suspicious exit events.
- All justifications reference the exact moments and identities used to make the decision

### Common failure modes
- **False alarms and missed detections**: Imperfect person detection produces ghost tracklets and also fails to detect real people, which pollutes the feature space and contaminates the identity catalogue.
- **Track fragmentation and identity splitting**: Continuous appearances get broken into multiple short tracklets, causing a single person to be represented by several catalogue entries.
- **Identity merging**: Distinct people are incorrectly grouped as the same individual, creating false positive matches.
- **Identity splitting**: One person is assigned multiple global identities across clips, creating false negative matches.
- **Clustering failures**: Noisy or insufficient embeddings lead to over-clustering or under-clustering, driving both merging and splitting errors.
- **Lack of contextual reasoning**: The system relies on visual cues alone and cannot infer intent or context, so ambiguous interactions are misclassified based only on proximity or brief contact.

## Requirements

-   Python 3.8+ 
-   `ultralytics`, `torch`, `torchvision`, `opencv-python`, `scikit-learn`, `numpy`, `scipy`, `pillow`, `tqdm`
-   `YOLOv12-violence-detector-roboflow` model trained to classify Violence vs NonViolence
-   `All_weapon.pt` model detects a wide range of weapons, including guns, knives, swords, sticks, axes, and more
-   `models/yolo/links_for_download_models.txt` file consist of the download links for these models 
## How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/rasvetik/person-reid-scene-classification.git
    cd person-reid-scene-classification
    ```
2.  **Create a virtual environment**:
    ```bash
    python -m venv myvenv
    ```
3.  **Activate the environment**:
    ```bash
    source idvenv/bin/activate # on Linux 
    # myvenv\Scripts\activate  # on Windows
    ```
4. **Upgrade pip**:
    ```bash
    pip install --upgrade pip
    ```
5. **Install PyTorch**: (visit https://pytorch.org for your specific system)
     ```bash
    # Example for GPU (CUDA 11.8):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```
6.  **Set up the environment**:
    ```bash
    pip install -r requirements.txt
    ```    
5.  **Input Video Data**: Place your video files `1.mp4`, `2.mp4`, `3.mp4`, `4.mp4` in the `data/videos/` directory.
6.  **Running the Pipeline**:
    ```bash
    # Run full pipeline with default settings
    python src/main.py
    ```
This will:
* Process all videos in `data/Videos/`
* Detect and track persons
* Generate identity catalogue
* Classify scenes as normal/crime
* The script will process the videos and generate the output files in the `outputs/` directory.

## Outputs

-   `detections{clip_id}.json`:	Raw detection data per clip
-   `tracklet_info.json`:	Aggregated tracklet metadata
-   `aggregated_embeddings.npy`:	Aggregated features as numpy file
-   `identity_catalogue.json`: Contains the person identity catalogue.
-   `scene_labels.json`: Contains the per-clip scene labels and justifications.

## Basic Usage with Options
```bash 
# Use GPU acceleration (if available)
python main.py --device cuda

# Process every 5th frame (faster, less accurate)
python main.py --sample_rate 5

# Specify custom paths
python main.py --video_dir ./my_videos --output_dir ./my_results

# Load cached data (skip reprocessing)
python main.py --load_data True

# Set random seed for reproducibility
python main.py --seed 42
## Limitations and Future Work
```
## Command-Line Arguments

| Argument | Type | Default| Description | 
|----------|------|--------|-------------|
|`--video_dir`	|str	|`../data/Videos`	|Directory containing input videos|
|`--output_dir`	|str	|`../outputs	`|Directory for output files|
|`--sample_rate`	|int	|`1`	|Process every Nth frame (`1`=all frames)|
|`--device`	|str	|`cpu`	|Device for inference (`cuda` or `cpu`)|
|`--seed`	|int	|`42`	|Random seed for reproducibility|
|`--load_data`	|bool	|`False`	|Load cached results instead of reprocessing|

## Running Individual Modules
**1. Person Re-Identification Only**
```bash
    # From main.py, comment out scene classification section
    # Or run video_processor.py directly:
    cd src
    python video_processor.py
```
Edit the` __main__` section in `video_processor.py` to customize.

**2. Scene Classification Only**
```bash
    # Requires pre-computed detection data and catalogue
    cd src
    python scene_classifier.py
```
Edit the `__main__` section to specify which clip to process.

**3. Visualization Only**
```bash
    # Requires detection data and catalogue
    cd src
    python video_annotator.py
```
## Configuration Options
### Detection Thresholds
Adjust in source files:

`video_processor.py`
```bash
    # Detection confidence threshold
    results = self.yolo_model.track(frame, persist=True, classes=[0], verbose=False, iou=0.3, conf=0.3)
    # Increase conf=0.5 for fewer false positives
```
`scene_classifier.py`
```bash
    # Weapon detection threshold
    if obj_conf > 0.45:  # Adjust between 0.3-0.7
```
`scene_classifier.py`
```bash
    # Violence detection threshold
    if vio_cls == 0 and vio_conf > 0.5:  # Adjust between 0.4-0.7
```
### Clustering Parameters
`main.py`
```bash
    # Person Re-ID clustering threshold
    agg_clust = AgglomerativeClustering(n_clusters=None, 
                                        distance_threshold=0.2,  # Adjust 0.1-0.4
                                        metric="cosine", 
                                        linkage="average")
```
- Lower threshold (0.1): More unique persons (may over-segment)
- Higher threshold (0.4): Fewer unique persons (may under-segment)

## Expected Runtime
* *GPU (NVIDIA RTX 3060)*: ~2-5 minutes for 4 clips
* *CPU*: ~10-20 minutes for 4 clips

## Limitations
-   **Scene Classification Ambiguity**: The heuristic-based scene classification can be ambiguous. For example, close proximity does not always mean a crime. A more robust solution would involve fine-tuning an action recognition model on a specific set of behaviors.
-   **Re-ID Robustness**: Occlusions, poor lighting, and camera angle changes can affect person re-identification. Using a stronger Re-ID model (e.g., a dedicated OSNet or TransReID model) could improve results.
-   **Computational Cost**: The current approach can be slow, especially for long videos. Optimizations like processing only keyframes or using a lighter-weight backbone could improve performance.
-   **Dataset Specificity**: The model weights are pre-trained on general datasets. Fine-tuning them on task-specific data would yield better results.

## Assumptions

-   The person re-identification model assumes that the appearance features are discriminative enough to distinguish between individuals.
-   The definition of a "crime" is based on a simple set of rules derived from common-sense assumptions (e.g., detecting weapons or fighting behavior).
-   `Agglomerative clustering` parameters (`distance_threshold` and `metric`) are chosen based on reasonable defaults and may require tuning for optimal performance.

## License and Citation
MIT License - See LICENSE file for details
```bash
# If you use this code in your research, please cite:

@software{person-reid-scene-classification,
  author = {Sveta Raboy},
  title = {Person Re-Identification and Crime Scene Classification System},
  year = {2025},
  url = {https://github.com/rasvetik/person-reid-scene-classification}
}
```

---

**Author**: Svetlana Raboy  
**Date**: 20/10/2025  
**Version**: 1.0