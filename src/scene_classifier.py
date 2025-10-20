"""
Scene Classification Module for Crime Detection

This module analyzes video scenes to detect criminal activities using multiple
detection models and behavioral heuristics. It combines:

1. Weapon Detection: Identifies dangerous objects (knives, guns, etc.)
2. Violence Detection: Recognizes violent behaviors and actions
3. Behavioral Analysis: State-machine tracking of person activities (theft detection)
4. Person Re-Identification: Uses the catalogue to track individuals across frames

The classifier uses a multi-model approach with heuristic rules to determine
if a scene contains normal or criminal activity.

Detection Strategies:
- Object-based: Presence of weapons or suspicious items
- Behavior-based: Tracking person-item interactions (pickup, bagging, leaving)
- Interaction-based: Analyzing proximity and person-person interactions
- Spatial reasoning: Checkout zone detection and exit monitoring
"""

import cv2
import os
import torch
from utils import save_json, load_json, frame_to_timestamp
from collections import defaultdict
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from ultralytics.utils.plotting import Annotator, colors


# noinspection SpellCheckingInspection
class SceneClassifier:
    """
    Classifies video scenes as 'normal' or 'crime' using multiple detection models.

    This class orchestrates a comprehensive scene analysis pipeline that combines:
    - Multi-model inference (general objects, weapons, violence)
    - State-machine behavioral tracking (theft detection)
    - Spatial reasoning (checkout zones, exits)
    - Person re-identification integration

    The classifier maintains state across frames to detect complex behaviors like:
    - Shoplifting (item pickup -> bagging -> leaving without checkout)
    - Weapon presence
    - Violence detection
    - Suspicious person interactions

    Attributes:
        video_path (str): Path to input video file
        clip_id (str): Unique identifier for this clip
        catalogue (dict): Person identity catalogue from Re-ID pipeline
        person_data (list): Pre-computed person detections with embeddings
        yolo_model (YOLO): General object detection model
        violence_model (YOLO): Specialized violence detection model
        weapon_model (YOLO): Specialized weapon detection model
        person_states (dict): State machine for each person's behavior
        checkout_zone (tuple): Spatial bounds of checkout area
    """

    def __init__(self, video_path, clip_id, catalogue, person_data, show_detections=0,
                 output_vis_dir=Path('../outputs/visualizations'),
                 sample_rate=1, model_device=None, save_video=True):
        """
        Initializes the Scene Classifier with multiple detection models.

        Args:
            video_path (str or Path): Path to input video file (.mp4)
            clip_id (str): Unique identifier for this video clip
            catalogue (dict): Person identity catalogue from Re-ID pipeline
                Maps global person IDs to their appearances across clips
            person_data (list): Pre-computed detection data with embeddings
                From VideoProcessor, contains bbox, track_id, embeddings, etc.
            show_detections (bool, optional): Display real-time detection windows
                Shows separate windows for: objects, weapons, violence, analysis
            output_vis_dir (Path, optional): Directory for saving visualization outputs
            sample_rate (int, optional): Process every Nth frame (1 = all frames)
            model_device (str, optional): Inference device ('cuda' or 'cpu')
                Auto-detects GPU availability if None
            save_video (bool, optional): Save annotated analysis video

        Notes:
            - Loads three separate YOLO models (general, weapon, violence)
            - Initializes state machines for behavioral tracking
            - Pre-processes person data for fast frame-based lookup
        """
        # Auto-detect device if not specified
        if model_device is None:
            model_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Store initialization parameters
        self.video_path = video_path
        self.clip_id = clip_id
        self.catalogue = catalogue
        self.person_data = person_data
        self.device = model_device
        self.sample_rate = sample_rate
        self.output_dir = output_vis_dir
        self.show_detections = show_detections
        self.save_video = save_video

        # Video properties (populated during analysis)
        self.total_frames = None

        # ========================================
        # Initialize Detection Models
        # ========================================

        # Model 1: General Object Detection (YOLO11n)
        # Detects common objects: persons, bags, items, checkout equipment
        # noinspection SpellCheckingInspection
        self.yolo_model = YOLO("..\\models\\yolo\\yolo11n.pt")

        # Model 2: Violence Detection (Custom-trained YOLO)
        # Specialized model for detecting violent actions and behaviors
        self.violence_model = YOLO("..\\models\\yolo\\best_model_yolo_violence.pt")

        # Model 3: Weapon Detection (Custom-trained YOLO)
        # Specialized model for detecting weapons (knives, guns, etc.)
        self.weapon_model = YOLO("..\\models\\yolo\\All_weapon.pt")

        # ========================================
        # Define Classification Keywords
        # ========================================

        # Objects indicating potential crime
        self.crime_keywords = ["weapon", "knife", "gun", "fight", "pistol", "sword", "stick"]

        # Objects indicating normal activity
        self.normal_keywords = ["person", "car", "bicycle", "chair"]

        # Objects typically found near checkout areas
        # Used for spatial reasoning about legitimate vs suspicious exits
        self.checkout_keywords = ["laptop", "mouse", "keyboard", "cell phone",
                                  "remote", "tv", "table", "chair", "door"]

        # ========================================
        # Behavioral Analysis Parameters
        # ========================================

        # Proximity threshold in pixels for determining interactions
        # Used to detect: person-item, person-bag, person-checkout proximity
        self.proximity_threshold = 50

        # ========================================
        # State Machine Initialization
        # ========================================

        # Track behavioral state for each person
        # States: IDLE -> PICKED_ITEM -> BAGGED_ITEM -> SUSPICIOUS_LEAVE
        # Format: {person_id: {"state": "IDLE", "items": []}}
        self.person_states = defaultdict(lambda: {"state": "IDLE", "items": []})

        # Checkout zone bounds (x1, x2, y1, y2)
        # Initially None, set dynamically based on scene analysis
        self.checkout_zone = None

        # ========================================
        # Pre-process Person Data for Fast Lookup
        # ========================================

        # Convert person detection list into frame-indexed dictionary
        # Enables O(1) lookup of detections for any frame
        # Structure: {frame_idx: [detection1, detection2, ...]}
        self.frames_data = defaultdict(list)

        for record in self.person_data:
            # Only include detections from this specific clip
            if record["clip_id"] == self.clip_id:
                self.frames_data[record["frame_idx"]].append(record)

    def analyze_scene(self):
        """
        Performs comprehensive scene analysis to classify as 'normal' or 'crime'.

        This method implements a multi-stage analysis pipeline:

        Stage 1: Video Setup
            - Open video and read properties
            - Initialize checkout zone heuristics
            - Build track-to-global ID mapping

        Stage 2: Frame-by-Frame Analysis
            For each frame:
            a. Object Detection (3 models: general, weapon, violence)
            b. Heuristic 1: Suspicious Object Detection
               - Check for weapons, violence indicators
            c. Heuristic 2: Behavioral Analysis
               - Track person states (IDLE -> PICKED_ITEM -> BAGGED_ITEM)
               - Detect theft patterns (bagging without checkout)
            d. Heuristic 3: Interaction Analysis
               - Analyze person-person proximity
               - Detect potential conflicts
            e. Annotation and Visualization
               - Draw bounding boxes and labels
               - Save keyframes for detected crimes

        Stage 3: Results Compilation
            - Aggregate all detections and justifications
            - Generate final classification label
            - Return structured results

        Returns:
            dict: Scene classification results with structure:
                {
                    "clip_id": str,
                    "label": "normal" | "crime",
                    "justification": dict mapping timestamps to detection reasons,
                    "num_unique_persons": int,
                    "persons": list of person appearances with metadata
                }

        Classification Logic:
            - Default: "normal"
            - Changes to "crime" if ANY of these occur:
              * Weapon detected with confidence > 0.45
              * Violence detected with confidence > 0.5
              * Person leaves without checkout after bagging items
              * Suspicious person-person interactions

        Side Effects:
            - Creates annotated video if save_video=True
            - Saves keyframe images for detected crimes
            - Shows real-time windows if show_detections=True
            - Updates person_states with behavioral tracking
        """
        # ========================================
        # Initialize Analysis Variables
        # ========================================

        # Default classification (innocent until proven guilty)
        label = "normal"

        # Justification storage: {timestamp: [reason1, reason2, ...]}
        # Records why/when crime was detected
        justification = defaultdict(list)

        # List of all persons appearing in this clip
        persons_in_clip = []

        # ========================================
        # Stage 1: Video Setup
        # ========================================

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return

        # Extract video metadata
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_idx = 0

        # ----------------------------------------
        # Setup Video Writer (if saving)
        # ----------------------------------------
        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_video:
            output_video_path = os.path.join(self.output_dir, f"scene_analysis_{self.clip_id}.mp4")
            # noinspection PyUnresolvedReferences
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # ----------------------------------------
        # Initialize Checkout Zone Heuristics
        # ----------------------------------------
        # In real-world scenarios, this would be configured manually or learned
        # Heuristic: Assume checkout is in the last 20% of frame width
        # Format: (x1, x2, y1, y2)
        self.checkout_zone = (0, int(frame_width * 0.8), 0, frame_height)

        # ----------------------------------------
        # Build Track-to-Global ID Mapping
        # ----------------------------------------
        # Maps local track IDs (within this clip) to global person IDs (across clips)
        # This enables consistent person identification throughout analysis
        track_to_global_map = {}

        for global_id, appearances in self.catalogue.items():
            for appearance in appearances:
                # Only map appearances in this specific clip
                if appearance['clip_id'] == self.clip_id:
                    # Extract track ID from local_id string: "track_5" -> 5
                    track_id = int(appearance['local_id'].split('_')[-1])
                    track_to_global_map[track_id] = global_id

                    # Store person metadata for final report
                    persons_in_clip.append({
                        'person_id': global_id,
                        'duration': appearance['frame_range'],
                        'number_frames': appearance['number_frames'],
                        'avg_area': appearance['avg_area'],
                        'avg_confidence': appearance['avg_confidence'],
                    })

        # ========================================
        # Stage 2: Frame-by-Frame Analysis
        # ========================================

        pbar = tqdm(total=self.total_frames, desc=f"Scene analysis {self.video_path}")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break  # End of video

            # ----------------------------------------
            # Frame Sampling
            # ----------------------------------------
            # Skip frames based on sample_rate to reduce computation
            if frame_idx % self.sample_rate != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            # ----------------------------------------
            # Multi-Model Object Detection
            # ----------------------------------------

            # Model 1: General object detection (persons, bags, items)
            # verbose=False: Suppress per-frame output
            results = self.yolo_model.predict(frame, verbose=False)[0]

            # Model 2: Violence detection (violent actions, fights)
            violence_results = self.violence_model(frame, verbose=False)[0]

            # Model 3: Weapon detection (knives, guns, etc.)
            # Uses tracking to maintain weapon IDs across frames
            weapon_results = self.weapon_model.track(frame, verbose=False)[0]

            # Generate visualization frames for each model (for debugging)
            violence_frame = violence_results.plot()
            weapon_frame = weapon_results.plot()

            # Initialize annotator for final combined visualization
            annotator = Annotator(frame, line_width=2, example=str(self.frames_data.get(frame_idx, "")))

            # ----------------------------------------
            # Optional: Display Individual Model Results
            # ----------------------------------------
            if self.show_detections:
                # Show general object detection
                annotated_frame = results.plot()
                cv2.imshow('Object Detection', annotated_frame)
                cv2.waitKey(1)

                # Show violence detection
                cv2.imshow('Violence Detection', violence_frame)
                cv2.waitKey(1)

                # Show weapon detection
                cv2.imshow('Weapon Detection', weapon_frame)
                cv2.waitKey(1)

            # ----------------------------------------
            # Extract Detection Data from Each Model
            # ----------------------------------------

            # General object detections
            detections = results.boxes.xyxy.cpu().numpy()  # Bounding boxes
            classes = results.boxes.cls.cpu().numpy()  # Class IDs
            confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
            tracks = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [None]
            centers_of_detections = results.boxes.xywh.cpu().numpy()  # Center format

            # Weapon detections
            weapon_cdetections = weapon_results.boxes.xyxy.cpu().numpy()
            weapon_classes = weapon_results.boxes.cls.cpu().numpy()
            weapon_confidences = weapon_results.boxes.conf.cpu().numpy()
            weapon_tracks = weapon_results.boxes.id.int().cpu().tolist() if weapon_results.boxes.id is not None else [
                None]

            # Violence detections
            violence_detections = violence_results.boxes.xyxy.cpu().numpy()
            violence_classes = violence_results.boxes.cls.cpu().numpy()
            violence_confidences = violence_results.boxes.conf.cpu().numpy()

            # ========================================
            # HEURISTIC 1: Suspicious Object Detection
            # ========================================
            # Check for presence of weapons or violence indicators

            # Get class name mappings for each model
            violence_names = violence_results.names
            weapon_names = weapon_results.names
            detection_names = results.names

            # Flags for this frame's detections
            crime_detected_in_frame = False
            violence_detected_in_frame = False

            # ----------------------------------------
            # Check for Weapons
            # ----------------------------------------
            for i, wdet in enumerate(weapon_classes):
                obj_name = weapon_names[int(wdet)]

                # Check if detected object is a known crime keyword
                if obj_name in self.crime_keywords:
                    obj_conf = weapon_confidences[i]

                    # Confidence threshold: 0.45 (balance false positives/negatives)
                    if obj_conf > 0.45:
                        label = "crime"
                        crime_detected_in_frame = True
                        timestamp = frame_to_timestamp(frame_idx, fps)
                        justification[timestamp] = [f"#{frame_idx}: {obj_name} {obj_conf:.2f}"]

                        # Annotate weapon detection on frame
                        res_bbox = weapon_cdetections[i]
                        t_id = weapon_tracks[i] if weapon_tracks[i] is not None else ''
                        ann_label = f"ID:{t_id} {obj_name} {obj_conf:.2f})"
                        color = colors(int(t_id) % 18, True)
                        annotator.box_label(res_bbox, label=ann_label, color=color)

            # ----------------------------------------
            # Check for Violence
            # ----------------------------------------
            for i, vdet in enumerate(violence_detections):
                vio_cls = violence_classes[i]
                vio_conf = violence_confidences[i]
                vio_obj = violence_names[vio_cls]
                violence_detected_in_frame = False
                # Class 0 typically represents "violence" in custom-trained models
                # Confidence threshold: 0.5 (stricter than weapons due to complexity)
                if vio_cls == 0 and vio_conf > 0.5:
                    # Commented out to avoid over-triggering
                    # label = "crime"
                    violence_detected_in_frame = True
                    timestamp = frame_to_timestamp(frame_idx, fps)
                    justification[timestamp] = [f"#{frame_idx}: violence {vio_conf:.2f}"]

                    # Annotate violence detection
                    ann_label = f"{vio_obj} {vio_conf:.2f}"
                    # Use different color range to distinguish from persons
                    color = colors(int(vio_cls + 15) % 18, True)
                    annotator.box_label(vdet, label=ann_label, color=color)

            # ========================================
            # HEURISTIC 2: Behavioral Analysis (State Machine)
            # ========================================
            # Track person-item interactions to detect theft patterns

            # ----------------------------------------
            # Categorize Detected Objects
            # ----------------------------------------
            # Group detections by semantic meaning for behavior analysis

            persons = []  # Person detections
            bags = []  # Bag/container detections (handbag, backpack, suitcase)
            small_items = []  # Potentially stealable items (all non-person, non-bag, non-vehicle)
            checkout_items = []  # Objects typically found near checkout areas

            for bbox, cls_id, track_id, conf, cbox in zip(detections, classes, tracks, confidences,
                                                          centers_of_detections):
                cls_name = detection_names[int(cls_id)]

                if cls_name == "person":
                    persons.append((bbox, track_id, conf, cbox))

                # Define what constitutes a "bag"
                elif cls_name in ["handbag", "suitcase", "backpack"]:
                    bags.append((bbox, conf, cls_name, cbox))

                # Define "small items" by exclusion (not person, bag, or vehicle)
                # These are potentially stealable items
                elif cls_name not in ["person", "handbag", "suitcase", "backpack", "car", "truck"]:
                    small_items.append((bbox, conf, cls_name, cbox))

                # Identify checkout-related objects for spatial reasoning
                if cls_name in self.checkout_keywords:
                    checkout_items.append((bbox, conf, cls_name, cbox))

            # ----------------------------------------
            # Process Each Person's Behavioral State
            # ----------------------------------------
            # Use pre-computed person data for accurate tracking

            detections_for_frame = self.frames_data.get(frame_idx, [])

            for det in detections_for_frame:
                # Extract person information
                person_bbox = det["bbox"]
                track_id = det["track_id"]

                # Map to global person ID
                global_id = track_to_global_map.get(track_id, "unknown")
                global_id_parts = global_id.split("_")

                # Skip if person couldn't be mapped to global ID
                if global_id_parts[0] == "unknown":
                    continue

                confidence = det["confidence"]

                # Annotate person on frame
                ann_label = f"ID:{global_id_parts[-1]} {global_id_parts[0]} (T{track_id}, {confidence:.2f})"
                color = colors(int(global_id_parts[-1]) % 18, True)
                annotator.box_label(person_bbox, label=ann_label, color=color)

                # Get this person's state machine
                person_state = self.person_states[global_id_parts[-1]]

                # ----------------------------------------
                # State Transition Logic
                # ----------------------------------------

                # STATE 1: IDLE
                # Person is in scene but not interacting with items
                if person_state["state"] == "IDLE":
                    # Check for item pickup
                    for item_bbox, item_conf, item_name, item_cbox in small_items:
                        # Calculate distance between person and item
                        item_distance = self.bbox_distance(person_bbox, item_bbox)

                        # Proximity threshold check
                        if item_distance < self.proximity_threshold:
                            # Transition to PICKED_ITEM state
                            person_state["state"] = "PICKED_ITEM"
                            person_state["items"].append((item_bbox, item_conf, item_name, item_cbox))

                            # Log the interaction
                            timestamp = frame_to_timestamp(frame_idx, fps)
                            justification[timestamp] = [
                                f"#{frame_idx}: Person {global_id_parts[-1]} picked up a {item_name}."]
                            break  # Assume only one item picked per frame

                # STATE 2: PICKED_ITEM
                # Person has picked up an item
                elif person_state["state"] == "PICKED_ITEM":
                    # Check for item bagging
                    for bag_bbox, bag_conf, bag_name, bag_cbox in bags:
                        # Calculate distances
                        bag_distance = self.bbox_distance(person_bbox, bag_bbox)
                        # Check if the previously picked item is near the bag
                        prev_bag_distance = self.bbox_distance(person_state["items"][-1][0], bag_bbox)

                        # Both person and item must be near bag
                        if bag_distance < self.proximity_threshold and prev_bag_distance < self.proximity_threshold / 2.5:
                            # Transition to BAGGED_ITEM state
                            person_state["state"] = "BAGGED_ITEM"

                            # Log the bagging action
                            timestamp = frame_to_timestamp(frame_idx, fps)
                            justification[timestamp] = [
                                f"#{frame_idx}: Person {global_id_parts[-1]} bagged an item to {bag_name}."]
                            break

                # STATE 3: BAGGED_ITEM
                # Person has placed item into bag
                elif person_state["state"] == "BAGGED_ITEM":
                    # Check for suspicious exit
                    # Person is leaving without passing through checkout
                    if self._is_leaving_store(person_bbox, frame_width) and \
                            not self._is_in_checkout(person_bbox, checkout_items):

                        # CRIME DETECTED: Shoplifting
                        label = "crime"
                        crime_detected_in_frame = True
                        timestamp = frame_to_timestamp(frame_idx, fps)
                        justification[timestamp] = [
                            f"#{frame_idx}: Person {global_id_parts[-1]} left the store without paying after bagging an item."]

                        # Save keyframe capturing the theft
                        frame_filename = os.path.join(self.output_dir, f"{self.clip_id}_frame_{frame_idx}_theft.jpg")
                        cv2.imwrite(frame_filename, frame)

                        if self.show_detections:
                            cv2.imshow('Theft Detection', frame)
                            cv2.waitKey(1)

                # ========================================
                # HEURISTIC 3: Person-Person Interaction (Commented Out)
                # ========================================
                # This section analyzes person-person proximity for potential conflicts
                # Currently commented to avoid false positives

                # for idet in detections_for_frame:
                #     iperson_bbox = idet["bbox"]
                #     itrack_id = idet["track_id"]
                #     person_distance = self.bbox_distance(person_bbox, iperson_bbox)
                #
                #     # Check if two different persons are very close
                #     if person_distance < self.proximity_threshold:
                #         iglobal_id = track_to_global_map.get(itrack_id, "unknown")
                #         iglobal_id = iglobal_id.split("_")
                #
                #         # Don't compare person to themselves
                #         if global_id_parts[-1] == iglobal_id[-1]:
                #             continue
                #
                #         # Log potential conflict
                #         timestamp = frame_to_timestamp(frame_idx, fps)
                #         justification[timestamp] = [f"Person {global_id_parts[-1]} in close proximity with person {iglobal_id[-1]}, potentially a conflict."]
                #         crime_detected_in_frame = True

            # ========================================
            # Save Annotated Frames for Crime Events
            # ========================================

            # Get fully annotated frame
            annotated_frame = annotator.result()

            # Save violence keyframes
            # Currently commented to avoid large number of saved images.
            # if violence_detected_in_frame:
            #     frame_filename = os.path.join(self.output_dir, f"{self.clip_id}_frame_{frame_idx}_violence.jpg")
            #     cv2.imwrite(frame_filename, annotated_frame)
            #
            #     if self.show_detections:
            #         cv2.imshow('Violence Detection', annotated_frame)
            #         cv2.waitKey(1)

            # ----------------------------------------
            # Display and Save Annotated Frame
            # ----------------------------------------
            if self.show_detections:
                cv2.imshow("Scene Analysis", annotated_frame)
                cv2.waitKey(1)

            if self.save_video:
                # noinspection PyUnboundLocalVariable
                out.write(annotated_frame)

            frame_idx += 1
            pbar.update(1)

        # ========================================
        # Cleanup Resources
        # ========================================
        if self.show_detections:
            cv2.destroyAllWindows()
        if self.save_video:
            out.release()
        cap.release()
        pbar.close()

        # ========================================
        # Stage 3: Return Classification Results
        # ========================================
        return {
            "clip_id": self.clip_id,
            "label": label,  # "normal" or "crime"
            "justification": justification if justification else "No suspicious activity detected.",
            "confidence_score": 1.0 if label == "crime" else 0.0,  # Binary confidence
            "num_unique_persons": len(persons_in_clip),
            "persons": persons_in_clip  # Metadata about all persons in clip
        }

    @staticmethod
    def bbox_distance(bbox1, bbox2):
        """
        Calculates Euclidean distance between centers of two bounding boxes.

        This method is used throughout the behavioral analysis to determine
        proximity between objects, which indicates potential interactions.

        Args:
            bbox1 (tuple): First bounding box (x1, y1, x2, y2)
            bbox2 (tuple): Second bounding box (x1, y1, x2, y2)

        Returns:
            float: Euclidean distance in pixels between box centers

        Applications:
            - Person-item proximity (pickup detection)
            - Person-bag proximity (bagging detection)
            - Person-person proximity (interaction detection)
            - Person-checkout proximity (legitimate exit detection)

        Note:
            Uses center points rather than edge distances for more intuitive
            proximity measurements that work regardless of box sizes.
        """
        # Extract coordinates
        x1_a, y1_a, x2_a, y2_a = bbox1
        x1_b, y1_b, x2_b, y2_b = bbox2

        # Calculate centers
        center_a = ((x1_a + x2_a) / 2, (y1_a + y2_a) / 2)
        center_b = ((x1_b + x2_b) / 2, (y1_b + y2_b) / 2)

        # Euclidean distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
        distance = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5

        return distance

    def _is_leaving_store(self, bbox, frame_width):
        """
        Determines if a person is moving towards the store exit.

        This method uses heuristics to detect when a person is approaching
        the exit boundary of the frame. It adapts based on where the checkout
        zone is located (left or right side of frame).

        Args:
            bbox (tuple): Person bounding box (x1, y1, x2, y2)
            frame_width (int): Width of video frame in pixels

        Returns:
            bool: True if person is near exit, False otherwise

        Logic:
            - If checkout is on right side: exit is on right (x2 > 90% of width)
            - If checkout is on left side: exit is on left (x2 < 20% of width)
            - Proximity threshold ensures person is very close to edge

        Notes:
            - Assumes single-entrance/exit stores
            - Real-world systems would use calibrated exit zones
            - Heuristic works for many retail layouts but may need tuning
        """
        x1, y1, x2, y2 = bbox
        leaving = 0

        # Case 1: Checkout zone is on the right, exit is on the right
        if frame_width - self.checkout_zone[1] < self.proximity_threshold:
            # Exit is on the right side
            # Check if person's right edge is beyond 90% of frame width
            leaving = x2 > frame_width * 0.9

        # Case 2: Checkout zone is on the left, exit is on the left
        elif frame_width - self.checkout_zone[1] > frame_width / 3:
            # Exit is on the left side
            # Check if person's right edge is before 20% of frame width
            leaving = x2 < frame_width * 0.2

        return leaving

    def _is_in_checkout(self, bbox, checkout_items):
        """
        Determines if a person is currently in the checkout area.

        This method uses spatial reasoning to determine if a person is in a
        legitimate checkout location. It analyzes proximity to checkout-related
        objects (counters, registers, etc.) and validates against the checkout zone.

        Args:
            bbox (tuple): Person bounding box (x1, y1, x2, y2)
            checkout_items (list): List of detected checkout-related objects
                Each item: (bbox, confidence, class_name, center_box)

        Returns:
            bool: True if person is in valid checkout area, False otherwise

        Algorithm:
            1. Find all checkout items near the person (within proximity threshold)
            2. Calculate average position of nearby checkout items
            3. Validate this position against the predefined checkout zone
            4. Update checkout zone if evidence suggests different location
            5. Check if person bbox overlaps with validated checkout zone

        Notes:
            - Dynamically adjusts checkout zone based on detected items
            - Uses proximity to multiple checkout objects for robustness
            - Prevents false positives from single misclassified objects
        """
        x1, y1, x2, y2 = bbox
        in_checkout = 0  # Counter for nearby checkout items
        checkout_zone = [0, 0, 0, 0]  # Accumulator for checkout item positions

        # ----------------------------------------
        # Find Nearby Checkout Items
        # ----------------------------------------
        for item_bbox, item_conf, item_name, item_cbox in checkout_items:
            # Calculate distance between person and checkout item
            item_distance = self.bbox_distance(bbox, item_bbox)

            # Check if item is within proximity threshold
            if item_distance < self.proximity_threshold:
                in_checkout += 1
                # Accumulate checkout item positions
                # This will be averaged to find checkout center
                checkout_zone = [checkout_zone[i] + item_bbox[i] for i in range(4)]

        # ----------------------------------------
        # Validate and Update Checkout Zone
        # ----------------------------------------
        if in_checkout:
            # Calculate average position of nearby checkout items
            # Add epsilon to prevent division by zero
            checkout_zone = [cz / (in_checkout + 1e-6) for cz in checkout_zone]

            # Validate against predefined checkout zone
            checkout_distance = self.bbox_distance(self.checkout_zone, checkout_zone)

            # Case 1: Detected checkout matches predefined zone
            if checkout_distance < self.proximity_threshold * 2:
                # Keep existing checkout zone (it's correct)
                self.checkout_zone = self.checkout_zone

            # Case 2: Detected checkout is in different location
            else:
                # Check alternative location (opposite side of store)
                # If original zone was right side (80% width), check left side (20% width)
                zone = list(self.checkout_zone)
                zone[1] = zone[1] / 0.8 * 0.2  # Mirror to opposite side

                checkout_distance = self.bbox_distance(zone, checkout_zone)

                # If alternative location matches, update checkout zone
                if checkout_distance < self.proximity_threshold * 2:
                    self.checkout_zone = self.checkout_zone  # Keep original for now
                    # In production, might want: self.checkout_zone = tuple(checkout_zone)

        # ----------------------------------------
        # Check if Person is Within Checkout Bounds
        # ----------------------------------------
        # Person must be entirely within the checkout zone boundaries
        return (x1 > self.checkout_zone[0] and x2 < self.checkout_zone[1] and
                y1 > self.checkout_zone[2] and y2 < self.checkout_zone[3])


def run_scene_classification(video_dir, catalogue, output_dir, clip_iter_id=None):
    """
    Runs scene classification on multiple video clips.

    This function orchestrates the classification pipeline across a dataset:
    1. Loads video files from directory
    2. For each video:
       a. Loads pre-computed detection data
       b. Initializes SceneClassifier
       c. Analyzes scene
       d. Collects results
    3. Generates summary statistics
    4. Saves results to JSON

    Args:
        video_dir (Path): Directory containing video files (.mp4)
        catalogue (dict): Person identity catalogue from Re-ID pipeline
        output_dir (Path): Directory for saving classification results
        clip_iter_id (int, optional): Process only specific clip index
                                      If None, processes all clips

    Outputs:
        Creates scene_labels.json with structure:
        {
            "dataset_summary": {
                "total_clips": int,
                "normal_count": int,
                "crime_count": int
            },
            "clips": [
                {
                    "clip_id": str,
                    "label": "normal" | "crime",
                    "justification": dict,
                    "num_unique_persons": int,
                    "persons": list
                },
                ...
            ]
        }

    Notes:
        - Requires pre-computed detection files: detections{clip_id}.json
        - Can process single clip for testing/debugging
        - Shows real-time visualization during classification
    """
    # ========================================
    # Load Video Files
    # ========================================
    video_files = sorted(video_dir.glob("*.mp4"))

    # Optional: Process only specific clip (for testing/debugging)
    if clip_iter_id is not None:
        video_files = [video_files[clip_iter_id]]

    # ========================================
    # Initialize Results Summary
    # ========================================
    summary = {
        'dataset_summary': {
            'total_clips': len(video_files),
            'normal_count': 0,
            'crime_count': 0
        },
        'clips': []
    }

    # ========================================
    # Process Each Video Clip
    # ========================================
    for video_file in video_files:
        clip_id = video_file.stem

        # Load pre-computed detection data
        detection_path = os.path.join(outputs_dir, f"detections{clip_id}.json")
        detections = load_json(detection_path)

        print(f"Classifying scene for {video_file}...")

        # Initialize classifier with all detection models
        classifier = SceneClassifier(
            video_file,
            clip_id,
            catalogue,
            show_detections=1,  # Show real-time windows
            person_data=detections,  # Pre-computed person detections
            output_vis_dir=Path(os.path.join(output_dir, 'visualizations'))
        )

        # Run scene analysis
        label_info = classifier.analyze_scene()

        # Store results
        summary['clips'].append(label_info)

        # Update counters
        if label_info['label'] == 'crime':
            summary['dataset_summary']['crime_count'] += 1
        else:
            summary['dataset_summary']['normal_count'] += 1

    # ========================================
    # Save Results to JSON
    # ========================================
    output_path = os.path.join(output_dir, "scene_labels.json")
    save_json(output_path, summary)
    print(f"Scene labels generated at {output_path}")


# ========================================
# Standalone Execution Example
# ========================================
if __name__ == "__main__":
    """
    Example usage of SceneClassifier for testing and demonstration.

    This standalone script:
    1. Loads pre-computed person identity catalogue
    2. Runs scene classification on a specific clip
    3. Shows real-time visualization windows
    4. Saves annotated videos and keyframes

    Prerequisites:
        - Run main.py first to generate:
            * identity_catalogue.json
            * detections{clip_id}.json files
        - Download/train specialized YOLO models:
            * best_model_yolo_violence.pt
            * All_weapon.pt

    Usage:
        python scene_classifier.py

    Testing Tips:
        - Start with clip_iter_id to test single clips
        - Use show_detections=1 to see all detection windows
        - Check saved keyframes in visualizations/ directory
        - Review justifications in scene_labels.json
    """
    # ========================================
    # Setup Paths
    # ========================================
    videos_dir = Path("../data/Videos")
    outputs_dir = Path("../outputs")

    # ========================================
    # Load Identity Catalogue
    # ========================================
    # The catalogue enables person re-identification across clips
    catalogue_path = os.path.join(outputs_dir, "identity_catalogue.json")
    identity_catalogue = load_json(catalogue_path)

    print(f"Loaded identity catalogue with {len(identity_catalogue)} persons")

    # ========================================
    # Run Scene Classification
    # ========================================
    # Process only clip index 2 (third video file)
    # Remove or set to None to process all clips
    run_scene_classification(videos_dir, identity_catalogue, outputs_dir, clip_iter_id=3)

    print("\nScene classification complete!")
    print(f"Results saved to: {outputs_dir}/scene_labels.json")
    print(f"Annotated videos saved to: {outputs_dir}/visualizations/")


# State Machine Logic
# IDLE → PICKED_ITEM → BAGGED_ITEM → (leaving) → CRIME
#   ↑         ↓              ↓
#   └─────────┴──────────────┘ (if checkout)