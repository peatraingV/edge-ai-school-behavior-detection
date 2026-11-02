# REQUIRED LIBRARIES
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Literal
import os
os.environ["HAILO_MONITOR"] = "1"

import sys
import re
import math
import time
import subprocess
import atexit
import threading
import queue
from datetime import datetime
import event_logger
import cv2
import numpy as np
import pygame
import csv

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst  # noqa: E402 (must be imported after gi.require_version)
import hailo  # noqa: E402

# [HW - Stage 2] The HailoRT Python API is separate from
# the main GStreamer pose-estimation pipeline above. It runs best.hef
# (the smoking-verification model) on cropped images in a dedicated thread.
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType,
    HailoSchedulingAlgorithm,
)

from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad, get_numpy_from_buffer,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)

PROJECT_DIR = Path(__file__).resolve().parent

# ============================================================
# EDIT THESE VARIABLES FOR EACH NEW VIDEO TEST
# ============================================================
SCENARIO_NAME = "smoking"           # Examples: "exercise_01", "hallway_smoking"
EXPECTED_LABEL = "smoking"          # Choose one of: "normal", "violence", "smoking"
VIDEO_PATH = "test/smoking.mp4"     # Path to the input video

# Initialize the counter here to avoid NameError if Ctrl+C is pressed early
smoking_confirm_stats = {"count": 0}


# /dev/shm — fixed location; do not change
SHM_BASE = Path("/dev/shm/cam_live")
FRAME_PATH = SHM_BASE / "latest_frame.jpg"
TEMP_PATH = SHM_BASE / "temp_frame.jpg"
SHM_BASE.mkdir(parents=True, exist_ok=True)

# Flask server path — relative to main.py
flask_proc = subprocess.Popen(
    ["python3", str(PROJECT_DIR / "flask_server.py")])


def cleanup():
    print("\nCleaning up the system.")
    flask_proc.terminate()
    flask_proc.wait()
    print("Web server stopped.")
    if 'verify_queue' in globals():
        verify_queue.put(None)

     # === ACCESS THE GLOBAL VARIABLES DECLARED ABOVE ===
    if 'user_data' in globals():
        duration = get_video_duration(VIDEO_PATH)

        # Read the smoking counter safely
        smk_count = smoking_confirm_stats.get("count", 0) if 'smoking_confirm_stats' in globals() else 0

        export_event_result_longform(
            SCENARIO_NAME, Path(VIDEO_PATH).stem,
            EXPECTED_LABEL, user_data, duration, smk_count
        )
        export_layer_stats(SCENARIO_NAME, user_data)

atexit.register(cleanup)



# [HW] Model precompiled for Hailo-8 (replaces the previous yolo26n-pose.pt model running on the CPU)
HEF_PATH = PROJECT_DIR / "models" / "yolov8s_pose_h8l.hef"

pygame.mixer.init()
vio_audio_path = str(PROJECT_DIR / "audio" / "bao_dong.mp3")
smok_sound_path = str(PROJECT_DIR / "audio" / "bao_dong_smoking.mp3")
vio_sound = pygame.mixer.Sound(vio_audio_path) if Path(vio_audio_path).exists() else None
smok_sound = pygame.mixer.Sound(smok_sound_path) if Path(smok_sound_path).exists() else None

# Violence thresholds
CONF_THRESH = 0.7
DIST_THRESH = 1.5
SPEED_THRESH = 3.6
JERK_THRESH = 1800
TIME_ID = 10
alpha = 0.5
NOISE_FILTER_THRESH = 5.0
ALARM_MAG_THRESH = 15
SCALE_FACTOR = 0.5          # Reduce the frame dimensions by half for Raspberry Pi 5
AREA_THRESH = 750           # A 0.5 scale reduces the area by a factor of four (3000 / 4 = 750)
MAX_LOCKED_FRAMES = 30      # Arena lock duration (30 frames ≈ 1 second)
COOLDOWN_DURATION = 6.0

# Smoking thresholds
SAVE_FOLDER = PROJECT_DIR / "output" / "smoking_crops"
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

EVENT_CSV = PROJECT_DIR / "output" / "event_test_results.csv"   # added counter update

pixel_crop = 95                         # Crop-region size (image dimension = 2 * pixel_crop)
min_time_thresh = 2                     # Time and repetition thresholds for smoking detection
max_time_thresh = 5
repeat_thresh = 5

# --- Stage 2 thresholds (verification using best.hef) ---
# [HW] best.hef: UINT8 input (320, 320, 3); output already includes NMS
# (yolov8_nms_postprocess); there is only one class, so no class name is needed; only
# the highest confidence among the detections for that class is required.
VERIFIER_HEF_PATH = PROJECT_DIR / "models" / "best.hef"
VERIFIER_INPUT_SIZE = 640
VERIFY_CONF_THRESH = 0.6                # TODO: tune this threshold using real-world data
VERIFY_AGG_METHOD = "max"               # "max" | "majority_vote" | "average"
ALERT_DISPLAY_DURATION = 3.0            # (seconds) to display the "SMOKING" overlay after Stage 2 confirmation

# [HW] Hailo-8L has only one physical device. The pose pipeline (hailonet in
# GStreamer) and SmokingVerifier (the separate VDevice configured here) MUST share
# the same "vdevice-group-id" and enable multi-process-service=true so HailoRT can
# share the device through the scheduler (hailort.service) instead of competing for exclusive
# access (HAILO_OUT_OF_PHYSICAL_DEVICES). This value "1" must EXACTLY MATCH
# the vdevice-group-id generated by hailo_apps in the pipeline
# string (verified on the Raspberry Pi 5 by inspecting pipeline_str).
HAILO_VDEVICE_GROUP_ID = "1"

# Active Zone thresholds (eye-to-upper-chest region)
eye_margin_ratio = 0.08
chest_ratio = 0.35
x_margin_ratio = 0.30

# EMA and stability thresholds
ema_alpha = 0.4
stability_max_disp_ratio = 0.06

# [HW] Thresholds for the standalone ByteTrack implementation (replaces the previous Ultralytics BoT-SORT tracker).
# See byte_tracker.py for details.
TRACK_HIGH_THRESH = 0.6
TRACK_LOW_THRESH = 0.1
TRACK_MATCH_IOU = 0.3
TRACK_MAX_AGE = 30

# --- COCO keypoint indices ---
NOSE_IDX = 0
LEFT_EYE_IDX = 1
RIGHT_EYE_IDX = 2
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12


# Violence detection class
class PersonHistory:
    def __init__(self, max_len=10):
        self.data = {}
        self.max_len = max_len
        self.last_speeds = {}
        self.age_count = {}
        self.kpts_map = {'l_wrist': 9, 'r_wrist': 10, 'l_ankle': 15, 'r_ankle': 16}

    def update(self, person_id, keypoints, kpts_conf, current_time):
        if person_id not in self.data:
            self.data[person_id] = {
                'l_wrist': deque(maxlen=self.max_len),
                'r_wrist': deque(maxlen=self.max_len),
                'l_ankle': deque(maxlen=self.max_len),
                'r_ankle': deque(maxlen=self.max_len)
            }

        self.age_count[person_id] = self.age_count.get(person_id, 0) + 1

        for part_name, idx in self.kpts_map.items():
            if kpts_conf[idx] > CONF_THRESH:
                raw_x = keypoints[idx][0]
                raw_y = keypoints[idx][1]

                if len(self.data[person_id][part_name]) > 0:
                    prev_x = self.data[person_id][part_name][-1][0]
                    prev_y = self.data[person_id][part_name][-1][1]

                    smooth_x = alpha * raw_x + (1 - alpha) * prev_x
                    smooth_y = alpha * raw_y + (1 - alpha) * prev_y
                else:
                    smooth_x = raw_x
                    smooth_y = raw_y

                self.data[person_id][part_name].append((smooth_x, smooth_y, current_time))

    def get_max_velocity(self, person_id, box_height):
        if person_id not in self.data or box_height == 0:
            return 0.0

        max_v = 0.0
        for part in ['l_wrist', 'r_wrist', 'l_ankle', 'r_ankle']:
            queue = self.data[person_id][part]

            if len(queue) >= 5:
                x2, y2, t2 = queue[-1]
                x1, y1, t1 = queue[-5]

                dist = math.hypot(x2 - x1, y2 - y1)
                dt = t2 - t1

                if dt > 0.05:
                    v_raw = dist / dt
                    v_norm = v_raw / box_height
                    if v_norm > max_v:
                        max_v = v_norm

        self.last_speeds[person_id] = max_v
        return max_v

    def jerk_calc(self, person_id, box_height):
        if person_id not in self.data or box_height == 0:
            return 0.0

        max_jerk = 0.0

        for part in ['l_wrist', 'r_wrist', 'l_ankle', 'r_ankle']:
            queue = self.data[person_id][part]

            if len(queue) >= 7:
                x4, y4, t4 = queue[-1]
                x3, y3, t3 = queue[-3]
                x2, y2, t2 = queue[-5]
                x1, y1, t1 = queue[-7]

                dt = (t4 - t1) / 3.0

                if dt < 0.02:
                    continue

                vx1 = (x2 - x1) / dt; vy1 = (y2 - y1) / dt
                vx2 = (x3 - x2) / dt; vy2 = (y3 - y2) / dt
                vx3 = (x4 - x3) / dt; vy3 = (y4 - y3) / dt

                ax1 = (vx2 - vx1) / dt; ay1 = (vy2 - vy1) / dt
                ax2 = (vx3 - vx2) / dt; ay2 = (vy3 - vy2) / dt

                jx = (ax2 - ax1) / dt
                jy = (ay2 - ay1) / dt

                j_magnitude = math.hypot(jx, jy)
                j_norm = j_magnitude / box_height

                if j_norm > max_jerk:
                    max_jerk = j_norm

        return max_jerk

    def cleanup_lost_ids(self, active_ids):
        lost_ids = [pid for pid in self.data.keys() if pid not in active_ids]
        for pid in lost_ids:
            del self.data[pid]
            self.last_speeds.pop(pid, None)


# ==========================================
# STAGE 2: SMOKING VERIFICATION USING best.hef (HailoRT)
# ==========================================
@dataclass
class EvidenceCrop:
    """An image crop around the wrist/mouth region, accumulated for each frame while the wrist is in the Active Zone. These crops provide evidence for Stage 2 when Stage 1 is triggered."""
    image: np.ndarray
    person_id: int
    side: Literal["LEFT", "RIGHT"]
    frame_timestamp: float


@dataclass
class VerificationResult:
    person_id: int
    is_smoking: bool
    confidence: float
    aggregation_method: Literal["max", "majority_vote", "average"]
    evaluated_crops: list = field(default_factory=list)


class SmokingVerifier:
    """Wrapper around HailoRT InferVStreams for best.hef. The VDevice and network group are initialized once and reused for each Stage 2 verification.

    [HW] best.hef runs alongside the pose model (the primary Hailo model in GStreamerPoseEstimationApp) on the same physical Hailo-8L device. To avoid HAILO_OUT_OF_PHYSICAL_DEVICES, this VDevice is created with a group_id, multi_process_service=True, and a scheduling algorithm that MATCH the "vdevice-group-id" and "multi-process-service" properties added to hailonet in HeadlessPoseApp.get_pipeline_string() below. Both models therefore share the chip through the hailort.service scheduler instead of competing for exclusive access. Requirement: hailort.service must be running (check with `systemctl is-active hailort`; enable it with `sudo systemctl enable --now hailort` if necessary)."""

    def __init__(self, hef_path, input_size=VERIFIER_INPUT_SIZE,
                 vdevice_group_id=HAILO_VDEVICE_GROUP_ID):
        self.input_size = input_size
        self.hef = HEF(str(hef_path))

        # [HW] Do NOT use an empty VDevice() (the default behavior claims the device exclusively and causes
        # HAILO_OUT_OF_PHYSICAL_DEVICES because the pose pipeline already holds the chip).
        # Create dedicated parameters and match group_id with the pose pipeline hailonet
        # vdevice-group-id, and enable multi_process_service so
        # HailoRT (through hailort.service) can schedule both models on the chip.
        vdevice_params = VDevice.create_params()
        vdevice_params.group_id = vdevice_group_id
        vdevice_params.multi_process_service = True
        vdevice_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.target = VDevice(vdevice_params)

        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)

        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name

    def infer_batch(self, images_bgr):
        """Input: a list of BGR images of any size. Output: the maximum confidence score for each image, in the same order as the input.

        [HW] Verified on the actual Raspberry Pi 5 using probe_hef_output.py: results[output_name] is a list of length batch_size. Each item contains a list of classes (one class in this model), and each class entry is an ndarray with shape (N_detections, 5). The final column (index 4) contains the detection score. This output structure was inspected directly rather than assumed."""
        n = len(images_bgr)
        batch = np.zeros((n, self.input_size, self.input_size, 3), dtype=np.uint8)
        for i, img in enumerate(images_bgr):
            # The model was trained/compiled using RGB images (matching frame_rgb
            # obtained directly from the Hailo buffer in app_callback), so convert
            # the system working frame from BGR back to RGB before
            # feeding it to the model; otherwise, color-channel mismatch may reduce accuracy.
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch[i] = cv2.resize(rgb, (self.input_size, self.input_size))

        input_data = {self.input_name: batch}

        with InferVStreams(self.network_group, self.input_vstreams_params,
                            self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                results = infer_pipeline.infer(input_data)

        per_image_conf = []
        for per_image_classes in results[self.output_name]:
            best_conf = 0.0
            for cls_dets in per_image_classes:
                if cls_dets is not None and len(cls_dets) > 0:
                    best_conf = max(best_conf, float(np.max(cls_dets[:, 4])))
            per_image_conf.append(best_conf)
        return per_image_conf


def verify_smoking(person_id, evidence_crops, verifier, confidence_threshold, method):
    """Run best.hef on the collected EvidenceCrop images and aggregate the results into one decision for this person_id. Call this function only after Stage 1 is triggered; do not run it on every frame."""
    if not evidence_crops:
        return VerificationResult(person_id, False, 0.0, method, [])

    images = [c.image for c in evidence_crops]
    per_crop_conf = verifier.infer_batch(images)

    if method == "max":
        final_conf = max(per_crop_conf)
        is_smoking = final_conf >= confidence_threshold
    elif method == "average":
        final_conf = sum(per_crop_conf) / len(per_crop_conf)
        is_smoking = final_conf >= confidence_threshold
    elif method == "majority_vote":
        positive_confs = [c for c in per_crop_conf if c >= confidence_threshold]
        is_smoking = len(positive_confs) > len(per_crop_conf) / 2
        final_conf = (sum(positive_confs) / len(positive_confs)
                      if positive_confs else sum(per_crop_conf) / len(per_crop_conf))
    else:
        raise ValueError(f"Invalid aggregation_method: {method}")

    return VerificationResult(person_id, is_smoking, final_conf, method, evidence_crops)


# --- Stage 2 asynchronous infrastructure (does not block the main pose pipeline) ---
smoking_verifier = SmokingVerifier(VERIFIER_HEF_PATH)
verify_queue = queue.Queue()
verify_lock = threading.Lock()
# person_id -> expiration timestamp for the "SMOKING" overlay (confirmed by Stage 2)
confirmed_smoking_until = {}


def smoking_verify_worker():
    """Run Stage 2 verification in a dedicated thread using the two wrist crops from the latest frame."""
    while True:
        job = verify_queue.get()
        if job is None:  # sentinel indicating that the worker thread should stop during shutdown
            verify_queue.task_done()
            break

        # 1. Receive l_wrist and r_wrist from the main processing loop
        person_id, frame_snapshot, _job_time, l_wrist, r_wrist = job
        try:
            h_img, w_img = frame_snapshot.shape[:2]
            evidence_crops = []

            # 2. Crop one image around each wrist (left and right)
            for side_name, wrist_pt in [("LEFT", l_wrist), ("RIGHT", r_wrist)]:
                cx, cy = wrist_pt
                if cx > 1 and cy > 1:
                    x1 = int(max(0, cx - pixel_crop))
                    y1 = int(max(0, cy - pixel_crop))
                    x2 = int(min(w_img, cx + pixel_crop))
                    y2 = int(min(h_img, cy + pixel_crop))

                    crop_img = frame_snapshot[y1:y2, x1:x2]
                    if crop_img.size > 0:
                        # Wrap the cropped image in an EvidenceCrop instance for inference
                        evidence_crops.append(EvidenceCrop(
                            image=crop_img,
                            person_id=person_id,
                            side=side_name,
                            frame_timestamp=_job_time
                        ))

            if not evidence_crops:
                verify_queue.task_done()
                continue

            # 3. Run the verification model on the cropped images
            result = verify_smoking(
                person_id, evidence_crops, smoking_verifier,
                VERIFY_CONF_THRESH, VERIFY_AGG_METHOD)

            if result.is_smoking:
                ts = datetime.now().strftime("%H-%M-%S in %d_%m_%y")

                # 4. Save only the crops confirmed by the AI
                for ev_crop in result.evaluated_crops:
                    path = SAVE_FOLDER / f"Alert_{ev_crop.side}_ID{person_id}_{ts}.jpg"
                    cv2.imwrite(str(path), ev_crop.image)

                event_logger.log_event(
                    "smoking", frame_snapshot,
                    {"person_id": int(person_id), "confidence": float(round(result.confidence, 2))})

                if smok_sound is not None:
                    smok_sound.play()

                with verify_lock:
                    confirmed_smoking_until[person_id] = time.time() + ALERT_DISPLAY_DURATION
                    smoking_confirm_stats["count"] += 1   # Update the counter while holding the lock

                print(f"[WARNING] ID {person_id}: Smoking confirmed (conf={result.confidence:.2f})")
            else:
                print(f"[DEBUG] ID {person_id}: Stage 1 triggered, but Stage 2 rejected the event (conf={result.confidence:.2f}) - ignoring event")
        except Exception as e:
            print(f"[ERROR] Stage 2 verification failed for ID {person_id}: {e}")
        finally:
            verify_queue.task_done()


verify_thread = threading.Thread(target=smoking_verify_worker, daemon=True)
verify_thread.start()


# Smoking detection class
class Smoking:
    def __init__(self, min_time_thresh, max_time_thresh, repeat_thresh, ema_alpha, stability_max_disp_ratio):
        self.min_time_thresh = min_time_thresh
        self.max_time_thresh = max_time_thresh
        self.repeat_thresh = repeat_thresh
        self.ema_alpha = ema_alpha
        self.stability_max_disp_ratio = stability_max_disp_ratio
        self.trackers = {}

    def update(self, person_id, wrist_near_nose, wrist_xy, shoulder_width, current_time):
        if person_id not in self.trackers:
            self.trackers[person_id] = {
                'start': None, 'count': 0, 'holding': False,
                'last_seen': current_time, 'ema_xy': None,
            }

        state = self.trackers[person_id]
        state['last_seen'] = current_time

        if wrist_near_nose is True and wrist_xy is not None:
            if state['start'] is None:
                state['start'] = current_time
                state['holding'] = False
                state['ema_xy'] = wrist_xy
                return False, state['count']

            alpha_ = self.ema_alpha
            prev_ema = state['ema_xy']
            new_ema = (
                alpha_ * wrist_xy[0] + (1 - alpha_) * prev_ema[0],
                alpha_ * wrist_xy[1] + (1 - alpha_) * prev_ema[1],
            )
            disp = math.hypot(new_ema[0] - prev_ema[0], new_ema[1] - prev_ema[1])
            state['ema_xy'] = new_ema

            max_disp_px = max(shoulder_width, 1e-3) * self.stability_max_disp_ratio
            if disp > max_disp_px:
                state['start'] = current_time
                state['holding'] = False
                return False, state['count']

            duration = current_time - state['start']
            if self.min_time_thresh < duration < self.max_time_thresh:
                if state['holding'] is False:
                    state['count'] += 1
                    state['holding'] = True

                if state['count'] == self.repeat_thresh:
                    state['count'] = 0
                    return True, self.repeat_thresh

            return False, state['count']

        else:
            state['start'] = None
            state['holding'] = False
            state['ema_xy'] = None
            return False, state['count']

    def cleanup(self, current_time, timeout=60):
        ids_to_delete = [pid for pid, data in self.trackers.items()
                          if current_time - data['last_seen'] > timeout]
        for pid in ids_to_delete:
            del self.trackers[pid]
            print(f"--- Da don dep bo nho cho ID {pid} (da roi di) ---")


# Helper functions for violence logic and optical flow
def center_and_width(kpts_xy, kpts_conf, box):
    width = box[2] - box[0]
    if kpts_conf[11] > CONF_THRESH and kpts_conf[12] > CONF_THRESH:
        return ((kpts_xy[11][0] + kpts_xy[12][0]) / 2, (kpts_xy[11][1] + kpts_xy[12][1]) / 2), width
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), width


def get_merge_box(boxA, boxB, frame_w, frame_h, scale_factor=0.3, min_pad=15, max_pad=80):
    wA = boxA[2] - boxA[0]; hA = boxA[3] - boxA[1]
    wB = boxB[2] - boxB[0]; hB = boxB[3] - boxB[1]

    base_size = max(wA, wB, hA, hB)
    dynamic_padding = int(base_size * scale_factor)
    padding = max(min_pad, min(dynamic_padding, max_pad))

    x1 = max(0, int(min(boxA[0], boxB[0]) - padding))
    y1 = max(0, int(min(boxA[1], boxB[1]) - padding))
    x2 = min(frame_w, int(max(boxA[2], boxB[2]) + padding))
    y2 = min(frame_h, int(max(boxA[3], boxB[3]) + padding))

    return [x1, y1, x2, y2]


def analyze_violence_features(prev_roi_gray, curr_roi_gray, mag_threshold, area_thresh):
    if prev_roi_gray.size == 0 or curr_roi_gray.size == 0:
        return 0.0

    prev_blur = cv2.GaussianBlur(prev_roi_gray, (5, 5), 0)
    curr_blur = cv2.GaussianBlur(curr_roi_gray, (5, 5), 0)

    flow = cv2.calcOpticalFlowFarneback(
        prev_blur, curr_blur, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0)

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    valid_pixels = mag > mag_threshold

    if np.sum(valid_pixels) < area_thresh:
        return 0.0
    active_mags = mag[valid_pixels]
    mean_magnitude = np.mean(active_mags)
    return mean_magnitude


def nose_wrist_dist(kpts_xy, kpts_conf):
    left_wrist_xy = tuple(kpts_xy[LEFT_WRIST_IDX])
    right_wrist_xy = tuple(kpts_xy[RIGHT_WRIST_IDX])

    required_conf_ok = (
        kpts_conf[LEFT_EYE_IDX] > CONF_THRESH
        and kpts_conf[RIGHT_EYE_IDX] > CONF_THRESH
        and kpts_conf[LEFT_SHOULDER_IDX] > CONF_THRESH
        and kpts_conf[RIGHT_SHOULDER_IDX] > CONF_THRESH
        and kpts_conf[LEFT_HIP_IDX] > CONF_THRESH
        and kpts_conf[RIGHT_HIP_IDX] > CONF_THRESH
    )
    if not required_conf_ok:
        return False, None, None, left_wrist_xy, right_wrist_xy

    left_eye_y = kpts_xy[LEFT_EYE_IDX][1]
    right_eye_y = kpts_xy[RIGHT_EYE_IDX][1]
    left_shoulder_x, left_shoulder_y = kpts_xy[LEFT_SHOULDER_IDX]
    right_shoulder_x, right_shoulder_y = kpts_xy[RIGHT_SHOULDER_IDX]
    left_hip_y = kpts_xy[LEFT_HIP_IDX][1]
    right_hip_y = kpts_xy[RIGHT_HIP_IDX][1]

    shoulder_width = abs(left_shoulder_x - right_shoulder_x)
    shoulder_cx = (left_shoulder_x + right_shoulder_x) / 2
    eye_y = min(left_eye_y, right_eye_y)
    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    hip_y = (left_hip_y + right_hip_y) / 2

    y_margin = shoulder_width * eye_margin_ratio
    x_margin = shoulder_width * x_margin_ratio

    top_y = eye_y - y_margin
    bottom_y = shoulder_y + (hip_y - shoulder_y) * chest_ratio
    left_x = shoulder_cx - (shoulder_width / 2 + x_margin)
    right_x = shoulder_cx + (shoulder_width / 2 + x_margin)

    def _in_zone(idx, wrist_xy):
        if kpts_conf[idx] <= CONF_THRESH:
            return False
        wx, wy = wrist_xy
        nose_y = kpts_xy[NOSE_IDX][1]
        if wy < nose_y:
            return False
        return left_x <= wx <= right_x and top_y <= wy <= bottom_y

    left_active = _in_zone(LEFT_WRIST_IDX, left_wrist_xy)
    right_active = _in_zone(RIGHT_WRIST_IDX, right_wrist_xy)

    # Add active_side ("LEFT"/"RIGHT"/None) so collect_evidence_crop (Stage 2) knows which wrist is being cropped; preserve the existing left/right priority logic.
    if left_active:
        return True, left_wrist_xy, "LEFT", left_wrist_xy, right_wrist_xy
    if right_active:
        return True, right_wrist_xy, "RIGHT", left_wrist_xy, right_wrist_xy
    return False, None, None, left_wrist_xy, right_wrist_xy

def get_video_duration(video_path):
    """Get the video duration using ffprobe."""
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0

def export_event_result_longform(scenario_name, clip_name, expected_label,
                                   user_data, video_duration_sec, smoking_count):
    """Write long-form video test results to a CSV file."""
    file_exists = EVENT_CSV.exists()
    with open(EVENT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Scenario", "Clip", "Ground-truth label",
                             "Duration (seconds)",
                             "Violence trigger count",
                             "Smoking trigger count",
                             "False alarm rate (events/minute)"])
        total_triggers = user_data.violence_trigger_count + smoking_count
        false_alarm_rate = (total_triggers / video_duration_sec) * 60 if expected_label == "normal" else None
        writer.writerow([
            scenario_name, clip_name, expected_label,
            round(video_duration_sec, 1),
            user_data.violence_trigger_count,
            smoking_count,
            round(false_alarm_rate, 2) if false_alarm_rate is not None else "N/A"
        ])

def draw_pose_overlay(frame, boxes, track_ids, kpts_xy_all, kpts_conf_all):
    """Draw bounding boxes, track IDs, and a minimal skeleton to replace Ultralytics results[0].plot() (no longer used after switching to Hailo). This is for visualization/debugging only and does not affect detection logic."""
    skeleton_pairs = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    for box, pid, kxy, kconf in zip(boxes, track_ids, kpts_xy_all, kpts_conf_all):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"ID {pid}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        for a, b in skeleton_pairs:
            if kconf[a] > CONF_THRESH and kconf[b] > CONF_THRESH:
                pa = (int(kxy[a][0]), int(kxy[a][1]))
                pb = (int(kxy[b][0]), int(kxy[b][1]))
                cv2.line(frame, pa, pb, (255, 200, 0), 1)
        for i in range(kxy.shape[0]):
            if kconf[i] > CONF_THRESH:
                cv2.circle(frame, (int(kxy[i][0]), int(kxy[i][1])), 2, (0, 0, 255), -1)


# Initialize shared monitoring objects (not recreated for each frame)
monitor_smoking = Smoking(min_time_thresh, max_time_thresh, repeat_thresh, ema_alpha, stability_max_disp_ratio)
tracker_velocity = PersonHistory()


class user_app_callback_class(app_callback_class):
    """System state, replacing the global variables previously used in the while True loop. State that must persist across frames (violence state machine, HUD logging, alarm cooldown, etc.) is stored here."""

    def __init__(self):
        super().__init__()
        self.use_frame = True

        # Violence state machine (Stage 2 -> Stage 3, the "arena" state)
        self.is_arena_locked = False
        self.locked_box = None
        self.arena_frame_count = 0
        self.roi_prev = None
        self.violence_buffer = deque(maxlen=15)

        # === Added: frame counters for Table 4.1 ===
        self.total_frames  = 0
        self.layer1_frames = 0   # frames processed by Stage 1 without escalation
        self.layer2_frames = 0   # frames passed to Stage 2 (Optical Flow)
        self.violence_trigger_count = 0
        self.smoking_trigger_count  = 0

        # HUD display state
        self.sec_peak_mag = 0.0
        self.last_display_mag = 0.0
        self.last_log_time = time.time()

        # Violence alarm cooldown
        self.last_alarm_time = 0.0

        # Per-second smoking status log
        self.states_per_second = {}
        self.last_print_time = time.time()


# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    user_data.total_frames += 1        # added counter update
    current_time = time.time()

    format, width, height = get_caps_from_pad(pad)
    if format is None or width is None or height is None:
        return Gst.PadProbeReturn.OK

    # Frames returned by Hailo are in RGB; convert them to BGR to match the
    # rest of the system (event_logger, cv2.imwrite, and the original cv2 loop,
    # which all expect BGR as when frames were read with cv2.VideoCapture).
    frame_rgb = get_numpy_from_buffer(buffer, format, width, height)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    id_track = []
    pixel_boxes = []
    kpts_xy = []
    kpts_conf = []

    for detection in detections:
        if detection.get_label() != "person":
            continue

        # Get the unique ID from the Hailo tracker
        unique_ids = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(unique_ids) == 0:
            continue # Skip detections that have not been assigned a tracker ID

        track_id = unique_ids[0].get_id()
        id_track.append(track_id)

        bbox = detection.get_bbox()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if len(landmarks) == 0:
            continue

        x1 = bbox.xmin() * w
        y1 = bbox.ymin() * h
        x2 = (bbox.xmin() + bbox.width()) * w
        y2 = (bbox.ymin() + bbox.height()) * h

        points = landmarks[0].get_points()
        kxy = np.zeros((17, 2), dtype=np.float32)
        kconf = np.zeros(17, dtype=np.float32)
        for k in range(min(17, len(points))):
            p = points[k]
            kxy[k, 0] = (p.x() * bbox.width() + bbox.xmin()) * w
            kxy[k, 1] = (p.y() * bbox.height() + bbox.ymin()) * h
            # [HW] According to the HailoPoint documentation (TAPPAS hailo-objects-api), p.confidence():
            # returns a value from 0.0 to 1.0, defaulting to 1.0 if the HEF does not output a separate confidence
            # for each keypoint. If all confidence values are 1.0 in practice
            # (meaning the HEF does not provide per-keypoint confidence), revisit the filtering strategy.
            kconf[k] = p.confidence()

        pixel_boxes.append([x1, y1, x2, y2])
        kpts_xy.append(kxy)
        kpts_conf.append(kconf)

    if len(pixel_boxes) == 0:
        _flush_frame(annotated_frame)
        return Gst.PadProbeReturn.OK

    id_track = np.array(id_track, dtype=int)
    pixel_boxes = np.array(pixel_boxes, dtype=np.float32)

    # Assign track IDs using the standalone ByteTrack implementation (replacing random/unstable IDs
    # from Hailo). An id_track value below 0 means the detection is treated as noise (score too
    # low) and excluded from subsequent processing.

    draw_pose_overlay(annotated_frame, pixel_boxes, id_track, kpts_xy, kpts_conf)

    current_data = {}
    current_cnt_display = 0
    display_id = ""

    for i, person_id in enumerate(id_track):
        wrist_near_nose, active_wrist, active_side, l_wrist, r_wrist = nose_wrist_dist(kpts_xy[i], kpts_conf[i])

        shoulder_width = abs(kpts_xy[i][LEFT_SHOULDER_IDX][0] - kpts_xy[i][RIGHT_SHOULDER_IDX][0])



        crop, count = monitor_smoking.update(person_id, wrist_near_nose, active_wrist, shoulder_width, current_time)

        if person_id not in user_data.states_per_second:
            user_data.states_per_second[person_id] = {'active': False, 'current_count': 0}
        user_data.states_per_second[person_id]['active'] = wrist_near_nose
        user_data.states_per_second[person_id]['current_count'] = count

        if crop:
            print(f"[WARNING] ID {person_id}: reached the Stage 1 threshold -> ALERT IMMEDIATELY (Stage 2 bypassed)")

            # 1. Log the event snapshot (confidence is set to 1.0 because this path confirms directly without another AI check)
            event_logger.log_event("smoking", frame, {"person_id": int(person_id), "confidence": 1.0})

            # 2. Play the alert sound
            if smok_sound is not None:
                smok_sound.play()

            # 3. Increment the CSV counter and set the on-screen alert duration
            with verify_lock:
                confirmed_smoking_until[person_id] = current_time + ALERT_DISPLAY_DURATION
                if 'smoking_confirm_stats' in globals():
                    smoking_confirm_stats["count"] += 1

        # If Stage 2 recently confirmed smoking (and the display duration has not expired),
        # draw the warning overlay on the current frame.
        with verify_lock:
            confirmed_until = confirmed_smoking_until.get(person_id, 0.0)
        if current_time < confirmed_until:
            cv2.putText(annotated_frame, f"!!! SMOKING (CONFIRMED) !!! ID {person_id}", (15, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 165, 255), 3)

        tracker_velocity.update(person_id, kpts_xy[i], kpts_conf[i], current_time)
        box_h = pixel_boxes[i][3] - pixel_boxes[i][1]
        vel = tracker_velocity.get_max_velocity(person_id, box_h)
        jerk = tracker_velocity.jerk_calc(person_id, box_h)
        current_data[person_id] = (vel, jerk)

    if current_time - user_data.last_print_time >= 1.0:
        if len(user_data.states_per_second) == 0:
            print("Normal.")
        user_data.states_per_second = {}
        user_data.last_print_time = current_time

    if len(id_track) > 0:
        first_person_id = id_track[0]
        display_id = str(first_person_id)
        if first_person_id in monitor_smoking.trackers:
            current_cnt_display = monitor_smoking.trackers[first_person_id]['count']
    # Keep display_id / current_cnt_display from the original code for future
    # HUD extensions (not currently drawn on the frame, matching the original behavior).
    _ = (display_id, current_cnt_display)

    monitor_smoking.cleanup(current_time)


    # STATE 0: INITIAL SCREENING
    if not user_data.is_arena_locked:
        user_data.layer1_frames += 1    # count this frame as processed by Stage 1
        num_people = len(id_track)
        trigger_violence = False

        if num_people >= 2:
            for i in range(num_people):
                for j in range(i + 1, num_people):
                    id_i, id_j = id_track[i], id_track[j]
                    if (tracker_velocity.age_count.get(id_i, 0) < TIME_ID
                            or tracker_velocity.age_count.get(id_j, 0) < TIME_ID):
                        continue

                    v1, j1 = current_data.get(id_i, (0, 0))
                    v2, j2 = current_data.get(id_j, (0, 0))
                    if max(v1, v2) < SPEED_THRESH or max(j1, j2) < JERK_THRESH:
                        continue

                    box_i, box_j = pixel_boxes[i], pixel_boxes[j]

                    (c1, w1) = center_and_width(kpts_xy[i], kpts_conf[i], box_i)
                    (c2, w2) = center_and_width(kpts_xy[j], kpts_conf[j], box_j)

                    dist_ratio = math.hypot(c1[0] - c2[0], c1[1] - c2[1]) / ((w1 + w2) / 2 + 1e-6)

                    if dist_ratio < DIST_THRESH:
                        trigger_violence = True
                        user_data.locked_box = get_merge_box(box_i, box_j, w, h, scale_factor=0.35)
                        break
                if trigger_violence:
                    break

        if trigger_violence:
            user_data.is_arena_locked = True
            user_data.arena_frame_count = 0
            lx1, ly1, lx2, ly2 = user_data.locked_box

            roi_gray_full = cv2.cvtColor(frame[ly1:ly2, lx1:lx2], cv2.COLOR_BGR2GRAY)
            user_data.roi_prev = cv2.resize(roi_gray_full, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            print("Scanning optical flow")

    # STATE 1: LOCKED ARENA (OPTICAL FLOW)
    else:
        user_data.layer2_frames += 1    # count this frame as passed to Stage 2
        lx1, ly1, lx2, ly2 = user_data.locked_box

        roi_gray_full = cv2.cvtColor(frame[ly1:ly2, lx1:lx2], cv2.COLOR_BGR2GRAY)
        roi_curr = cv2.resize(roi_gray_full, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        mag = analyze_violence_features(user_data.roi_prev, roi_curr,
                                         NOISE_FILTER_THRESH * SCALE_FACTOR, AREA_THRESH)
        mag = mag / SCALE_FACTOR

        is_frame_violent = (mag > ALARM_MAG_THRESH)
        user_data.violence_buffer.append(is_frame_violent)

        if (len(user_data.violence_buffer) == user_data.violence_buffer.maxlen
                and sum(user_data.violence_buffer) > (0.7 * user_data.violence_buffer.maxlen)):
            is_violence = True
        else:
            is_violence = False

        if mag > user_data.sec_peak_mag:
            user_data.sec_peak_mag = mag

        curr_t = time.time()
        if curr_t - user_data.last_log_time >= 1.0:
            user_data.last_display_mag = user_data.sec_peak_mag
            user_data.sec_peak_mag = 0.0
            user_data.last_log_time = curr_t

        if is_violence:
            cv2.rectangle(annotated_frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 6)
            cv2.putText(annotated_frame, "!!! VIOLENCE !!!", (lx1, ly1 - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

            if current_time - user_data.last_alarm_time >= COOLDOWN_DURATION:
                if vio_sound is not None:
                    vio_sound.play()
                user_data.last_alarm_time = current_time
                user_data.violence_trigger_count += 1
                event_logger.log_event("violence", annotated_frame, {"mag": float(round(mag, 2))})

        else:
            cv2.rectangle(annotated_frame, (lx1, ly1), (lx2, ly2), (0, 165, 255), 3)
            cv2.putText(annotated_frame,
                        f"Analyzing... {MAX_LOCKED_FRAMES - user_data.arena_frame_count}",
                        (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        text_color = (0, 0, 255) if is_violence else (0, 255, 0)
        cv2.putText(annotated_frame, f"LIVE Mag: {mag:.2f}", (lx1 + 10, ly1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(annotated_frame, f"PEAK Mag: {user_data.last_display_mag:.2f}",
                    (lx1 + 10, ly1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        user_data.roi_prev = roi_curr.copy()
        user_data.arena_frame_count += 1

        if user_data.arena_frame_count >= MAX_LOCKED_FRAMES:
            print("Time limit reached -> returning to Stage 1")
            user_data.is_arena_locked = False
            user_data.locked_box = None
            user_data.roi_prev = None
            user_data.violence_buffer.clear()

    tracker_velocity.cleanup_lost_ids(set(id_track.tolist()))

    _flush_frame(annotated_frame)
    return Gst.PadProbeReturn.OK


def _flush_frame(annotated_frame):
    """Write the annotated frame to /dev/shm for flask_server to read. Use an atomic write-then-rename operation, as in the original implementation, to prevent Flask from reading a partially written file."""
    display_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imwrite(str(TEMP_PATH), display_frame)
    os.replace(TEMP_PATH, FRAME_PATH)


class HeadlessPoseApp(GStreamerPoseEstimationApp):
    def get_pipeline_string(self):
        pipeline_str = super().get_pipeline_string()
        # Replace every autovideosink element with fakesink
        pipeline_str = pipeline_str.replace("autovideosink", "fakesink")

        # [HW] Enable multi-process-service and scheduling-algorithm on the hailonet
        # in the pose pipeline to MATCH the VDevice configuration used by SmokingVerifier
        # (Stage 2) above. Both models then share one Hailo-8L through the scheduler
        # provided by hailort.service instead of competing for exclusive device access. This was
        # verified on the Raspberry Pi 5: the original pipeline_str already contains "vdevice-group-id=1"
        # in the hailonet section, so the two properties only need to be inserted after it.
        pipeline_str, n_subs = re.subn(
            r'(vdevice-group-id=\S+)',
            r'\1 multi-process-service=true scheduling-algorithm=1',
            pipeline_str,
        )
        if n_subs == 0:
            print("[WARNING] 'vdevice-group-id=' was not found in pipeline_str "
                  "-> multi-process-service is NOT enabled for the pose pipeline hailonet. "
                  "Stage 2 may still encounter HAILO_OUT_OF_PHYSICAL_DEVICES.")

        return pipeline_str


STATS_CSV = PROJECT_DIR / "output" / "layer1_stats.csv"
STATS_CSV.parent.mkdir(parents=True, exist_ok=True)

def export_layer_stats(scenario_name, user_data):
    """Write one row of Stage 1/Stage 2 statistics for a video scenario."""
    file_exists = STATS_CSV.exists()
    with open(STATS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Scenario name", "Total frames",
                             "Frames handled by Stage 1", "Frames passed to Stage 2"])
        writer.writerow([
            scenario_name,
            user_data.total_frames,
            user_data.layer1_frames,
            user_data.layer2_frames,
        ])
    print(f"[STATS] Written: {scenario_name} -> "
          f"total={user_data.total_frames}, "
          f"L1={user_data.layer1_frames}, L2={user_data.layer2_frames}")


if __name__ == "__main__":
    env_file = PROJECT_DIR / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    # [HW] GStreamerPoseEstimationApp parses sys.argv internally (using argparse),
    # so pass the camera input and HEF path through sys.argv instead of function arguments.
    # TODO: run `python3 main.py --help` on the Raspberry Pi 5 (after installing
    # hailo_apps) to confirm the correct --input/-i and --hef-path flags, because
    # flag names may differ between hailo-apps/hailo-apps-infra versions.
    sys.argv = [
        sys.argv[0],
        "--input", str(VIDEO_PATH),
        "--hef-path", str(HEF_PATH),
    ]

    user_data = user_app_callback_class()
    app = HeadlessPoseApp(app_callback, user_data)
    app.run()
