# System Architecture

## 1. Overview

The **School Misbehavior Detection** system is an edge-AI monitoring application designed to run on a Raspberry Pi 5 with a Hailo-8L AI accelerator. The repository currently provides two execution modes:

- **Live-camera mode** (`main.py`): reads frames from a camera, performs pose-based processing and behavior analysis, and publishes the annotated live view.
- **Offline-video mode** (`main_video_test.py`): runs the processing pipeline against a selected test video and exports experiment statistics.

The monitoring interface is provided by a separate Streamlit process (`dashboard.py`). A Flask process (`flask_server.py`), started by the main processing application, serves the latest annotated frame to the dashboard's embedded live-view page.

The implementation contains two behavior-analysis paths:

1. **Smoking detection:** a pose-based first-stage trigger followed by a HailoRT model-verification stage.
2. **Violence detection:** person-interaction and motion heuristics followed by optical-flow analysis within a selected region of interest (ROI).

This document describes the implementation in this repository. It does not claim that the heuristics or models provide a final or universally reliable determination of behavior.

## 2. High-Level Architecture

```mermaid
flowchart TD
    A[Camera: /dev/video0] --> B[GStreamer pose-estimation pipeline]
    T[Test video] --> V[main_video_test.py]
    B --> C[Hailo pose model and tracking]
    V --> C
    C --> D[Application callback: detections, IDs, boxes, keypoints]
    D --> S[Smoking detection - Stage 1]
    S -->|Trigger| Q[Evidence queue]
    Q --> W[Background verification worker]
    W --> G[SmokingVerifier: best.hef via HailoRT]
    G -->|Confirmed| E[Smoking alert, crops, event and overlay]
    D --> H[Violence candidate filtering]
    H --> I[ROI optical-flow analysis]
    I -->|Confirmed by temporal rule| J[Violence alert and event]
    E --> K[Annotated frame]
    J --> K
    D --> K
    K --> M[/dev/shm/cam_live/latest_frame.jpg]
    M --> F[Flask live-frame endpoint]
    F --> R[Browser live view]
    R --> SD[Streamlit dashboard iframe]
    E --> EL[event_logger.py]
    J --> EL
    EL --> JS[shared/events.json and shared/snapshots/]
    JS --> SD
    V --> CSV[output experiment CSV files]
```

## 3. Runtime Components

| Component | Responsibility |
|---|---|
| `main.py` | Live-camera entry point. Starts the Flask child process, configures the Hailo pose pipeline, processes frames, runs smoking and violence analysis, and writes the latest annotated frame. |
| `main_video_test.py` | Offline-video test entry point. Uses a configurable scenario name, expected label, and video path; runs the analysis pipeline and writes experiment statistics. |
| Hailo GStreamer pose pipeline | Runs the compiled pose-estimation model and provides person detections, bounding boxes, landmarks, and tracker IDs to the application callback. |
| `SmokingVerifier` | Runs the second-stage smoking verifier (`models/best.hef`) through the HailoRT Python API. |
| `event_logger.py` | Saves event snapshots and appends event metadata to `shared/events.json`. |
| `flask_server.py` | Serves a small web page and the latest JPEG frame written to shared memory. |
| `dashboard.py` | Reads event data and snapshots, displays monitoring statistics and history, and embeds the Flask live view. |
| `scripts/diagnostics/` | Contains hardware/model inspection utilities; these are diagnostic tools, not part of the normal runtime path. |

The live processing application and the Streamlit dashboard are separate processes. The Flask server is launched by the processing application as a child process. The dashboard reads the local `shared/` data files and embeds the Flask URL configured through `CAMERA_SERVER_URL`.

## 4. Frame Processing and Pose Estimation

In live-camera mode, `main.py` configures the input as `/dev/video0`. The GStreamer pose-estimation application uses the compiled model at `models/yolov8s_pose_h8l.hef`.

For each available frame, the application callback:

1. Retrieves the GStreamer buffer and frame dimensions.
2. Converts the Hailo buffer to a NumPy image and converts RGB to BGR for the OpenCV processing path.
3. Retrieves Hailo person detections, tracker IDs, bounding boxes, and pose landmarks.
4. Converts pose landmarks into image-pixel coordinates and retains their confidence values.
5. Draws bounding boxes, person IDs, and a simplified pose skeleton for visualization.
6. Passes per-person pose data to the smoking and violence analysis paths.
7. Writes the annotated frame to shared memory for the Flask service.

The pose model and its GStreamer integration are provided by the Hailo software stack. The project uses Hailo Python modules for buffer conversion, the GStreamer application wrapper, and the pose-estimation pipeline.

## 5. Smoking Detection Pipeline

### 5.1 Stage 1: Pose-Based Trigger

Stage 1 uses pose keypoints to identify a hand/wrist position inside an active region around the person's face and upper torso. The implementation checks the confidence of relevant facial, shoulder, hip, and wrist keypoints before evaluating the region.

The trigger logic includes:

- A geometric active-zone check based on eye, shoulder, and hip positions.
- A wrist-position condition relative to the nose and the active-zone boundaries.
- Exponential moving average (EMA) smoothing of wrist coordinates.
- A displacement/stability check to reduce unstable position changes.
- A duration window and repeated-event counter before issuing a Stage 1 trigger.

Stage 1 is a pose-and-temporal heuristic. It does not by itself establish that a person is smoking; in the current live configuration, its trigger is passed to Stage 2 for verification.

### 5.2 Stage 2: HailoRT Verification

The current live-camera configuration enables Stage 2. When Stage 1 triggers, the main callback places an evidence job on a queue instead of running the verifier synchronously in the main frame-processing path.

A background worker then:

1. Receives the person ID, a frame snapshot, timestamp, and left/right wrist coordinates.
2. Extracts image crops around the available wrist positions.
3. Converts each crop from BGR to RGB and resizes it to the verifier input size.
4. Runs batch inference using `models/best.hef` through HailoRT `InferVStreams`.
5. Reads the detection confidence values and aggregates them using the configured method.
6. Compares the final confidence with `VERIFY_CONF_THRESH`.

The current source configuration uses a 640-pixel verifier input size, a confidence threshold of `0.6`, and the `max` aggregation method. These are configurable parameters and may be tuned in the source code.

When Stage 2 confirms smoking, the application saves the evaluated evidence crops under `output/smoking_crops/`, records a smoking event and snapshot through `event_logger.py`, plays the smoking alert audio when available, and displays a temporary confirmation overlay.

### 5.3 Sharing the Hailo-8L Device

The pose pipeline and the Stage 2 verifier use the same physical Hailo-8L device. The implementation configures the verifier with a VDevice group ID and multi-process service, and modifies the pose pipeline string to enable matching multi-process scheduling properties. This is intended to let both inference paths share the device through `hailort.service` instead of competing for exclusive access.

The Hailo service, HailoRT version, model compatibility, and VDevice configuration must match the deployment environment.

## 6. Violence Detection Pipeline

The violence path is implemented as a staged motion-analysis heuristic rather than a separate end-to-end violence-classification model.

### 6.1 Candidate Filtering

For each tracked person, the application maintains short histories of selected pose keypoints (wrists and ankles). It derives normalized movement measures, including velocity and jerk, from keypoint motion over time.

A candidate interaction is considered when:

- At least two tracked people are present.
- The tracks have accumulated the configured minimum age.
- The pair satisfies the configured motion thresholds.
- The normalized distance between the two person centers is below the configured proximity threshold.

When these conditions are met, the application creates a merged bounding region around the pair and enters the ROI-analysis state.

### 6.2 Optical-Flow Analysis

The selected ROI is converted to grayscale and downscaled. Consecutive ROI frames are blurred and passed to OpenCV's Farnebäck dense optical-flow calculation.

The implementation calculates motion magnitude, filters pixels below a configured magnitude threshold, and checks the number of remaining active pixels. The mean magnitude of the active pixels is used as a frame-level motion score.

A rolling temporal buffer is used to reduce one-frame triggers. The current implementation marks the ROI as violent when the configured proportion of recent frames exceeds the motion threshold. When an alert is emitted, the application draws an overlay, plays the violence alert audio subject to a cooldown, and logs a `violence` event with the motion magnitude.

The ROI state is released after the configured frame window, after which the application returns to the initial candidate-filtering state.

## 7. Live Frame Transport

The processing application writes each annotated frame to a RAM-backed directory:

```text
/dev/shm/cam_live/
├── latest_frame.jpg
└── temp_frame.jpg
```

The frame is first written to a temporary file and then moved into place with `os.replace`, reducing the chance that the web server reads a partially written JPEG.

`flask_server.py` exposes:

- `/`: a small HTML page containing the live image element.
- `/latest_frame.jpg`: the most recently written JPEG frame.

The page periodically requests the JPEG endpoint with a cache-busting query parameter. `dashboard.py` embeds the Flask page in an iframe. The dashboard obtains the Flask address from the `CAMERA_SERVER_URL` environment variable.

The `/dev/shm` frame buffer is transient RAM storage. It is not a persistent recording archive.

## 8. Event Logging and Dashboard Data

When a smoking or violence event is emitted, `event_logger.py`:

1. Creates a timestamped JPEG snapshot in `shared/snapshots/`.
2. Builds an event record containing the timestamp, event type, snapshot filename, and event-specific metadata.
3. Appends the record to `shared/events.json`.
4. Retains the most recent 500 records in the JSON file.

The metadata currently used by the application includes fields such as `person_id`, `confidence`, or `mag`, depending on the event path.

`dashboard.py` reads `shared/events.json` and the associated snapshots. It displays the latest event, event counts, a filterable history, and selected snapshots. The dashboard refreshes periodically and embeds the Flask live view separately from the event-history data.

Because the dashboard reads `shared/` from the local project directory, the dashboard process is expected to run on the same Raspberry Pi/project filesystem as the event logger. The browser can access the Streamlit and Flask services over the network, but the two Python processes share local event files.

## 9. Offline Video Evaluation

`main_video_test.py` is the offline evaluation entry point. The operator configures:

- `SCENARIO_NAME`: the experiment/scenario identifier.
- `EXPECTED_LABEL`: the ground-truth scenario label (`normal`, `violence`, or `smoking`).
- `VIDEO_PATH`: the input video file.

The video test application runs the Hailo pose pipeline and the behavior-analysis logic against the selected clip. On shutdown, it exports event-level results and layer statistics to CSV files under `output/`, including video duration, trigger counts, false-alarm-rate information for normal clips, and Stage 1/Stage 2 frame counts.

These CSV files are runtime experiment outputs. They are not required as source files for the live monitoring application and should not be committed unless a curated evaluation result is intentionally prepared for publication.

## 10. Data and Runtime Directories

| Path | Purpose | Persistence |
|---|---|---|
| `models/yolov8s_pose_h8l.hef` | Hailo-compiled pose-estimation model | Stored in the project; ensure the file is available after cloning. |
| `models/best.hef` | Custom smoking-verification model | Stored in the project; ensure the file is available after cloning. |
| `audio/` | Alert audio files | Stored in the project. |
| `test/` | Input videos for offline evaluation | Large video files are tracked with Git LFS. |
| `output/smoking_crops/` | Evidence crops saved after Stage 2 confirmation | Runtime output; image contents are excluded from Git. |
| `shared/events.json` | Event metadata consumed by the dashboard | Runtime data; should not contain committed runtime records. |
| `shared/snapshots/` | Event snapshot images | Runtime output; snapshot images are excluded from Git. |
| `/dev/shm/cam_live/` | Latest annotated frame and temporary frame | Transient RAM storage. |
| `output/` CSV files | Offline experiment statistics | Runtime output; excluded from Git unless curated results are intentionally published. |

## 11. Execution Modes

### Live Camera

1. Activate the Python environment that has access to the installed Hailo and GStreamer components.
2. Run `main.py` from the project root. The application starts its Flask child process and initializes the GStreamer/Hailo pose pipeline.
3. Start the Streamlit dashboard as a separate process.
4. Configure `CAMERA_SERVER_URL` to point to the Flask server address so the dashboard can display the live view.

### Offline Video Test

1. Set `SCENARIO_NAME`, `EXPECTED_LABEL`, and `VIDEO_PATH` in `main_video_test.py`.
2. Run the script from the project root in the same Hailo-enabled environment used for the processing pipeline.
3. Review the generated CSV statistics under `output/` after the run.

For repository cloning, Git LFS setup, Python dependencies, and Hailo environment prerequisites, see [Installation](installation.md).

## 12. Implementation Notes and Limitations

- The system is designed around Raspberry Pi 5 and Hailo-8L; it is not a hardware-independent Python application.
- HailoRT, the Hailo Apps Python modules, GStreamer, PyGObject, and the compiled models must be compatible with one another.
- The live frame buffer uses `/dev/shm`, so the live image is not retained after system restart.
- Event metadata is stored in a local JSON file. This implementation is suitable for a single-device deployment but is not a multi-node database architecture.
- The violence detector uses motion and interaction heuristics. Its outputs should be evaluated against representative footage and should not be interpreted as definitive proof of violent behavior.
- The smoking detector is a two-stage pipeline in the current live configuration. Stage 1 generates candidate evidence; Stage 2 performs model-based verification.
- Detection thresholds and model confidence settings are implementation parameters. Their performance depends on camera placement, image quality, scene conditions, and the evaluation dataset.
