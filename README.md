# Edge-AI-Based Detection of Smoking Behavior and Potentially Violent Interactions in School Environments

An edge-AI monitoring system designed for **Raspberry Pi 5** with a **Hailo-8L AI accelerator**. The project processes live camera streams or recorded videos to analyze two types of behavior: smoking and potential violent interactions. It also provides a web dashboard for live-view monitoring and event review.

> **Note:** This repository documents a working project implementation. The detection methods are not a substitute for human review and should not be treated as definitive proof of misconduct.

## Features

- **Pose estimation and person tracking** using a Hailo-compiled pose model in a GStreamer pipeline.
- **Smoking detection:** a pose-based and temporal Stage 1 trigger followed by Stage 2 verification using a separate HailoRT model.
- **Violence detection:** person-interaction and motion heuristics, followed by Farnebäck optical-flow analysis within a selected region of interest (ROI).
- **Event logging:** timestamped snapshots and event metadata stored locally for dashboard review.
- **Live monitoring:** annotated camera frames served by Flask and displayed in a Streamlit dashboard.
- **Offline video evaluation:** configurable test scenarios with CSV summaries of trigger counts and processing statistics.

## System Overview

```text
Live camera ─┐
             ├─> GStreamer + Hailo pose estimation and tracking
Test video ──┘                         │
                                      ├─> Smoking analysis
                                      │     └─> Stage 2 HailoRT verification
                                      │
                                      └─> Violence candidate analysis
                                            └─> ROI optical-flow analysis
                                                     │
                                      Event logging and annotated frames
                                          │                    │
                                   shared/ events         Flask live view
                                          │                    │
                                          └────> Streamlit dashboard
```

For the detailed processing flow, component responsibilities, and implementation limitations, see [`docs/system_architecture.md`](docs/system_architecture.md).

## Hardware

| Component | Configuration |
|---|---|
| Companion computer | Raspberry Pi 5 |
| AI accelerator | Hailo-8L (13 TOPS) |
| Camera | USB camera (live configuration uses `/dev/video0`) |

The application depends on a compatible Hailo software stack, including HailoRT, Hailo Apps, GStreamer, and the required Python bindings. It is not a hardware-independent Python application.

## Repository Structure

```text
.
├── audio/                 # Alert audio assets
├── docs/                  # Installation and architecture documentation
├── models/                # Compiled Hailo models (.hef)
├── output/                # Runtime crops and evaluation CSV files
├── requirements/          # Python dependency lists
├── scripts/diagnostics/   # Hailo/model diagnostic utilities
├── shared/                # Event metadata and snapshots used by the dashboard
├── test/                  # Offline evaluation videos (Git LFS)
├── dashboard.py           # Streamlit monitoring dashboard
├── event_logger.py        # Event and snapshot logging
├── flask_server.py        # Live-frame web server
├── main.py                # Live-camera application
└── main_video_test.py     # Offline-video evaluation application
```

## Installation

See **[Installation Guide](docs/installation.md)** for the full setup instructions, including Git LFS, Python dependencies, Hailo/GStreamer prerequisites, camera checks, and dashboard configuration.

Quick repository setup:

```bash
git lfs install
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
```

The test videos are managed with **Git LFS**. A normal clone downloads them; they can also be skipped initially and fetched later when offline-video evaluation is needed. Refer to the installation guide for both options.

## Running the Application

Run the commands from the project root, using the Hailo-enabled Python environment described in the installation guide.

### Live-camera mode

Connect the camera and make sure it is available at `/dev/video0` (or update the input path in `main.py`). Then run:

```bash
python main.py
```

The application starts the Flask live-frame server as a child process. Start the Streamlit dashboard in a separate terminal:

```bash
streamlit run dashboard.py
```

Configure `CAMERA_SERVER_URL` for the dashboard so the browser can reach the Flask server. See the installation guide for the configuration format and deployment notes.

### Offline-video evaluation

Before running a test, edit `SCENARIO_NAME`, `EXPECTED_LABEL`, and `VIDEO_PATH` near the top of `main_video_test.py`. The supported ground-truth labels in the current script are `normal`, `violence`, and `smoking`.

Run the test with:

```bash
python main_video_test.py
```

The script writes experiment statistics to `output/`. These generated files are runtime outputs; only commit selected, reviewed results when they are intentionally prepared for sharing.

## Dependencies and Models

- `requirements/main.txt` lists pip-installed dependencies for the processing application.
- `requirements/dashboard.txt` lists the Streamlit dashboard dependencies.
- HailoRT, Hailo Apps, GStreamer, the Hailo GStreamer plugin, and PyGObject must be installed through a compatible system/Hailo setup; they are not installed by the pip requirements files.
- The current application expects `models/yolov8s_pose_h8l.hef` and `models/best.hef`.

Keep the HailoRT version, Hailo Apps version, GStreamer integration, Python environment, and compiled `.hef` models compatible. Check the model and audio asset licenses before redistributing them.

## Evaluation and Limitations

The offline-video workflow records trigger counts, video duration, false-alarm-rate information for normal clips, and Stage 1/Stage 2 frame statistics. These are project-level measurements; this README does not claim a particular accuracy, FPS, or latency.

The violence path is a motion- and interaction-based heuristic rather than a dedicated end-to-end violence classifier. Smoking detection uses a two-stage pipeline in the current live configuration. Performance depends on camera placement, image quality, scene conditions, and the evaluation data. Review alerts before taking action.

## Documentation

- [System Architecture](docs/system_architecture.md)
- [Installation](docs/installation.md)
