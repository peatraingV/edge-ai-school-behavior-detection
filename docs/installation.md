# Installation

This guide describes how to prepare the project on a Raspberry Pi 5 with a Hailo-8L AI accelerator. The Hailo software stack must be installed and configured for the target Raspberry Pi OS before running the application.

## 1. Hardware Requirements

| Component | Requirement |
|---|---|
| Companion computer | Raspberry Pi 5 |
| AI accelerator | Hailo-8L (13 TOPS) |
| Camera | USB camera available as a Linux video device (the live configuration uses `/dev/video0`) |
| Storage | Enough free space for the compiled models, application files, and test videos |

## 2. Software Requirements

The project depends on the following software components:

- Raspberry Pi OS or a compatible Linux environment configured for the Hailo-8L.
- Python 3 and Python virtual-environment support.
- HailoRT and the Hailo Apps Python modules.
- GStreamer and the Hailo GStreamer plugin.
- PyGObject, Flask, and Pygame.
- OpenCV and NumPy for image processing.
- Streamlit and Pandas for the monitoring dashboard.
- Git and Git LFS for cloning the repository and downloading large test videos.

The dependency files in `requirements/` separate pip-installed packages from components supplied by the Raspberry Pi/Hailo system environment:

- `requirements/main.txt`: NumPy `1.26.4` and OpenCV `5.0.0.93`. Its comments list Flask `2.2.2`, Pygame `2.1.2`, PyGObject `3.42.2`, HailoRT `4.20.0`, GStreamer, and Hailo Apps as system-provided components.
- `requirements/dashboard.txt`: Streamlit `1.60.0` and Pandas `3.0.5`.

Install HailoRT, Hailo Apps, GStreamer, PyGObject, and the Hailo GStreamer plugin using the installation procedure appropriate for the Raspberry Pi OS and Hailo software release in use. Do not expect these components to be installed by the pip requirements files.

> **Compatibility note:** Keep the HailoRT, Hailo Apps, GStreamer plugin, Python environment, and compiled `.hef` models compatible with one another. The versions listed in the requirements comments describe the project's recorded environment and should be checked against the actual target device before deployment.

## 3. Install Git LFS and Clone the Repository

The repository uses Git LFS for large test video files. Install Git LFS before cloning so Git can download the actual video files rather than leaving LFS pointer files in their place.

```bash
sudo apt update
sudo apt install git git-lfs
git lfs install
```

Clone the repository:

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
```

Git LFS downloads files matched by the repository's `.gitattributes` during a normal clone. Verify that the test videos are listed by Git LFS:

```bash
git lfs ls-files
```

If the repository was cloned without downloading LFS content, fetch it from the project root:

```bash
git lfs pull
```

### Optional: Clone Without Downloading Test Videos

If you only need the live-camera application initially and want to defer downloading the large test videos, skip the automatic LFS download:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
```

Download the test videos later:

```bash
git lfs pull --include="test/**"
```

The `test/` video files are needed for offline-video evaluation, but not for the live-camera mode.

## 4. Prepare the Python Environment

The main application imports Python modules provided by the Hailo and Raspberry Pi system installation. A virtual environment that can access system-installed packages may therefore be required.

If you are creating a new environment, install Python venv support if it is not already available, then create and activate an environment with system site packages enabled:

```bash
sudo apt install python3-venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Install the pip dependencies from the project root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements/main.txt
python -m pip install -r requirements/dashboard.txt
```

If you already have a tested Hailo-enabled Python environment, you can use that environment instead. Make sure it can import the Hailo, GStreamer, and PyGObject modules required by the application. Do not commit local virtual-environment directories such as `hailo_env/`, `pi5_env/`, or `.venv/` to the repository.

## 5. Check the Hailo and Camera Setup

Check that the Hailo service is active:

```bash
systemctl is-active hailort
```

Check that the Hailo GStreamer element is available:

```bash
gst-inspect-1.0 hailonet
```

Check that the required Python modules can be imported from the activated environment:

```bash
python -c "import cv2, numpy, gi, hailo, hailo_platform; print('Required Python imports: OK')"
```

Check that a camera device is present:

```bash
ls -l /dev/video*
```

The current live-camera configuration uses `/dev/video0`. If the camera is assigned to a different device path, update the camera input in `main.py` before running the application.

## 6. Check the Project Files

Confirm that the compiled Hailo models and alert audio files are present:

```text
models/
├── yolov8s_pose_h8l.hef
└── best.hef

audio/
├── bao_dong.mp3
└── bao_dong_smoking.mp3
```

The model filenames above are the paths expected by the current application. Make sure you have the right to use and redistribute any model or audio assets included in your deployment or repository.

The project writes runtime data under `output/` and `shared/`. These directories may need to exist and be writable by the user running the application. Runtime snapshots, smoking evidence crops, and experiment CSV files are generated during execution.

## 7. Dashboard Configuration

The dashboard embeds the live view served by `flask_server.py`. Configure the `CAMERA_SERVER_URL` environment variable to the address of the Flask server that the browser can reach.

For example, set the variable using the actual Raspberry Pi address and Flask port configured by the project:

```bash
export CAMERA_SERVER_URL="http://<RASPBERRY_PI_IP>:<FLASK_PORT>"
```

Replace both placeholders with the values for your deployment. The dashboard and event logger use the local project files under `shared/`, so the dashboard process should run on the same Raspberry Pi/project filesystem as the processing application.

## 8. Next Step

After completing these installation checks, follow the execution-mode instructions in [System Architecture](system_architecture.md) to start live-camera monitoring or run offline-video testing. For offline evaluation, make sure the Git LFS test videos have been downloaded before starting a test run.
