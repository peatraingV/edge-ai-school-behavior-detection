import json
from datetime import datetime
from pathlib import Path

import cv2


# The project root is the directory containing this file.
PROJECT_DIR = Path(__file__).resolve().parent

BASE = PROJECT_DIR / "shared"
EVENTS = BASE / "events.json"
SNAPS = BASE / "snapshots"

BASE.mkdir(parents=True, exist_ok=True)
SNAPS.mkdir(parents=True, exist_ok=True)


def log_event(event_type: str, frame, meta: dict):
    """Save an event snapshot and append its metadata to the event log.

    Args:
        event_type: "violence" or "smoking".
        frame: Image frame to save as the event snapshot.
        meta: Additional event metadata, such as {"mag": 8.2} or
            {"person_id": 3}.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{event_type}.jpg"
    )
    snapshot_path = SNAPS / snapshot_name
    cv2.imwrite(str(snapshot_path), frame)

    safe_meta = {
        key: float(value) if hasattr(value, "item") else value
        for key, value in meta.items()
    }

    event = {
        "time": timestamp,
        "type": event_type,
        "snapshot": snapshot_name,
        **safe_meta,
    }

    events = []
    if EVENTS.exists():
        try:
            with EVENTS.open(encoding="utf-8") as event_file:
                events = json.load(event_file)
        except (OSError, json.JSONDecodeError):
            events = []

    events.append(event)
    with EVENTS.open("w", encoding="utf-8") as event_file:
        json.dump(events[-500:], event_file, ensure_ascii=False)
