import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


PROJECT_DIR = Path(__file__).resolve().parent
BASE = PROJECT_DIR / "shared"
EVENTS = BASE / "events.json"
SNAPS = BASE / "snapshots"
CAMERA_SERVER_URL = os.getenv("CAMERA_SERVER_URL", "").strip().rstrip("/")

st.set_page_config(
    page_title="AI Camera Monitor",
    page_icon="🎥",
    layout="wide",
)


def load_events():
    """Load recorded detection events from the shared JSON file."""
    if not EVENTS.exists():
        return []

    try:
        with EVENTS.open(encoding="utf-8") as event_file:
            return json.load(event_file)
    except (OSError, json.JSONDecodeError):
        return []


events = load_events()

# Header
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("AI Camera Monitor")
with col_status:
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = sum(
        1 for event in events if event.get("time", "").startswith(today)
    )
    st.metric("Today's Alerts", today_count)

st.divider()

# Main layout: live camera feed and most recent alert
col_live, col_alert = st.columns([3, 2])

with col_live:
    st.subheader("Live Camera")
    if CAMERA_SERVER_URL:
        components.iframe(
            src=CAMERA_SERVER_URL,
            height=750,
            scrolling=False,
        )
        st.caption("🟢 Live stream")
    else:
        st.warning(
            "The camera stream URL is not configured. Set the "
            "CAMERA_SERVER_URL environment variable to the Raspberry Pi "
            "Flask server URL, for example: http://<PI_IP>:5000."
        )

with col_alert:
    st.subheader("Most Recent Alert")
    if events:
        last_event = events[-1]
        event_type = last_event.get("type", "")
        event_time = last_event.get("time", "")

        if event_type == "violence":
            st.error(f"**VIOLENCE** — {event_time}")
            magnitude = last_event.get("mag")
            if magnitude is not None:
                st.metric("Motion Magnitude", magnitude)
        elif event_type == "smoking":
            st.warning(f"**SMOKING** — {event_time}")
            person_id = last_event.get("person_id")
            if person_id is not None:
                st.metric("Person ID", person_id)

        snapshot_path = SNAPS / last_event.get("snapshot", "")
        if snapshot_path.is_file():
            st.image(str(snapshot_path), use_container_width=True)
        else:
            st.caption("Snapshot not found.")
    else:
        st.success("No alerts recorded yet.")

st.divider()

# Summary statistics
if events:
    all_events_df = pd.DataFrame(events)
    col_total, col_violence, col_smoking = st.columns(3)

    with col_total:
        st.metric("Total Incidents", len(all_events_df))
    with col_violence:
        violence_count = len(all_events_df[all_events_df["type"] == "violence"])
        st.metric("Violence Incidents", violence_count)
    with col_smoking:
        smoking_count = len(all_events_df[all_events_df["type"] == "smoking"])
        st.metric("Smoking Incidents", smoking_count)

st.divider()

# Incident history
st.subheader("Incident History")

if events:
    events_df = pd.DataFrame(events[::-1])  # Most recent events appear first.

    col_filter_type, col_filter_date = st.columns([2, 3])
    with col_filter_type:
        filter_type = st.selectbox(
            "Filter by event type",
            ["All", "violence", "smoking"],
        )
    with col_filter_date:
        filter_date = st.text_input(
            "Filter by date (YYYY-MM-DD; leave blank to show all dates)",
            value="",
        )

    if filter_type != "All":
        events_df = events_df[events_df["type"] == filter_type]
    if filter_date.strip():
        events_df = events_df[
            events_df["time"].str.startswith(filter_date.strip())
        ]

    visible_columns = ["time", "type"]
    if "mag" in events_df.columns:
        visible_columns.append("mag")
    if "person_id" in events_df.columns:
        visible_columns.append("person_id")

    st.dataframe(
        events_df[visible_columns].reset_index(drop=True),
        use_container_width=True,
        height=280,
    )

    # Display the snapshot associated with the selected event.
    st.subheader("View Snapshot")
    max_index = max(len(events_df) - 1, 0)
    selected_index = st.number_input(
        f"Select a row (0 = most recent; maximum {max_index})",
        min_value=0,
        max_value=max_index,
        value=0,
    )

    if not events_df.empty:
        selected_event = events_df.iloc[selected_index]
        snapshot_name = selected_event.get("snapshot", "")
        snapshot_path = SNAPS / snapshot_name if snapshot_name else None

        if snapshot_path is not None and snapshot_path.is_file():
            st.image(
                str(snapshot_path),
                caption=(
                    f"{selected_event['type'].upper()} — "
                    f"{selected_event['time']}"
                ),
                use_container_width=True,
            )

            col_delete, _ = st.columns([1, 3])
            with col_delete:
                if st.button("🗑️ Delete This Snapshot"):
                    try:
                        snapshot_path.unlink(missing_ok=True)
                        if EVENTS.exists():
                            all_events = json.loads(
                                EVENTS.read_text(encoding="utf-8")
                            )
                            all_events = [
                                event
                                for event in all_events
                                if event.get("snapshot") != snapshot_name
                            ]
                            EVENTS.write_text(
                                json.dumps(all_events, ensure_ascii=False),
                                encoding="utf-8",
                            )
                        st.toast("✅ Snapshot deleted.")
                    except (OSError, json.JSONDecodeError) as error:
                        st.toast(f"❌ Error: {error}")
                    # The timed refresh below updates the dashboard.
        else:
            st.warning("No snapshot is available for the selected event.")

    st.divider()

    # Clear all event history and snapshots.
    st.subheader("⚠️ Clear All History")
    if st.button("🗑️ Delete All Events and Snapshots"):
        try:
            deleted_count = 0
            if SNAPS.exists():
                for snapshot_file in SNAPS.glob("*.jpg"):
                    snapshot_file.unlink()
                    deleted_count += 1
            if EVENTS.exists():
                EVENTS.write_text("[]", encoding="utf-8")
            st.toast(
                f"✅ Deleted {deleted_count} snapshots and cleared all event history."
            )
        except OSError as error:
            st.toast(f"❌ Error: {error}")

# Refresh the dashboard every three seconds.
try:
    time.sleep(3)
    st.rerun()
except Exception:
    pass
