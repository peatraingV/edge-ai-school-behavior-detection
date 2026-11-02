"""Flask server that exposes the latest camera frame to the dashboard."""

import logging
import os

from flask import Flask, render_template_string, send_file


app = Flask(__name__)

# Suppress routine Werkzeug access logs; keep error messages visible.
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.ERROR)

SHM_PATH = "/dev/shm/cam_live/latest_frame.jpg"

# The browser polls this endpoint for the latest frame approximately every 75 ms.
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Camera Monitoring System</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #1e1e1e; color: white; text-align: center; margin-top: 50px; }
        .cam-container { border: 4px solid #444; border-radius: 12px; display: inline-block; overflow: hidden; box-shadow: 0 10px 20px rgba(0,0,0,0.5); }
        img { max-width: 1000px; width: 100%; height: auto; display: block; }
        .status { margin-bottom: 20px; font-size: 1.2rem; color: #4ade80; }
    </style>
</head>
<body>
    <h1>🎥 Intelligent Surveillance System</h1>
    <div class="cam-container">
        <!-- The image is continuously refreshed by JavaScript. -->
        <img id="cam_feed" src="/latest_frame.jpg" alt="Camera is initializing...">
    </div>

    <script>
        const imgElement = document.getElementById("cam_feed");
        // Append a timestamp to prevent the browser from reusing a cached frame.
        setInterval(() => {
            imgElement.src = "/latest_frame.jpg?t=" + Date.now();
        }, 75);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/latest_frame.jpg")
def get_frame():
    # Check whether the AI pipeline has written a frame to shared memory.
    if not os.path.exists(SHM_PATH):
        # The browser will retry when the next polling interval starts.
        return "Not Ready", 404

    # Serve the JPEG directly from shared memory.
    return send_file(SHM_PATH, mimetype="image/jpeg")


if __name__ == "__main__":
    print("🚀 Flask server starting at http://localhost:5000")
    # Disable debug mode and the reloader to avoid spawning extra processes.
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
