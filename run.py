"""
run.py  —  Launcher for Face Recognition Attendance System v4.0
================================================================
Usage:
    python run.py

This will:
  1. Load teacher + student face photos and build embeddings
  2. Start the Flask backend on http://localhost:5000
  3. Automatically open your browser to the UI
"""

import sys, os, time, threading, webbrowser
from pathlib import Path

# ── Disable ALL Python output buffering ──────────────────────────
# This is critical for SSE: Python's default stdout buffer can hold
# back server-sent events before flushing them to the browser.
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

# Make sure we run from the project root directory
os.chdir(Path(__file__).parent)

# ── Check minimum dependencies ────────────────────────────────────
def check_deps():
    missing = []
    for pkg, import_name in [
        ("flask",          "flask"),
        ("opencv-python",  "cv2"),
        ("numpy",          "numpy"),
        ("Pillow",         "PIL"),
        ("onnxruntime",    "onnxruntime"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n[ERROR] Missing packages. Install with:")
        print(f"  pip install {' '.join(missing)}\n")
        sys.exit(1)

check_deps()

# ── Import backend after dep check ────────────────────────────────
from backend import app, load_and_start, log

def open_browser():
    time.sleep(1.5)   # wait for Flask to be ready
    webbrowser.open("http://localhost:5000")

if __name__ == "__main__":
    print("=" * 62)
    print("  Face Recognition Attendance System v4.0")
    print("  Sharda School of Computing Science & Engineering")
    print("=" * 62)
    print()

    ok = load_and_start()
    if not ok:
        print("\n[SETUP REQUIRED]")
        print("  Place teacher photo  in: registered_faces/teacher/")
        print("  Place student photos in: registered_faces/students/")
        print("  Photo naming: <ID>_<FirstName>_<LastName>.jpg")
        print("  Example: 2024457224_Rishabh_Saurabh.jpg\n")
        print("  ArcFace model (optional, better accuracy):")
        print("  Download: https://github.com/yakhyo/face-reidentification"
              "/releases/download/v0.0.1/w600k_mbf.onnx")
        print("  Place at:  models/arcface/w600k_mbf.onnx\n")
        sys.exit(1)

    print()
    print("  → Opening browser at http://localhost:5000")
    print("  → Press Ctrl+C to stop\n")

    threading.Thread(target=open_browser, daemon=True).start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
