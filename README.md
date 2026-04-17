# Face Recognition Attendance System v4.0
**PBL-2 (CSP-297) | Sharda School of Computing Science and Engineering**
Members: Rishabh Saurabh (2024457224) | Adnan Rahman (2024126451)
Faculty: Dr. Manmohan Singh

---

## Architecture (Why it's fast now)

| Component | Language | Why |
|---|---|---|
| Face detection + ArcFace inference | Python / OpenCV / ONNX | Must be Python — no other option |
| HTTP server + REST API | Python / Flask | Thin layer — negligible overhead |
| Camera MJPEG stream | Python → Browser | Browser renders natively at 30fps |
| **GUI / Layout / Theme** | **HTML + CSS** | GPU-accelerated in browser — much faster than Tkinter |
| **Real-time updates** | **JavaScript (SSE)** | Zero Python polling — push-only |
| **Dark/Light toggle** | **CSS variables** | Instant — single attribute change |
| **Attendance table** | **JS + HTML** | Native DOM — renders in <1ms |
| Database | SQLite (Python) | Stays Python — async write thread |

**Result:** Python only runs camera + AI. Everything visual is HTML/CSS/JS.

---

## Project Structure

```
face_attendance_v4/
├── run.py                       ← START HERE — launches everything
├── backend.py                   ← Python: camera, ArcFace, DB, Flask API
├── templates/
│   └── index.html               ← HTML: UI structure
├── static/
│   ├── style.css                ← CSS: dark/light theme, layout
│   ├── app.js                   ← JS: SSE events, REST calls, UI updates
│   └── logo.png                 ← Sharda logo (copy sharda_logo.png here)
├── models/
│   └── arcface/
│       └── w600k_mbf.onnx       ← ArcFace model (download below)
├── registered_faces/
│   ├── teacher/
│   │   └── T001_Dr_Manmohan_Singh.jpg
│   └── students/
│       ├── 2024457224_Rishabh_Saurabh.jpg
│       └── 2024126451_Adnan_Rahman.jpg
└── attendance.db                ← auto-created SQLite database
```

---

## Setup

### 1. Install Python dependencies
```bash
pip install flask opencv-python numpy onnxruntime pillow
```

### 2. Download ArcFace model (~13 MB) — optional but recommended
```
URL: https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx
Save to: models/arcface/w600k_mbf.onnx
```
If missing, system auto-falls back to ORB recogniser.

### 3. Add face photos
- **Teacher:** `registered_faces/teacher/T001_Dr_Manmohan_Singh.jpg`
- **Students:** `registered_faces/students/2024457224_Rishabh_Saurabh.jpg`
- Naming rule: `<ID>_<FirstName>_<LastName>.jpg`

### 4. Add logo
Copy `sharda_logo.png` → `static/logo.png`

### 5. Run
```bash
python run.py
```
Browser opens automatically at `http://localhost:5000`

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the UI |
| `/stream` | GET | MJPEG camera stream |
| `/events` | GET | Server-Sent Events stream |
| `/api/start` | POST | Start attendance session |
| `/api/end` | POST | End session |
| `/api/attendance` | GET | Today's attendance (JSON) |
| `/api/status` | GET | Current system state (JSON) |
