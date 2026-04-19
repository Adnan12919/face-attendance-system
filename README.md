🎓 Face Recognition Attendance System

An AI-powered smart attendance system that uses face recognition + anti-proxy detection to automate classroom attendance.

---
```
🚀 Features
✅ Real-time face detection using OpenCV
✅ High-accuracy recognition using ArcFace (ONNX)
✅ ORB fallback (works even without model)
✅ Teacher verification before session start
✅ Live student attendance tracking
✅ Anti-proxy detection (prevents fake attendance)
✅ Real-time UI updates using SSE (no refresh needed)
✅ SQLite database for storage
✅ Dark / Light theme UI
```
---

🏗️ Tech Stack

|Component | Technology |
|---|---|
|Backend | Python (Flask, OpenCV, ONNX Runtime) |
|Frontend |	HTML, CSS, JavaScript |
|Database |	SQLite |
|AI Model |	ArcFace |

---

📂 Project Structure
```
face_attendance_system/
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

⚙️ Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/Adnan12919/face-attendance-system.git
cd face-attendance-system

---

2️⃣ Install dependencies

pip install flask opencv-python numpy onnxruntime pillow

---

3️⃣ Add face images

Add images in this format:

registered_faces/
  teacher/
    teacherID_Name.jpg
  students/
    studentID_Name.jpg

📌 Naming rule:

<ID>_<FirstName>_<LastName>.jpg

---

## 📁 Face Registration Guidelines

### 👨‍🏫 Teacher

* File format:

```id="x4b2np"
teacherID_Name.jpg
```

---

### 🎓 Students

* File format:

```id="p3l8zs"
studentID_Name.jpg
```

📌 The filename (without extension) is used as the **attendance ID**

---

### 🖼️ Supported Formats

```id="y9k1df"
.jpg  .jpeg  .png  .bmp  .webp
```

---

### ⚠️ Important Rules

* Each image must contain **only ONE clearly visible face**
* Avoid:

  * Blurry images
  * Multiple faces
  * Side angles
  * Low lighting

✔ Best results:

* Front-facing image
* Good lighting
* Clear face

---

4️⃣ Download ArcFace model (Optional but recommended)

Download:
https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx

Place it in:

models/arcface/

👉 If not added, system will use ORB fallback.

---

5️⃣ Run the project

python run.py

Open in browser:

http://localhost:5000

---

🧠 How It Works

1.Teacher face is verified first
2.Attendance session starts
3.Students are detected via webcam
4.Faces are matched using AI model
5.Attendance is marked in database
6.Proxy detection prevents fake faces (photo/screens)

---

📸 Screenshots

Add your project screenshots here

screenshots/ui.png
pbl2/screenshots/ui1.png 
pbl2/screenshots/ui2.png

screenshots/attendance.png
pbl2/screenshots/attendance.png 
pbl2/screenshots/teacher_verification.png


---

👨‍💻 Authors

* Adnan Rahman
* Rishabh Saurabh

---

🎯 Future Improvements
📱 Mobile app integration
☁️ Cloud deployment
🧑‍🏫 Multi-class support
📊 Analytics dashboard

---

⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!