🎓 Face Recognition Attendance System

An AI-powered smart attendance system that uses face recognition + anti-proxy detection to automate classroom attendance.

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
🏗️ Tech Stack
Component	Technology
Backend	Python (Flask, OpenCV, ONNX Runtime)
Frontend	HTML, CSS, JavaScript
Database	SQLite
AI Model	ArcFace
📂 Project Structure
face-attendance-system/
│
├── backend.py              # Main backend logic
├── run.py                  # Entry point
├── README.md
│
├── templates/
│   └── index.html          # UI layout
│
├── static/
│   ├── style.css           # Styling
│   ├── app.js              # Frontend logic
│   └── logo.png
│
├── models/
│   └── arcface/
│       └── w600k_mbf.onnx  # (download manually)
│
├── registered_faces/
│   ├── teacher/
│   └── students/
⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Adnan12919/face-attendance-system.git
cd face-attendance-system
2️⃣ Install dependencies
pip install flask opencv-python numpy onnxruntime pillow
3️⃣ Add face images

Add images in this format:

registered_faces/
  teacher/
    T001_Name.jpg
  students/
    12345_Name.jpg

📌 Naming rule:

<ID>_<FirstName>_<LastName>.jpg
4️⃣ Download ArcFace model (Optional but recommended)

Download:
https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx

Place it in:

models/arcface/

👉 If not added, system will use ORB fallback.

5️⃣ Run the project
python run.py

Open in browser:

http://localhost:5000
🧠 How It Works
Teacher face is verified first
Attendance session starts
Students are detected via webcam
Faces are matched using AI model
Attendance is marked in database
Proxy detection prevents fake faces (photo/screens)
📸 Screenshots

Add your project screenshots here

screenshots/ui.png
screenshots/attendance.png
screenshots/proxy.png
👨‍💻 Authors
Adnan Rahman
Rishabh Saurabh
🎯 Future Improvements
📱 Mobile app integration
☁️ Cloud deployment
🧑‍🏫 Multi-class support
📊 Analytics dashboard
⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!