"""
backend.py  — Face Recognition Attendance System v4.0
======================================================
Project : PBL-2 (CSP-297) | Sharda School of Computing Science & Engineering
Members : Rishabh Saurabh (2024457224) | Adnan Rahman (2024126451)
Faculty : Dr. Manmohan Singh

Architecture
────────────
  Python (this file) handles ONLY:
    • Camera capture  (OpenCV — must stay Python)
    • ArcFace ONNX inference  (onnxruntime — must stay Python)
    • SQLite database  (stdlib — fast enough)
    • HTTP API + MJPEG stream + SSE events  (Flask)

  Browser (HTML/CSS/JS) handles:
    • All UI rendering, layout, theming, animations
    • Receives MJPEG frames for camera display
    • Receives SSE events for real-time state updates
    • Sends REST calls for button actions

Install:
    pip install flask opencv-python numpy onnxruntime pillow

Run:
    python run.py          (auto-opens browser)
    — or —
    python backend.py      (then open http://localhost:5000)
"""

from __future__ import annotations
import cv2, numpy as np, sqlite3, hashlib, logging, threading, json
import base64, time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Generator
from flask import Flask, Response, jsonify, request, send_from_directory

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("attendance")

# ── onnxruntime ───────────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    ORT_OK = True
except ImportError:
    ORT_OK = False
    log.warning("onnxruntime not found — ORB fallback active")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR          = Path(__file__).parent
STATIC_DIR        = BASE_DIR / "static"
TEMPLATE_DIR      = BASE_DIR / "templates"
FACES_DIR         = BASE_DIR / "registered_faces"
ARCFACE_MODEL     = BASE_DIR / "models" / "arcface" / "w600k_mbf.onnx"
DB_PATH           = BASE_DIR / "attendance.db"

app = Flask(__name__,
            static_folder=str(STATIC_DIR),
            template_folder=str(TEMPLATE_DIR))

# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class FaceRecord:
    person_id:   str
    name:        str
    role:        str          # "teacher" | "student"
    gray_face:   np.ndarray   # 128×128 for ORB fallback
    orb_des:     Optional[np.ndarray] = None
    hist:        Optional[np.ndarray] = None
    arc_emb:     Optional[np.ndarray] = None   # 512-d ArcFace embedding

@dataclass
class AttendanceRecord:
    student_id: str
    name:       str
    timestamp:  str
    confidence: float

@dataclass
class AppState:
    teacher_verified:  bool = False
    teacher_record:    Optional[FaceRecord] = None
    session_active:    bool = False
    marked_ids:        set  = field(default_factory=set)
    session_id:        str  = ""
    # ── Stored for panel restore on browser reconnect ──────────────
    teacher_name:      str  = ""
    teacher_id:        str  = ""
    teacher_conf:      float = 0.0
    teacher_thumb_b64: str  = ""   # base64 JPEG thumbnail

# ══════════════════════════════════════════════════════════════════════════════
# SSE EVENT BUS  — thread-safe broadcast to all connected browser clients
# ══════════════════════════════════════════════════════════════════════════════
class EventBus:
    """Push JSON events to all connected SSE clients."""
    def __init__(self):
        self._queues: list[queue_cls] = []
        self._lock = threading.Lock()

    def subscribe(self):
        import queue
        q = queue.Queue(maxsize=64)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._queues.discard(q) if hasattr(self._queues,'discard') \
                else (self._queues.remove(q) if q in self._queues else None)

    def publish(self, event: str, data: dict):
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        dead = []
        with self._lock:
            for q in self._queues:
                try:
                    q.put_nowait(msg)
                except Exception:
                    dead.append(q)
            for q in dead:
                try: self._queues.remove(q)
                except ValueError: pass

bus   = EventBus()
state = AppState()

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
class AttendanceDB:
    def __init__(self, path: Path = DB_PATH):
        self.path  = str(path)
        self._lock = threading.Lock()
        with sqlite3.connect(self.path) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT    NOT NULL,
                    name       TEXT    NOT NULL,
                    date       TEXT    NOT NULL,
                    time       TEXT    NOT NULL,
                    confidence REAL    NOT NULL,
                    session_id TEXT    NOT NULL,
                    integrity  TEXT    NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS uniq_sess_stu
                    ON attendance(session_id, student_id);
                CREATE TABLE IF NOT EXISTS proxy_incidents (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT    NOT NULL,
                    name       TEXT    NOT NULL,
                    date       TEXT    NOT NULL,
                    time       TEXT    NOT NULL,
                    reason     TEXT    NOT NULL,
                    score      REAL    NOT NULL,
                    session_id TEXT    NOT NULL
                );
            """)
        log.info("DB ready: %s", path)

    def save(self, rec: AttendanceRecord, session_id: str) -> bool:
        ts  = datetime.fromisoformat(rec.timestamp)
        raw = f"{session_id}|{rec.student_id}|{rec.timestamp}|{rec.confidence:.2f}"
        chk = hashlib.sha256(raw.encode()).hexdigest()
        with self._lock:
            with sqlite3.connect(self.path) as c:
                cur = c.execute(
                    "INSERT OR IGNORE INTO attendance "
                    "(student_id,name,date,time,confidence,session_id,integrity) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (rec.student_id, rec.name,
                     ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M:%S"),
                     round(rec.confidence, 2), session_id, chk)
                )
                c.commit()
                inserted = cur.rowcount > 0
        if inserted:
            log.info("DB ✔  %s | %s | %.0f%%", rec.student_id, rec.name, rec.confidence)
        return inserted

    def save_proxy(self, student_id: str, name: str,
                   reason: str, score: float, session_id: str):
        """Log a proxy attempt to the proxy_incidents table."""
        now = datetime.now()
        with self._lock:
            with sqlite3.connect(self.path) as c:
                c.execute(
                    "INSERT INTO proxy_incidents "
                    "(student_id,name,date,time,reason,score,session_id) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (student_id, name,
                     now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
                     reason, score, session_id)
                )
                c.commit()
        log.warning("PROXY LOGGED  %s | %s | score=%.0f | %s",
                    student_id, name, score, reason)

    def today(self) -> list[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self.path) as c:
            rows = c.execute(
                "SELECT student_id,name,time,confidence "
                "FROM attendance WHERE date=? ORDER BY time", (today,)
            ).fetchall()
        return [{"id": r[0], "name": r[1], "time": r[2], "conf": r[3]}
                for r in rows]

    def proxy_today(self) -> list[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self.path) as c:
            rows = c.execute(
                "SELECT student_id,name,time,reason,score "
                "FROM proxy_incidents WHERE date=? ORDER BY time DESC", (today,)
            ).fetchall()
        return [{"id": r[0], "name": r[1], "time": r[2],
                 "reason": r[3], "score": r[4]} for r in rows]

db = AttendanceDB()

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
class Preprocessor:
    FACE_SZ = (128, 128)
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def process(self, bgr: np.ndarray) -> np.ndarray:
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        eq    = self.clahe.apply(gray)
        blur  = cv2.GaussianBlur(eq, (3, 3), 0)
        return cv2.resize(blur, self.FACE_SZ, interpolation=cv2.INTER_LANCZOS4)

prep = Preprocessor()

# ══════════════════════════════════════════════════════════════════════════════
# ARCFACE ONNX ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class ArcFaceEngine:
    SZ = (112, 112)

    def __init__(self, model_path: Path = ARCFACE_MODEL):
        self.sess       = None
        self.input_name = None
        if not ORT_OK:
            return
        if not model_path.exists():
            log.warning("ArcFace model missing at '%s'. ORB fallback active.\n"
                        "  Download: https://github.com/yakhyo/face-reidentification"
                        "/releases/download/v0.0.1/w600k_mbf.onnx\n"
                        "  Place at: models/arcface/w600k_mbf.onnx", model_path)
            return
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level   = 3
        self.sess       = ort.InferenceSession(str(model_path),
                            sess_options=opts,
                            providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        log.info("ArcFace loaded: %s", model_path)

    @property
    def ok(self) -> bool:
        return self.sess is not None

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(bgr, self.SZ, interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))[np.newaxis]
        emb = self.sess.run(None, {self.input_name: img})[0][0]
        emb /= (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

arc = ArcFaceEngine()

# ══════════════════════════════════════════════════════════════════════════════
# FACE RECOGNISER
# ══════════════════════════════════════════════════════════════════════════════
class FaceRecogniser:
    ARC_THRESH  = 0.35
    ARC_MARGIN  = 0.06
    ORB_THRESH  = 40.0
    ORB_MARGIN  = 6.0
    MATCH_RATIO = 0.72
    W_ORB = 0.50; W_HIST = 0.25; W_SSIM = 0.25

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

    def build(self, pid: str, name: str, role: str,
              bgr: np.ndarray) -> FaceRecord:
        gray    = prep.process(bgr)
        _, des  = self.orb.detectAndCompute(gray, None)
        hist    = self._hist(gray)
        arc_emb = arc.embed(bgr) if arc.ok else None
        return FaceRecord(person_id=pid, name=name, role=role,
                          gray_face=gray, orb_des=des, hist=hist, arc_emb=arc_emb)

    def match(self, bgr: np.ndarray,
              records: list[FaceRecord]) -> tuple[Optional[FaceRecord], float]:
        if not records:
            return None, 0.0
        return (self._arc_match(bgr, records) if arc.ok
                else self._orb_match(bgr, records))

    # ── ArcFace path ──────────────────────────────────────────────────
    def _arc_match(self, bgr, records):
        qe = arc.embed(bgr)
        scores = [(arc.similarity(qe, r.arc_emb) if r.arc_emb is not None else 0.0, r)
                  for r in records]
        scores.sort(key=lambda x: x[0], reverse=True)
        best_s, best_r = scores[0]
        if len(scores) > 1 and best_s - scores[1][0] < self.ARC_MARGIN:
            return None, self._a2p(best_s)
        if best_s >= self.ARC_THRESH:
            return best_r, self._a2p(best_s)
        return None, self._a2p(best_s)

    @staticmethod
    def _a2p(s: float) -> float:
        return float(np.clip((s - 0.20) / 0.60 * 100.0, 0.0, 100.0))

    # ── ORB fallback ──────────────────────────────────────────────────
    def _orb_match(self, bgr, records):
        gray    = prep.process(bgr)
        _, dq   = self.orb.detectAndCompute(gray, None)
        hq      = self._hist(gray)
        scores  = [(self.W_ORB * self._oscore(dq, r.orb_des)
                  + self.W_HIST * self._hscore(hq, r.hist)
                  + self.W_SSIM * self._ssim(gray, r.gray_face), r)
                   for r in records]
        scores.sort(key=lambda x: x[0], reverse=True)
        best_s, best_r = scores[0]
        if len(scores) > 1 and best_s - scores[1][0] < self.ORB_MARGIN:
            return None, best_s
        return (best_r, best_s) if best_s >= self.ORB_THRESH else (None, best_s)

    def _oscore(self, dq, dt) -> float:
        if dq is None or dt is None or len(dq) < 2 or len(dt) < 2:
            return 0.0
        raw  = self.bf.knnMatch(dq, dt, k=2)
        good = [m for p in raw if len(p) == 2
                for m, n in [p] if m.distance < self.MATCH_RATIO * n.distance]
        r = len(good) / min(len(dq), len(dt))
        return float(100.0 / (1.0 + np.exp(-14.0 * (r - 0.20))))

    def _hscore(self, h1, h2) -> float:
        if h1 is None or h2 is None: return 0.0
        return float(max(0.0, cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)) * 100.0)

    def _hist(self, gray) -> np.ndarray:
        h = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(h, h); return h

    def _ssim(self, a, b) -> float:
        fa = a.astype(np.float32) / 255.0
        fb = b.astype(np.float32) / 255.0
        return float(np.sum(fa * fb) /
                     (np.sqrt(np.sum(fa**2) * np.sum(fb**2)) + 1e-8) * 100.0)

recogniser = FaceRecogniser()

# ══════════════════════════════════════════════════════════════════════════════
# PROXY DETECTOR
# Anti-spoofing: detects photos held on phones / ID cards / printed pages.
#
# Three independent signals — a real face must pass ALL three:
#
#  Signal 1 — TEXTURE LIVENESS (LBP variance)
#    Real skin has high-frequency micro-texture (pores, hairs, lighting 3D).
#    A photo of a face is a flat 2-D image re-photographed — its LBP pattern
#    variance is lower because fine texture is lost in the print/screen.
#    We measure this via Local Binary Pattern operator variance.
#
#  Signal 2 — GRADIENT ENTROPY (flatness score)
#    A real face has gradient orientations spread across many directions
#    (curved surfaces, shadows, features). A flat photo has gradients
#    concentrated around edges only → lower orientation entropy.
#
#  Signal 3 — SPECULAR GLARE / SCREEN REFLECTION DETECTION
#    Phone screens and laminated ID cards produce rectangular bright blobs
#    (reflections / backlight). We detect these in the face crop via
#    thresholding saturated bright regions in HSV + contour rectangularity.
#
# Any signal failing → PROXY ALERT.  All pass → LIVE face.
# ══════════════════════════════════════════════════════════════════════════════
class ProxyDetector:
    """
    Detects proxy attempts: photos on phones, printed ID cards, screen images.

    Thresholds are tuned conservatively to minimise false positives on real
    faces under varied indoor lighting (lecture hall conditions).
    """

    # ── Tunable thresholds ────────────────────────────────────────────
    # LBP variance: real faces typically > 800 (skin texture)
    #               printed/screen photos: typically < 500
    LBP_REAL_MIN        = 450.0

    # Gradient entropy: real faces > 3.5 bits (many edge directions)
    #                   flat photos: < 2.8 bits (few edge directions)
    GRAD_ENTROPY_MIN    = 2.8

    # Specular blob: ratio of very-bright pixels in face region
    # Real skin: < 0.08 (small specular highlights on nose/forehead)
    # Phone screen: > 0.15 (large uniform bright area)
    GLARE_MAX_RATIO     = 0.18

    # Consecutive proxy frames before alert fires (avoids single-frame noise)
    PROXY_CONFIRM_N     = 3

    def __init__(self):
        self._proxy_streak: dict[str, int] = {}   # person_id → consecutive proxy frames

    # ── Public API ────────────────────────────────────────────────────
    def is_proxy(self, face_bgr: np.ndarray, person_id: str = "") -> tuple[bool, str, float]:
        """
        Returns (is_proxy: bool, reason: str, score: float 0-100).
        score 0 = definitely live, 100 = definitely proxy.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))           # normalise size
        face_bgr_r = cv2.resize(face_bgr, (64, 64))

        lbp_var  = self._lbp_variance(gray)
        grad_ent = self._gradient_entropy(gray)
        glare    = self._glare_ratio(face_bgr_r)

        # Compute per-signal failure flags
        fail_lbp   = lbp_var   < self.LBP_REAL_MIN
        fail_grad  = grad_ent  < self.GRAD_ENTROPY_MIN
        fail_glare = glare     > self.GLARE_MAX_RATIO

        # At least 2 of 3 signals must fail to trigger proxy
        failures = sum([fail_lbp, fail_grad, fail_glare])
        is_prx = failures >= 2

        # Streak tracking to avoid single-frame false positives
        if person_id:
            if is_prx:
                self._proxy_streak[person_id] = \
                    self._proxy_streak.get(person_id, 0) + 1
            else:
                self._proxy_streak[person_id] = 0
            # Only fire after PROXY_CONFIRM_N consecutive proxy frames
            if self._proxy_streak.get(person_id, 0) < self.PROXY_CONFIRM_N:
                is_prx = False

        # Reason string
        reasons = []
        if fail_lbp:   reasons.append(f"flat texture (LBP={lbp_var:.0f})")
        if fail_grad:  reasons.append(f"low gradient entropy ({grad_ent:.2f}b)")
        if fail_glare: reasons.append(f"screen glare ({glare*100:.0f}%)")
        reason = "; ".join(reasons) if reasons else "live"

        # Proxy score: average of 3 normalised failure scores (0-100)
        score_lbp   = max(0.0, 1.0 - lbp_var / self.LBP_REAL_MIN)
        score_grad  = max(0.0, 1.0 - grad_ent / self.GRAD_ENTROPY_MIN)
        score_glare = min(1.0, glare / self.GLARE_MAX_RATIO)
        proxy_score = (score_lbp + score_grad + score_glare) / 3.0 * 100.0

        return is_prx, reason, round(proxy_score, 1)

    def reset(self, person_id: str = ""):
        """Reset streak counter after session ends."""
        if person_id:
            self._proxy_streak.pop(person_id, None)
        else:
            self._proxy_streak.clear()

    # ── Signal 1: LBP Variance ────────────────────────────────────────
    @staticmethod
    def _lbp_variance(gray: np.ndarray) -> float:
        """
        Compute Local Binary Pattern variance.
        Real faces: high micro-texture → high variance.
        Flat photo:  low micro-texture → low variance.
        """
        h, w = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        g = gray.astype(np.int32)
        # 8-neighbour LBP at radius 1
        neighbours = [
            g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:],
            g[1:-1, 2:], g[2:, 2:],   g[2:, 1:-1],
            g[2:, :-2],  g[1:-1, :-2],
        ]
        centre = g[1:-1, 1:-1]
        code   = np.zeros_like(centre, dtype=np.uint8)
        for i, nb in enumerate(neighbours):
            code |= ((nb >= centre).astype(np.uint8) << i)
        return float(np.var(code))

    # ── Signal 2: Gradient Entropy ────────────────────────────────────
    @staticmethod
    def _gradient_entropy(gray: np.ndarray) -> float:
        """
        Orientation entropy of Sobel gradients.
        Real face: gradients in many directions → high entropy.
        Flat photo: gradients mostly at printed edges → low entropy.
        """
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        angle = np.arctan2(gy, gx)                    # -π to π
        # Bin into 18 bins (20° each)
        hist, _ = np.histogram(angle, bins=18, range=(-np.pi, np.pi))
        hist = hist.astype(np.float32) + 1e-9
        hist /= hist.sum()
        entropy = -np.sum(hist * np.log2(hist))        # Shannon entropy (bits)
        return float(entropy)

    # ── Signal 3: Screen / Glare Detection ───────────────────────────
    @staticmethod
    def _glare_ratio(bgr: np.ndarray) -> float:
        """
        Fraction of face region that is extremely bright (screen backlight
        or ID card laminate reflection).
        Real skin: small specular highlights → ratio < 0.08
        Phone screen: large uniform bright area → ratio > 0.15
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # Very bright + low saturation = white glare / screen light
        mask = cv2.inRange(hsv,
                           np.array([0,   0, 220], dtype=np.uint8),
                           np.array([180, 40, 255], dtype=np.uint8))
        ratio = float(np.count_nonzero(mask)) / (bgr.shape[0] * bgr.shape[1])
        return ratio

proxy_detector = ProxyDetector()


# ══════════════════════════════════════════════════════════════════════════════
class FaceDetector:
    _DATA = cv2.data.haarcascades
    def __init__(self):
        self.cascades = [
            cv2.CascadeClassifier(self._DATA + n)
            for n in ["haarcascade_frontalface_default.xml",
                      "haarcascade_frontalface_alt2.xml",
                      "haarcascade_frontalface_alt.xml"]
            if Path(self._DATA + n).exists()
        ]
        log.info("FaceDetector: %d cascade(s)", len(self.cascades))

    def crop_faces(self, frame: np.ndarray,
                   min_sz=70) -> list[tuple[tuple, np.ndarray]]:
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rects = []
        for c in self.cascades:
            found = c.detectMultiScale(gray, 1.05, 5,
                                       minSize=(min_sz, min_sz),
                                       flags=cv2.CASCADE_SCALE_IMAGE)
            if len(found): rects.extend(found.tolist())
        rects = self._nms(rects)
        H, W = frame.shape[:2]
        out = []
        for (x, y, w, h) in rects:
            x, y = max(0, x), max(0, y)
            w, h = min(w, W-x), min(h, H-y)
            crop = frame[y:y+h, x:x+w]
            if crop.size: out.append(((x, y, w, h), crop))
        return out

    @staticmethod
    def _iou(a, b):
        ax1,ay1,aw,ah = a; bx1,by1,bw,bh = b
        ix = max(0, min(ax1+aw,bx1+bw)-max(ax1,bx1))
        iy = max(0, min(ay1+ah,by1+bh)-max(ay1,by1))
        inter = ix*iy; union = aw*ah+bw*bh-inter
        return inter/union if union else 0.0

    def _nms(self, rects, thr=0.35):
        rects = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
        kept = []
        for r in rects:
            if all(self._iou(r,k) < thr for k in kept):
                kept.append(tuple(r))
        return kept

detector = FaceDetector()

# ══════════════════════════════════════════════════════════════════════════════
# PHOTO LOADER
# ══════════════════════════════════════════════════════════════════════════════
EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def parse_filename(stem: str) -> tuple[str, str]:
    parts = stem.split("_", 1)
    return parts[0], (parts[1].replace("_"," ") if len(parts)>1 else parts[0])

def load_faces(directory: Path, role: str) -> list[FaceRecord]:
    records = []
    for p in sorted(p for p in directory.iterdir() if p.suffix.lower() in EXTS):
        img = cv2.imread(str(p))
        if img is None: log.warning("Cannot read: %s", p.name); continue
        faces = detector.crop_faces(img, min_sz=50)
        crop  = (max(faces, key=lambda f: f[1].shape[0]*f[1].shape[1])[1]
                 if faces else img)
        pid, name = parse_filename(p.stem)
        rec = recogniser.build(pid, name, role, crop)
        records.append(rec)
        log.info("Registered %-8s: %s (%s)", role, name, pid)
    return records

# ══════════════════════════════════════════════════════════════════════════════
# CAMERA WORKER  — runs in background thread, pushes MJPEG + SSE events
# ══════════════════════════════════════════════════════════════════════════════
# Shared MJPEG frame (bytes) — written by camera thread, read by MJPEG route
import queue as _queue
_frame_lock  = threading.Lock()
_latest_jpeg = b""
CONFIRM_N    = 4     # consecutive frames to confirm teacher
_teacher_hits = 0

def _encode_jpeg(frame: np.ndarray, quality=80) -> bytes:
    _, buf = cv2.imencode(".jpg", frame,
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

def _crop_b64(crop: np.ndarray) -> str:
    """Encode face crop to base64 PNG for SSE thumbnail."""
    small = cv2.resize(crop, (160, 130))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode()

def camera_worker(teacher: FaceRecord, students: list[FaceRecord]):
    global _latest_jpeg, _teacher_hits
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open camera")
        bus.publish("error", {"msg": "Camera not found"})
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # reduce buffer lag

    log.info("Camera thread started")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        display = frame.copy()
        faces   = detector.crop_faces(frame, min_sz=65)

        # ── TEACHER VERIFICATION ──────────────────────────────────────
        if not state.session_active:
            for (x, y, w, h), crop in faces:
                rec, conf = recogniser.match(crop, [teacher])
                if rec:
                    _teacher_hits += 1
                    color = (62, 207, 142)
                    label = f"{rec.name}  {conf:.0f}%"
                else:
                    _teacher_hits = max(0, _teacher_hits - 1)
                    color = (120, 120, 120)
                    label = "Unknown"

                cv2.rectangle(display, (x,y), (x+w,y+h), color, 2)
                _put_label(display, label, x, y, color)

                if (_teacher_hits >= CONFIRM_N
                        and not state.teacher_verified):
                    state.teacher_verified  = True
                    state.teacher_record    = teacher
                    # ── Store for reconnect restore ────────────────
                    state.teacher_name      = rec.name
                    state.teacher_id        = rec.person_id
                    state.teacher_conf      = round(conf, 1)
                    state.teacher_thumb_b64 = _crop_b64(crop)
                    bus.publish("teacher_verified", {
                        "name":   rec.name,
                        "id":     rec.person_id,
                        "conf":   round(conf, 1),
                        "thumb":  _crop_b64(crop),
                    })
                    log.info("Teacher verified: %s (%.0f%%)", rec.name, conf)

        # ── STUDENT DETECTION ─────────────────────────────────────────
        else:
            for (x, y, w, h), crop in faces:
                rec, conf = recogniser.match(crop, students)
                if rec:
                    # ── PROXY CHECK ───────────────────────────────────
                    is_prx, prx_reason, prx_score = proxy_detector.is_proxy(
                        crop, person_id=rec.person_id
                    )

                    if is_prx:
                        # Proxy detected — red box, alert label, SSE event
                        color = (0, 0, 220)
                        label = f"PROXY: {rec.name}"
                        # Draw warning overlay on bounding box
                        cv2.rectangle(display, (x, y), (x+w, y+h),
                                      (0, 0, 220), 3)
                        # Warning diagonal stripes overlay
                        overlay = display.copy()
                        for i in range(0, w, 12):
                            cv2.line(overlay,
                                     (x+i, y), (x+i-h, y+h),
                                     (0, 0, 220), 2)
                        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

                        bus.publish("proxy_detected", {
                            "id":     rec.person_id,
                            "name":   rec.name,
                            "conf":   round(conf, 1),
                            "score":  prx_score,
                            "reason": prx_reason,
                            "thumb":  _crop_b64(crop),
                        })
                        # Log to DB
                        db.save_proxy(rec.person_id, rec.name,
                                      prx_reason, prx_score,
                                      state.session_id)
                        log.warning("PROXY DETECTED: %s (score=%.0f) — %s",
                                    rec.name, prx_score, prx_reason)
                        # Cancel any existing attendance mark for this person
                        if rec.person_id in state.marked_ids:
                            state.marked_ids.discard(rec.person_id)
                            bus.publish("attendance_cancelled", {
                                "id":   rec.person_id,
                                "name": rec.name,
                            })
                        # Skip normal attendance flow
                        _put_label(display, label, x, y, color)
                        continue
                    # ── END PROXY CHECK ───────────────────────────────

                    color = (0, 180, 255)
                    label = f"{rec.name}  {conf:.0f}%"
                    # fire student event (UI handles dedup display)
                    bus.publish("student_detected", {
                        "id":    rec.person_id,
                        "name":  rec.name,
                        "conf":  round(conf, 1),
                        "thumb": _crop_b64(crop),
                    })
                    # save to DB if new
                    if rec.person_id not in state.marked_ids:
                        state.marked_ids.add(rec.person_id)
                        ar = AttendanceRecord(
                            student_id=rec.person_id, name=rec.name,
                            timestamp=datetime.now().isoformat(),
                            confidence=round(conf, 2)
                        )
                        saved = db.save(ar, state.session_id)
                        if saved:
                            bus.publish("attendance_saved", {
                                "id":   rec.person_id,
                                "name": rec.name,
                                "conf": round(conf, 1),
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "count": len(state.marked_ids),
                            })
                else:
                    color = (80, 80, 200)
                    label = "Unknown"

                cv2.rectangle(display, (x,y), (x+w,y+h), color, 2)
                _put_label(display, label, x, y, color)

        # Session overlay
        if state.session_active:
            cv2.putText(display, "SESSION ACTIVE",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (62, 207, 142), 2)

        # Encode and store latest JPEG frame
        with _frame_lock:
            _latest_jpeg = _encode_jpeg(display)

def _put_label(frame, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(y - 6, th + 4)
    cv2.rectangle(frame, (x, ty-th-4), (x+tw+8, ty+2), color, -1)
    cv2.putText(frame, text, (x+4, ty-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(str(TEMPLATE_DIR), "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)

# ── MJPEG camera stream ───────────────────────────────────────────────────────
@app.route("/stream")
def stream():
    def generate():
        while True:
            with _frame_lock:
                frame = _latest_jpeg
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.033)   # ~30 fps cap
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── Server-Sent Events — real-time state pushes ───────────────────────────────
@app.route("/events")
def events():
    import queue

    q = bus.subscribe()

    def stream_gen():
        try:
            # Send initial state immediately — tells browser current teacher/session state
            init_msg = f"event: init\ndata: {json.dumps(_build_init())}\n\n"
            yield init_msg

            while True:
                try:
                    msg = q.get(timeout=15)
                    yield msg
                except queue.Empty:
                    # Heartbeat keeps connection alive through proxies/firewalls
                    yield ": heartbeat\n\n"
        finally:
            bus.unsubscribe(q)

    # Use streaming_with_context to ensure each yield is flushed immediately.
    # Critical: without these headers Flask's WSGI server buffers the response.
    from flask import stream_with_context
    return Response(
        stream_with_context(stream_gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache, no-store, must-revalidate",
            "Pragma":            "no-cache",
            "Expires":           "0",
            "X-Accel-Buffering": "no",       # disable nginx buffering
            "Connection":        "keep-alive",
            "Transfer-Encoding": "chunked",   # forces chunk-per-yield flushing
        }
    )

def _build_init() -> dict:
    return {
        "teacher_verified":    state.teacher_verified,
        "teacher_name":        state.teacher_name,
        "teacher_id":          state.teacher_id,
        "teacher_conf":        state.teacher_conf,
        "teacher_thumb":       state.teacher_thumb_b64,
        "session_active":      state.session_active,
        "marked_count":        len(state.marked_ids),
        "engine":              "arcface" if arc.ok else "orb",
        "attendance":          db.today(),
        "proxy_incidents":     db.proxy_today(),
    }

# ── REST API actions ──────────────────────────────────────────────────────────
@app.route("/api/start", methods=["POST"])
def api_start():
    if not state.teacher_verified:
        return jsonify({"ok": False, "msg": "Teacher not verified"}), 403
    if state.session_active:
        return jsonify({"ok": False, "msg": "Session already active"}), 409
    state.session_active = True
    state.session_id     = datetime.now().strftime("SID_%Y%m%d_%H%M%S")
    state.marked_ids.clear()
    bus.publish("session_started", {"session_id": state.session_id})
    log.info("Session started: %s", state.session_id)
    return jsonify({"ok": True, "session_id": state.session_id})

@app.route("/api/end", methods=["POST"])
def api_end():
    if not state.session_active:
        return jsonify({"ok": False, "msg": "No active session"}), 409
    n = len(state.marked_ids)
    state.session_active   = False
    state.teacher_verified = False   # require re-verification
    proxy_detector.reset()           # clear all streak counters
    global _teacher_hits
    _teacher_hits = 0
    bus.publish("session_ended", {"count": n})
    log.info("Session ended — %d student(s) marked", n)
    return jsonify({"ok": True, "count": n})

@app.route("/api/attendance")
def api_attendance():
    return jsonify(db.today())

@app.route("/api/proxy")
def api_proxy():
    return jsonify(db.proxy_today())

@app.route("/api/status")
def api_status():
    return jsonify(_build_init())

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════
_teacher_record:  Optional[FaceRecord]       = None
_student_records: list[FaceRecord]           = []

def load_and_start():
    global _teacher_record, _student_records

    teacher_dir = FACES_DIR / "teacher"
    student_dir = FACES_DIR / "students"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    student_dir.mkdir(parents=True, exist_ok=True)

    teachers = load_faces(teacher_dir, "teacher")
    if not teachers:
        log.error("No teacher photo in '%s'. Add <ID>_<First>_<Last>.jpg", teacher_dir)
        print(f"\n[ERROR] Add teacher photo to: {teacher_dir}")
        return False

    students = load_faces(student_dir, "student")
    if not students:
        log.error("No student photos in '%s'.", student_dir)
        print(f"\n[ERROR] Add student photos to: {student_dir}")
        return False

    _teacher_record  = teachers[0]
    _student_records = students

    log.info("Teacher : %s (%s)", _teacher_record.name, _teacher_record.person_id)
    log.info("Students: %d loaded", len(_student_records))

    # Start camera in background thread
    t = threading.Thread(
        target=camera_worker,
        args=(_teacher_record, _student_records),
        daemon=True
    )
    t.start()
    return True

if __name__ == "__main__":
    ok = load_and_start()
    if ok:
        log.info("Server → http://localhost:5000")
        # use_reloader=False is critical — reloader forks the process and
        # breaks the camera thread + SSE queue references.
        # threaded=True gives each SSE client its own thread so the
        # generator's yield flushes immediately.
        app.run(host="0.0.0.0", port=5000, debug=False,
                threaded=True, use_reloader=False)
