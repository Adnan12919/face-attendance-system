/**
 * app.js  —  Face Recognition Attendance System v4.1
 * ====================================================
 * Single clean file — no duplicates.
 */

"use strict";

// ── DOM refs ──────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const el = {
  themeBtn:         $("themeBtn"),
  engineBadge:      $("engineBadge"),
  teacherThumb:     $("teacherThumb"),
  tName:            $("tName"),
  tId:              $("tId"),
  tConf:            $("tConf"),
  tBadge:           $("tBadge"),
  tBarWrap:         $("tBarWrap"),
  tBarFill:         $("tBarFill"),
  tBarLabel:        $("tBarLabel"),
  studentThumb:     $("studentThumb"),
  sName:            $("sName"),
  sId:              $("sId"),
  sConf:            $("sConf"),
  attBody:          $("attBody"),
  attCount:         $("attCount"),
  proxyBody:        $("proxyBody"),
  proxyCount:       $("proxyCount"),
  proxyBadge:       $("proxyBadge"),
  proxyAlert:       $("proxyAlert"),
  proxyAlertName:   $("proxyAlertName"),
  proxyAlertReason: $("proxyAlertReason"),
  proxyAlertScore:  $("proxyAlertScore"),
  sessionStatus:    $("sessionStatus"),
  statusMsg:        $("statusMsg"),
  clock:            $("clock"),
  mainBtn:          $("mainBtn"),
};

// ── App state ─────────────────────────────────────────────────────
let teacherVerified = false;
let sessionActive   = false;
let markedIds       = new Set();
let proxyIds        = new Set();
let proxyCount      = 0;
let toastWrap       = null;
let proxyAlertTimer = null;

// ── Theme ─────────────────────────────────────────────────────────
function applyTheme(mode) {
  document.documentElement.setAttribute("data-theme", mode);
  el.themeBtn.textContent = (mode === "dark") ? "☀ Light" : "🌙 Dark";
  localStorage.setItem("theme", mode);
}
applyTheme(localStorage.getItem("theme") || "dark");
el.themeBtn.addEventListener("click", () => {
  applyTheme(
    document.documentElement.getAttribute("data-theme") === "dark"
      ? "light" : "dark"
  );
});

// ── Clock ─────────────────────────────────────────────────────────
(function tick() {
  const n = new Date();
  const D = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
  const M = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  el.clock.textContent =
    `${D[n.getDay()]} ${String(n.getDate()).padStart(2,"0")} ${M[n.getMonth()]} ${n.getFullYear()}` +
    `   ${String(n.getHours()).padStart(2,"0")}:${String(n.getMinutes()).padStart(2,"0")}:${String(n.getSeconds()).padStart(2,"0")}`;
  setTimeout(tick, 1000);
})();

// ── Toasts ────────────────────────────────────────────────────────
function initToastWrap() {
  toastWrap = document.createElement("div");
  toastWrap.className = "toast-wrap";
  document.body.appendChild(toastWrap);
}
function toast(msg, type) {
  const t = document.createElement("div");
  t.className = "toast " + (type || "info");
  t.textContent = msg;
  toastWrap.appendChild(t);
  setTimeout(() => { if (t.parentNode) t.remove(); }, 3200);
}

// ── Tab switching ─────────────────────────────────────────────────
window.switchTab = function(tab) {
  $("attTab").style.display   = tab === "att" ? "flex" : "none";
  $("proxyTab").style.display = tab === "proxy" ? "flex" : "none";
  $("tabAttBtn").classList.toggle("active",   tab === "att");
  $("tabProxyBtn").classList.toggle("active", tab === "proxy");
};

// ── Helpers ───────────────────────────────────────────────────────
function setThumb(wrap, b64) {
  wrap.innerHTML = "";
  const img = document.createElement("img");
  img.src = "data:image/jpeg;base64," + b64;
  img.alt = "Face";
  wrap.appendChild(img);
}

function setConfBar(fill, label, pct) {
  fill.style.width  = Math.min(100, pct) + "%";
  label.textContent = Math.round(pct) + "%";
}

function setStatus(msg, type) {
  el.statusMsg.textContent     = msg;
  el.sessionStatus.className   = "sess-dot sess-" + (type || "idle");
  el.sessionStatus.textContent = (type === "active")
    ? "● SESSION ACTIVE — scanning students…"
    : "● " + msg;
}

function setEngineBadge(engine) {
  if (engine === "arcface") {
    el.engineBadge.textContent = "⚡ ArcFace ONNX";
    el.engineBadge.className   = "engine-badge arc";
  } else {
    el.engineBadge.textContent = "⚙ ORB Fallback";
    el.engineBadge.className   = "engine-badge orb";
  }
}

function setBtnStart() {
  el.mainBtn.textContent = "▶ START ATTENDANCE";
  el.mainBtn.classList.remove("end-mode");
  el.mainBtn.disabled = !teacherVerified || sessionActive;
}

function setBtnEnd() {
  el.mainBtn.textContent = "⏹ END SESSION";
  el.mainBtn.classList.add("end-mode");
  el.mainBtn.disabled = false;
}

function populateTeacherPanel(name, id, conf, thumb) {
  el.tName.textContent  = name || "—";
  el.tId.textContent    = id   || "—";
  el.tConf.textContent  = conf ? conf + "%" : "—";
  el.tBadge.textContent = "✔  Verified";
  el.tBadge.classList.add("verified");
  if (thumb) setThumb(el.teacherThumb, thumb);
  el.tBarWrap.style.display = "flex";
  setConfBar(el.tBarFill, el.tBarLabel, conf || 0);
}

function resetTeacherPanel() {
  el.tName.textContent      = "—";
  el.tId.textContent        = "—";
  el.tConf.textContent      = "—";
  el.tBadge.textContent     = "Re-verification required";
  el.tBadge.classList.remove("verified");
  el.tBarWrap.style.display = "none";
  el.teacherThumb.innerHTML = '<div class="thumb-placeholder">No face<br>detected</div>';
}

function addAttendanceRow(data, animate) {
  if (markedIds.has(data.id)) return;
  markedIds.add(data.id);
  var tr = document.createElement("tr");
  tr.id  = "att-" + data.id.replace(/[^a-zA-Z0-9]/g, "_");
  if (animate) tr.classList.add("row-new");
  tr.innerHTML =
    "<td title=\"" + data.name + "\">" + data.name + "</td>" +
    "<td>" + (data.time || "--:--") + "</td>" +
    "<td>" + Math.round(data.conf) + "%</td>";
  el.attBody.insertBefore(tr, el.attBody.firstChild);
  el.attCount.textContent = markedIds.size + " marked";
}

function removeAttendanceRow(sid) {
  var safeId = "att-" + sid.replace(/[^a-zA-Z0-9]/g, "_");
  var row    = document.getElementById(safeId);
  if (row) {
    row.style.opacity    = "0";
    row.style.transition = "opacity 0.3s";
    setTimeout(function() { if (row.parentNode) row.remove(); }, 350);
  }
  markedIds.delete(sid);
  el.attCount.textContent = markedIds.size + " marked";
}

function showProxyAlert(data) {
  el.proxyAlertName.textContent   = "⚠ PROXY: " + data.name + " (" + data.id + ")";
  el.proxyAlertReason.textContent = data.reason || "Flat image / screen detected";
  el.proxyAlertScore.textContent  = Math.round(data.score || 0) + "%";
  el.proxyAlert.style.display     = "flex";
  clearTimeout(proxyAlertTimer);
  proxyAlertTimer = setTimeout(function() {
    el.proxyAlert.style.display = "none";
  }, 6000);
}

function addProxyRow(data, animate) {
  proxyCount++;
  proxyIds.add(data.id);
  el.proxyBadge.textContent   = proxyCount;
  el.proxyBadge.style.display = "inline-block";
  var tr = document.createElement("tr");
  tr.classList.add("proxy-row");
  if (animate) tr.classList.add("row-new");
  tr.title     = data.reason || "";
  tr.innerHTML =
    "<td title=\"" + data.name + "\">" + data.name + "</td>" +
    "<td>" + (data.time || "--:--") + "</td>" +
    "<td>" + Math.round(data.score || 0) + "%</td>";
  el.proxyBody.insertBefore(tr, el.proxyBody.firstChild);
  el.proxyCount.textContent =
    proxyCount + " incident" + (proxyCount !== 1 ? "s" : "");
}

// ── SSE connection ────────────────────────────────────────────────
function connectSSE() {
  var es = new EventSource("/events");

  es.addEventListener("init", function(e) {
    var d = JSON.parse(e.data);
    setEngineBadge(d.engine || "orb");
    teacherVerified = !!d.teacher_verified;
    sessionActive   = !!d.session_active;

    if (d.teacher_verified && d.teacher_name) {
      populateTeacherPanel(d.teacher_name, d.teacher_id,
                           d.teacher_conf,  d.teacher_thumb);
    }

    if (sessionActive) {
      setBtnEnd();
      setStatus("Session active — scanning for students…", "active");
    } else {
      setBtnStart();
      setStatus(
        d.teacher_verified
          ? "✔ Teacher verified: " + d.teacher_name + " — Click 'Start Attendance' to begin."
          : "Awaiting teacher verification…",
        "idle"
      );
    }

    (d.attendance || []).forEach(function(r) {
      addAttendanceRow({ id: r.id, name: r.name, conf: r.conf, time: r.time }, false);
    });
    (d.proxy_incidents || []).forEach(function(r) {
      addProxyRow({ id: r.id, name: r.name, time: r.time,
                    reason: r.reason, score: r.score }, false);
    });
  });

  es.addEventListener("teacher_verified", function(e) {
    var d = JSON.parse(e.data);
    teacherVerified = true;
    populateTeacherPanel(d.name, d.id, d.conf, d.thumb);
    setBtnStart();
    setStatus("✔ Teacher verified: " + d.name + " — Click 'Start Attendance' to begin.", "idle");
    toast("✔ Teacher verified: " + d.name, "success");
  });

  es.addEventListener("student_detected", function(e) {
    var d = JSON.parse(e.data);
    el.sName.textContent = d.name;
    el.sId.textContent   = d.id;
    el.sConf.textContent = d.conf + "%";
    if (d.thumb) setThumb(el.studentThumb, d.thumb);
  });

  es.addEventListener("attendance_saved", function(e) {
    var d = JSON.parse(e.data);
    addAttendanceRow(d, true);
    setStatus("✔ Attendance saved: " + d.name + " (" + d.id + ")  " + Math.round(d.conf) + "%", "active");
    toast("✔ " + d.name + " marked — " + Math.round(d.conf) + "%", "success");
  });

  es.addEventListener("proxy_detected", function(e) {
    var d = JSON.parse(e.data);
    el.sName.textContent = d.name;
    el.sId.textContent   = d.id;
    el.sConf.textContent = d.conf + "% ⚠ PROXY";
    if (d.thumb) setThumb(el.studentThumb, d.thumb);
    showProxyAlert(d);
    addProxyRow({
      id: d.id, name: d.name,
      time: new Date().toTimeString().slice(0, 8),
      reason: d.reason, score: d.score
    }, true);
    setStatus("⚠ PROXY: " + d.name + " — " + d.reason, "active");
    toast("⚠ PROXY: " + d.name + " — attendance BLOCKED", "proxy");
  });

  es.addEventListener("attendance_cancelled", function(e) {
    var d = JSON.parse(e.data);
    removeAttendanceRow(d.id);
    toast("⚠ Attendance cancelled for " + d.name + " (proxy)", "proxy");
  });

  es.addEventListener("session_started", function() {
    sessionActive = true;
    markedIds.clear();
    el.attBody.innerHTML        = "";
    el.attCount.textContent     = "0 marked";
    el.studentThumb.innerHTML   = '<div class="thumb-placeholder">Scanning…</div>';
    el.sName.textContent        = "—";
    el.sId.textContent          = "—";
    el.sConf.textContent        = "—";
    el.proxyAlert.style.display = "none";
    setBtnEnd();
    setStatus("● Session active — scanning for students…", "active");
    toast("Attendance session started", "info");
  });

  es.addEventListener("session_ended", function(e) {
    var d = JSON.parse(e.data);
    sessionActive   = false;
    teacherVerified = false;
    resetTeacherPanel();
    el.proxyAlert.style.display = "none";
    setBtnStart();
    setStatus(
      "Session ended — " + d.count + " student(s) marked.  Show teacher face to start new session.",
      "idle"
    );
    toast("Session ended — " + d.count + " student(s) marked", "info");
  });

  es.addEventListener("error", function() {
    es.close();
    setTimeout(connectSSE, 2000);
  });
}

// ── Start / End button ────────────────────────────────────────────
el.mainBtn.addEventListener("click", async function() {
  el.mainBtn.disabled = true;
  var url = sessionActive ? "/api/end" : "/api/start";
  try {
    var res  = await fetch(url, { method: "POST" });
    var data = await res.json();
    if (!data.ok) {
      toast(data.msg || "Request failed", "info");
      el.mainBtn.disabled = false;
    }
  } catch(err) {
    toast("Network error — server unreachable", "info");
    el.mainBtn.disabled = false;
  }
});

// ── Bootstrap ─────────────────────────────────────────────────────
initToastWrap();
connectSSE();
