#!/usr/bin/env python3
"""
Live view of the Drive-JEPA closed loop in the browser: arena map with the obstacles, rover and its trail,
the trajectory the model is predicting RIGHT NOW (anchored in the world at the pose of its frame), the
previous predictions (fading), the goal chain with the look-ahead point of the command, the camera image,
and the numbers (command, speed, model rate / latency, controller state, motor commands, goals reached).

It only LISTENS (never publishes): start and stop it at any time. On the Jetson, next to the other services:

    docker compose up monitor             # or: docker compose up  (starts it with the others)
    -> on the PC: http://192.168.27.108:8081

The map is the controller's safety map (config/controller.yaml map_file, or --map): the obstacles drawn
here are NOT known to the model nor to the navigator. Without ROS: --fake draws synthetic data (to check
the page). Pure Python + numpy/PyYAML + rclpy; the page is plain HTML/JS (no internet needed).
Options: --port 8081, --config-dir /ws/config, --map, --trail 3000, --camera-hz 2, --fake.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rover_inference.commands import point_along  # noqa: E402


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def load_map(path):
    """Bounds and obstacles of a rover_collector map file (walls -> boxes as the controller does)."""
    if not path or not os.path.exists(path):
        return {"bounds": None, "obstacles": [], "name": f"no map ({path})"}
    from rover_inference.safety import SafetyMap
    m = SafetyMap.from_yaml(path)
    obs = []
    for o in m.obstacles:
        if o.kind == "circle":
            obs.append({"type": "circle", "c": o.center.tolist(), "r": o.radius})
        else:
            obs.append({"type": "box", "c": o.center.tolist(), "h": o.half_size.tolist(), "yaw": o.yaw})
    return {"bounds": list(m.bounds), "obstacles": obs, "name": os.path.basename(path)}


class LiveState:
    """Everything the page shows. ROS callbacks write, HTTP threads read (one lock)."""

    def __init__(self, trail=3000, trail_step=0.02, lookahead=1.5):
        self.lock = threading.Lock()
        self.pose, self.pose_t = None, 0.0
        self.trail = deque(maxlen=trail)
        self.trail_step, self.lookahead = trail_step, lookahead
        self.preds = deque(maxlen=4)                   # newest last: dict(world pts, seq, ...)
        self.pred_times = deque(maxlen=20)
        self.goals, self.reached, self.goals_t = [], 0, 0.0
        self.state, self.state_t = "-", 0.0
        self.cmd = {"v": 0.0, "w": 0.0, "t": 0.0, "topic": "-"}
        self.cmd_hist = deque(maxlen=400)              # (t, v, w) for the strip chart (20 Hz x 20 s)
        self.battery = None
        self.summary = None
        self.jpeg, self.jpeg_t = None, 0.0
        self.t0 = time.time()

    # ------------------------------------------------------------------ writers
    def on_pose(self, x, y, yaw):
        with self.lock:
            self.pose, self.pose_t = (x, y, yaw), time.time()
            if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) >= self.trail_step:
                self.trail.append((round(x, 3), round(y, 3)))

    def on_trajectory(self, d):
        x0, y0, h0 = d["pose_world"]
        tr = np.asarray(d["trajectory"], float)
        c, s = math.cos(h0), math.sin(h0)
        pts = np.stack([x0 + c * tr[:, 0] - s * tr[:, 1], y0 + s * tr[:, 0] + c * tr[:, 1]], 1)
        L = float(np.hypot(*np.diff(np.vstack([[0, 0], tr[:, :2]]), axis=0).T).sum())
        now = time.time()
        with self.lock:
            self.preds.append({"pts": [[x0, y0]] + pts.round(3).tolist(), "seq": d.get("seq"),
                               "command": d.get("command"), "v": d["status"][4] if d.get("status") else None,
                               "len4s": L, "infer_ms": d.get("infer_ms"), "age_ms": d.get("age_ms"), "t": now})
            self.pred_times.append(now)

    def on_goals(self, d):
        with self.lock:
            self.goals, self.reached, self.goals_t = d.get("goals") or [], d.get("reached", 0), time.time()

    def on_state(self, s):
        with self.lock:
            self.state, self.state_t = s, time.time()

    def on_cmd(self, v, w, topic):
        now = time.time()
        with self.lock:
            self.cmd = {"v": v, "w": w, "t": now, "topic": topic}
            self.cmd_hist.append((now, v, w))

    def on_battery(self, v):
        with self.lock:
            self.battery = v

    def on_summary(self, d):
        with self.lock:
            self.summary = d

    def on_image(self, jpeg):
        with self.lock:
            self.jpeg, self.jpeg_t = jpeg, time.time()

    def reset_trail(self):
        with self.lock:
            self.trail.clear()

    # ------------------------------------------------------------------ reader
    def snapshot(self):
        now = time.time()
        with self.lock:
            lap = None
            if self.pose is not None and self.goals:
                p = point_along([self.pose[:2]] + [g[:2] for g in self.goals], self.lookahead)
                lap = None if p is None else [round(float(p[0]), 3), round(float(p[1]), 3)]
            pt = [t for t in self.pred_times if now - t < 5.0]
            newest = self.preds[-1] if self.preds else None
            return {
                "pose": self.pose, "pose_age": now - self.pose_t if self.pose else None,
                "trail": list(self.trail),
                "preds": [{"pts": p["pts"], "age": now - p["t"]} for p in self.preds],
                "pred": None if newest is None else {k: v for k, v in newest.items() if k != "pts"} | {"age": now - newest["t"]},
                "rate_hz": len(pt) / 5.0,
                "goals": self.goals, "reached": self.reached, "lookahead_pt": lap,
                "state": self.state, "state_age": now - self.state_t,
                "cmd": self.cmd | {"age": now - self.cmd["t"]},
                "cmd_hist": [(round(t - now, 2), round(v, 3), round(w, 3)) for t, v, w in self.cmd_hist if now - t < 20],
                "battery": self.battery, "summary": self.summary,
                "camera_age": now - self.jpeg_t if self.jpeg is not None else None,
                "uptime": now - self.t0,
            }

    def image(self):
        with self.lock:
            return self.jpeg


# ---------------------------------------------------------------------------------------------- ROS
def start_ros(state, topics):
    import rclpy
    from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import Float64, String

    rclpy.init()
    node = rclpy.create_node("drive_jepa_live_view")
    pose_type = PoseStamped if topics.get("pose_type") == "PoseStamped" else TransformStamped

    def on_pose(m):
        if hasattr(m, "transform"):
            p, q = m.transform.translation, m.transform.rotation
        else:
            p, q = m.pose.position, m.pose.orientation
        state.on_pose(p.x, p.y, _yaw(q))

    def js(fn):
        def cb(m):
            try:
                fn(json.loads(m.data))
            except (ValueError, KeyError, TypeError):
                pass
        return cb

    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(pose_type, topics["pose"], on_pose, qos_profile_sensor_data)
    node.create_subscription(String, topics["trajectory"], js(state.on_trajectory), 10)
    node.create_subscription(String, topics["goals"], js(state.on_goals), 10)
    node.create_subscription(String, topics["state"], lambda m: state.on_state(m.data), 10)
    node.create_subscription(String, topics["summary"], js(state.on_summary), latched)
    for name in ("cmd_vel", "cmd_vel_dry"):
        node.create_subscription(Twist, topics[name], lambda m, n=topics[name]: state.on_cmd(m.linear.x, m.angular.z, n), 10)
    node.create_subscription(Float64, topics["battery"], lambda m: state.on_battery(m.data), 10)
    node.create_subscription(CompressedImage, topics["camera"], lambda m: state.on_image(bytes(m.data)),
                             qos_profile_sensor_data)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    return node


def start_fake(state):
    """Synthetic data: the rover drives an oval, the 'model' predicts the next 4 s with some noise."""
    def run():
        rng = np.random.default_rng(0)
        try:
            import cv2
            img = np.full((540, 960, 3), 60, np.uint8)
            cv2.putText(img, "fake camera", (330, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
            state.on_image(cv2.imencode(".jpg", img)[1].tobytes())
        except ImportError:
            pass
        goals = [[0.5, 0.0], [-2.5, 1.0], [-5.5, -0.8], [-2.5, -1.0]]
        t, k = 0.0, 0
        while True:
            a = 0.2 * t / 2.2
            x, y = -2.5 + 3.0 * math.cos(a), 1.0 * math.sin(a)
            yaw = math.atan2(1.0 * math.cos(a), -3.0 * math.sin(a))
            state.on_pose(x, y, yaw)
            state.on_cmd(0.2, 0.1 * math.sin(3 * t), "/drive_jepa/cmd_vel_dry")
            if k % 7 == 0:
                tr = []
                for i in range(1, 9):
                    b = a + 0.2 * 0.5 * i / 2.2
                    wx, wy = -2.5 + 3.0 * math.cos(b), 1.0 * math.sin(b) + rng.normal(0, 0.01)
                    dx, dy = wx - x, wy - y
                    tr.append([math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy, 0.0])
                state.on_trajectory({"pose_world": [x, y, yaw], "trajectory": tr, "seq": k // 7, "command": "left",
                                     "status": [1, 0, 0, 0, 0.2, 0, 0, 0.02], "infer_ms": 277.0, "age_ms": 300.0})
                state.on_goals({"goals": goals[(k // 300) % 4:] + goals[:(k // 300) % 4], "reached": k // 300})
                state.on_state("running")
            state.on_battery(11.8)
            t += 0.05
            k += 1
            time.sleep(0.05)
    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------------------------------------- HTTP
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Drive-JEPA live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#14171c;--panel:#1c2027;--fg:#e6e8eb;--muted:#9aa3ad;--line:#2c323b}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
header{padding:8px 14px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
header b{font-size:16px} #status{color:var(--muted)}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:10px;padding:10px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px}
canvas{width:100%;display:block}
table{width:100%;border-collapse:collapse} td{padding:3px 4px;border-bottom:1px solid var(--line)} td:first-child{color:var(--muted)}
.big{font-size:22px;font-weight:600} .ok{color:#5fd38d} .warn{color:#f2c14e} .bad{color:#ff6b6b}
img{width:100%;border-radius:6px;background:#000;min-height:60px}
.legend span{display:inline-block;margin-right:12px;color:var(--muted)} .sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:4px}
button{background:#2c323b;color:var(--fg);border:1px solid #3a414c;border-radius:5px;padding:3px 9px;cursor:pointer}
</style></head><body>
<header><b>Drive-JEPA closed loop</b><span id="mapname"></span><span id="status"></span>
<span style="margin-left:auto"><label><input type="checkbox" id="cam" checked> camera</label> <button id="reset">clear trail</button></span></header>
<main>
<div>
 <div class="panel"><canvas id="map"></canvas>
  <div class="legend"><span><i class="sw" style="background:#ff9f43"></i>rover trail</span><span><i class="sw" style="background:#4dabf7;height:4px"></i>model prediction (4 s)</span><span><i class="sw" style="background:#4dabf755"></i>previous predictions</span><span><i class="sw" style="background:#e64980"></i>goal chain</span><span>&#9675; reach radius</span><span>&#9670; look-ahead point of the command</span></div></div>
 <div class="panel" style="margin-top:10px"><canvas id="strip" height="110"></canvas>
  <div class="legend"><span><i class="sw" style="background:#5fd38d"></i>v [m/s]</span><span><i class="sw" style="background:#f2c14e"></i>w motor command [rad/s, x cmd_w_scale]</span><span>last 20 s</span></div></div>
</div>
<div>
 <div class="panel"><table>
  <tr><td>command (intention)</td><td class="big" id="command">-</td></tr>
  <tr><td>controller</td><td id="state">-</td></tr>
  <tr><td>goals reached</td><td class="big" id="reached">-</td></tr>
  <tr><td>current goal / distance</td><td id="goal">-</td></tr>
  <tr><td>speed in the model status</td><td id="vst">-</td></tr>
  <tr><td>predicted path in 4 s</td><td id="len">-</td></tr>
  <tr><td>model rate / inference</td><td id="rate">-</td></tr>
  <tr><td>prediction age</td><td id="age">-</td></tr>
  <tr><td>motors (v, w)</td><td id="cmd">-</td></tr>
  <tr><td>Vicon / battery</td><td id="vicon">-</td></tr>
 </table></div>
 <div class="panel" style="margin-top:10px"><img id="img" alt="camera" style="display:none"><div id="camage" style="color:var(--muted);font-size:12px"></div></div>
 <div class="panel" style="margin-top:10px;font-size:12px;color:var(--muted)" id="summary">no SUMMARY yet (published by the controller when it stops)</div>
</div></main>
<script>
const $=id=>document.getElementById(id); let MAP=null, S=null;
fetch('map.json').then(r=>r.json()).then(m=>{MAP=m; $('mapname').textContent='map: '+m.name+' (safety only, unknown to model and navigator)';});
$('reset').onclick=()=>fetch('reset',{method:'POST'});
function fmt(x,d=2){return x==null||isNaN(x)?'-':Number(x).toFixed(d)}
function view(cv){const b=MAP&&MAP.bounds?MAP.bounds:[-10,2,-2,2]; const pad=0.3;
  const W=cv.clientWidth, sx=(b[1]-b[0])+2*pad, sy=(b[3]-b[2])+2*pad, k=W/sx; cv.width=W*devicePixelRatio; cv.height=sy*k*devicePixelRatio; cv.style.height=(sy*k)+'px';
  const c=cv.getContext('2d'); c.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  return {c,k,P:(x,y)=>[(x-b[0]+pad)*k,(b[3]+pad-y)*k]}}
function draw(){ if(!S) return; const cv=$('map'); const {c,k,P}=view(cv); c.clearRect(0,0,cv.width,cv.height);
  if(MAP&&MAP.bounds){const b=MAP.bounds; const [x0,y0]=P(b[0],b[3]),[x1,y1]=P(b[1],b[2]); c.strokeStyle='#59626e'; c.lineWidth=1.5; c.strokeRect(x0,y0,x1-x0,y1-y0);
    c.strokeStyle='#2a3039'; c.lineWidth=1; for(let x=Math.ceil(b[0]);x<=b[1];x++){const [a,_]=P(x,0); c.beginPath(); c.moveTo(a,y0); c.lineTo(a,y1); c.stroke();}
    for(let y=Math.ceil(b[2]);y<=b[3];y++){const [_,a]=P(0,y); c.beginPath(); c.moveTo(x0,a); c.lineTo(x1,a); c.stroke();}}
  if(MAP) for(const o of MAP.obstacles){c.fillStyle='#8a5a44'; c.strokeStyle='#c08060';
    if(o.type=='circle'){const [x,y]=P(o.c[0],o.c[1]); c.beginPath(); c.arc(x,y,o.r*k,0,7); c.fill(); c.stroke();}
    else{const [x,y]=P(o.c[0],o.c[1]); c.save(); c.translate(x,y); c.rotate(-o.yaw); c.fillRect(-o.h[0]*k,-o.h[1]*k,2*o.h[0]*k,2*o.h[1]*k); c.strokeRect(-o.h[0]*k,-o.h[1]*k,2*o.h[0]*k,2*o.h[1]*k); c.restore();}}
  // goal chain
  if(S.goals.length){c.strokeStyle='#e64980'; c.setLineDash([6,5]); c.lineWidth=1.5; c.beginPath(); let st=S.pose?P(S.pose[0],S.pose[1]):P(...S.goals[0]); c.moveTo(...st);
    for(const g of S.goals){c.lineTo(...P(g[0],g[1]));} c.stroke(); c.setLineDash([]);
    S.goals.forEach((g,i)=>{const [x,y]=P(g[0],g[1]); c.fillStyle=i==0?'#e64980':'#e6498088'; c.beginPath(); c.arc(x,y,i==0?6:4,0,7); c.fill();
      if(i==0){c.strokeStyle='#e6498088'; c.beginPath(); c.arc(x,y,(window.REACH||0.5)*k,0,7); c.stroke();}
      c.fillStyle='#e6e8eb'; c.font='12px system-ui'; c.fillText('G'+(i+1),x+7,y-7);});}
  if(S.lookahead_pt){const [x,y]=P(...S.lookahead_pt); c.fillStyle='#f2c14e'; c.beginPath(); c.moveTo(x,y-6); c.lineTo(x+6,y); c.lineTo(x,y+6); c.lineTo(x-6,y); c.fill();}
  // trail
  c.strokeStyle='#ff9f43'; c.lineWidth=2; c.beginPath(); S.trail.forEach((p,i)=>{const q=P(p[0],p[1]); i?c.lineTo(...q):c.moveTo(...q)}); c.stroke();
  // predictions: older faded, newest bold
  S.preds.forEach((p,i)=>{const newest=i==S.preds.length-1; c.strokeStyle=newest?'#4dabf7':'rgba(77,171,247,'+(0.15+0.12*i)+')'; c.lineWidth=newest?3.5:1.5; c.beginPath();
    p.pts.forEach((q,j)=>{const r=P(q[0],q[1]); j?c.lineTo(...r):c.moveTo(...r)}); c.stroke();
    if(newest){c.fillStyle='#4dabf7'; p.pts.slice(1).forEach(q=>{const r=P(q[0],q[1]); c.beginPath(); c.arc(r[0],r[1],3,0,7); c.fill();});
      const a=P(p.pts[0][0],p.pts[0][1]); c.strokeStyle='#4dabf7'; c.lineWidth=1; c.beginPath(); c.arc(a[0],a[1],4,0,7); c.stroke();}});
  // rover outline 20x20 cm
  if(S.pose){const [x,y]=P(S.pose[0],S.pose[1]); c.save(); c.translate(x,y); c.rotate(-S.pose[2]); c.fillStyle='#ffffffcc'; c.fillRect(-0.1*k,-0.1*k,0.2*k,0.2*k);
    c.fillStyle='#14171c'; c.beginPath(); c.moveTo(0.1*k,0); c.lineTo(0.0,-0.06*k); c.lineTo(0.0,0.06*k); c.fill(); c.restore();}
  // strip chart
  const sc=$('strip'), W=sc.clientWidth; sc.width=W*devicePixelRatio; sc.height=110*devicePixelRatio; const g=sc.getContext('2d'); g.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  const X=t=>W*(1+t/20), Y=v=>55-v*35; g.strokeStyle='#2c323b'; g.beginPath(); g.moveTo(0,55); g.lineTo(W,55); g.stroke();
  for(const [col,idx] of [['#5fd38d',1],['#f2c14e',2]]){g.strokeStyle=col; g.lineWidth=1.5; g.beginPath(); S.cmd_hist.forEach((h,i)=>{const p=[X(h[0]),Y(h[idx])]; i?g.lineTo(...p):g.moveTo(...p)}); g.stroke();}
  g.fillStyle='#9aa3ad'; g.font='11px system-ui'; g.fillText('+1',2,22); g.fillText('-1',2,92);
}
function cls(e,ok,warn){e.className=ok?'ok':warn?'warn':'bad'}
async function tick(){ try{ S=await (await fetch('state.json')).json(); }catch(e){ $('status').textContent='no connection to the live view'; return; }
  const p=S.pred; $('status').textContent='up '+fmt(S.uptime,0)+' s';
  const cm=p?p.command:'-'; $('command').textContent=cm; $('command').style.color={left:'#4dabf7',right:'#ff9f43',straight:'#5fd38d'}[cm]||'#9aa3ad';
  $('state').textContent=S.state; cls($('state'),S.state=='running',S.state.startsWith('waiting'));
  $('reached').textContent=S.reached;
  if(S.goals.length&&S.pose){const g=S.goals[0]; $('goal').textContent='['+g.map(v=>fmt(v,1))+']  '+fmt(Math.hypot(g[0]-S.pose[0],g[1]-S.pose[1]))+' m'}else $('goal').textContent='-';
  $('vst').textContent=p?fmt(p.v)+' m/s':'-'; $('len').textContent=p?fmt(p.len4s)+' m':'-';
  $('rate').textContent=fmt(S.rate_hz,1)+' Hz / '+(p?fmt(p.infer_ms,0):'-')+' ms'; cls($('rate'),S.rate_hz>2,S.rate_hz>0.5);
  $('age').textContent=p?fmt((p.age_ms+p.infer_ms)/1000+p.age,2)+' s (frame -> now)':'-';
  $('cmd').textContent=fmt(S.cmd.v)+' m/s, '+fmt(S.cmd.w)+' rad/s  '+(S.cmd.topic.endsWith('_dry')?'(DRY RUN)':'')+(S.cmd.age>1?' [stale]':'');
  $('vicon').textContent=(S.pose_age==null?'no pose':fmt(S.pose_age*1000,0)+' ms')+' / '+(S.battery==null?'-':fmt(S.battery,2)+' V'); cls($('vicon'),S.pose_age!=null&&S.pose_age<0.3,false);
  if(S.summary){const s=S.summary; $('summary').textContent='SUMMARY: '+fmt(s.distance,1)+' m in '+fmt(s.time_running,0)+' s, predictions '+s.predictions_used+
     ', min gap obstacles '+fmt((s.min_obstacle_gap??s.min_gap)*100,1)+' cm, safety stop: '+(s.latched||'none')}
  $('camage').textContent=S.camera_age==null?'no camera image':'camera image '+fmt(S.camera_age,1)+' s old';
  draw(); }
setInterval(tick,200); window.addEventListener('resize',draw);
let camBusy=false; setInterval(()=>{ if(!$('cam').checked||camBusy) return; camBusy=true; const im=new Image(); im.onload=()=>{$('img').src=im.src; $('img').style.display='block'; camBusy=false}; im.onerror=()=>{camBusy=false}; im.src='camera.jpg?'+Date.now(); }, CAM_MS);
fetch('cfg.json').then(r=>r.json()).then(c=>{window.REACH=c.reach_radius});
</script></body></html>"""


def make_handler(state, map_info, cfg, cam_ms):
    page = PAGE.replace("CAM_MS", str(int(cam_ms))).encode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            elif path == "/state.json":
                self._send(json.dumps(state.snapshot(), default=float).encode(), "application/json")
            elif path == "/map.json":
                self._send(json.dumps(map_info).encode(), "application/json")
            elif path == "/cfg.json":
                self._send(json.dumps(cfg).encode(), "application/json")
            elif path == "/camera.jpg":
                img = state.image()
                if img is None:
                    self._send(b"", "image/jpeg", 404)
                else:
                    self._send(img, "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path == "/reset":
                state.reset_trail()
            self._send(b"ok", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--config-dir", default="/ws/config")
    ap.add_argument("--map", help="map file (default: map_file of config/controller.yaml)")
    ap.add_argument("--trail", type=int, default=3000)
    ap.add_argument("--camera-hz", type=float, default=2.0)
    ap.add_argument("--fake", action="store_true", help="synthetic data, no ROS")
    args = ap.parse_args()

    def read(name):
        p = Path(args.config_dir) / name
        return yaml.safe_load(p.read_text()) if p.exists() else {}
    ctrl, nav, inf = read("controller.yaml"), read("navigator.yaml"), read("inference.yaml")
    map_path = args.map or ctrl.get("map_file")
    map_info = load_map(map_path)
    topics = {"pose": "/vicon/toki/toki", "pose_type": "TransformStamped", "trajectory": "/drive_jepa/trajectory",
              "goals": "/drive_jepa/goals", "state": "/drive_jepa/controller_state", "summary": "/drive_jepa/summary",
              "cmd_vel": "/toki/cmd_vel", "cmd_vel_dry": "/drive_jepa/cmd_vel_dry", "battery": "/toki/battery",
              "camera": (inf.get("topics") or {}).get("camera", "/camera/image_raw/compressed")}
    topics.update({k: v for k, v in (ctrl.get("topics") or {}).items() if k in topics})
    cfg = {"reach_radius": (nav.get("navigator") or {}).get("reach_radius", 0.3),
           "cmd_w_scale": ctrl.get("cmd_w_scale", 2.2)}
    lookahead = (inf.get("pipeline") or {}).get("route_lookahead", 1.5)

    state = LiveState(trail=args.trail, lookahead=lookahead)
    if args.fake:
        start_fake(state)
    else:
        start_ros(state, topics)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(state, map_info, cfg, 1000.0 / args.camera_hz))
    print(f"live view on http://<this machine>:{args.port}  | map {map_info['name']} | topics {topics}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
