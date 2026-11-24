"""
ROS 2 node `drive_jepa_inference`: runs Drive-JEPA on the rover in real time. It never drives the motors:
it only publishes the predicted trajectory; the controller (separate container) turns it into /toki/cmd_vel.

    subscribes  /camera/image_raw/compressed   sensor_msgs/CompressedImage (sensors container, same Jetson)
                /vicon/toki/toki               geometry_msgs/TransformStamped (Vicon bridge, lab PC)
                /drive_jepa/goals              std_msgs/String JSON {"goals": [[x, y], ...]} current goal first
                                               (navigator; command source 'goals')
                /drive_jepa/command            std_msgs/String "left|straight|right|unknown" (source 'topic')
    publishes   /drive_jepa/trajectory         std_msgs/String JSON, one per prediction:
                  t              time the frame was taken (camera clock, s)
                  pose_world     Vicon pose (x, y, yaw) at t: anchor of the trajectory in the world
                  trajectory     8 x (x, y, heading) at t + 0.5 ... 4.0 s, rover frame at t [m, rad]
                  command, status (the 8 inputs given to the model), infer_ms, age_ms, seq
                /drive_jepa/info               std_msgs/String JSON (latched): model, configuration

    python3 -m rover_inference.node --ros-args -p config:=/ws/config/inference.yaml
    (in the container: docker compose run --rm inference)

The model runs in a worker thread, so the subscriptions keep filling the buffers while it computes.
"""
import json
import math
import signal
import threading
import time

import yaml

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .model_runner import ConstantVelocityRunner, ModelRunner
from .pipeline import InferencePipeline


def _stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9, header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class InferenceNode(Node):
    def __init__(self):
        super().__init__("drive_jepa_inference")
        path = self.declare_parameter("config", "/ws/config/inference.yaml").value
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        m, tp = cfg["model"], cfg["topics"]
        if m.get("runner", "network") == "constant_velocity":
            runner = ConstantVelocityRunner()
            self.get_logger().warn("runner = constant_velocity: NO network, only to check topics and timing")
        else:
            self.get_logger().info(f"loading {m['checkpoint']} ...")
            runner = ModelRunner(m["checkpoint"], m["rover_finetune"], m.get("navsim_dir"), m.get("weights"),
                                 m.get("device", "cuda"), m.get("precision", "fp16"))
        self.pipe = InferencePipeline(runner, **cfg.get("pipeline", {}))
        self.get_logger().info(f"model ready: {runner.info} | pipeline {self.pipe.cfg}")

        self.create_subscription(CompressedImage, tp["camera"], self.on_image, qos_profile_sensor_data)
        pose_type = PoseStamped if tp.get("pose_type", "TransformStamped") == "PoseStamped" else TransformStamped
        self.create_subscription(pose_type, tp["pose"], self.on_pose, qos_profile_sensor_data)
        self.create_subscription(String, tp["goals"], self.on_goals, 10)
        self.create_subscription(String, tp["command"], self.on_command, 10)
        self.pub = self.create_publisher(String, tp["trajectory"], 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.info_pub = self.create_publisher(String, tp["info"], latched)
        self.info_pub.publish(String(data=json.dumps({"model": runner.info, "pipeline": self.pipe.cfg,
                                                      "topics": tp}, default=str)))

        self.max_rate = float(cfg.get("max_rate_hz", 10.0))
        self.seq, self.stats = 0, {"n": 0, "infer": 0.0, "age": 0.0, "reasons": {}}
        self.stop = False
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()
        self.create_timer(5.0, self.report)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ callbacks: only fill the buffers
    def on_image(self, msg):
        stamp, stamp_ns = _stamp(msg.header)
        self.pipe.on_image(stamp, self.now(), bytes(msg.data), stamp_ns)

    def on_pose(self, msg):
        stamp, _ = _stamp(msg.header)
        if hasattr(msg, "transform"):
            p, q = msg.transform.translation, msg.transform.rotation
        else:
            p, q = msg.pose.position, msg.pose.orientation
        self.pipe.on_vicon(stamp, self.now(), p.x, p.y, _yaw(q))

    def on_goals(self, msg):
        try:
            d = json.loads(msg.data)
            self.pipe.on_goals(d.get("goals") if isinstance(d, dict) else d)
        except (ValueError, AttributeError) as e:
            self.get_logger().warn(f"bad goals message: {e}")

    def on_command(self, msg):
        self.pipe.on_command(msg.data.strip().lower(), self.now())

    # ------------------------------------------------------------------ worker
    def loop(self):
        period = 1.0 / self.max_rate
        while not self.stop and rclpy.ok():
            t0 = time.monotonic()
            res, why = self.pipe.step(self.now())
            if res is None:
                self.stats["reasons"][why] = self.stats["reasons"].get(why, 0) + 1
                time.sleep(0.005)
                continue
            self.seq += 1
            out = {"seq": self.seq, "t": res["t"], "frame_stamp_ns": res["frame_stamp_ns"],
                   "prev_stamp_ns": res["prev_stamp_ns"], "pose_world": [round(float(v), 5) for v in res["pose_world"]],
                   "trajectory": [[round(float(v), 5) for v in p] for p in res["trajectory"]],
                   "command": res["command"], "status": [round(float(v), 5) for v in res["status"]],
                   "infer_ms": round(res["infer_ms"], 1), "age_ms": round(res.get("age_ms", float("nan")), 1)}
            self.pub.publish(String(data=json.dumps(out)))
            self.stats["n"] += 1
            self.stats["infer"] += res["infer_ms"]
            self.stats["age"] += res.get("age_ms", 0.0)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def report(self):
        s, n = self.stats, max(self.stats["n"], 1)
        waits = ", ".join(f"{k}: {v}" for k, v in s["reasons"].items() if k != "no new frame")
        self.get_logger().info(f"{s['n'] / 5.0:.1f} Hz | inference {s['infer'] / n:.0f} ms | frame age at output "
                               f"{s['age'] / n:.0f} ms | clock {self.pipe.clock.report()}"
                               + (f" | waiting: {waits}" if waits else ""))
        self.stats = {"n": 0, "infer": 0.0, "age": 0.0, "reasons": {}}


def main():
    rclpy.init()

    def on_term(*_):                       # docker stop sends SIGTERM, ignored by a PID-1 process
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop = True
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
