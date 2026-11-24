"""
ROS 2 node `drive_jepa_navigator`: gives the model its intention (chain of goals). It never drives and never
sees the images; the inference node turns the chain into the command (training rule).

    subscribes  /vicon/toki/toki      geometry_msgs/TransformStamped (current position: goal reached?)
    publishes   /drive_jepa/goals     std_msgs/String JSON, at `rate_hz` and at every change:
                                      {"goals": [[x, y], ...] current goal first, "reached": n, "mode": ...}
                                      no fresh pose -> nothing published (the inference keeps the last chain)
                /drive_jepa/navigator_summary  std_msgs/String JSON (latched): goals reached with their times

    python3 -m rover_inference.navigator_node --ros-args -p config:=/ws/config/navigator.yaml

Modes (config/navigator.yaml): list (fixed route, repeatable tests) | random (roam: next goal 2-5 m ahead).
"""
import json
import math
import signal

import yaml

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from .navigator import Navigator, NavigatorParams
from .safety import SafetyMap


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NavigatorNode(Node):
    def __init__(self):
        super().__init__("drive_jepa_navigator")
        path = self.declare_parameter("config", "/ws/config/navigator.yaml").value
        with open(path) as f:
            cfg = yaml.safe_load(f)
        tp = cfg["topics"]
        p = NavigatorParams(**(cfg.get("navigator") or {}))
        smap = SafetyMap.from_yaml(cfg["map_file"]) if cfg.get("map_file") else None
        self.nav = Navigator(p, cfg.get("goals"), smap)
        self.pose_timeout = float(cfg.get("pose_timeout", 0.5))
        self.pose, self.pose_time, self.last_chain = None, None, None

        pose_type = PoseStamped if tp.get("pose_type", "TransformStamped") == "PoseStamped" else TransformStamped
        self.create_subscription(pose_type, tp["pose"], self.on_pose, qos_profile_sensor_data)
        self.pub = self.create_publisher(String, tp["goals"], 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_summary = self.create_publisher(String, tp["summary"], latched)
        route = f"{len(self.nav.route)} goals, loop {p.loop}" if p.mode == "list" else f"seed {p.seed}"
        self.get_logger().info(f"navigator {p.mode} ({route}) | map {cfg.get('map_file') or 'none'} | {p}")
        self.create_timer(1.0 / float(cfg.get("rate_hz", 2.0)), self.on_timer)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_pose(self, msg):
        if hasattr(msg, "transform"):
            p, q = msg.transform.translation, msg.transform.rotation
        else:
            p, q = msg.pose.position, msg.pose.orientation
        self.pose, self.pose_time = (p.x, p.y, _yaw(q)), self.now()
        n = self.nav.reached
        chain = self.nav.update(self.pose_time, self.pose)
        if self.nav.reached != n:                              # publish at once when a goal is reached
            t, g = self.nav.log[-1]
            self.get_logger().info(f"goal {self.nav.reached} reached: {g}  ->  next {chain[:1]}")
            self.publish(chain)

    def on_timer(self):
        if self.pose is None or self.now() - self.pose_time > self.pose_timeout:
            return
        self.publish(self.nav.update(self.pose_time, self.pose))

    def publish(self, chain):
        if chain != self.last_chain and not chain:
            self.get_logger().warn("route finished: empty goal chain (command 'unknown')")
        self.last_chain = chain
        self.pub.publish(String(data=json.dumps({"goals": [[round(v, 3) for v in g] for g in chain],
                                                 "reached": self.nav.reached, "mode": self.nav.p.mode})))

    def stop(self):
        s = {"reached": self.nav.reached, "log": self.nav.log}
        self.pub_summary.publish(String(data=json.dumps(s)))
        self.get_logger().info(f"SUMMARY navigator: {self.nav.reached} goals reached {self.nav.log}")


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # as the controller: summary still published

    def on_term(*_):                       # docker stop sends SIGTERM, ignored by a PID-1 process
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
