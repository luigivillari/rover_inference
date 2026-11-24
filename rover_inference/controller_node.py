"""
ROS 2 node `drive_jepa_controller`: follows the predicted trajectory and drives the motors. The ONLY process
that publishes /toki/cmd_vel during the Drive-JEPA tests (the rover_collector driver must not run).

    subscribes  /drive_jepa/trajectory   std_msgs/String JSON (inference node)
                /vicon/toki/toki         geometry_msgs/TransformStamped (freshness by receive time, as the collector)
                /toki/battery            std_msgs/Float64
    publishes   /toki/cmd_vel            geometry_msgs/Twist, 20 Hz; angular.z = w x cmd_w_scale (2.2 for toki:
                                         the tracks slip, same factor as the data collection)
                                         enabled: false -> /drive_jepa/cmd_vel_dry instead (the rover does not move)
                /drive_jepa/controller_state  std_msgs/String: running, no_pose, waiting: ..., stopped: ...
                /drive_jepa/summary      std_msgs/String JSON (latched): distance, interventions, min gap ...

    python3 -m rover_inference.controller_node --ros-args -p config:=/ws/config/controller.yaml

Ctrl+C (or docker stop) always leaves the rover with a zero command.
"""
import json
import math
import signal

import yaml

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float64, String

from .follower import Follower, FollowerParams
from .safety import SafetyMap


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ControllerNode(Node):
    def __init__(self):
        super().__init__("drive_jepa_controller")
        path = self.declare_parameter("config", "/ws/config/controller.yaml").value
        with open(path) as f:
            cfg = yaml.safe_load(f)
        tp = cfg["topics"]
        p = FollowerParams(**{k: v for k, v in (cfg.get("follower") or {}).items()})
        smap = None
        if cfg.get("map_file"):
            smap = SafetyMap.from_yaml(cfg["map_file"])
            self.get_logger().info(f"safety map {cfg['map_file']}: {len(smap.obstacles)} obstacles, bounds {smap.bounds}")
        elif cfg.get("require_map", True):
            raise RuntimeError("map_file is required (geofence + obstacle stop); require_map: false only off the rover")
        else:
            self.get_logger().warn("NO safety map: no geofence and no obstacle stop")
        self.f = Follower(p, smap)
        self.cmd_w_scale = float(cfg.get("cmd_w_scale", 2.2))
        self.enabled = bool(cfg.get("enabled", False))
        self.pose, self.pose_time, self.battery = None, None, None

        pose_type = PoseStamped if tp.get("pose_type", "TransformStamped") == "PoseStamped" else TransformStamped
        self.create_subscription(pose_type, tp["pose"], self.on_pose, qos_profile_sensor_data)
        self.create_subscription(String, tp["trajectory"], self.on_traj, 10)
        if tp.get("battery") not in (None, "", "none"):
            self.create_subscription(Float64, tp["battery"], self.on_battery, 10)
        out_topic = tp["cmd_vel"] if self.enabled else tp["cmd_vel_dry"]
        self.pub_cmd = self.create_publisher(Twist, out_topic, 10)
        self.pub_state = self.create_publisher(String, tp["state"], 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_summary = self.create_publisher(String, tp["summary"], latched)
        (self.get_logger().warn if self.enabled else self.get_logger().info)(
            f"{'ENABLED: driving the rover on ' if self.enabled else 'DRY RUN (rover does not move): '}{out_topic} | "
            f"cmd_w_scale {self.cmd_w_scale} | {p}")
        self.last_state, self.last_state_pub = None, 0.0
        self.create_timer(1.0 / float(cfg.get("control_rate", 20.0)), self.on_timer)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_pose(self, msg):
        if hasattr(msg, "transform"):
            p, q = msg.transform.translation, msg.transform.rotation
        else:
            p, q = msg.pose.position, msg.pose.orientation
        self.pose, self.pose_time = (p.x, p.y, _yaw(q)), self.now()

    def on_traj(self, msg):
        try:
            self.f.set_prediction(json.loads(msg.data))
        except (ValueError, KeyError) as e:
            self.get_logger().warn(f"bad trajectory message: {e}")

    def on_battery(self, msg):
        self.battery = msg.data

    def on_timer(self):
        t = self.now()
        fresh = self.pose is not None and t - self.pose_time <= self.f.p.pose_timeout
        out = self.f.update(t, self.pose if fresh else None, self.battery)
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(out.v), float(out.w) * self.cmd_w_scale
        self.pub_cmd.publish(cmd)
        if out.state != self.last_state or t - self.last_state_pub > 1.0:
            self.pub_state.publish(String(data=out.state))
            self.last_state_pub = t
            if out.state != self.last_state:
                if out.state.startswith("stopped"):          # one call site per severity (rclpy)
                    self.get_logger().warn(f"state: {out.state}")
                    self.publish_summary()
                else:
                    self.get_logger().info(f"state: {out.state}")
            self.last_state = out.state

    def publish_summary(self):
        s = self.f.summary()
        self.pub_summary.publish(String(data=json.dumps(s, default=str)))
        holds = ", ".join(f"{k} {v}x ({s['hold_time'].get(k, 0):.1f} s)" for k, v in s["holds"].items()) or "none"
        self.get_logger().info(
            f"SUMMARY: driven {s['distance']:.1f} m in {s['time_running']:.0f} s | predictions used "
            f"{s['predictions_used']} (mean age {s['mean_prediction_age_ms']:.0f} ms) | min gap "
            f"{s['min_gap'] * 100:.1f} cm | max distance from the predicted path {s['max_cross_track'] * 100:.1f} cm | "
            f"brakes: {holds} | safety stop: {s['latched'] or 'none'}")

    def stop(self):
        for _ in range(3):
            self.pub_cmd.publish(Twist())
        self.publish_summary()


def main():
    # no rclpy signal handler: Ctrl+C raises KeyboardInterrupt while the context is still valid,
    # so the node can still publish the final zero command (as the rover_collector driver)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    # docker stop sends SIGTERM, which a PID-1 process ignores (killed 10 s later WITHOUT a zero command):
    # turn it into the same path as Ctrl+C
    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    node = ControllerNode()
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
