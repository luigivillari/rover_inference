#!/usr/bin/env python3
"""
Replay a recorded bag through the inference pipeline that runs on the rover, and compare what it builds
with the extracted training sample of the same frame (index.json of extract_samples.py).

Checks, without the rover:
  1. inputs: same frames, same command, same velocity / acceleration as in training (pipeline = training
     data -> differences ~0 with dynamics 'central');
  2. dynamics 'backward' (no wait for later poses): how different the status is, and how much the
     prediction error grows with it (--checkpoint) -> which mode to use on the rover;
  3. with --checkpoint: ADE / FDE of the model on this session with the inputs built online (should match
     evaluate.py on the same session when the inputs match).

The navigator is emulated from /collector/route (chain of goals in the order they were driven, as the
extraction does), so the 'goals' command source is tested too.

    python3 tools/replay_bag.py <bag> --dataset <dataset>/<session> --no-model
    python3 tools/replay_bag.py <bag> --dataset <dataset>/<session> \
        --checkpoint runs/<run>/best.pt --rover-finetune ~/storage/rover_finetune [--precision fp16]

Needs mcap + rosbags (as extract_samples.py); no ROS installation.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rover_inference.commands import CMD_NAMES  # noqa: E402
from rover_inference.model_runner import ConstantVelocityRunner, ModelRunner  # noqa: E402
from rover_inference.pipeline import InferencePipeline  # noqa: E402

POSE_TYPES = {"geometry_msgs/msg/TransformStamped", "geometry_msgs/msg/PoseStamped"}


def header_stamp_ns(raw):
    """Header stamp without deserialising the message (CDR: 4-byte encapsulation, sec int32, nanosec uint32)."""
    sec, nsec = np.frombuffer(raw[4:12], dtype="<i4")
    return int(sec) * 1_000_000_000 + int(np.uint32(nsec))


def mcap_files(bag):
    bag = Path(bag)
    return [bag] if bag.is_file() else sorted(bag.glob("*.mcap"))


def iter_messages(bag, topics=None):
    from mcap.reader import make_reader
    for f in mcap_files(bag):
        with open(f, "rb") as fh:
            for schema, channel, msg in make_reader(fh).iter_messages(topics=topics, log_time_order=True):
                yield channel.topic, schema.name, msg.log_time, msg.data


def topic_types(bag):
    from mcap.reader import make_reader
    out = {}
    for f in mcap_files(bag):
        with open(f, "rb") as fh:
            summ = make_reader(fh).get_summary()
            for ch in summ.channels.values():
                out[ch.topic] = summ.schemas[ch.schema_id].name
    return out


def goal_sequence(route_values):
    """Copy of extract_samples.goal_sequence: goals in the order they were driven."""
    seq, pos, last = [], [], "start"
    for r in route_values:
        xy = (r or {}).get("goal_xy")
        key = None if xy is None else (round(float(xy[0]), 3), round(float(xy[1]), 3))
        if key != last:
            seq.append(None if key is None else np.array(key, float))
            last = key
        pos.append(len(seq) - 1)
    return seq, pos


def goal_chain(seq, k):
    chain = []
    for g in seq[k:]:
        if g is None:
            break
        chain.append(g)
    return chain


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--dataset", required=True, help="extracted session folder (index.json + frames/) of this bag")
    ap.add_argument("--checkpoint", help="fine-tuned checkpoint (best.pt / last.pt); without: inputs only")
    ap.add_argument("--rover-finetune", default="../rover_finetune", help="rover_finetune folder")
    ap.add_argument("--weights", help="base Drive-JEPA weights (default: <rover-finetune>/weights/...)")
    ap.add_argument("--no-model", action="store_true", help="compare the inputs only (no GPU needed)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="fp32 = as evaluate.py; fp16 = as on the Jetson")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--modes", nargs="+", default=["central", "backward"], choices=["central", "backward"])
    ap.add_argument("--limit", type=int, default=0, help="stop after this many samples (0 = all)")
    ap.add_argument("--out", help="per-sample results (JSON)")
    args = ap.parse_args()

    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)

    index = json.loads((Path(args.dataset) / "index.json").read_text())
    params = index.get("params", {})
    samples = sorted(index["samples"], key=lambda s: int(Path(s["frame_now"]).stem))
    if args.limit:
        samples = samples[:args.limit]
    need = {int(Path(s[k]).stem) for s in samples for k in ("frame_now", "frame_prev")}
    latency = float(params.get("camera_latency", 0.017))
    pose_topic, image_topic = index["pose_topic"], index["image_topic"]

    if args.no_model or not args.checkpoint:
        runner = ConstantVelocityRunner()
    else:
        runner = ModelRunner(args.checkpoint, args.rover_finetune, weights=args.weights, device=args.device,
                             precision=args.precision, warmup=1)
    common = dict(camera_latency=latency, route_lookahead=float(params.get("route_lookahead", 1.5)),
                  route_threshold=float(params.get("route_threshold", 0.1)),
                  h_vel=float(params.get("h_vel", 0.1)), h_acc=float(params.get("h_acc", 0.25)),
                  smooth=float(params.get("smooth", 0.05)), max_pose_gap=float(params.get("max_pose_gap", 0.1)),
                  command_source="goals")
    pipes = {m: InferencePipeline(runner, dynamics=m, **common) for m in args.modes}

    # navigator emulation: all route messages first (hindsight chain of goals, as the extraction)
    route_msgs = [(lt, json.loads(ts.deserialize_cdr(raw, mt).data)) for _, mt, lt, raw in
                  iter_messages(args.bag, ["/collector/route"])]
    seq, pos = goal_sequence([r for _, r in route_msgs])
    print(f"bag {args.bag}: {len(route_msgs)} route messages, {sum(g is not None for g in seq)} goals | "
          f"{len(samples)} samples to rebuild | camera latency {latency * 1000:.0f} ms")

    pending = list(samples)
    results = {m: [] for m in args.modes}
    r_i = 0
    for topic, mtype, log_ns, raw in iter_messages(args.bag, [pose_topic, image_topic]):
        now = log_ns * 1e-9
        while r_i < len(route_msgs) and route_msgs[r_i][0] <= log_ns:   # navigator: goals at this time
            chain = goal_chain(seq, pos[r_i])
            for p in pipes.values():
                p.on_goals([g.tolist() for g in chain])
            r_i += 1
        if topic == pose_topic:
            msg = ts.deserialize_cdr(raw, mtype)
            st = header_stamp_ns(raw) * 1e-9
            if hasattr(msg, "transform"):
                pp, q = msg.transform.translation, msg.transform.rotation
            else:
                pp, q = msg.pose.position, msg.pose.orientation
            for p in pipes.values():
                p.on_vicon(st, now, pp.x, pp.y, yaw_of(q))
        else:
            sns = header_stamp_ns(raw)
            jpeg = bytes(ts.deserialize_cdr(raw, mtype).data) if sns in need else None
            for p in pipes.values():
                p.on_image(sns * 1e-9, now, jpeg, sns)
        # rebuild every sample as soon as the poses it needs have arrived (central: ~0.3 s after the frame)
        while pending:
            s = pending[0]
            stamp = int(Path(s["frame_now"]).stem)
            t_frame = stamp * 1e-9 - latency
            ready = all(p.latest_pose_time() is not None and
                        p.latest_pose_time() >= t_frame + p.poses.needs(m, p.cfg["h_vel"], p.cfg["h_acc"])[1] + 0.02
                        for m, p in pipes.items())
            if not ready:
                break
            pending.pop(0)
            for m, p in pipes.items():
                fr = p.frames.by_stamp(stamp)
                built, why = p.build(now, frame=fr) if fr is not None else (None, "frame not in buffer")
                if built is None:
                    results[m].append({"frame": stamp, "error": why})
                    continue
                out = p.run(built)
                gt = np.asarray(s["trajectory"], float)
                st_tr = np.asarray(s["status"], float)
                d = np.hypot(*(out["trajectory"][:, :2] - gt[:, :2]).T)
                results[m].append({
                    "frame": stamp, "prev_ok": out["prev_stamp_ns"] == int(Path(s["frame_prev"]).stem),
                    "cmd": out["command"], "cmd_train": CMD_NAMES[int(np.argmax(st_tr[:4]))],
                    "dv": float(np.hypot(*(out["status"][4:6] - st_tr[4:6]))),
                    "da": float(np.hypot(*(out["status"][6:8] - st_tr[6:8]))),
                    "ade": float(d.mean()), "fde": float(d[-1]),
                    "obstacle": s.get("obstacle_ahead_m") is not None})
        if not pending:
            break

    model = "constant velocity (inputs check)" if isinstance(runner, ConstantVelocityRunner) else runner.info["checkpoint"]
    print(f"\npredictions: {model}\n")
    print(f"  {'dynamics':9s} {'n':>5s} {'skipped':>8s} {'prev frame =':>12s} {'command =':>10s} "
          f"{'|dv| mean/max':>15s} {'|da| mean/max':>15s} {'ADE':>7s} {'FDE':>7s} {'ADE obst.':>9s}")
    for m, rs in results.items():
        ok = [r for r in rs if "error" not in r]
        if not ok:
            print(f"  {m:9s} no sample rebuilt ({rs[0]['error'] if rs else 'none reached'})")
            continue
        f = lambda k: np.array([r[k] for r in ok], float)
        obst = [r["ade"] for r in ok if r["obstacle"]]
        print(f"  {m:9s} {len(ok):5d} {len(rs) - len(ok):8d} {100 * f('prev_ok').mean():11.1f}% "
              f"{100 * np.mean([r['cmd'] == r['cmd_train'] for r in ok]):9.1f}% "
              f"{f('dv').mean():7.3f}/{f('dv').max():<7.3f} {f('da').mean():7.3f}/{f('da').max():<7.3f} "
              f"{f('ade').mean():7.3f} {f('fde').mean():7.3f} {np.mean(obst) if obst else float('nan'):9.3f}")
    print("\n  dv, da: difference from the training status [m/s, m/s^2]; ADE/FDE vs the recorded trajectory [m].")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
