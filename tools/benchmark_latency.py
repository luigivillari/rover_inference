#!/usr/bin/env python3
"""
Latency of Drive-JEPA on this machine (Jetson Orin Nano on the rover), as the inference node runs it:
JPEG decode + preprocessing of one frame (the other one is cached) + network forward.

Also compares the precisions with fp32 (the precision of the offline evaluation): the difference of the
predicted poses tells whether fp16 / bf16 can be used on the rover.

    python3 tools/benchmark_latency.py --checkpoint /checkpoints/best.pt --rover-finetune /rover_finetune
    python3 tools/benchmark_latency.py ... --frames /data/test_dataset/Ltest_s1 --runs 50

Read: mean / p95 of the total time -> achievable re-planning rate (1000 / total ms). With --frames also the
ADE / FDE on those samples (checks that the checkpoint loads and runs as in evaluate.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rover_inference.frames import decode_jpeg_rgb, preprocess_image  # noqa: E402
from rover_inference.model_runner import ModelRunner  # noqa: E402


def load_inputs(folder, n):
    """(jpeg_prev, jpeg_now, status) of n samples of an extracted session, or synthetic 1920x1080 frames."""
    if folder:
        idx = json.loads((Path(folder) / "index.json").read_text())
        out = []
        for s in idx["samples"][::max(1, len(idx["samples"]) // n)][:n]:
            out.append(((Path(folder) / s["frame_prev"]).read_bytes(), (Path(folder) / s["frame_now"]).read_bytes(),
                        np.asarray(s["status"], np.float32), np.asarray(s["trajectory"], float)))
        return out
    import cv2
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        img = cv2.GaussianBlur(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8), (15, 15), 5)
        jpg = cv2.imencode(".jpg", img)[1].tobytes()
        out.append((jpg, jpg, np.array([0, 1, 0, 0, 0.25, 0, 0, 0], np.float32), None))
    return out


def stats(x):
    x = np.asarray(x)
    return f"mean {x.mean():6.1f}  p95 {np.percentile(x, 95):6.1f}  max {x.max():6.1f} ms"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--rover-finetune", required=True)
    ap.add_argument("--weights", help="base Drive-JEPA weights (default: <rover-finetune>/weights/...)")
    ap.add_argument("--frames", help="extracted session folder (real frames); default: synthetic 1920x1080 JPEGs")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--precisions", nargs="+", default=["fp16", "fp32"], choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    inputs = load_inputs(args.frames, args.runs)
    preds = {}
    for prec in args.precisions:
        runner = ModelRunner(args.checkpoint, args.rover_finetune, weights=args.weights, device=args.device,
                             precision=prec, warmup=3)
        t_dec, t_net, t_tot, out = [], [], [], []
        for jp, jn, st, _ in inputs:
            t0 = time.perf_counter()
            img_prev = preprocess_image(decode_jpeg_rgb(jp))     # on the rover the previous frame is cached:
            t1 = time.perf_counter()                              # one decode per prediction
            img_now = preprocess_image(decode_jpeg_rgb(jn))
            t2 = time.perf_counter()
            out.append(runner(img_prev, img_now, st))
            t3 = time.perf_counter()
            t_dec.append((t2 - t1) * 1000)
            t_net.append((t3 - t2) * 1000)
            t_tot.append((t3 - t1) * 1000)
        preds[prec] = np.array(out)
        print(f"\n{prec} on {runner.info['device']} ({runner.info['trainable']}, load {runner.info['load_s']} s)")
        print(f"  decode + preprocess  {stats(t_dec)}")
        print(f"  network              {stats(t_net)}")
        print(f"  total per prediction {stats(t_tot)}  ->  {1000 / np.mean(t_tot):.1f} Hz")
        if inputs[0][3] is not None:                          # real samples: error vs the recorded trajectory
            d = np.hypot(*(preds[prec][:, :, :2] - np.array([x[3][:, :2] for x in inputs])).transpose(2, 0, 1))
            print(f"  ADE {d.mean():.3f} m  FDE {d[:, -1].mean():.3f} m on these {len(inputs)} samples "
                  f"(compare with evaluate.py on the same session)")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  GPU memory peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
        except ImportError:
            pass
        del runner
    if "fp32" in preds:
        for prec, p in preds.items():
            if prec != "fp32":
                d = np.hypot(*(p[:, :, :2] - preds["fp32"][:, :, :2]).transpose(2, 0, 1))
                print(f"\n{prec} vs fp32: pose difference mean {d.mean() * 1000:.1f} mm, max {d.max() * 1000:.1f} mm")


if __name__ == "__main__":
    main()
