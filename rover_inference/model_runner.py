"""
Drive-JEPA model for inference: the fine-tuned checkpoint of rover_finetune, rebuilt with the same code.

The model definition is NOT copied here: it is loaded from the rover_finetune folder (mounted read-only in
the container), so the network on the rover is exactly the one that was trained and evaluated
(base Drive-JEPA weights + the trained parameters of best.pt / last.pt, LoRA merged if any).

    runner = ModelRunner("/checkpoints/best.pt", "/rover_finetune")
    traj = runner(img_prev, img_now, status)     # (8, 3) x, y, heading in the rover frame at t [m, rad]
"""
import os
import sys
import time

import numpy as np


class ModelRunner:
    def __init__(self, ckpt, rover_finetune_dir, navsim_dir=None, weights=None, device="cuda",
                 precision="fp16", warmup=2):
        import torch
        self.torch = torch
        rf = os.path.abspath(os.path.expanduser(rover_finetune_dir))
        if rf not in sys.path:
            sys.path.insert(0, rf)
        from rover_finetune.losses import TargetScaler
        from rover_finetune.model import load_finetuned

        navsim_dir = navsim_dir or os.path.join(rf, "third_party", "navsim_v1")
        weights = weights or os.path.join(rf, "weights", "drive_jepa_pf_vitl_weights.pt")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available (container started without the NVIDIA runtime?)")
        self.device = torch.device(device)
        t0 = time.time()
        self.model, self.cfg, ck = load_finetuned(os.path.expanduser(ckpt), navsim_dir, weights, device=self.device)
        self.scaler = TargetScaler(self.cfg.get("targets", {}).get("scale", 1.0))
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[precision]
        self.info = {"checkpoint": os.path.abspath(os.path.expanduser(ckpt)), "epoch": ck.get("epoch"),
                     "trainable": self.cfg["model"]["trainable"], "precision": precision, "device": str(self.device),
                     "load_s": round(time.time() - t0, 1)}
        dummy = np.zeros((3, 256, 512), np.float32)
        for _ in range(warmup):                       # CUDA kernels / cuDNN autotuning before the first real frame
            self(dummy, dummy, np.array([0, 1, 0, 0, 0.2, 0, 0, 0], np.float32))

    def __call__(self, img_prev, img_now, status):
        torch = self.torch
        with torch.inference_mode():
            cam = torch.from_numpy(np.stack([img_prev, img_now], axis=1)[None]).to(self.device)  # (1, 3, 2, H, W)
            st = torch.as_tensor(np.asarray(status, np.float32)[None]).to(self.device)
            if self.dtype is not None and self.device.type == "cuda":
                with torch.autocast("cuda", dtype=self.dtype):
                    out = self.model(cam, st)["trajectory"]
            else:
                out = self.model(cam, st)["trajectory"]
            traj = self.scaler.to_metric(out.float())
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        return traj[0].cpu().numpy()


class ConstantVelocityRunner:
    """Stand-in without a model (tests, dry runs of the pipeline): constant velocity from the status."""

    info = {"checkpoint": None, "model": "constant velocity (no network)"}

    def __call__(self, img_prev, img_now, status):
        v = np.asarray(status[4:6], float)
        t = 0.5 * np.arange(1, 9)
        return np.stack([v[0] * t, v[1] * t, np.zeros(8)], axis=1)
