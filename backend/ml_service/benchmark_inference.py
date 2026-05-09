"""
benchmark_inference.py — Measure per-clip CPU inference latency for CoughSense V4.

Usage:
    python3 ml_service/benchmark_inference.py \
        --checkpoint checkpoints_v4/fold1_best.pt \
        --n_runs 50
"""

import argparse, time
import numpy as np
import torch
import torch.nn.functional as F

# Lazy import to avoid pulling in full training script
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def benchmark(checkpoint: str, n_runs: int = 50):
    device = torch.device('cpu')

    # Build model
    from train_v4 import CoughSenseV4, SYMPTOM_COLS, CLASS_NAMES
    model = CoughSenseV4(n_classes=len(CLASS_NAMES), sym_dim=len(SYMPTOM_COLS))
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    # Dummy inputs
    mel   = torch.randn(1, 1, 64, 128)
    sym   = torch.zeros(1, len(SYMPTOM_COLS))
    has_s = torch.ones(1)

    latencies = []
    with torch.no_grad():
        for i in range(n_runs + 10):  # 10 warm-up
            t0 = time.perf_counter()
            _ = model(mel, mel, has_s, sym, lam=0.0)
            t1 = time.perf_counter()
            if i >= 10:
                latencies.append((t1 - t0) * 1000)

    print(f"Checkpoint : {checkpoint}")
    print(f"Runs       : {n_runs}")
    print(f"Mean       : {np.mean(latencies):.1f} ms")
    print(f"P50        : {np.percentile(latencies, 50):.1f} ms")
    print(f"P95        : {np.percentile(latencies, 95):.1f} ms")
    print(f"P99        : {np.percentile(latencies, 99):.1f} ms")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints_v4/fold1_best.pt')
    p.add_argument('--n_runs', type=int, default=50)
    args = p.parse_args()
    benchmark(args.checkpoint, args.n_runs)
