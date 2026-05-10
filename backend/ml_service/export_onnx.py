"""
export_onnx.py — Export a trained CoughSense V4 checkpoint to ONNX.

Usage:
    python3 ml_service/export_onnx.py \
        --checkpoint checkpoints_v4/fold1_best.pt \
        --output coughsense_v4.onnx
"""

import argparse
import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from train_v4 import CoughSenseV4, SYMPTOM_COLS, CLASS_NAMES


def export(checkpoint: str, output: str, opset: int = 17):
    model = CoughSenseV4(n_classes=len(CLASS_NAMES), sym_dim=len(SYMPTOM_COLS))
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    # Representative dummy inputs
    mel   = torch.randn(1, 1, 64, 128)
    sym   = torch.zeros(1, len(SYMPTOM_COLS))
    has_s = torch.ones(1)
    lam   = torch.zeros(1)

    torch.onnx.export(
        model,
        (mel, mel, has_s, sym, 0.0),
        output,
        opset_version=opset,
        input_names=['mel_heavy', 'mel_shallow', 'has_shallow', 'symptoms'],
        output_names=['disease_logits', 'domain_logits', 'embedding'],
        dynamic_axes={
            'mel_heavy':   {0: 'batch'},
            'mel_shallow': {0: 'batch'},
            'has_shallow': {0: 'batch'},
            'symptoms':    {0: 'batch'},
        },
    )
    print(f"Exported to {output}  (opset {opset})")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints_v4/fold1_best.pt')
    p.add_argument('--output',     default='coughsense_v4.onnx')
    p.add_argument('--opset',      type=int, default=17)
    args = p.parse_args()
    export(args.checkpoint, args.output, args.opset)
