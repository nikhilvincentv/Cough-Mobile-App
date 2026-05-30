#!/bin/bash
# run_experiments.sh — Full CoughSense V6 experiment pipeline
#
# Run order:
#   1. Precompute Whisper mels (once, ~15 min)
#   2. Train Whisper-tiny encoder (main contribution, ~3–4 hrs on MPS)
#   3. Train EfficientNet-B2 baseline (~2 hrs)
#   4. [Optional] Download OPERA-CT, then train OPERA fine-tune (~3 hrs)
#   5. [Optional] Train Dual-Encoder (Whisper+OPERA) — best expected results
#
# IMPORTANT: First run disk cleanup commands below if disk is full:
#   rm -rf ml_service/whisper_mels/       # remove stale mels (8.8GB)
#   rm bronchitis_pneumonia_data/*.rar    # already extracted (18MB)

set -e
cd "$(dirname "$0")"

echo "=== CoughSense V6 Experiment Pipeline ==="
echo "Classes: healthy / covid / respiratory_cond / bronchitis / pneumonia"
echo "Dataset: $(wc -l < ml_service/v6_data.csv) rows in v6_data.csv"
echo ""

# ── Step 1: Precompute Whisper mels ───────────────────────────────────────────
echo "Step 1: Precomputing Whisper mels..."
python3 -u ml_service/precompute_whisper_mels.py \
    --csv ml_service/v6_data.csv \
    --out_dir ml_service/whisper_mels
echo ""

# ── Step 2: Whisper encoder training ──────────────────────────────────────────
echo "Step 2: Training Whisper-tiny encoder (5-class)..."
mkdir -p checkpoints_whisper
python3 -u ml_service/train_whisper.py \
    --csv ml_service/v6_data.csv \
    --output_dir checkpoints_whisper \
    --epochs 40 --batch_size 32 --folds 5 --seed 42 \
    2>&1 | tee checkpoints_whisper/training_log_whisper.txt
echo ""

# ── Step 3: EfficientNet-B2 baseline ──────────────────────────────────────────
echo "Step 3: Training EfficientNet-B2 baseline..."
mkdir -p checkpoints_effnet
python3 -u ml_service/train_opera.py \
    --csv ml_service/v6_data.csv \
    --model effnet \
    --output_dir checkpoints_effnet \
    --epochs 40 --batch_size 32 --folds 5 --seed 42 \
    2>&1 | tee checkpoints_effnet/training_log_effnet.txt
echo ""

# ── Step 4: OPERA fine-tune (requires checkpoint) ─────────────────────────────
OPERA_CKPT="checkpoints_opera/opera-ct.pth"
if [ -f "$OPERA_CKPT" ]; then
    echo "Step 4: Training OPERA-CT fine-tune..."
    mkdir -p checkpoints_opera
    python3 -u ml_service/train_opera.py \
        --csv ml_service/v6_data.csv \
        --model opera \
        --checkpoint "$OPERA_CKPT" \
        --output_dir checkpoints_opera \
        --epochs 40 --batch_size 32 --folds 5 --seed 42 \
        2>&1 | tee checkpoints_opera/training_log_opera.txt
else
    echo "Step 4: OPERA checkpoint not found. Download with:"
    echo "  python3 -c \"from huggingface_hub import hf_hub_download; \\"
    echo "    hf_hub_download('evelyn0414/OPERA', 'opera-ct.pth', local_dir='checkpoints_opera')\""
    echo "  Then re-run this script or run train_opera.py directly."
fi
echo ""

# ── Step 5: Dual-encoder (Whisper + OPERA) ───────────────────────────────────
if [ -f "$OPERA_CKPT" ]; then
    echo "Step 5: Training Dual-Encoder (Whisper + OPERA)..."
    mkdir -p checkpoints_dual
    python3 -u ml_service/train_dual_encoder.py \
        --csv ml_service/v6_data.csv \
        --opera_ckpt "$OPERA_CKPT" \
        --output_dir checkpoints_dual \
        --epochs 40 --batch_size 16 --folds 5 --seed 42 \
        2>&1 | tee checkpoints_dual/training_log_dual.txt
else
    echo "Step 5: Skipping dual-encoder (need OPERA checkpoint)."
fi

echo ""
echo "=== All experiments complete! ==="
echo "Results:"
for d in checkpoints_whisper checkpoints_effnet checkpoints_opera checkpoints_dual; do
    if [ -f "$d/cv_summary_"*.json ]; then
        echo "  $d:"
        python3 -c "
import json, glob
f = glob.glob('$d/cv_summary_*.json')[0]
s = json.load(open(f))
print(f'    bal_acc={s[\"mean_bal_acc\"]*100:.1f}% ±{s[\"std_bal_acc\"]*100:.1f}%  F1={s[\"mean_f1\"]:.3f}  AUC={s[\"mean_auc\"]:.3f}')
" 2>/dev/null || true
    fi
done
