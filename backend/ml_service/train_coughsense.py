"""
CoughSense Training Pipeline

Data sources:
  Coswara  (IISc Bangalore) — healthy + COVID-19 cough recordings
  CoughVID (EPFL/Zenodo)   — bronchitis + COVID-19 recordings

Novel training techniques:
  - Domain-adversarial training (GRL: Coswara vs CoughVID)
  - Class-balanced supervised contrastive loss (per-class temperature)
  - Targeted augmentation (COVID 6x, others 2x)
  - 5-fold stratified cross-validation
  - DANN lambda annealing schedule

Usage:
  python train_coughsense.py \
    --coswara_dir /path/to/Coswara-Data/all_audio \
    --coughvid_dir /path/to/coughvid \
    --output_dir ./checkpoints \
    --epochs 60 \
    --batch_size 32

  # Quick smoke test on your existing multi_disease_data CSVs:
  python train_coughsense.py --use_csv --csv_path ../../multi_disease_data/balanced_3class.csv

Author: Nikhil Vincent
"""

import os
import argparse
import random
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import librosa
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

from coughsense_model import (
    build_model, CoughSenseLoss, get_lambda_schedule, count_parameters
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 16000
CLIP_DURATION = 3.5
CLIP_SAMPLES  = int(SAMPLE_RATE * CLIP_DURATION)

CLASS_NAMES = ['healthy', 'covid', 'bronchitis']
CLASS_MAP   = {c: i for i, c in enumerate(CLASS_NAMES)}
DOMAIN_MAP  = {'coswara': 0, 'coughvid': 1, 'unknown': 0}

# Augmentation multipliers — COVID minority gets hardest push
AUG_MULT = {'healthy': 2, 'covid': 6, 'bronchitis': 2}


# ─────────────────────────────────────────────────────────────────────────────
# Audio Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class AudioAugmentor:

    @staticmethod
    def add_gaussian_noise(wave, snr_db_range=(15, 35)):
        snr_db       = random.uniform(*snr_db_range)
        signal_power = wave.pow(2).mean()
        noise_power  = signal_power / (10 ** (snr_db / 10))
        return wave + torch.randn_like(wave) * noise_power.sqrt()

    @staticmethod
    def time_shift(wave, max_frac=0.2):
        shift = random.randint(-int(wave.size(-1) * max_frac),
                                int(wave.size(-1) * max_frac))
        return torch.roll(wave, shift, dims=-1)

    @staticmethod
    def volume_scale(wave, lo=0.7, hi=1.3):
        return wave * random.uniform(lo, hi)

    @staticmethod
    def time_stretch(wave, rate_range=(0.85, 1.15)):
        rate     = random.uniform(*rate_range)
        orig_len = wave.size(-1)
        new_freq = int(SAMPLE_RATE * rate)
        stretched = torchaudio.functional.resample(wave, SAMPLE_RATE, new_freq)
        if stretched.size(-1) >= orig_len:
            return stretched[..., :orig_len]
        return F.pad(stretched, (0, orig_len - stretched.size(-1)))

    @staticmethod
    def apply_random(wave, num_augs=2):
        fns = [AudioAugmentor.add_gaussian_noise,
               AudioAugmentor.time_shift,
               AudioAugmentor.volume_scale,
               AudioAugmentor.time_stretch]
        for fn in random.sample(fns, min(num_augs, len(fns))):
            wave = fn(wave)
        return wave


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CoughDataset(Dataset):

    def __init__(self, samples, augment=True):
        self.augment = augment
        expanded = []
        for s in samples:
            mult = AUG_MULT[s['label']] if augment else 1
            for i in range(mult):
                expanded.append({**s, 'aug_idx': i})
        random.shuffle(expanded)
        self.samples = expanded

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        try:
            wave, sr = torchaudio.load(path)
        except Exception:
            y, sr = librosa.load(path, sr=None, mono=True)
            wave  = torch.tensor(y).unsqueeze(0)

        if sr != SAMPLE_RATE:
            wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)

        T = wave.shape[-1]
        if T >= CLIP_SAMPLES:
            start = random.randint(0, T - CLIP_SAMPLES) if self.augment else (T - CLIP_SAMPLES) // 2
            wave  = wave[..., start:start + CLIP_SAMPLES]
        else:
            wave = F.pad(wave, (0, CLIP_SAMPLES - T))

        peak = wave.abs().max()
        if peak > 0:
            wave = wave / (peak + 1e-8)
        return wave  # (1, CLIP_SAMPLES)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        wave = self._load(s['path'])
        if self.augment and s.get('aug_idx', 0) > 0:
            wave = AudioAugmentor.apply_random(wave, num_augs=2)
        return {
            'waveform': wave,
            'label':    CLASS_MAP[s['label']],
            'domain':   DOMAIN_MAP.get(s.get('domain', 'unknown'), 0),
        }


def collate_fn(batch):
    waveforms = torch.stack([b['waveform'] for b in batch])
    labels    = torch.tensor([b['label']  for b in batch], dtype=torch.long)
    domains   = torch.tensor([b['domain'] for b in batch], dtype=torch.long)
    return waveforms, labels, domains


# ─────────────────────────────────────────────────────────────────────────────
# Data Discovery
# ─────────────────────────────────────────────────────────────────────────────

def load_from_csv(csv_path):
    """
    Load samples from the existing multi_disease_data CSVs.
    Columns: disease, audio_path, source
    """
    import csv
    samples = []
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return samples

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get('disease', '').lower().strip()
            path  = row.get('audio_path', '').strip()
            src   = row.get('source', 'unknown').lower().strip()

            # Normalize label
            if label in ('covid-19', 'covid'):
                label = 'covid'
            elif label in ('healthy', 'normal'):
                label = 'healthy'
            elif label in ('bronchitis',):
                label = 'bronchitis'
            else:
                continue  # skip unknown labels

            # Resolve relative paths from CSV location
            p = Path(path)
            if not p.is_absolute():
                p = (csv_path.parent / path).resolve()

            if p.exists():
                samples.append({'path': str(p), 'label': label, 'domain': src})
            # else: silently skip missing files

    label_counts = Counter(s['label'] for s in samples)
    print(f"[CSV] Loaded {len(samples)} samples: {dict(label_counts)}")
    return samples


def discover_coswara(coswara_dir):
    samples = []
    p = Path(coswara_dir)
    if not p.exists():
        print(f"[WARN] Coswara dir not found: {coswara_dir}")
        return samples

    # Try metadata CSV for labels
    patient_labels = {}
    for mf in list(p.parent.rglob('combined_data.csv')) + list(p.parent.rglob('metadata*.csv')):
        try:
            import pandas as pd
            df = pd.read_csv(mf)
            if 'covid_status' in df.columns and 'id' in df.columns:
                for _, row in df.iterrows():
                    pid    = str(row['id'])
                    status = str(row['covid_status']).lower()
                    if 'positive' in status:
                        patient_labels[pid] = 'covid'
                    elif 'healthy' in status or 'no_resp' in status:
                        patient_labels[pid] = 'healthy'
        except Exception:
            pass

    for f in list(p.rglob('*cough-heavy*.wav')) + list(p.rglob('*cough_heavy*.wav')):
        fname = f.stem.lower()
        label = None
        for pid, lbl in patient_labels.items():
            if pid in fname:
                label = lbl
                break
        if label is None:
            if 'covid' in fname or 'positive' in fname:
                label = 'covid'
            elif 'healthy' in fname or 'negative' in fname:
                label = 'healthy'
        if label in ('covid', 'healthy'):
            samples.append({'path': str(f), 'label': label, 'domain': 'coswara'})

    print(f"[Coswara] {len(samples)} samples")
    return samples


def discover_coughvid(coughvid_dir):
    samples = []
    p = Path(coughvid_dir)
    if not p.exists():
        print(f"[WARN] CoughVID dir not found: {coughvid_dir}")
        return samples

    meta = p / 'metadata_compiled.csv'
    if not meta.exists():
        meta = next(p.glob('metadata*.csv'), None)

    if meta is None:
        # No metadata — treat all as bronchitis (original approach)
        for ext in ['*.webm', '*.ogg', '*.wav']:
            for f in p.rglob(ext):
                samples.append({'path': str(f), 'label': 'bronchitis', 'domain': 'coughvid'})
        print(f"[CoughVID] {len(samples)} samples (no metadata, all bronchitis)")
        return samples

    try:
        import pandas as pd
        df = pd.read_csv(meta)
        label_map = {
            'COVID-19': 'covid', 'covid-19': 'covid', 'covid': 'covid',
            'healthy':  'healthy',
            'bronchitis': 'bronchitis',
            'asthma':   'bronchitis',
            'COPD':     'bronchitis',
        }
        cough_col  = 'cough_detected' if 'cough_detected' in df.columns else None
        status_col = 'status'         if 'status'         in df.columns else None
        if status_col is None:
            candidates = [c for c in df.columns if 'status' in c.lower() or 'expert' in c.lower()]
            status_col = candidates[0] if candidates else None

        for _, row in df.iterrows():
            uuid = str(row.get('uuid', row.get('filename', ''))).strip()
            if not uuid:
                continue
            if cough_col and float(row.get(cough_col, 1.0) or 1.0) < 0.8:
                continue
            raw = str(row.get(status_col, '') or '') if status_col else ''
            label = label_map.get(raw)
            if label is None:
                continue
            for ext in ['.webm', '.ogg', '.wav', '.mp3']:
                fpath = p / (uuid + ext)
                if fpath.exists():
                    samples.append({'path': str(fpath), 'label': label, 'domain': 'coughvid'})
                    break
    except Exception as e:
        print(f"[ERROR] CoughVID metadata: {e}")

    print(f"[CoughVID] {len(samples)} labeled samples")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Training / Eval Loops
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    lambda_d = get_lambda_schedule(epoch, total_epochs)

    for waveform, labels, domains in tqdm(loader, desc=f'Train {epoch}', leave=False):
        waveform = waveform.to(device)
        labels   = labels.to(device)
        domains  = domains.to(device)

        optimizer.zero_grad()
        outputs = model(waveform, lambda_d=lambda_d)
        losses  = criterion(outputs, labels, domains, lambda_d)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses['total'].item()
        all_preds.extend(outputs['logits'].argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), acc, f1, lambda_d


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for waveform, labels, domains in tqdm(loader, desc='Eval', leave=False):
        waveform = waveform.to(device)
        labels   = labels.to(device)
        domains  = domains.to(device)

        outputs = model(waveform, lambda_d=0.0)
        losses  = criterion(outputs, labels, domains, lambda_d=0.0)
        total_loss += losses['total'].item()

        probs = F.softmax(outputs['logits'], dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(probs.argmax(axis=1))
        all_labels.extend(labels.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm  = confusion_matrix(all_labels, all_preds)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        per_class[name] = float(accuracy_score(all_labels[mask], all_preds[mask])) if mask.sum() > 0 else 0.0

    try:
        auc = float(roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro'))
    except Exception:
        auc = 0.0

    return {
        'loss':          total_loss / len(loader),
        'accuracy':      float(acc),
        'f1_macro':      float(f1),
        'auc':           auc,
        'per_class_acc': per_class,
        'confusion':     cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  Output: {output_dir}")

    # ── Load data ─────────────────────────────────────────────────────────
    if args.use_csv:
        all_samples = load_from_csv(args.csv_path)
    else:
        all_samples  = discover_coswara(args.coswara_dir)
        all_samples += discover_coughvid(args.coughvid_dir)

    if not all_samples:
        print("[ERROR] No samples found. Check paths or use --use_csv.")
        return

    label_counts  = Counter(s['label']  for s in all_samples)
    domain_counts = Counter(s.get('domain', 'unknown') for s in all_samples)
    print(f"\nDataset (pre-augmentation):")
    print(f"  Labels:  {dict(label_counts)}")
    print(f"  Domains: {dict(domain_counts)}")
    print(f"  Total:   {len(all_samples)}\n")

    # ── Cross-validation ──────────────────────────────────────────────────
    labels_arr  = np.array([s['label'] for s in all_samples])
    skf         = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, labels_arr)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.folds}")
        print(f"{'='*60}")

        train_s = [all_samples[i] for i in train_idx]
        val_s   = [all_samples[i] for i in val_idx]

        train_ds = CoughDataset(train_s, augment=True)
        val_ds   = CoughDataset(val_s,   augment=False)

        # Weighted sampler for balanced batches
        tr_labels    = [CLASS_MAP[s['label']] for s in train_s]
        class_counts = Counter(tr_labels)
        weights      = [1.0 / class_counts[l] for l in tr_labels]
        sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=2,
                                  collate_fn=collate_fn, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=2,
                                  collate_fn=collate_fn, pin_memory=True)

        model     = build_model().to(device)
        if fold == 0:
            count_parameters(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        criterion = CoughSenseLoss(alpha=0.3, label_smoothing=0.1)

        best_f1, best_state = 0.0, None
        history = []

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc, tr_f1, lam = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, args.epochs
            )
            val_m = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            history.append({'epoch': epoch, 'tr_loss': tr_loss,
                            'tr_acc': tr_acc, 'tr_f1': tr_f1,
                            'val_acc': val_m['accuracy'],
                            'val_f1': val_m['f1_macro'],
                            'val_auc': val_m['auc']})

            print(f"Ep {epoch:3d} | loss={tr_loss:.3f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "
                  f"val_acc={val_m['accuracy']:.3f} val_f1={val_m['f1_macro']:.3f} "
                  f"val_auc={val_m['auc']:.3f} λ={lam:.3f}")
            for cls, acc in val_m['per_class_acc'].items():
                print(f"       {cls}: {acc:.3f}")

            if val_m['f1_macro'] > best_f1:
                best_f1    = val_m['f1_macro']
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"  ✓ Best F1: {best_f1:.4f}")

        # Final eval
        model.load_state_dict(best_state)
        model.to(device)
        final = eval_epoch(model, val_loader, criterion, device)
        fold_results.append(final)

        ckpt = output_dir / f'fold{fold+1}_best.pt'
        torch.save({
            'model_state':   best_state,
            'fold':          fold + 1,
            'best_val_f1':   best_f1,
            'final_metrics': final,
            'history':       history,
            'class_names':   CLASS_NAMES,
            'args':          vars(args),
        }, ckpt)
        print(f"\nSaved: {ckpt}")
        print(f"Final: acc={final['accuracy']:.4f} f1={final['f1_macro']:.4f} auc={final['auc']:.4f}")
        cm = np.array(final['confusion'])
        print("Confusion matrix:")
        for row, name in zip(cm, CLASS_NAMES):
            print(f"  {name:12s}: {row}")

    # ── CV Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    for k in ['accuracy', 'f1_macro', 'auc']:
        vals = [r[k] for r in fold_results]
        print(f"  {k:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\nPer-class accuracy (mean ± std):")
    for cls in CLASS_NAMES:
        vals = [r['per_class_acc'].get(cls, 0) for r in fold_results]
        print(f"  {cls:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    summary = {
        'mean_accuracy': float(np.mean([r['accuracy'] for r in fold_results])),
        'mean_f1':       float(np.mean([r['f1_macro'] for r in fold_results])),
        'mean_auc':      float(np.mean([r['auc']      for r in fold_results])),
        'std_accuracy':  float(np.std( [r['accuracy'] for r in fold_results])),
        'std_f1':        float(np.std( [r['f1_macro'] for r in fold_results])),
        'fold_results':  fold_results,
        'config':        vars(args),
    }
    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {output_dir}/cv_summary.json")
    print("\nDone. To use trained model, run: python coughsense_service.py")


def parse_args():
    p = argparse.ArgumentParser(description='Train CoughSense')
    p.add_argument('--coswara_dir',  default='../../Coswara-Data/all_audio')
    p.add_argument('--coughvid_dir', default='../../coughvid')
    p.add_argument('--output_dir',   default='./checkpoints')
    p.add_argument('--epochs',       type=int,   default=60)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=8e-4)
    p.add_argument('--folds',        type=int,   default=5)
    p.add_argument('--seed',         type=int,   default=42)
    # Quick-start with existing CSVs
    p.add_argument('--use_csv',      action='store_true',
                   help='Load from existing multi_disease_data CSV instead of raw audio dirs')
    p.add_argument('--csv_path',     default='../../multi_disease_data/balanced_3class.csv')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
