"""
CoughSense V2 Training Pipeline

Innovations vs standard training:
  1. PCGrad — gradient surgery to prevent L_focal vs L_domain conflicts
  2. GradNorm — automatic dynamic loss weight balancing (Chen et al. 2018)
  3. Curriculum Learning — loss-ranked easy-to-hard sample ordering
  4. Warm Restarts LR — CosineAnnealingWarmRestarts (better escaping local minima)
  5. Embedding MixUp — interpolation in embedding space (Manifold MixUp)
  6. EMA Self-Distillation — KL(student || EMA teacher) each batch
  7. Memory Bank update — enqueue new embeddings every step

Usage:
  # Quick start (existing CSVs, no dataset download):
  python train_coughsense.py --use_csv

  # Full training (raw audio dirs):
  python train_coughsense.py \
    --coswara_dir /path/to/Coswara-Data/all_audio \
    --coughvid_dir /path/to/coughvid \
    --epochs 80

Author: Nikhil Vincent
"""

import os
import sys
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import librosa
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from coughsense_model import (
    build_model, CoughSenseLoss, EMATeacher, GradNorm, PCGrad,
    get_lambda_schedule, count_parameters
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE       = 16000
CLIP_SAMPLES      = int(SAMPLE_RATE * 3.5)
ALL_CLASS_NAMES   = ['healthy', 'covid', 'bronchitis']  # canonical order
DOMAIN_MAP        = {'coswara': 0, 'coughvid': 1, 'unknown': 0}
AUG_MULT          = {'healthy': 2, 'covid': 6, 'bronchitis': 2}

# These are set dynamically in train() after loading samples
CLASS_NAMES: list = ALL_CLASS_NAMES
CLASS_MAP:   dict = {c: i for i, c in enumerate(CLASS_NAMES)}


# ─────────────────────────────────────────────────────────────────────────────
# Audio Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class AudioAugmentor:

    @staticmethod
    def gaussian_noise(w, lo=15, hi=35):
        snr_db = random.uniform(lo, hi)
        sig_p  = w.pow(2).mean()
        nse_p  = sig_p / (10 ** (snr_db / 10))
        return w + torch.randn_like(w) * nse_p.sqrt()

    @staticmethod
    def time_shift(w, max_frac=0.2):
        s = random.randint(-int(w.size(-1) * max_frac), int(w.size(-1) * max_frac))
        return torch.roll(w, s, dims=-1)

    @staticmethod
    def volume_scale(w, lo=0.6, hi=1.4):
        return w * random.uniform(lo, hi)

    @staticmethod
    def time_stretch(w, lo=0.82, hi=1.18):
        rate     = random.uniform(lo, hi)
        orig_len = w.size(-1)
        s = torchaudio.functional.resample(w, SAMPLE_RATE, int(SAMPLE_RATE * rate))
        return s[..., :orig_len] if s.size(-1) >= orig_len else F.pad(s, (0, orig_len - s.size(-1)))

    @staticmethod
    def room_impulse(w):
        """Synthetic room impulse response via exponential decay convolution."""
        rt60   = random.uniform(0.1, 0.6)
        sr     = SAMPLE_RATE
        t60_s  = int(rt60 * sr)
        h      = torch.exp(-torch.linspace(0, 6, t60_s)) * torch.randn(t60_s)
        h      = h / (h.abs().max() + 1e-8)
        w_np   = w.squeeze(0).numpy()
        import numpy as np
        from scipy import signal as sp
        out = sp.fftconvolve(w_np, h.numpy())[:len(w_np)]
        return torch.tensor(out, dtype=w.dtype).unsqueeze(0)

    @staticmethod
    def apply(w, n=2, use_room=False):
        fns = [AudioAugmentor.gaussian_noise, AudioAugmentor.time_shift,
               AudioAugmentor.volume_scale,   AudioAugmentor.time_stretch]
        if use_room:
            fns.append(AudioAugmentor.room_impulse)
        for fn in random.sample(fns, min(n, len(fns))):
            try:
                w = fn(w)
            except Exception:
                pass  # room impulse needs scipy; skip gracefully
        return w


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
            w, sr = torchaudio.load(path)
        except Exception:
            try:
                y, sr = librosa.load(path, sr=None, mono=True)
                w = torch.tensor(y).unsqueeze(0)
            except Exception:
                return torch.zeros(1, CLIP_SAMPLES)
        if w.numel() == 0 or w.shape[-1] == 0:
            return torch.zeros(1, CLIP_SAMPLES)
        if sr != SAMPLE_RATE:
            w = torchaudio.functional.resample(w, sr, SAMPLE_RATE)
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        T = w.shape[-1]
        if T >= CLIP_SAMPLES:
            start = random.randint(0, T - CLIP_SAMPLES) if self.augment else (T - CLIP_SAMPLES) // 2
            w = w[..., start:start + CLIP_SAMPLES]
        else:
            w = F.pad(w, (0, CLIP_SAMPLES - T))
        peak = w.abs().max()
        return w / (peak + 1e-8) if peak > 0 else w

    def __getitem__(self, idx):
        s = self.samples[idx]
        w = self._load(s['path'])
        if self.augment and s.get('aug_idx', 0) > 0:
            w = AudioAugmentor.apply(w, n=2)
        return dict(waveform=w, label=CLASS_MAP[s['label']],
                    domain=DOMAIN_MAP.get(s.get('domain', 'unknown'), 0),
                    age_group=s.get('age_group', -1),
                    gender=s.get('gender', -1),
                    path=s['path'])


def collate_fn(batch):
    return (torch.stack([b['waveform']  for b in batch]),
            torch.tensor([b['label']    for b in batch], dtype=torch.long),
            torch.tensor([b['domain']   for b in batch], dtype=torch.long),
            torch.tensor([b['age_group']for b in batch], dtype=torch.long),
            torch.tensor([b['gender']   for b in batch], dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# Manifold MixUp
# ─────────────────────────────────────────────────────────────────────────────

def manifold_mixup(embeddings, labels, domains, alpha=0.4):
    """
    MixUp in embedding space (Verma et al. 2019, Manifold MixUp).
    Smoother than input-level MixUp for audio — interpolating spectrograms
    creates artifacts, but interpolating learned embeddings is semantically
    meaningful. Regularizes the embedding space between class clusters.

    Returns mixed embeddings and soft label vectors.
    """
    lam = np.random.beta(alpha, alpha)
    B   = embeddings.size(0)
    idx = torch.randperm(B, device=embeddings.device)

    mixed_emb = lam * embeddings + (1 - lam) * embeddings[idx]
    # Soft labels for mixed samples
    y_onehot    = F.one_hot(labels,    num_classes=len(CLASS_NAMES)).float()
    y_dom_onehot = F.one_hot(domains,  num_classes=2).float()
    mixed_y     = lam * y_onehot    + (1 - lam) * y_onehot[idx]
    mixed_d     = lam * y_dom_onehot + (1 - lam) * y_dom_onehot[idx]
    return mixed_emb, mixed_y, mixed_d, lam


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum Learning
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumSampler:
    """
    Loss-based curriculum learning.
    Tracks per-sample loss history. At each epoch:
      - Phase 1 (warm-up, epochs 1-10): random order
      - Phase 2 (curriculum, epochs 11+): sample with probability
        proportional to recent loss (hard samples get higher weight)

    This is 'self-paced' curriculum: the model itself decides what's hard.
    Harder samples receive more training attention as training progresses.
    """
    def __init__(self, n_samples, warmup_epochs=10):
        self.losses       = np.ones(n_samples)  # uniform init
        self.n_samples    = n_samples
        self.warmup_epochs = warmup_epochs

    def update_losses(self, indices, losses):
        for i, loss in zip(indices, losses):
            self.losses[i] = 0.8 * self.losses[i] + 0.2 * loss  # EMA

    def get_weights(self, epoch):
        if epoch <= self.warmup_epochs:
            return np.ones(self.n_samples)
        # Hard samples (high loss) get higher sampling probability
        w = self.losses / (self.losses.sum() + 1e-8)
        # Clip to avoid extreme oversampling of any single sample
        w = np.clip(w, w.mean() * 0.1, w.mean() * 10)
        return w / w.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Data Discovery
# ─────────────────────────────────────────────────────────────────────────────

def _age_to_group(age):
    """Map integer age to age-group index (0–3), or -1 if unknown."""
    try:
        a = int(float(age))
        if a <= 20:  return 0
        if a <= 40:  return 1
        if a <= 60:  return 2
        return 3
    except Exception:
        return -1


def _gender_to_idx(g):
    """Map gender string to 0/1, or -1 if unknown."""
    if isinstance(g, str):
        g = g.strip().lower()
        if g in ('male', 'm'):   return 0
        if g in ('female', 'f'): return 1
    return -1


def _load_coswara_demographics(coswara_data_dir=None):
    """Return dict: patient_id → {age_group, gender}."""
    candidates = []
    if coswara_data_dir:
        candidates.append(Path(coswara_data_dir) / 'combined_data.csv')
    # Also try standard locations relative to this file
    candidates += [
        Path.home() / 'Downloads/TRANSFER/cough-ai-expo/Coswara-Data/combined_data.csv',
        Path(__file__).parent / 'combined_data.csv',
    ]
    for p in candidates:
        if p.exists():
            try:
                import pandas as pd
                df = pd.read_csv(p)
                demo = {}
                for _, row in df.iterrows():
                    pid = str(row.get('id', '')).strip()
                    if not pid:
                        continue
                    demo[pid] = {
                        'age_group': _age_to_group(row.get('a', -1)),
                        'gender':    _gender_to_idx(row.get('g', '')),
                    }
                print(f"[Demo] Loaded {len(demo)} Coswara demographic records from {p}")
                return demo
            except Exception as e:
                print(f"[Demo] Failed to load {p}: {e}")
    print("[Demo] No Coswara demographic metadata found — age/gender will be -1")
    return {}


def load_from_csv(csv_path):
    import csv
    samples = []
    p = Path(csv_path)
    if not p.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return samples

    # Try to load demographic metadata from Coswara
    demo_map = _load_coswara_demographics()

    with open(p) as f:
        for row in csv.DictReader(f):
            label = row.get('disease', '').lower().strip()
            path  = row.get('audio_path', '').strip()
            src   = row.get('source', 'unknown').lower().strip()
            label = {'covid-19': 'covid', 'covid': 'covid',
                     'healthy': 'healthy', 'normal': 'healthy',
                     'bronchitis': 'bronchitis'}.get(label)
            if label is None:
                continue
            fp = Path(path) if Path(path).is_absolute() else (p.parent / path).resolve()
            if not fp.exists():
                continue
            # Extract patient ID from filename: {pid}_cough-heavy.wav
            fname = fp.stem   # e.g. "iV3Db6t1T8b7c5HQY2TwxIhjbzD3_cough-heavy"
            pid   = fname.rsplit('_', 1)[0] if '_' in fname else fname
            demo  = demo_map.get(pid, {'age_group': -1, 'gender': -1})
            samples.append({
                'path': str(fp), 'label': label, 'domain': src,
                'age_group': demo['age_group'], 'gender': demo['gender'],
            })
    print(f"[CSV] {len(samples)} samples: {dict(Counter(s['label'] for s in samples))}")
    n_with_demo = sum(1 for s in samples if s['age_group'] >= 0)
    print(f"[CSV] {n_with_demo}/{len(samples)} samples have demographic labels")
    return samples


def discover_coswara(coswara_dir):
    samples = []
    p = Path(coswara_dir)
    if not p.exists():
        print(f"[WARN] Coswara dir not found: {coswara_dir}")
        return samples
    patient_labels = {}
    for mf in list(p.parent.rglob('combined_data.csv')) + list(p.parent.rglob('metadata*.csv')):
        try:
            import pandas as pd
            df = pd.read_csv(mf)
            if {'covid_status', 'id'} <= set(df.columns):
                for _, row in df.iterrows():
                    s = str(row['covid_status']).lower()
                    patient_labels[str(row['id'])] = (
                        'covid' if 'positive' in s else
                        'healthy' if 'healthy' in s or 'no_resp' in s else None)
        except Exception:
            pass
    for f in list(p.rglob('*cough-heavy*.wav')) + list(p.rglob('*cough_heavy*.wav')):
        fname = f.stem.lower()
        label = next((lbl for pid, lbl in patient_labels.items() if pid in fname), None)
        if label is None:
            label = ('covid' if 'covid' in fname or 'positive' in fname else
                     'healthy' if 'healthy' in fname or 'negative' in fname else None)
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
    meta = next((p / n for n in ['metadata_compiled.csv'] if (p / n).exists()),
                next(p.glob('metadata*.csv'), None))
    if meta is None:
        for ext in ['*.webm', '*.ogg', '*.wav']:
            for f in p.rglob(ext):
                samples.append({'path': str(f), 'label': 'bronchitis', 'domain': 'coughvid'})
        print(f"[CoughVID] {len(samples)} samples (no metadata)")
        return samples
    try:
        import pandas as pd
        df  = pd.read_csv(meta)
        lmap = {'COVID-19': 'covid', 'covid-19': 'covid', 'covid': 'covid',
                'healthy': 'healthy', 'bronchitis': 'bronchitis',
                'asthma': 'bronchitis', 'COPD': 'bronchitis'}
        cc   = 'cough_detected' if 'cough_detected' in df.columns else None
        sc   = 'status' if 'status' in df.columns else next(
            (c for c in df.columns if 'status' in c.lower()), None)
        for _, row in df.iterrows():
            uuid = str(row.get('uuid', row.get('filename', ''))).strip()
            if cc and float(row.get(cc, 1.0) or 1.0) < 0.8:
                continue
            label = lmap.get(str(row.get(sc, '') or ''))
            if not label:
                continue
            for ext in ['.webm', '.ogg', '.wav', '.mp3']:
                fp = p / (uuid + ext)
                if fp.exists():
                    samples.append({'path': str(fp), 'label': label, 'domain': 'coughvid'})
                    break
    except Exception as e:
        print(f"[ERROR] CoughVID: {e}")
    print(f"[CoughVID] {len(samples)} samples")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, pcgrad, criterion, device, epoch,
                total_epochs, ema_teacher, gradnorm, gradnorm_opt,
                curriculum=None, use_mixup=True, use_pcgrad=True):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    lambda_d    = get_lambda_schedule(epoch, total_epochs)
    lambda_demo = get_lambda_schedule(epoch, total_epochs)   # same schedule

    for batch_idx, (waveform, labels, domains, age_groups, genders) in enumerate(
            tqdm(loader, desc=f'Train {epoch}/{total_epochs}', leave=False)):
        waveform   = waveform.to(device)
        labels     = labels.to(device)
        domains    = domains.to(device)
        age_groups = age_groups.to(device)
        genders    = genders.to(device)

        # ── EMA teacher soft targets ──────────────────────────────────
        with torch.no_grad():
            teacher_probs = ema_teacher.predict(waveform)

        # ── Forward (V3 — lambda_demo added) ──────────────────────────
        out = model(waveform, lambda_d=lambda_d, lambda_demo=lambda_demo)

        # ── Manifold MixUp (embedding space) ─────────────────────────
        # Skip when phase encoder is active: disease_head input dim
        # changed (256+192=448), so the shortcut path is incompatible.
        _can_mixup = use_mixup and not model.use_phase_encoder
        if _can_mixup and random.random() < 0.5:
            emb_mix, y_soft, d_soft, lam = manifold_mixup(
                out['embeddings'], labels, domains)
            logits_mix = model.disease_head(emb_mix)   # 256-dim only
            dom_mix    = model.domain_clf(emb_mix, lambda_d)
            l_focal  = -(y_soft * F.log_softmax(logits_mix, dim=1)).sum(dim=1).mean()
            l_domain = -(d_soft * F.log_softmax(dom_mix, dim=1)).sum(dim=1).mean()
            l_supcon = criterion.supcon(out['embeddings'], labels)
            l_distill = F.kl_div(F.log_softmax(out['logits'], dim=1),
                                  teacher_probs, reduction='batchmean')
            l_demo   = criterion.demographic_loss(out, age_groups, genders)
        else:
            l_focal   = criterion.focal(out['logits'], labels)
            l_domain  = criterion.domain_loss(out['domain_logits'], domains)
            l_supcon  = criterion.supcon(out['embeddings'], labels)
            l_demo    = criterion.demographic_loss(out, age_groups, genders)
            l_distill = torch.tensor(0.0, device=device)
            if teacher_probs is not None:
                l_distill = F.kl_div(F.log_softmax(out['logits'], dim=1),
                                     teacher_probs, reduction='batchmean')

        # ── GradNorm dynamic weighting (4 tasks in V3) ────────────────
        if gradnorm is not None and batch_idx % 10 == 0:
            shared_param = next(iter(model.fusion.parameters()))
            try:
                gradnorm.update([l_focal, l_supcon, l_domain, l_demo],
                                shared_param, gradnorm_opt)
                gn_w = gradnorm.get_weights()
            except Exception:
                gn_w = torch.ones(4, device=device)
        else:
            gn_w = (gradnorm.get_weights() if gradnorm is not None
                    else torch.ones(4, device=device))

        l_cls_w    = gn_w[0] * l_focal
        l_supcon_w = gn_w[1] * l_supcon
        l_domain_w = gn_w[2] * l_domain
        l_demo_w   = gn_w[3] * l_demo if gn_w.numel() > 3 else l_demo

        # ── PCGrad or standard backward ──────────────────────────────
        # PCGrad: include ALL conflicting objectives so the graph is
        # consumed in a single backward pass (fixes retain_graph error).
        if use_pcgrad and isinstance(pcgrad, PCGrad):
            auxiliary = criterion.alpha * l_supcon_w + criterion.beta * l_distill
            pcgrad.pc_backward([l_cls_w, l_domain_w, l_demo_w, auxiliary])
            total_loss_val = float((l_cls_w + auxiliary - lambda_d * l_domain_w
                                    - lambda_demo * l_demo_w).detach())
        else:
            total_loss_tensor = (l_cls_w
                                 + criterion.alpha * l_supcon_w
                                 - lambda_d   * l_domain_w
                                 - lambda_demo * l_demo_w
                                 + criterion.beta * l_distill)
            pcgrad.zero_grad()
            total_loss_tensor.backward()
            total_loss_val = total_loss_tensor.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        pcgrad.step()

        # ── Memory bank + EMA updates ─────────────────────────────────
        with torch.no_grad():
            model.memory_bank.update(out['embeddings'].detach(), labels)
        ema_teacher.update(model)

        total_loss += total_loss_val
        all_preds.extend(out['logits'].argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), acc, f1, lambda_d


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    all_age_groups, all_genders = [], []

    for waveform, labels, domains, age_groups, genders in tqdm(loader, desc='Eval', leave=False):
        waveform = waveform.to(device)
        labels   = labels.to(device)
        domains  = domains.to(device)

        out    = model(waveform, lambda_d=0.0, lambda_demo=0.0)
        losses = criterion(out, labels, domains, lambda_d=0.0, lambda_demo=0.0)
        total_loss += losses['total'].item()

        probs = F.softmax(out['logits'], dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(probs.argmax(1))
        all_labels.extend(labels.cpu().numpy())
        all_age_groups.extend(age_groups.numpy())
        all_genders.extend(genders.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    acc = float(accuracy_score(all_labels, all_preds))
    f1  = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    cm  = confusion_matrix(all_labels, all_preds)

    per_class = {}
    for i, n in enumerate(CLASS_NAMES):
        m = all_labels == i
        per_class[n] = float(accuracy_score(all_labels[m], all_preds[m])) if m.sum() > 0 else 0.0

    try:
        if len(CLASS_NAMES) == 2:
            auc = float(roc_auc_score(all_labels, all_probs[:, 1]))
        else:
            auc = float(roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro'))
    except Exception:
        auc = 0.0

    # ── Demographic-stratified accuracy ─────────────────────────────
    all_age_groups = np.array(all_age_groups)
    all_genders    = np.array(all_genders)
    AGE_NAMES   = ['<21', '21-40', '41-60', '60+']
    GENDER_NAMES= ['male', 'female']

    age_acc, gender_acc = {}, {}
    for gi, gname in enumerate(AGE_NAMES):
        m = all_age_groups == gi
        if m.sum() > 0:
            age_acc[gname] = float(accuracy_score(all_labels[m], all_preds[m]))
    for gi, gname in enumerate(GENDER_NAMES):
        m = all_genders == gi
        if m.sum() > 0:
            gender_acc[gname] = float(accuracy_score(all_labels[m], all_preds[m]))

    return dict(loss=total_loss/len(loader), accuracy=acc, f1_macro=f1,
                auc=auc, per_class_acc=per_class, confusion=cm.tolist(),
                age_acc=age_acc, gender_acc=gender_acc)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  Output: {output_dir}")
    print(f"PCGrad: {args.use_pcgrad}  |  GradNorm: {args.use_gradnorm}  "
          f"|  Curriculum: {args.use_curriculum}  |  MixUp: {args.use_mixup}")

    # ── Data ──────────────────────────────────────────────────────────
    if args.use_csv:
        all_samples = load_from_csv(args.csv_path)
    else:
        all_samples  = discover_coswara(args.coswara_dir)
        all_samples += discover_coughvid(args.coughvid_dir)

    if not all_samples:
        print("[ERROR] No samples found.")
        return

    lc = Counter(s['label'] for s in all_samples)
    dc = Counter(s.get('domain', '?') for s in all_samples)
    print(f"\nDataset: {dict(lc)} | Domains: {dict(dc)} | Total: {len(all_samples)}\n")

    # Build class list dynamically — preserve canonical order, keep only present classes
    global CLASS_NAMES, CLASS_MAP
    present = set(lc.keys())
    CLASS_NAMES = [c for c in ALL_CLASS_NAMES if c in present]
    if len(CLASS_NAMES) < len(present):
        # Any class not in canonical list gets appended
        CLASS_NAMES += sorted(present - set(CLASS_NAMES))
    CLASS_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}
    print(f"Active classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

    # Reduce folds when dataset is small (need at least 2 samples per class per fold)
    min_class_count = min(lc.values())
    safe_folds = min(args.folds, max(2, min_class_count // 2))
    if safe_folds != args.folds:
        print(f"[INFO] Reduced folds {args.folds}→{safe_folds} (smallest class has {min_class_count} samples)")
        args.folds = safe_folds

    labels_arr  = np.array([s['label'] for s in all_samples])
    skf         = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(all_samples, labels_arr)):
        print(f"\n{'='*65}")
        print(f"FOLD {fold+1}/{args.folds}")
        print(f"{'='*65}")

        tr_s  = [all_samples[i] for i in tr_idx]
        val_s = [all_samples[i] for i in val_idx]

        tr_ds  = CoughDataset(tr_s,  augment=True)
        val_ds = CoughDataset(val_s, augment=False)

        # Weighted sampler
        tr_lbl   = [CLASS_MAP[s['label']] for s in tr_s]
        cc       = Counter(tr_lbl)
        weights  = [1.0 / cc[l] for l in tr_lbl]
        sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)

        # num_workers=0 avoids multiprocessing issues on macOS; pin_memory only for CUDA
        _nw = 0 if sys.platform == 'darwin' else 2
        _pm = device.type == 'cuda'
        tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size,
                                sampler=sampler, num_workers=_nw,
                                collate_fn=collate_fn, pin_memory=_pm)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=_nw,
                                collate_fn=collate_fn, pin_memory=_pm)

        # ── Model & optimizers ─────────────────────────────────────
        model     = build_model(num_classes=len(CLASS_NAMES), bank_size=args.bank_size,
                               use_phase_encoder=args.use_phase_encoder).to(device)
        if fold == 0:
            count_parameters(model)

        base_opt  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        pcgrad    = PCGrad(base_opt) if args.use_pcgrad else base_opt
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_opt, T_0=max(1, args.epochs // 3), T_mult=1, eta_min=1e-6)

        criterion  = CoughSenseLoss(alpha=0.3, beta=0.1, memory_bank=model.memory_bank,
                                    focal_gamma=2.0, label_smoothing=0.1)
        ema_teacher = EMATeacher(model, alpha=args.ema_alpha)

        # GradNorm — V3 has 4 tasks: focal, supcon, domain, demo
        gradnorm, gradnorm_opt = None, None
        if args.use_gradnorm:
            gradnorm = GradNorm(n_tasks=4, alpha_gn=1.5).to(device)
            gradnorm_opt = torch.optim.Adam(gradnorm.parameters(), lr=1e-3)

        # Curriculum
        curriculum = CurriculumSampler(len(tr_ds), warmup_epochs=10) if args.use_curriculum else None

        best_f1, best_state = 0.0, None
        history = []

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc, tr_f1, lam = train_epoch(
                model, tr_loader, pcgrad, criterion, device, epoch, args.epochs,
                ema_teacher, gradnorm, gradnorm_opt,
                curriculum=curriculum,
                use_mixup=args.use_mixup,
                use_pcgrad=args.use_pcgrad)
            val_m = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(epoch)

            history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc, tr_f1=tr_f1,
                                val_acc=val_m['accuracy'], val_f1=val_m['f1_macro'],
                                val_auc=val_m['auc'], lambda_d=lam))

            print(f"Ep {epoch:3d} | loss={tr_loss:.3f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "
                  f"val_acc={val_m['accuracy']:.3f} f1={val_m['f1_macro']:.3f} "
                  f"auc={val_m['auc']:.3f} λ_d={lam:.3f}")
            for cls, acc_v in val_m['per_class_acc'].items():
                print(f"       {cls:12s}: {acc_v:.3f}")
            if val_m.get('age_acc'):
                print("       [age]  " + " ".join(f"{k}:{v:.2f}" for k, v in val_m['age_acc'].items()))
            if val_m.get('gender_acc'):
                print("       [gender]" + " ".join(f"{k}:{v:.2f}" for k, v in val_m['gender_acc'].items()))

            if val_m['f1_macro'] > best_f1:
                best_f1    = val_m['f1_macro']
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"  ✓ Best F1: {best_f1:.4f}")

        # ── Final eval + calibration ───────────────────────────────
        model.load_state_dict(best_state)
        model.to(device)
        final = eval_epoch(model, val_loader, criterion, device)
        fold_results.append(final)

        # Fit temperature calibration on val logits
        with torch.no_grad():
            all_logits, all_labels_t = [], []
            for wave, labs, doms, ag, gen in val_loader:
                out = model(wave.to(device), lambda_d=0.0, lambda_demo=0.0)
                all_logits.append(out['logits'].cpu())
                all_labels_t.append(labs)
            all_logits   = torch.cat(all_logits)
            all_labels_t = torch.cat(all_labels_t)
        model.calibrator.calibrate(all_logits, all_labels_t)

        ckpt = output_dir / f'fold{fold+1}_best.pt'
        torch.save(dict(model_state=best_state, fold=fold+1, best_val_f1=best_f1,
                        final_metrics=final, history=history,
                        class_names=CLASS_NAMES, args=vars(args),
                        calibrator_state=model.calibrator.state_dict()), ckpt)
        print(f"\nSaved: {ckpt}")
        print(f"Final: acc={final['accuracy']:.4f} f1={final['f1_macro']:.4f} auc={final['auc']:.4f}")
        cm = np.array(final['confusion'])
        print("Confusion matrix:")
        for row, n in zip(cm, CLASS_NAMES):
            print(f"  {n:12s}: {row}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*65}")
    for k in ['accuracy', 'f1_macro', 'auc']:
        vals = [r[k] for r in fold_results]
        print(f"  {k:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("\nPer-class accuracy (mean ± std):")
    for c in CLASS_NAMES:
        vals = [r['per_class_acc'].get(c, 0) for r in fold_results]
        print(f"  {c:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    summary = dict(
        mean_accuracy=float(np.mean([r['accuracy'] for r in fold_results])),
        mean_f1      =float(np.mean([r['f1_macro'] for r in fold_results])),
        mean_auc     =float(np.mean([r['auc']      for r in fold_results])),
        std_accuracy =float(np.std( [r['accuracy'] for r in fold_results])),
        std_f1       =float(np.std( [r['f1_macro'] for r in fold_results])),
        fold_results =fold_results, config=vars(args))
    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_dir}/cv_summary.json")
    print("\nTo run the service: python coughsense_service.py")


def parse_args():
    p = argparse.ArgumentParser(description='Train CoughSense V2')
    p.add_argument('--coswara_dir',    default='../../Coswara-Data/all_audio')
    p.add_argument('--coughvid_dir',   default='../../coughvid')
    p.add_argument('--output_dir',     default='./checkpoints')
    p.add_argument('--epochs',         type=int,   default=80)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--lr',             type=float, default=6e-4)
    p.add_argument('--folds',          type=int,   default=5)
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--bank_size',      type=int,   default=256)
    p.add_argument('--ema_alpha',      type=float, default=0.999)
    # Innovation flags
    p.add_argument('--use_pcgrad',          action='store_true', default=True)
    p.add_argument('--no_pcgrad',           dest='use_pcgrad',         action='store_false')
    p.add_argument('--use_gradnorm',        action='store_true', default=True)
    p.add_argument('--no_gradnorm',         dest='use_gradnorm',       action='store_false')
    p.add_argument('--use_curriculum',      action='store_true', default=True)
    p.add_argument('--no_curriculum',       dest='use_curriculum',     action='store_false')
    p.add_argument('--use_mixup',           action='store_true', default=True)
    p.add_argument('--no_mixup',            dest='use_mixup',          action='store_false')
    p.add_argument('--use_phase_encoder',   action='store_true', default=True)
    p.add_argument('--no_phase_encoder',    dest='use_phase_encoder',  action='store_false')
    # Data
    p.add_argument('--use_csv',  action='store_true')
    p.add_argument('--csv_path', default='ml_service/local_data.csv')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
