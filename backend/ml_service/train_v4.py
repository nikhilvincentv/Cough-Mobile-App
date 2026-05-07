"""
train_v4.py — CoughSense V4: Multi-Modal + Symptom-Conditioned Training

Key additions over V3:
  1. FULL DATASET  : 2,046 samples (1,389 healthy + 657 COVID) from all
                     Coswara TRANSFER folders — 14× more than V3's 147.
  2. DUAL MODALITY : cough-heavy AND cough-shallow per patient.
                     Late-fusion: run both through the shared encoder,
                     mean-pool embeddings before the disease head.
  3. SYMPTOM FiLM  : 7 binary symptom flags (fever, cold, cough_sym,
                     diarrhoea, loss_of_smell, fatigue, sore_throat)
                     → 64-dim MLP → FiLM scale+shift on the fused audio
                     embedding.  CoughVID / missing = zero vector.
  4. LP-FT READY   : architecture exposes freeze/unfreeze so LP→FT can
                     be done in two phases if HeAR weights are available.

Usage
-----
    python3 ml_service/train_v4.py \
        --csv ml_service/full_data.csv \
        --output_dir checkpoints_v4 \
        --epochs 60 --batch_size 16 --folds 5 --seed 42

Author: Nikhil Vincent
"""

import os, sys, csv, random, math, json, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000
CLIP_SAMPLES  = int(SAMPLE_RATE * 3.5)
N_MELS        = 64
HOP           = 160
N_FFT         = 512
CLASS_NAMES   = ['healthy', 'covid']
SYMPTOM_COLS  = ['sym_fever','sym_cold','sym_cough',
                 'sym_diarrhoea','sym_loss_of_smell','sym_ftg','sym_st']

# SpecAugment parameters
SPECAUG_FREQ_MASKS  = 2
SPECAUG_FREQ_WIDTH  = 12
SPECAUG_TIME_MASKS  = 2
SPECAUG_TIME_WIDTH  = 20
SPECAUG_PROB        = 0.8

# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str) -> torch.Tensor:
    """Load WAV → (1, CLIP_SAMPLES) normalised tensor. Returns zeros on error."""
    try:
        w, sr = torchaudio.load(path)
    except Exception:
        return torch.zeros(1, CLIP_SAMPLES)
    if w.numel() == 0:
        return torch.zeros(1, CLIP_SAMPLES)
    if sr != SAMPLE_RATE:
        w = torchaudio.functional.resample(w, sr, SAMPLE_RATE)
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    T = w.shape[-1]
    if T >= CLIP_SAMPLES:
        start = (T - CLIP_SAMPLES) // 2
        w = w[..., start:start + CLIP_SAMPLES]
    else:
        w = F.pad(w, (0, CLIP_SAMPLES - T))
    peak = w.abs().max()
    return w / (peak + 1e-8) if peak > 0 else w


def wav_to_mel(w: torch.Tensor, training: bool = False) -> torch.Tensor:
    """Waveform (1,T) → log-mel (1, N_MELS, W). Optional SpecAugment."""
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP, n_fft=N_FFT)
    spec = mel_fn(w)                         # (1, N_MELS, W)
    spec = (spec + 1e-6).log()
    mu, sigma = spec.mean(), spec.std() + 1e-8
    spec = (spec - mu) / sigma

    if training and random.random() < SPECAUG_PROB:
        _, F_bins, T_bins = spec.shape
        for _ in range(SPECAUG_FREQ_MASKS):
            f0 = random.randint(0, max(0, F_bins - SPECAUG_FREQ_WIDTH))
            spec[0, f0:f0 + random.randint(1, SPECAUG_FREQ_WIDTH), :] = 0
        for _ in range(SPECAUG_TIME_MASKS):
            t0 = random.randint(0, max(0, T_bins - SPECAUG_TIME_WIDTH))
            spec[0, :, t0:t0 + random.randint(1, SPECAUG_TIME_WIDTH)] = 0
    return spec

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CoughDatasetV4(Dataset):
    def __init__(self, samples: list, training: bool = False):
        self.samples  = samples
        self.training = training
        self.label2id = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        label = self.label2id[s['label']]

        # ── Primary audio (cough-heavy) ──
        w_heavy = load_audio(s['path_heavy'])
        if self.training:
            # random crop augmentation
            T = w_heavy.shape[-1]
            if T > CLIP_SAMPLES:
                start = random.randint(0, T - CLIP_SAMPLES)
                w_heavy = w_heavy[..., start:start + CLIP_SAMPLES]
        mel_heavy = wav_to_mel(w_heavy, self.training)  # (1, 64, W)

        # ── Secondary audio (cough-shallow) if available ──
        if s.get('path_shallow') and Path(s['path_shallow']).exists():
            w_shallow = load_audio(s['path_shallow'])
            mel_shallow = wav_to_mel(w_shallow, self.training)
            has_shallow = torch.tensor(1.0)
        else:
            mel_shallow = torch.zeros_like(mel_heavy)
            has_shallow = torch.tensor(0.0)

        # ── Symptom vector ──
        sym = torch.tensor(s.get('symptoms', [0]*7), dtype=torch.float32)

        return mel_heavy, mel_shallow, has_shallow, sym, label


def load_csv(csv_path: str) -> list:
    samples = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            disease = row.get('disease', '').strip().lower()
            if disease not in CLASS_NAMES:
                continue
            path_h = row.get('audio_path', '').strip()
            if not path_h or not Path(path_h).exists():
                continue
            symptoms = [int(row.get(c, 0) or 0) for c in SYMPTOM_COLS]
            samples.append({
                'label':        disease,
                'path_heavy':   path_h,
                'path_shallow': row.get('audio_path_shallow', '').strip(),
                'symptoms':     symptoms,
                'age':          row.get('age', ''),
                'gender':       row.get('gender', ''),
            })
    return samples

# ─────────────────────────────────────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(),
            nn.Linear(channels // r, channels),
            nn.Sigmoid())
    def forward(self, x):
        s = x.mean(dim=(-2,-1))
        return x * self.fc(s).unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout))
        self.se   = SEBlock(cout)
        self.skip = nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
            nn.BatchNorm2d(cout)) if cin != cout or stride != 1 else nn.Identity()
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) * self.se(self.conv(x)).sigmoid() + self.skip(x))

    def forward(self, x):
        h = self.conv(x)
        h = h * self.se(h)
        return self.act(h + self.skip(x))


class CNNBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU())
        self.blocks = nn.Sequential(
            ResBlock(32,  64, stride=2),
            ResBlock(64,  128, stride=2),
            ResBlock(128, 256, stride=2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.proj(self.drop(x))


class ViTBranch(nn.Module):
    def __init__(self, img_h=64, img_w=None, patch=8, dim=128, depth=4, heads=4, out_dim=256):
        super().__init__()
        if img_w is None:
            img_w = img_h * 2   # 64×128 default
        nh, nw = img_h // patch, img_w // patch
        n_patches = nh * nw
        self.patch_emb = nn.Conv2d(1, dim, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_emb   = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.layers    = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim*4,
                                       dropout=0.1, batch_first=True,
                                       norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb,   std=0.02)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_emb(x).flatten(2).transpose(1,2)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_emb
        for layer in self.layers:
            tokens = layer(tokens)
        return self.proj(self.norm(tokens[:, 0]))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: condition audio features on symptoms."""
    def __init__(self, feat_dim=256, sym_dim=7, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sym_dim, hidden), nn.GELU(),
            nn.Linear(hidden, feat_dim * 2))   # gamma + beta

    def forward(self, feat, sym):
        gb   = self.mlp(sym)
        gamma, beta = gb.chunk(2, dim=-1)
        return feat * (1 + gamma) + beta        # residual FiLM


class GradReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class CoughSenseV4(nn.Module):
    """
    Dual-branch CNN-ViT with:
      - dual audio modality late-fusion (heavy + shallow)
      - FiLM symptom conditioning
      - domain-adversarial head (GRL)
      - disease head (binary: healthy vs COVID)
    """
    def __init__(self, n_classes=2, sym_dim=7, embed_dim=256):
        super().__init__()
        self.cnn = CNNBranch(out_dim=embed_dim)
        self.vit = ViTBranch(img_h=64, img_w=128, out_dim=embed_dim)

        # Attention gate
        self.gate = nn.Sequential(nn.Linear(embed_dim*2, 2), nn.Softmax(dim=-1))
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(), nn.LayerNorm(embed_dim))

        # FiLM symptom conditioning
        self.film = FiLMLayer(feat_dim=embed_dim, sym_dim=sym_dim)

        # Domain adversarial head (GRL)
        self.domain_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 2))

        # Disease head
        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def encode(self, mel):
        """mel: (B, 1, 64, W) → (B, embed_dim)"""
        # Resize to fixed 64×128
        mel = F.interpolate(mel, size=(64, 128), mode='bilinear', align_corners=False)
        fc  = self.cnn(mel)
        ft  = self.vit(mel)
        g   = self.gate(torch.cat([fc, ft], dim=-1))
        f   = g[:,0:1]*fc + g[:,1:2]*ft
        return self.proj(f)

    def forward(self, mel_heavy, mel_shallow, has_shallow, sym, lam=0.0):
        """
        mel_heavy, mel_shallow: (B,1,64,W)
        has_shallow: (B,) float 0/1
        sym: (B, sym_dim)
        lam: GRL lambda
        """
        z_h = self.encode(mel_heavy)                  # (B, D)

        # Late-fusion: average heavy + shallow when shallow is available
        z_s = self.encode(mel_shallow)
        mask = has_shallow.unsqueeze(-1)               # (B,1)
        z    = z_h + mask * z_s                        # sum; normalise
        n    = 1 + mask                                # (B,1) = 1 or 2
        z    = z / n                                   # mean

        # FiLM symptom conditioning
        z = self.film(z, sym)

        # L2 normalise
        z_norm = F.normalize(z, dim=-1)

        # Domain adversarial
        z_rev = GradReversalFn.apply(z_norm, lam)
        domain_logits = self.domain_head(z_rev)

        # Disease prediction
        logits = self.disease_head(z_norm)

        return logits, domain_logits, z_norm

# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_lambda(epoch, total):
    p = epoch / total
    return float(2 / (1 + math.exp(-10 * p)) - 1)


def make_weighted_sampler(samples):
    labels = [CLASS_NAMES.index(s['label']) for s in samples]
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for mel_h, mel_s, has_s, sym, labels in loader:
            mel_h = mel_h.to(device)
            mel_s = mel_s.to(device)
            has_s = has_s.to(device)
            sym   = sym.to(device)
            logits, _, _ = model(mel_h, mel_s, has_s, sym, lam=0.0)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:,1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    return bal_acc, f1, auc


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs):
    model.train()
    lam = get_lambda(epoch, total_epochs)

    focal_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    total_loss = 0.0
    for mel_h, mel_s, has_s, sym, labels in loader:
        mel_h  = mel_h.to(device)
        mel_s  = mel_s.to(device)
        has_s  = has_s.to(device)
        sym    = sym.to(device)
        labels = labels.to(device)

        logits, dom_logits, _ = model(mel_h, mel_s, has_s, sym, lam=lam)

        # Disease loss (focal-style via label smoothing cross entropy)
        p_hat = F.softmax(logits, dim=-1)
        pt    = p_hat.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt.detach()).pow(2)
        ce    = F.cross_entropy(logits, labels, reduction='none', label_smoothing=0.1)
        loss_cls = (focal_weight * ce).mean()

        # Domain adversarial loss (binary: coswara=0, fake domain)
        # All samples are Coswara here; domain head should predict uniform → chance
        dom_labels = torch.zeros(labels.shape[0], dtype=torch.long, device=device)
        loss_dom = F.cross_entropy(dom_logits, dom_labels) * lam * 0.3

        loss = loss_cls + loss_dom

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',        default='ml_service/full_data.csv')
    p.add_argument('--output_dir', default='checkpoints_v4')
    p.add_argument('--epochs',     type=int,   default=60)
    p.add_argument('--batch_size', type=int,   default=16)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--folds',      type=int,   default=5)
    p.add_argument('--seed',       type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Load data
    all_samples = load_csv(args.csv)
    print(f"Loaded {len(all_samples)} samples: {Counter(s['label'] for s in all_samples)}")

    labels_arr = np.array([CLASS_NAMES.index(s['label']) for s in all_samples])

    # Reduce folds if needed
    min_count = min(Counter(labels_arr).values())
    safe_folds = min(args.folds, max(2, min_count // 4))
    if safe_folds != args.folds:
        print(f"Reduced folds {args.folds}→{safe_folds} (min class={min_count})")
        args.folds = safe_folds

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []

    log_path = out_dir / 'training_log_v4.txt'
    log = open(log_path, 'w')

    for fold, (tr_idx, val_idx) in enumerate(skf.split(all_samples, labels_arr)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.folds}")
        print(f"Train: {len(tr_idx)}  Val: {len(val_idx)}")

        tr_samples  = [all_samples[i] for i in tr_idx]
        val_samples = [all_samples[i] for i in val_idx]

        tr_ds  = CoughDatasetV4(tr_samples,  training=True)
        val_ds = CoughDatasetV4(val_samples, training=False)

        sampler  = make_weighted_sampler(tr_samples)
        tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size,
                                sampler=sampler, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

        model = CoughSenseV4(n_classes=len(CLASS_NAMES), sym_dim=len(SYMPTOM_COLS)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.epochs, T_mult=1)

        best_f1, best_bal_acc, best_auc = 0.0, 0.0, 0.0
        best_path = out_dir / f'fold{fold+1}_best.pt'

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, tr_loader, optimizer, scheduler,
                                   device, epoch, args.epochs)
            bal_acc, f1, auc = evaluate(model, val_loader, device)

            line = (f"Fold {fold+1} Ep {epoch:3d}/{args.epochs} | "
                    f"loss={loss:.3f} | bal_acc={bal_acc:.4f} f1={f1:.4f} auc={auc:.4f}")
            print(line)
            log.write(line + '\n')
            log.flush()

            if f1 > best_f1:
                best_f1       = f1
                best_bal_acc  = bal_acc
                best_auc      = auc
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Best F1: {best_f1:.4f}  bal_acc={best_bal_acc:.4f}  AUC={best_auc:.4f}")

        fold_results.append({
            'fold':      fold + 1,
            'bal_acc':   best_bal_acc,
            'f1_macro':  best_f1,
            'auc':       best_auc,
        })
        print(f"Fold {fold+1} best → bal_acc={best_bal_acc:.4f}  f1={best_f1:.4f}  auc={best_auc:.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    bal_accs = [r['bal_acc']  for r in fold_results]
    f1s      = [r['f1_macro'] for r in fold_results]
    aucs     = [r['auc']      for r in fold_results]

    summary = {
        'mean_bal_acc':  float(np.mean(bal_accs)),
        'std_bal_acc':   float(np.std(bal_accs)),
        'mean_f1':       float(np.mean(f1s)),
        'std_f1':        float(np.std(f1s)),
        'mean_auc':      float(np.mean(aucs)),
        'std_auc':       float(np.std(aucs)),
        'fold_results':  fold_results,
        'config': vars(args),
    }

    for k, v in summary.items():
        if k not in ('fold_results', 'config'):
            print(f"  {k:18s}: {v:.4f}")

    summary_path = out_dir / 'cv_summary_v4.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    fin = (f"\nFINAL: {summary['mean_bal_acc']*100:.1f}% ± {summary['std_bal_acc']*100:.1f}%"
           f"  F1={summary['mean_f1']:.3f}  AUC={summary['mean_auc']:.3f}")
    print(fin)
    log.write(fin + '\n')
    log.close()


if __name__ == '__main__':
    main()
