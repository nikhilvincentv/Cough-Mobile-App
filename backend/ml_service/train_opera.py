"""
train_opera.py — CoughSense V6: OPERA-CT Fine-tuning (5-class)

OPERA (NeurIPS 2024): Open Respiratory Acoustic Foundation Model
  - Pretrained on 136K health audio recordings
  - Architecture: ViT-based audio masked autoencoder (Audio-MAE)
  - Input: 128-bin mel spectrogram, 128 time steps (~1.3 sec)
  - Checkpoint: evelyn0414/OPERA on HuggingFace (opera-ct.pth, 355MB)

Novelty for paper: Comparison between two domain-specific pretrained models:
  1. Whisper encoder (speech domain, 680k hours)  ← train_whisper.py
  2. OPERA encoder (respiratory health domain, 136k samples)  ← this script

Usage:
    # Download OPERA-CT checkpoint first:
    python3 -c "from huggingface_hub import hf_hub_download; \
        hf_hub_download('evelyn0414/OPERA', 'opera-ct.pth', \
        local_dir='checkpoints_opera')"

    # Fine-tune OPERA on 5-class cough data:
    python3 -u ml_service/train_opera.py \
        --csv ml_service/v6_data.csv \
        --output_dir checkpoints_opera \
        --checkpoint checkpoints_opera/opera-ct.pth \
        --epochs 40 --batch_size 32 --folds 5

    # Or run EfficientNet-B2 CNN baseline (no external checkpoint):
    python3 -u ml_service/train_opera.py \
        --csv ml_service/v6_data.csv \
        --output_dir checkpoints_effnet \
        --model effnet \
        --epochs 40 --batch_size 32 --folds 5

Author: Nikhil Vincent
"""

import os, sys, csv, random, math, json, argparse, hashlib, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import timm

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES   = ['healthy', 'covid', 'respiratory_cond', 'bronchitis', 'pneumonia']
N_CLASSES     = len(CLASS_NAMES)
DISEASE_REMAP = {'asthma': 'respiratory_cond'}
SYMPTOM_COLS  = ['sym_fever','sym_cold','sym_cough',
                 'sym_diarrhoea','sym_loss_of_smell','sym_ftg','sym_st']

SAMPLE_RATE   = 16_000
# OPERA mel params (from OPERA paper / repo)
OPERA_N_MELS  = 128
OPERA_N_FFT   = 1024
OPERA_HOP     = 160    # 10ms at 16kHz
OPERA_FRAMES  = 128    # ~1.28 seconds per crop
OPERA_DIM     = 768    # ViT-base dimension (OPERA-CT)

# EfficientNet-B2 params (CNN baseline)
EFFNET_N_MELS = 128
EFFNET_FRAMES = 256    # ~2.56 seconds

HEAD_LR       = 3e-4
ENC_LR        = 3e-5
FREEZE_EPOCHS = 5

# ─── Mel computation ─────────────────────────────────────────────────────────

def _load_audio_np(path: str, max_seconds: float = 4.0) -> np.ndarray:
    """Load audio with librosa → (max_samples,) float32 numpy array."""
    max_samples = int(max_seconds * SAMPLE_RATE)
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True,
                             duration=max_seconds)
        peak = np.abs(y).max()
        if peak > 0:
            y = y / (peak + 1e-8)
        T = len(y)
        if T >= max_samples:
            y = y[:max_samples]
        else:
            y = np.pad(y, (0, max_samples - T))
        return y.astype(np.float32)
    except Exception:
        return np.zeros(max_samples, dtype=np.float32)


def wav_to_opera_mel(y: np.ndarray) -> torch.Tensor:
    """numpy audio → (128, 128) normalized log-mel tensor."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=OPERA_N_MELS,
        n_fft=OPERA_N_FFT, hop_length=OPERA_HOP,
        win_length=OPERA_N_FFT, window='hann',
        center=True, pad_mode='reflect', power=2.0,
        norm='slaney', htk=True)
    mel = np.log(mel + 1e-6)
    mel = mel[:, :OPERA_FRAMES]
    if mel.shape[1] < OPERA_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, OPERA_FRAMES - mel.shape[1])))
    mel = torch.from_numpy(mel.astype(np.float32))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel   # (128, 128)


def wav_to_effnet_mel(y: np.ndarray) -> torch.Tensor:
    """numpy audio → (1, 128, 256) normalized log-mel tensor for EfficientNet."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=EFFNET_N_MELS,
        n_fft=1024, hop_length=160, win_length=1024,
        window='hann', center=True, pad_mode='reflect',
        power=2.0, norm='slaney', htk=True)
    mel = np.log(mel + 1e-6)
    mel = mel[:, :EFFNET_FRAMES]
    if mel.shape[1] < EFFNET_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, EFFNET_FRAMES - mel.shape[1])))
    mel = torch.from_numpy(mel.astype(np.float32))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0)  # (1, 128, 256)


def _specaugment(mel: torch.Tensor, is_minority: bool = False) -> torch.Tensor:
    if random.random() < 0.2:
        return mel
    mel = mel.clone()
    # mel shape: either (128, 128) or (1, 128, 256)
    if mel.dim() == 3:
        _, F_bins, T_bins = mel.shape
        for _ in range(2 + is_minority):
            f0 = random.randint(0, max(0, F_bins - 10))
            mel[:, f0:f0 + random.randint(1, 10), :] = 0
        for _ in range(2 + is_minority):
            t0 = random.randint(0, max(0, T_bins - 20))
            mel[:, :, t0:t0 + random.randint(1, 20)] = 0
    else:
        F_bins, T_bins = mel.shape
        for _ in range(2 + is_minority):
            f0 = random.randint(0, max(0, F_bins - 10))
            mel[f0:f0 + random.randint(1, 10), :] = 0
        for _ in range(2 + is_minority):
            t0 = random.randint(0, max(0, T_bins - 20))
            mel[:, t0:t0 + random.randint(1, 20)] = 0
    return mel


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CoughDataset(Dataset):
    def __init__(self, samples: list, model_type: str, training: bool = False):
        self.samples    = samples
        self.model_type = model_type
        self.training   = training
        self.label2id   = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        label = self.label2id[s['label']]
        is_minority = s['label'] in ('bronchitis', 'pneumonia', 'covid')

        y = _load_audio_np(s['path'], max_seconds=4.0)
        if self.model_type == 'opera':
            mel = wav_to_opera_mel(y)
        else:
            mel = wav_to_effnet_mel(y)

        if self.training:
            mel = _specaugment(mel, is_minority)

        sym    = torch.tensor(s.get('symptoms', [0]*7), dtype=torch.float32)
        domain = torch.tensor(s.get('domain', 0), dtype=torch.long)
        return mel, sym, domain, label


def load_csv(csv_path: str) -> list:
    samples = []
    missing = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            disease = row.get('disease', '').strip().lower()
            disease = DISEASE_REMAP.get(disease, disease)
            if disease not in CLASS_NAMES:
                continue
            path = row.get('audio_path', '').strip()
            if not path or not Path(path).exists():
                missing += 1
                continue
            symptoms = [int(row.get(c, 0) or 0) for c in SYMPTOM_COLS]
            samples.append({
                'label':    disease,
                'path':     path,
                'symptoms': symptoms,
                'domain':   int(row.get('domain', 0) or 0),
            })
    if missing:
        print(f"  Warning: skipped {missing} rows with missing audio")
    return samples


def make_weighted_sampler(samples: list) -> WeightedRandomSampler:
    counts = Counter(s['label'] for s in samples)
    base_weight = {c: 1.0 / max(cnt, 1) for c, cnt in counts.items()}
    weights = [base_weight[s['label']] for s in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


# ─── OPERA Model ─────────────────────────────────────────────────────────────

class OperaClassifier(nn.Module):
    """
    OPERA-CT encoder (Audio-MAE / ViT-base) fine-tuned for cough classification.

    The OPERA encoder expects (B, 1, 128, 128) log-mel spectrograms.
    We load the pretrained Audio-MAE weights, strip the decoder, and add a
    classification head on top of the CLS token.
    """

    def __init__(self, checkpoint_path: str, n_classes: int = N_CLASSES,
                 sym_dim: int = 7, n_domains: int = 3):
        super().__init__()
        # Build ViT-base using timm (matches OPERA-CT architecture)
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0,          # remove classifier head
            img_size=(128, 128),    # OPERA mel input size
            in_chans=1,             # single-channel mel
        )
        dim = self.encoder.embed_dim  # 768 for ViT-base

        # Load OPERA-CT pretrained weights (partial load — encoder only)
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state = ckpt.get('model', ckpt)
            # Filter to encoder keys only (OPERA checkpoint includes decoder)
            enc_state = {k.replace('encoder.', '').replace('module.', ''): v
                         for k, v in state.items()
                         if 'decoder' not in k}
            missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
            print(f"  OPERA checkpoint loaded: {len(enc_state)} keys, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            print("  WARNING: No OPERA checkpoint found — training from scratch.")
            print("  Download: python3 -c \"from huggingface_hub import "
                  "hf_hub_download; hf_hub_download('evelyn0414/OPERA', "
                  "'opera-ct.pth', local_dir='checkpoints_opera')\"")

        self.film = nn.Sequential(
            nn.Linear(sym_dim, 128), nn.GELU(),
            nn.Linear(128, dim * 2))

        self.domain_head = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, n_domains))

        self.disease_head = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def forward(self, mel, sym, lam=0.0):
        # mel: (B, 128, 128) → unsqueeze to (B, 1, 128, 128) for timm ViT
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        z = self.encoder(mel)       # (B, 768) — CLS token

        # FiLM conditioning
        gb = self.film(sym)
        gamma, beta = gb.chunk(2, dim=-1)
        z = z * (1 + gamma) + beta
        z_n = F.normalize(z, dim=-1)

        # GRL domain head
        from train_whisper import GradReversalFn
        z_rev = GradReversalFn.apply(z_n, lam)
        domain_logits  = self.domain_head(z_rev)
        disease_logits = self.disease_head(z_n)
        return disease_logits, domain_logits, z_n


# ─── EfficientNet-B2 Baseline ─────────────────────────────────────────────────

class EffNetClassifier(nn.Module):
    """
    EfficientNet-B2 baseline: 2D CNN on log-mel spectrogram.
    Pretrained on ImageNet, fine-tuned on (1, 128, 256) mel images.
    Used as CNN baseline for comparison with Whisper encoder in paper.
    """

    def __init__(self, n_classes: int = N_CLASSES, sym_dim: int = 7,
                 n_domains: int = 3):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b2',
            pretrained=True,
            num_classes=0,       # no head
            in_chans=1,
        )
        dim = self.backbone.num_features  # 1408 for EffNet-B2

        self.film = nn.Sequential(
            nn.Linear(sym_dim, 128), nn.GELU(),
            nn.Linear(128, dim * 2))

        self.domain_head = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, n_domains))

        self.disease_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1) if False else nn.Identity(),
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def freeze_encoder(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def forward(self, mel, sym, lam=0.0):
        # mel: (B, 1, 128, 256)
        z = self.backbone(mel)   # (B, 1408)

        gb = self.film(sym)
        gamma, beta = gb.chunk(2, dim=-1)
        z = z * (1 + gamma) + beta
        z_n = F.normalize(z, dim=-1)

        from train_whisper import GradReversalFn
        z_rev = GradReversalFn.apply(z_n, lam)
        domain_logits  = self.domain_head(z_rev)
        disease_logits = self.disease_head(z_n)
        return disease_logits, domain_logits, z_n


# ─── Loss / Eval ─────────────────────────────────────────────────────────────

def get_lambda(epoch, total_epochs, gamma=10.0):
    p = epoch / total_epochs
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def focal_loss(logits, labels, gamma=2.0, eps=0.1):
    n = logits.shape[-1]
    smooth = torch.full_like(logits, eps / (n - 1))
    smooth.scatter_(1, labels.unsqueeze(1), 1.0 - eps)
    log_p  = F.log_softmax(logits, dim=-1)
    focal  = (1 - log_p.exp()) ** gamma
    return -(focal * smooth * log_p).sum(dim=-1).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for mel, sym, _, labels in loader:
        mel    = mel.to(device)
        sym    = sym.to(device)
        logits, _, _ = model(mel, sym, lam=0.0)
        probs  = F.softmax(logits, dim=-1)
        preds  = probs.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_true == i
        if mask.sum() > 0:
            per_class[cls] = float((y_pred[mask] == i).mean())
    return bal_acc, f1, auc, per_class


def train_one_epoch(model, loader, opt_head, opt_enc, device,
                    epoch, total_epochs, frozen):
    model.train()
    lam = get_lambda(epoch, total_epochs)
    total_loss = 0.0
    for mel, sym, domain, labels in tqdm(loader, desc=f'Ep {epoch}', leave=False):
        mel    = mel.to(device)
        sym    = sym.to(device)
        domain = domain.to(device)
        labels = labels.to(device)
        disease_logits, domain_logits, _ = model(mel, sym, lam=lam)
        loss_cls    = focal_loss(disease_logits, labels)
        loss_domain = F.cross_entropy(domain_logits, domain)
        loss        = loss_cls + 0.3 * lam * loss_domain
        opt_head.zero_grad()
        if opt_enc is not None:
            opt_enc.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_head.step()
        if opt_enc is not None:
            opt_enc.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',        default='ml_service/v6_data.csv')
    p.add_argument('--output_dir', default='checkpoints_opera')
    p.add_argument('--model',      default='opera',
                   choices=['opera', 'effnet'],
                   help='opera = OPERA-CT (ViT), effnet = EfficientNet-B2 baseline')
    p.add_argument('--checkpoint', default='',
                   help='Path to OPERA-CT pretrained checkpoint (.pth)')
    p.add_argument('--epochs',     type=int,   default=40)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=HEAD_LR)
    p.add_argument('--enc_lr',     type=float, default=ENC_LR)
    p.add_argument('--folds',      type=int,   default=5)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--freeze_epochs', type=int, default=FREEZE_EPOCHS)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available()
              else 'cpu')
    print(f"Model: {args.model}  Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    all_samples = load_csv(args.csv)
    counts = Counter(s['label'] for s in all_samples)
    print(f"Loaded {len(all_samples)} samples")
    for cls in CLASS_NAMES:
        print(f"  {cls}: {counts.get(cls, 0)}")

    labels_arr = np.array([CLASS_NAMES.index(s['label']) for s in all_samples])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    log_path = out_dir / f'training_log_{args.model}.txt'
    log = open(log_path, 'w')

    for fold, (tr_idx, val_idx) in enumerate(skf.split(all_samples, labels_arr)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.folds}  ({args.model.upper()})")
        tr_samples  = [all_samples[i] for i in tr_idx]
        val_samples = [all_samples[i] for i in val_idx]
        print(f"  Train: {len(tr_samples)}  Val: {len(val_samples)}")

        model_type = 'opera' if args.model == 'opera' else 'effnet'
        tr_ds  = CoughDataset(tr_samples,  model_type, training=True)
        val_ds = CoughDataset(val_samples, model_type, training=False)
        sampler    = make_weighted_sampler(tr_samples)
        tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, sampler=sampler,
                                num_workers=2, pin_memory=(device != 'cpu'),
                                drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, pin_memory=(device != 'cpu'))

        if args.model == 'opera':
            model = OperaClassifier(
                checkpoint_path=args.checkpoint,
                n_classes=N_CLASSES).to(device)
            enc_params = list(model.encoder.parameters())
            head_params = (list(model.film.parameters()) +
                           list(model.domain_head.parameters()) +
                           list(model.disease_head.parameters()))
        else:
            model = EffNetClassifier(n_classes=N_CLASSES).to(device)
            enc_params = list(model.backbone.parameters())
            head_params = (list(model.film.parameters()) +
                           list(model.domain_head.parameters()) +
                           list(model.disease_head.parameters()))

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {total_params/1e6:.1f}M")

        opt_head = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
        opt_enc  = torch.optim.AdamW(enc_params, lr=args.enc_lr, weight_decay=1e-4)
        sched_head = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_head, T_0=args.epochs)
        sched_enc  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_enc, T_0=max(args.epochs - args.freeze_epochs, 1))

        best_bal_acc = best_f1 = best_auc = 0.0
        best_per_class = {}
        best_path = out_dir / f'fold{fold+1}_{args.model}_best.pt'

        for epoch in range(1, args.epochs + 1):
            frozen = (epoch <= args.freeze_epochs)
            if epoch == 1:
                model.freeze_encoder()
                print(f"  Phase 1: encoder FROZEN for {args.freeze_epochs} epochs")
            if epoch == args.freeze_epochs + 1:
                model.unfreeze_encoder()
                print(f"  Phase 2: encoder UNFROZEN")

            loss = train_one_epoch(model, tr_loader, opt_head,
                                   None if frozen else opt_enc,
                                   device, epoch, args.epochs, frozen)
            sched_head.step()
            if not frozen:
                sched_enc.step()

            bal_acc, f1, auc, per_class = evaluate(model, val_loader, device)
            line = (f"Fold {fold+1} Ep {epoch:3d}/{args.epochs} | "
                    f"loss={loss:.3f} bal_acc={bal_acc:.4f} f1={f1:.4f} auc={auc:.4f} | "
                    + '  '.join(f"{c[:6]}={per_class.get(c,0):.3f}" for c in CLASS_NAMES)
                    + (f"  [frozen]" if frozen else ""))
            print(line); log.write(line + '\n'); log.flush()

            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_f1 = f1; best_auc = auc; best_per_class = per_class
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Best bal_acc={best_bal_acc:.4f}")

        fold_results.append({'fold': fold+1, 'bal_acc': best_bal_acc,
                             'f1': best_f1, 'auc': best_auc,
                             'per_class': best_per_class})
        print(f"Fold {fold+1} best → bal_acc={best_bal_acc:.4f} f1={best_f1:.4f}")

    bal_accs = [r['bal_acc'] for r in fold_results]
    f1s      = [r['f1']      for r in fold_results]
    aucs     = [r['auc']     for r in fold_results]
    summary  = {
        'model': args.model,
        'mean_bal_acc': float(np.mean(bal_accs)),
        'std_bal_acc':  float(np.std(bal_accs)),
        'mean_f1':      float(np.mean(f1s)),
        'std_f1':       float(np.std(f1s)),
        'mean_auc':     float(np.mean(aucs)),
        'std_auc':      float(np.std(aucs)),
        'fold_results': fold_results,
        'class_names':  CLASS_NAMES,
        'config':       vars(args),
    }
    fin = (f"\nFINAL ({args.model}): "
           f"bal_acc={summary['mean_bal_acc']*100:.1f}% "
           f"±{summary['std_bal_acc']*100:.1f}%  "
           f"F1={summary['mean_f1']:.3f}  AUC={summary['mean_auc']:.3f}")
    print(fin); log.write(fin + '\n'); log.close()
    with open(out_dir / f'cv_summary_{args.model}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_dir}/cv_summary_{args.model}.json")


if __name__ == '__main__':
    main()
