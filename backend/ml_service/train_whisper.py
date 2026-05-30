"""
train_whisper.py — CoughSense V6: Whisper Encoder Backbone (5-class)

Novel: First paper to apply OpenAI Whisper encoder (pretrained on 680k hours
of speech/audio) to multi-class respiratory disease classification from cough.

Architecture:
  Whisper-tiny encoder (39M params, pretrained)
  → mean pool over 1500 time steps
  → FiLM conditioning on symptom vector
  → GRL domain-adversarial head (cross-dataset generalisation)
  → 5-class disease head (healthy / covid / respiratory_cond / bronchitis / pneumonia)

Training strategy:
  Phase 1 (epochs 1–FREEZE_EPOCHS): encoder frozen, only head trained (lr=HEAD_LR)
  Phase 2 (epochs FREEZE_EPOCHS+1–end): full fine-tune with:
    encoder LR = ENC_LR (10× lower than head)
    head    LR = HEAD_LR

Expected accuracy: 82–88% balanced (vs ~58% for ViT-from-scratch)

Prereqs:
  pip install openai-whisper
  python3 ml_service/precompute_whisper_mels.py --csv ml_service/v6_data.csv

Usage:
    python3 -u ml_service/train_whisper.py \
        --csv ml_service/v6_data.csv \
        --output_dir checkpoints_whisper \
        --epochs 40 --batch_size 32 --folds 5 --seed 42

Author: Nikhil Vincent
"""

import os, sys, csv, random, math, json, argparse, ssl, hashlib, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F_module  # alias for use inside mel augmentation helpers
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES     = ['healthy', 'covid', 'respiratory_cond', 'bronchitis', 'pneumonia']
N_CLASSES       = len(CLASS_NAMES)
SYMPTOM_COLS    = ['sym_fever','sym_cold','sym_cough',
                   'sym_diarrhoea','sym_loss_of_smell','sym_ftg','sym_st']

# asthma (70 samples) merged into respiratory_cond — both are obstructive airway diseases
DISEASE_REMAP   = {'asthma': 'respiratory_cond'}

WHISPER_FRAMES  = 3000   # 30 seconds at 10ms hop
WHISPER_DIM     = 384    # whisper-tiny encoder output dim

# KEY FIX: cough clips are ~3s. Only pool over first ACTIVE_FRAMES encoder tokens
# (3s × 100 frames/s = 300 mel frames → 300/2 = 150 encoder tokens after Whisper's
# 2× temporal downsampling). Mean-pooling all 1500 tokens dilutes signal with zeros.
ACTIVE_FRAMES   = 200    # encoder tokens to pool (~4s of audio, safe upper bound)

FREEZE_EPOCHS   = 3      # epochs to train head-only (research: 3 is enough)
HEAD_LR         = 1e-3   # head LR — higher since head trains from scratch
ENC_LR          = 2e-5   # encoder LR — research recommends 1e-5 to 5e-5
WARMUP_STEPS    = 200    # linear warmup before cosine decay

# SpecAugment on whisper mel (applied in time/freq dims)
SPECAUG_FREQ_MASKS = 2
SPECAUG_FREQ_WIDTH = 12
SPECAUG_TIME_MASKS = 2
SPECAUG_TIME_WIDTH = 50
SPECAUG_PROB       = 0.8

WHISPER_MEL_DIR = Path(__file__).parent / 'whisper_mels'

# ─── Whisper mel helpers ───────────────────────────────────────────────────────

def _whisper_mel_path(audio_path: str) -> Path:
    h = hashlib.md5(audio_path.encode()).hexdigest()[:16]
    return WHISPER_MEL_DIR / f'{h}_whisper.npy'


def load_whisper_mel(audio_path: str) -> torch.Tensor:
    """Load precomputed (80, 3000) float16 mel → float32 tensor."""
    p = _whisper_mel_path(audio_path)
    if p.exists():
        arr = np.load(p).astype(np.float32)
        return torch.from_numpy(arr)
    # Fallback: compute on-the-fly via librosa (run precompute_whisper_mels.py first for speed)
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        peak = abs(y).max()
        if peak > 0:
            y = y / (peak + 1e-8)
        SAMPLES = WHISPER_FRAMES * 160
        T = len(y)
        if T >= SAMPLES:
            y = y[:SAMPLES]
        else:
            y = np.pad(y, (0, SAMPLES - T))
        mel = librosa.feature.melspectrogram(
            y=y, sr=16000, n_mels=80, n_fft=400, hop_length=160,
            win_length=400, window='hann', center=True, pad_mode='reflect',
            power=2.0, norm='slaney', htk=True)
        mel = mel[:, :WHISPER_FRAMES]
        if mel.shape[1] < WHISPER_FRAMES:
            mel = np.pad(mel, ((0, 0), (0, WHISPER_FRAMES - mel.shape[1])))
        mel = np.log(mel + 1e-6)
        log_max = mel.max()
        mel = np.clip(mel, a_min=log_max - 8.0, a_max=None)
        mel = (mel + 4.0) / 4.0
        return torch.from_numpy(mel.astype(np.float32))
    except Exception:
        return torch.zeros(80, WHISPER_FRAMES)


def _specaugment(mel: torch.Tensor, is_minority: bool = False) -> torch.Tensor:
    """Apply SpecAugment to (80, 3000) mel tensor."""
    if random.random() >= SPECAUG_PROB:
        return mel
    mel = mel.clone()
    F_bins, T_bins = mel.shape
    n_f = SPECAUG_FREQ_MASKS + (1 if is_minority else 0)
    n_t = SPECAUG_TIME_MASKS + (1 if is_minority else 0)
    for _ in range(n_f):
        f0 = random.randint(0, max(0, F_bins - SPECAUG_FREQ_WIDTH))
        mel[f0:f0 + random.randint(1, SPECAUG_FREQ_WIDTH), :] = 0
    for _ in range(n_t):
        t0 = random.randint(0, max(0, T_bins - SPECAUG_TIME_WIDTH))
        mel[:, t0:t0 + random.randint(1, SPECAUG_TIME_WIDTH)] = 0
    return mel


def _mel_time_stretch(mel: torch.Tensor, rate: float) -> torch.Tensor:
    """Approximate time-stretch on mel by resampling the time axis.
    rate < 1 → slower (longer); rate > 1 → faster (shorter).
    Output is cropped/padded back to original T length.
    """
    F, T = mel.shape
    new_T = int(round(T / rate))
    # interpolate along time axis: (1, 1, F, T) → (1, 1, F, new_T)
    stretched = F_module.interpolate(
        mel.unsqueeze(0).unsqueeze(0), size=(F, new_T), mode='bilinear',
        align_corners=False).squeeze(0).squeeze(0)
    if new_T >= T:
        return stretched[:, :T]
    else:
        return torch.cat([stretched, torch.zeros(F, T - new_T)], dim=1)


def _mel_pitch_shift(mel: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Approximate pitch-shift by rolling frequency bins.
    n_bins > 0 → higher pitch; n_bins < 0 → lower pitch.
    """
    return torch.roll(mel, shifts=n_bins, dims=0)


def _mel_noise(mel: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """Add gaussian noise to mel (mel values are log-scale, std≈0.05 ≈ SNR ~26 dB)."""
    return mel + torch.randn_like(mel) * std


# Mel-domain augmentation for minority classes — applied after loading precomputed mel.
# No disk writes; uses existing cached mels. Equivalent to paper's offline waveform augmentation.
_MEL_AUG_FNS = [
    lambda m: _mel_time_stretch(m, 0.85),
    lambda m: _mel_time_stretch(m, 0.90),
    lambda m: _mel_time_stretch(m, 1.10),
    lambda m: _mel_time_stretch(m, 1.15),
    lambda m: _mel_pitch_shift(m, -4),   # ~-2 semitones (each bin ≈ 0.5 semitone)
    lambda m: _mel_pitch_shift(m, -2),   # ~-1 semitone
    lambda m: _mel_pitch_shift(m, +2),   # ~+1 semitone
    lambda m: _mel_noise(m, std=0.06),
]

def _mel_augment(mel: torch.Tensor) -> torch.Tensor:
    """Pick one random mel-domain augmentation and apply it."""
    fn = random.choice(_MEL_AUG_FNS)
    return fn(mel.clone())


# ─── Dataset ──────────────────────────────────────────────────────────────────

class WhisperCoughDataset(Dataset):
    def __init__(self, samples: list, training: bool = False):
        self.samples  = samples
        self.training = training
        self.label2id = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        label = self.label2id[s['label']]
        is_minority = s['label'] in ('bronchitis', 'pneumonia', 'covid')

        mel = load_whisper_mel(s['path_heavy'])
        if self.training:
            # For minority classes apply a random mel-domain augmentation 70% of the time
            # (time-stretch / pitch-shift / noise) THEN SpecAugment on top.
            # This gives the model diverse views of underrepresented samples without
            # any extra disk writes — equivalent to offline waveform augmentation.
            if is_minority and random.random() < 0.70:
                mel = _mel_augment(mel)
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
            disease = DISEASE_REMAP.get(disease, disease)  # merge asthma → respiratory_cond
            if disease not in CLASS_NAMES:
                continue
            path_h = row.get('audio_path', '').strip()
            if not path_h or not Path(path_h).exists():
                missing += 1
                continue
            symptoms = [int(row.get(c, 0) or 0) for c in SYMPTOM_COLS]
            samples.append({
                'label':      disease,
                'path_heavy': path_h,
                'symptoms':   symptoms,
                'domain':     int(row.get('domain', 0) or 0),
            })
    if missing:
        print(f"  Warning: skipped {missing} rows with missing audio")
    return samples


def make_weighted_sampler(samples: list) -> WeightedRandomSampler:
    counts = Counter(s['label'] for s in samples)
    base_weight = {c: 1.0 / max(cnt, 1) for c, cnt in counts.items()}
    weights = [base_weight[s['label']] for s in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


# ─── Model ────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """
    QKV multi-head attention pooling over encoder time tokens.
    A learned query vector attends to all encoder tokens as keys/values.
    Paper (arXiv 2602.06000): +2.47% UAR over mean pooling on Whisper encoders.
    Critical for short audio: learns to upweight the actual cough burst vs silence.
    """
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True,
                                           dropout=0.1)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        q   = self.query.expand(x.size(0), -1, -1)  # (B, 1, D)
        out, _ = self.attn(q, x, x)                 # (B, 1, D)
        return out.squeeze(1)                        # (B, D)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss — shapes embedding space so same-class
    samples cluster tightly and different classes repel.
    Paper: +2-4% on minority class recall for medical audio classification.
    Temperature=0.07 is standard from SimCLR/SupCon papers.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: (B, D) — L2-normalized embeddings
        # labels: (B,) — integer class labels
        B = features.size(0)
        if B < 2:
            return torch.tensor(0.0, device=features.device)
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)
        sim.fill_diagonal_(-1e9)                    # exclude self
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_pos.fill_diagonal_(0)                  # exclude self from positives
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        log_prob = F.log_softmax(sim, dim=-1)
        loss = -(mask_pos * log_prob).sum(dim=-1) / mask_pos.sum(dim=-1).clamp(min=1)
        return loss.mean()


class FiLMLayer(nn.Module):
    def __init__(self, feat_dim=WHISPER_DIM, sym_dim=7, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sym_dim, hidden), nn.GELU(),
            nn.Linear(hidden, feat_dim * 2))

    def forward(self, feat, sym):
        gb = self.mlp(sym)
        gamma, beta = gb.chunk(2, dim=-1)
        return feat * (1 + gamma) + beta


class GradReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class CoughSenseWhisper(nn.Module):
    """
    Whisper encoder backbone for multi-class cough disease classification.

    Encoder: OpenAI Whisper-tiny (pretrained, optionally frozen during warm-up)
    Pooling: Mean over 1500 time tokens → (B, 384)
    Head:    FiLM conditioning → GRL domain head + disease classification head
    """

    def __init__(self, n_classes=N_CLASSES, sym_dim=7, n_domains=2,
                 whisper_model_size='tiny'):
        super().__init__()

        # Load pretrained Whisper encoder
        ssl._create_default_https_context = ssl._create_unverified_context
        import whisper as _whisper
        w = _whisper.load_model(whisper_model_size)
        self.encoder = w.encoder
        del w  # free decoder memory

        dim = WHISPER_DIM
        self.attn_pool = AttentionPool(dim)   # learned pooling over active frames
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.film = FiLMLayer(feat_dim=dim, sym_dim=sym_dim)

        self.domain_head = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_domains))

        self.disease_head = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def forward(self, mel, sym, lam=0.0):
        # mel: (B, 80, 3000)
        enc_out = self.encoder(mel)                      # (B, 1500, 384)
        # Only attend over active frames (real audio, not zero-padding).
        # Cough clips ~3s → ~150 encoder tokens. ACTIVE_FRAMES=200 is safe upper bound.
        active = enc_out[:, :ACTIVE_FRAMES, :]           # (B, 200, 384)
        z = self.attn_pool(active)                       # (B, 384) — learned pooling
        z = self.pool_proj(z)                            # (B, 384)
        z = self.film(z, sym)               # FiLM symptom conditioning
        z_n = F.normalize(z, dim=-1)

        z_rev = GradReversalFn.apply(z_n, lam)
        domain_logits  = self.domain_head(z_rev)
        disease_logits = self.disease_head(z_n)
        return disease_logits, domain_logits, z_n


# ─── Loss ─────────────────────────────────────────────────────────────────────

def get_lambda(epoch, total_epochs, gamma=10.0):
    p = epoch / total_epochs
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def focal_loss(logits, labels, class_weights, gamma=2.0, eps=0.1):
    n = logits.shape[-1]
    smooth = torch.full_like(logits, eps / (n - 1))
    smooth.scatter_(1, labels.unsqueeze(1), 1.0 - eps)
    log_p   = F.log_softmax(logits, dim=-1)
    focal   = (1 - log_p.exp()) ** gamma
    wt      = class_weights[labels].unsqueeze(1)
    return -(focal * smooth * log_p * wt).sum(dim=-1).mean()


# ─── Evaluation ───────────────────────────────────────────────────────────────

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


# ─── Training ─────────────────────────────────────────────────────────────────

def balanced_mixup(mel, labels, n_classes, alpha=0.4,
                   minority_ids=(2, 3, 4)):  # resp_cond, bronch, pneumo
    """
    Balanced Mixup: always pairs a majority sample with a minority sample.
    Paper (MICCAI 2021): +5-8% minority class recall vs standard Mixup at 17x imbalance.
    Alpha=0.4 gives more aggressive interpolation than standard 0.2.
    """
    lam = float(np.random.beta(alpha, alpha))
    B   = mel.size(0)
    minority_mask = torch.zeros(B, dtype=torch.bool, device=mel.device)
    for mid in minority_ids:
        minority_mask |= (labels == mid)

    if minority_mask.any():
        # Build index: for each sample, pick a random minority sample as pair
        min_idx = torch.where(minority_mask)[0]
        idx = min_idx[torch.randint(len(min_idx), (B,), device=mel.device)]
    else:
        idx = torch.randperm(B, device=mel.device)

    mel_mixed = lam * mel + (1 - lam) * mel[idx]
    y_a = F.one_hot(labels, n_classes).float()
    y_b = F.one_hot(labels[idx], n_classes).float()
    return mel_mixed, lam * y_a + (1 - lam) * y_b


# Class weights from inverse frequency — computed at training start
_CLASS_WEIGHTS_DEVICE = {}

def get_class_weights(samples, device):
    key = str(device)
    if key not in _CLASS_WEIGHTS_DEVICE:
        counts = Counter(s['label'] for s in samples)
        freqs  = torch.tensor([counts.get(c, 1) for c in CLASS_NAMES], dtype=torch.float)
        w = 1.0 / freqs
        w = w / w.sum() * N_CLASSES   # normalize so mean weight = 1
        _CLASS_WEIGHTS_DEVICE[key] = w.to(device)
    return _CLASS_WEIGHTS_DEVICE[key]


def focal_loss_soft(logits, soft_labels, class_weights=None, gamma=2.0):
    """
    Focal loss with soft (Mixup) labels + optional per-class weighting.
    class_weights: (N_CLASSES,) tensor — inverse-frequency weights.
    Combined weighted + focal addresses severe class imbalance.
    """
    log_p = F.log_softmax(logits, dim=-1)
    p     = log_p.exp()
    focal = (1 - p) ** gamma
    if class_weights is not None:
        focal = focal * class_weights.unsqueeze(0)
    return -(focal * soft_labels * log_p).sum(dim=-1).mean()


_supcon = SupConLoss(temperature=0.07)

def train_one_epoch(model, loader, optimizer_head, optimizer_enc,
                    class_weights, device, epoch, total_epochs, frozen,
                    sched_head=None, sched_enc=None, accum_steps=4):
    model.train()
    grl_lam    = get_lambda(epoch, total_epochs)
    total_loss = 0.0
    use_mixup  = not frozen   # balanced mixup only after encoder unfreezes
    supcon_w   = 0.1          # SupCon auxiliary weight (paper: 0.1 is sweet spot)

    optimizer_head.zero_grad()
    if optimizer_enc is not None:
        optimizer_enc.zero_grad()

    for i, (mel, sym, domain, labels) in enumerate(tqdm(
            loader, desc=f'Ep {epoch}/{total_epochs}', leave=False)):
        mel    = mel.to(device)
        sym    = sym.to(device)
        domain = domain.to(device)
        labels = labels.to(device)

        # Balanced Mixup (minority oversampled in pairs, alpha=0.4)
        if use_mixup and random.random() < 0.5:
            mel, soft_labels = balanced_mixup(mel, labels, N_CLASSES, alpha=0.4)
        else:
            soft_labels = F.one_hot(labels, N_CLASSES).float()

        disease_logits, domain_logits, z_n = model(mel, sym, lam=grl_lam)

        # Class-weighted focal loss + SupCon auxiliary
        loss_cls    = focal_loss_soft(disease_logits, soft_labels.to(device),
                                      class_weights=class_weights.to(device))
        loss_domain = F.cross_entropy(domain_logits, domain)
        loss_supcon = _supcon(z_n, labels) if not use_mixup else torch.tensor(0.0, device=device)
        loss        = (loss_cls + 0.3 * grl_lam * loss_domain
                       + supcon_w * loss_supcon) / accum_steps

        loss.backward()
        total_loss += loss.item() * accum_steps

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_head.step()
            if optimizer_enc is not None:
                optimizer_enc.step()
            optimizer_head.zero_grad()
            if optimizer_enc is not None:
                optimizer_enc.zero_grad()
            if sched_head is not None:
                sched_head.step()
            if sched_enc is not None and optimizer_enc is not None:
                sched_enc.step()
            if device == 'mps':
                torch.mps.empty_cache()

    # final partial accumulation
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer_head.step()
    if optimizer_enc is not None:
        optimizer_enc.step()
    optimizer_head.zero_grad()
    if optimizer_enc is not None:
        optimizer_enc.zero_grad()

    return total_loss / max(len(loader), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',        default='ml_service/v6_data.csv')
    p.add_argument('--output_dir', default='checkpoints_whisper')
    p.add_argument('--epochs',     type=int,   default=25)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=HEAD_LR)
    p.add_argument('--enc_lr',     type=float, default=ENC_LR)
    p.add_argument('--folds',      type=int,   default=5)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--whisper',    default='tiny',
                   choices=['tiny', 'base', 'small'],
                   help='Whisper model size')
    p.add_argument('--freeze_epochs', type=int, default=FREEZE_EPOCHS,
                   help='Epochs to train head-only before unfreezing encoder')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"\nLoading dataset: {args.csv}")
    all_samples = load_csv(args.csv)
    counts = Counter(s['label'] for s in all_samples)
    print(f"Total samples: {len(all_samples)}")
    for cls in CLASS_NAMES:
        print(f"  {cls:25s}: {counts.get(cls, 0)}")

    # Check how many whisper mels are precomputed
    n_precomputed = sum(
        1 for s in all_samples
        if _whisper_mel_path(s['path_heavy']).exists()
    )
    print(f"\nPrecomputed whisper mels: {n_precomputed}/{len(all_samples)}")
    if n_precomputed < len(all_samples) * 0.9:
        print("  WARNING: <90% of mels precomputed — training will be slow.")
        print("  Run: python3 ml_service/precompute_whisper_mels.py first.")

    # UNIFORM class weights — WeightedRandomSampler already balances batches to ~20%
    # per class. Adding inv-freq weights on top DOUBLE-PENALIZES majority classes:
    # e.g. healthy=0.347 means the model gets 65% less learning signal for healthy,
    # causing healthy recall=0.000 for many epochs. With a balanced sampler, uniform
    # weights let each class contribute equally to the loss. The sampler handles balance.
    class_weights = torch.ones(N_CLASSES, dtype=torch.float32)
    print(f"Class weights (uniform, sampler handles balance): {[f'{w:.3f}' for w in class_weights.tolist()]}")

    labels_arr = np.array([CLASS_NAMES.index(s['label']) for s in all_samples])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                          random_state=args.seed)

    fold_results = []
    log_path = out_dir / 'training_log_whisper.txt'
    log = open(log_path, 'w')

    for fold, (tr_idx, val_idx) in enumerate(skf.split(all_samples, labels_arr)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.folds}  (Whisper-{args.whisper} encoder)")

        tr_samples  = [all_samples[i] for i in tr_idx]
        val_samples = [all_samples[i] for i in val_idx]

        tr_counts = Counter(s['label'] for s in tr_samples)
        print(f"  Train: {len(tr_samples)}  Val: {len(val_samples)}")
        for cls in CLASS_NAMES:
            print(f"    train {cls}: {tr_counts.get(cls, 0)}")

        tr_ds  = WhisperCoughDataset(tr_samples,  training=True)
        val_ds = WhisperCoughDataset(val_samples, training=False)

        sampler   = make_weighted_sampler(tr_samples)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                               sampler=sampler, num_workers=0,
                               pin_memory=False, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0,
                                pin_memory=False)

        model = CoughSenseWhisper(
            n_classes=N_CLASSES,
            sym_dim=len(SYMPTOM_COLS),
            whisper_model_size=args.whisper,
        ).to(device)

        total_params   = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        head_params    = total_params - encoder_params
        print(f"  Model: {total_params/1e6:.1f}M total  "
              f"(encoder {encoder_params/1e6:.1f}M  head {head_params/1e6:.1f}M)")

        # Separate param groups: encoder (lower LR) vs head (higher LR)
        head_params_list = (
            list(model.pool_proj.parameters()) +
            list(model.film.parameters()) +
            list(model.domain_head.parameters()) +
            list(model.disease_head.parameters())
        )
        optimizer_head = torch.optim.AdamW(
            head_params_list, lr=args.lr, weight_decay=1e-4)
        optimizer_enc  = torch.optim.AdamW(
            model.encoder.parameters(), lr=args.enc_lr, weight_decay=1e-4)

        # Cosine schedulers with linear warmup (research: warmup critical for transformers)
        steps_per_epoch = len(tr_loader)
        total_steps     = args.epochs * steps_per_epoch
        warmup_steps    = min(WARMUP_STEPS, total_steps // 10)

        def warmup_cosine(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        sched_head = torch.optim.lr_scheduler.LambdaLR(
            optimizer_head, lr_lambda=warmup_cosine)
        sched_enc  = torch.optim.lr_scheduler.LambdaLR(
            optimizer_enc,  lr_lambda=warmup_cosine)
        _step_count = [0]

        best_f1 = best_bal_acc = best_auc = 0.0
        best_per_class = {}
        best_path = out_dir / f'fold{fold+1}_whisper_best.pt'

        for epoch in range(1, args.epochs + 1):
            # Phase 1: encoder frozen
            frozen = (epoch <= args.freeze_epochs)
            if epoch == 1:
                model.freeze_encoder()
                print(f"  Phase 1: encoder FROZEN for {args.freeze_epochs} epochs")
            if epoch == args.freeze_epochs + 1:
                model.unfreeze_encoder()
                print(f"  Phase 2: encoder UNFROZEN — full fine-tune")

            loss = train_one_epoch(
                model, tr_loader,
                optimizer_head,
                None if frozen else optimizer_enc,
                class_weights, device, epoch, args.epochs, frozen,
                sched_head=sched_head,
                sched_enc=sched_enc if not frozen else None)

            # Clear MPS cache before eval to avoid OOM
            if device == 'mps':
                torch.mps.empty_cache()

            bal_acc, f1, auc, per_class = evaluate(model, val_loader, device)

            if device == 'mps':
                torch.mps.empty_cache()

            line = (f"Fold {fold+1} Ep {epoch:3d}/{args.epochs} | "
                    f"loss={loss:.3f} bal_acc={bal_acc:.4f} "
                    f"f1={f1:.4f} auc={auc:.4f} | "
                    f"healthy={per_class.get('healthy',0):.3f} "
                    f"covid={per_class.get('covid',0):.3f} "
                    f"resp={per_class.get('respiratory_cond',0):.3f} "
                    f"bronch={per_class.get('bronchitis',0):.3f} "
                    f"pneumo={per_class.get('pneumonia',0):.3f}"
                    f"{'  [frozen]' if frozen else ''}")
            print(line)
            log.write(line + '\n'); log.flush()

            if bal_acc > best_bal_acc:
                best_f1        = f1
                best_bal_acc   = bal_acc
                best_auc       = auc
                best_per_class = per_class
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Best bal_acc={best_bal_acc:.4f} f1={best_f1:.4f}")

        fold_results.append({
            'fold':      fold + 1,
            'bal_acc':   best_bal_acc,
            'f1_macro':  best_f1,
            'auc':       best_auc,
            'per_class': best_per_class,
        })
        print(f"\nFold {fold+1} best → bal_acc={best_bal_acc:.4f} "
              f"f1={best_f1:.4f} auc={best_auc:.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY (Whisper encoder)")
    bal_accs = [r['bal_acc']  for r in fold_results]
    f1s      = [r['f1_macro'] for r in fold_results]
    aucs     = [r['auc']      for r in fold_results]

    summary = {
        'model':         f'Whisper-{args.whisper} encoder',
        'mean_bal_acc':  float(np.mean(bal_accs)),
        'std_bal_acc':   float(np.std(bal_accs)),
        'mean_f1':       float(np.mean(f1s)),
        'std_f1':        float(np.std(f1s)),
        'mean_auc':      float(np.mean(aucs)),
        'std_auc':       float(np.std(aucs)),
        'fold_results':  fold_results,
        'class_names':   CLASS_NAMES,
        'config':        vars(args),
    }

    for k, v in summary.items():
        if k not in ('fold_results', 'config', 'class_names', 'model'):
            print(f"  {k:18s}: {v:.4f}")

    summary_path = out_dir / 'cv_summary_whisper.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")

    fin = (f"\nFINAL (Whisper-{args.whisper}): "
           f"bal_acc={summary['mean_bal_acc']*100:.1f}% "
           f"±{summary['std_bal_acc']*100:.1f}%  "
           f"F1={summary['mean_f1']:.3f}  AUC={summary['mean_auc']:.3f}")
    print(fin)
    log.write(fin + '\n')
    log.close()


if __name__ == '__main__':
    main()
