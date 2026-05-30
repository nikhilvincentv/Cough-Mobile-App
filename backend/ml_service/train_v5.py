"""
train_v5.py — CoughSense V5: 4-Class Multi-Disease Respiratory Classifier

Classes:
  0  healthy           — Coswara healthy + CoughVID healthy
  1  covid             — Coswara PCR-COVID + CoughVID COVID
  2  asthma            — Coswara asthma patients (heavy aug, 70 samples)
  3  respiratory_cond  — CoughVID symptomatic

Key innovations over V4:
  1. 4-class multi-disease (first unified COVID/asthma/resp-cond/healthy from cough)
  2. Cross-dataset fusion: Coswara + CoughVID with domain adversarial training
  3. webm audio decoding via ffmpeg
  4. Extreme minority class handling: CBS-weighted focal + 10x asthma oversampling
  5. Per-class temperature scaling calibration extended to 4 classes

Usage:
    python3 -u ml_service/train_v5.py \
        --csv ml_service/v5_data.csv \
        --output_dir checkpoints_v5 \
        --epochs 80 --batch_size 16 --folds 5 --seed 42

Author: Nikhil Vincent
"""

import os, sys, csv, random, math, json, argparse, subprocess, tempfile
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# ─── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
CLIP_SAMPLES = int(SAMPLE_RATE * 3.5)
N_MELS       = 64
HOP          = 160
N_FFT        = 512
CLASS_NAMES  = ['healthy', 'covid', 'asthma', 'respiratory_cond']
N_CLASSES    = len(CLASS_NAMES)
SYMPTOM_COLS = ['sym_fever','sym_cold','sym_cough',
                'sym_diarrhoea','sym_loss_of_smell','sym_ftg','sym_st']

# Augmentation
SPECAUG_FREQ_MASKS = 2
SPECAUG_FREQ_WIDTH = 12
SPECAUG_TIME_MASKS = 2
SPECAUG_TIME_WIDTH = 20
SPECAUG_PROB       = 0.8

# Asthma oversampling multiplier — 1 = perfectly balanced (25% each class)
# Setting >1 risks drowning out healthy; 1 is the right value here.
ASTHMA_OVERSAMPLE = 1

# ─── Audio helpers ────────────────────────────────────────────────────────────

def _ffmpeg_to_wav(path: str) -> str:
    """Convert any audio format to a temp wav via ffmpeg. Returns temp path."""
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-ar', str(SAMPLE_RATE),
             '-ac', '1', '-f', 'wav', tmp.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=15, check=True)
        return tmp.name
    except Exception:
        os.unlink(tmp.name)
        return ''


def _load_with_soundfile(path: str):
    """Load audio using soundfile, returns (samples_np, sr)."""
    import soundfile as sf
    data, sr = sf.read(path, always_2d=True)
    # data shape: (T, channels) → convert to (channels, T)
    data = data.T
    return data, sr


def load_audio(path: str) -> torch.Tensor:
    """Load audio → (1, CLIP_SAMPLES) normalised tensor. Handles wav/ogg/webm."""
    if not path or not Path(path).exists():
        return torch.zeros(1, CLIP_SAMPLES)

    tmp_path = ''
    try:
        # Try soundfile first (fast, no torchcodec dependency)
        try:
            data, sr = _load_with_soundfile(path)
            w = torch.from_numpy(data).float()
        except Exception:
            # Fall back to ffmpeg for webm / unsupported formats
            tmp_path = _ffmpeg_to_wav(path)
            if not tmp_path:
                return torch.zeros(1, CLIP_SAMPLES)
            data, sr = _load_with_soundfile(tmp_path)
            w = torch.from_numpy(data).float()

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
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


MEL_H = 64   # target height (= N_MELS)
MEL_W = 128  # target width for ViT patch grid

def wav_to_mel(w: torch.Tensor) -> torch.Tensor:
    """Waveform (1,T) → log-mel (1, MEL_H, MEL_W) normalised tensor.
    SpecAugment is applied separately via _specaugment() during training."""
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP, n_fft=N_FFT)
    spec = mel_fn(w)
    spec = (spec + 1e-6).log()
    mu, sigma = spec.mean(), spec.std() + 1e-8
    spec = (spec - mu) / sigma
    # Resize to fixed (MEL_H × MEL_W) for ViT patch grid
    spec = F.interpolate(spec.unsqueeze(0), size=(MEL_H, MEL_W),
                         mode='bilinear', align_corners=False).squeeze(0)
    return spec


def augment_waveform(w: torch.Tensor) -> torch.Tensor:
    """Extra waveform augmentations for minority classes."""
    aug = random.choice(['noise', 'shift', 'speed', 'none'])
    if aug == 'noise':
        snr_db = random.uniform(10, 30)
        sig_p  = w.pow(2).mean()
        nse_p  = sig_p / (10 ** (snr_db / 10))
        w = w + torch.randn_like(w) * nse_p.sqrt()
    elif aug == 'shift':
        shift = random.randint(-int(CLIP_SAMPLES * 0.15),
                                int(CLIP_SAMPLES * 0.15))
        w = torch.roll(w, shift, dims=-1)
    elif aug == 'speed':
        factor = random.uniform(0.85, 1.15)
        new_len = int(CLIP_SAMPLES * factor)
        if new_len > 0:
            w = torchaudio.functional.resample(w, CLIP_SAMPLES, new_len)
            T = w.shape[-1]
            if T >= CLIP_SAMPLES:
                w = w[..., :CLIP_SAMPLES]
            else:
                w = F.pad(w, (0, CLIP_SAMPLES - T))
    peak = w.abs().max()
    return w / (peak + 1e-8) if peak > 0 else w


# ─── Dataset ──────────────────────────────────────────────────────────────────

def _specaugment(mel: torch.Tensor, is_minority: bool = False) -> torch.Tensor:
    """Apply SpecAugment to a (1,H,W) mel tensor. Extra masks for minority."""
    if random.random() >= SPECAUG_PROB:
        return mel
    mel = mel.clone()
    _, F_bins, T_bins = mel.shape
    n_f = SPECAUG_FREQ_MASKS + (1 if is_minority else 0)
    n_t = SPECAUG_TIME_MASKS + (1 if is_minority else 0)
    for _ in range(n_f):
        f0 = random.randint(0, max(0, F_bins - SPECAUG_FREQ_WIDTH))
        mel[0, f0:f0 + random.randint(1, SPECAUG_FREQ_WIDTH), :] = 0
    for _ in range(n_t):
        t0 = random.randint(0, max(0, T_bins - SPECAUG_TIME_WIDTH))
        mel[0, :, t0:t0 + random.randint(1, SPECAUG_TIME_WIDTH)] = 0
    return mel


MEL_DIR = Path(__file__).parent / 'precomputed_mels'

def _mel_out_path(audio_path: str, suffix: str) -> Path:
    """Deterministic path used by both precompute_mels.py and this dataset."""
    h = abs(hash(audio_path)) % 10**9
    return MEL_DIR / f'{h}{suffix}.npy'

def _load_mel(path: str, suffix: str) -> torch.Tensor:
    """Load precomputed mel. Fall back to live compute if missing."""
    npy = _mel_out_path(path, suffix)
    if npy.exists():
        arr = np.load(npy).astype(np.float32)
        return torch.from_numpy(arr)
    # Fallback: live compute (slow — run precompute_mels.py first)
    return wav_to_mel(load_audio(path))


class CoughDatasetV5(Dataset):
    def __init__(self, samples: list, training: bool = False):
        self.samples  = samples
        self.training = training
        self.label2id = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        label       = self.label2id[s['label']]
        is_minority = (s['label'] == 'asthma')

        # Load precomputed mel (fast numpy read)
        mel_heavy = _load_mel(s['path_heavy'], '_heavy')

        # SpecAugment on mel (cheap, no audio I/O)
        if self.training:
            mel_heavy = _specaugment(mel_heavy, is_minority)

        # Shallow mel
        p_shallow = s.get('path_shallow', '')
        if p_shallow and Path(p_shallow).exists():
            mel_shallow = _load_mel(p_shallow, '_shallow')
            if self.training:
                mel_shallow = _specaugment(mel_shallow, is_minority)
            has_shallow = torch.tensor(1.0)
        else:
            mel_shallow = torch.zeros_like(mel_heavy)
            has_shallow = torch.tensor(0.0)

        sym    = torch.tensor(s.get('symptoms', [0]*7), dtype=torch.float32)
        domain = torch.tensor(s.get('domain', 0), dtype=torch.long)

        return mel_heavy, mel_shallow, has_shallow, sym, domain, label


def load_csv(csv_path: str) -> list:
    samples = []
    missing = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            disease = row.get('disease', '').strip().lower()
            if disease not in CLASS_NAMES:
                continue
            path_h = row.get('audio_path', '').strip()
            if not path_h or not Path(path_h).exists():
                missing += 1
                continue
            symptoms = [int(row.get(c, 0) or 0) for c in SYMPTOM_COLS]
            samples.append({
                'label':        disease,
                'path_heavy':   path_h,
                'path_shallow': row.get('audio_path_shallow', '').strip(),
                'symptoms':     symptoms,
                'age':          row.get('age', ''),
                'gender':       row.get('gender', ''),
                'domain':       int(row.get('domain', 0) or 0),
            })
    if missing:
        print(f"  Warning: skipped {missing} rows with missing audio")
    return samples


def make_weighted_sampler(samples: list) -> WeightedRandomSampler:
    """
    Balanced sampler: each class contributes equally to each epoch.
    Asthma gets a mild ASTHMA_OVERSAMPLE boost on top.
    This prevents dominant classes (healthy) from drowning out minority classes.
    """
    counts = Counter(s['label'] for s in samples)
    # Equal probability per class, then mild boost for asthma
    base_weight = {c: 1.0 / max(cnt, 1) for c, cnt in counts.items()}
    base_weight['asthma'] *= ASTHMA_OVERSAMPLE

    # Normalise so each class total weight is equal (except asthma slight boost)
    # Without this, 12364 healthy samples * (1/12364) = 1.0 total,
    # 70 asthma * (3/70) = 3.0 total → asthma is 3x, not 80x
    weights = [base_weight[s['label']] for s in samples]

    # Epoch size: use the actual training-set length so the number of batches
    # is sensible (~len(samples)/batch_size steps).  Each class still appears
    # with equal probability because of the per-sample weights above.
    n_samples = len(samples)
    return WeightedRandomSampler(weights, num_samples=n_samples, replacement=True)


# ─── Model components ─────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r), nn.ReLU(),
            nn.Linear(channels // r, channels), nn.Sigmoid())

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
            ResBlock(32,  64,  stride=2),
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
    def __init__(self, img_h=64, img_w=128, patch=8, dim=128, depth=4, heads=4, out_dim=256):
        super().__init__()
        nh, nw     = img_h // patch, img_w // patch
        n_patches  = nh * nw
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
    def __init__(self, feat_dim=256, sym_dim=7, hidden=64):
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


class CoughSenseV5(nn.Module):
    def __init__(self, n_classes=4, sym_dim=7, n_domains=2, embed_dim=256):
        super().__init__()
        self.cnn  = CNNBranch(out_dim=embed_dim)
        self.vit  = ViTBranch(img_h=64, img_w=128, out_dim=embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim*2, 2), nn.Softmax(dim=-1))
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(),
                                  nn.LayerNorm(embed_dim))
        self.film = FiLMLayer(feat_dim=embed_dim, sym_dim=sym_dim)
        self.domain_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Linear(64, n_domains))
        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def encode(self, mel):
        f_cnn = self.cnn(mel)
        f_vit = self.vit(mel)
        g = self.gate(torch.cat([f_cnn, f_vit], dim=-1))
        fused = g[:,0:1] * f_cnn + g[:,1:2] * f_vit
        return self.proj(fused)

    def forward(self, mel_heavy, mel_shallow, has_shallow, sym, lam=0.0):
        z_h = self.encode(mel_heavy)
        z_s = self.encode(mel_shallow)
        m   = has_shallow.unsqueeze(-1)
        z   = (z_h + m * z_s) / (1 + m)
        z   = self.film(z, sym)
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
    smooth_labels = torch.full_like(logits, eps / (n - 1))
    smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - eps)
    log_p  = F.log_softmax(logits, dim=-1)
    probs  = log_p.exp()
    focal  = (1 - probs) ** gamma
    wt = class_weights[labels].unsqueeze(1)
    loss = -(focal * smooth_labels * log_p * wt).sum(dim=-1)
    return loss.mean()


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for mel_h, mel_s, has_s, sym, _, labels in loader:
        mel_h = mel_h.to(device); mel_s = mel_s.to(device)
        has_s = has_s.to(device); sym   = sym.to(device)
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
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    # Per-class recall
    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_true == i
        if mask.sum() > 0:
            per_class[cls] = float((y_pred[mask] == i).mean())

    return bal_acc, f1, auc, per_class


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, class_weights,
                    device, epoch, total_epochs):
    model.train()
    lam = get_lambda(epoch, total_epochs)
    total_loss = 0.0

    for mel_h, mel_s, has_s, sym, domain, labels in tqdm(
            loader, desc=f'Ep {epoch}/{total_epochs}', leave=False):
        mel_h  = mel_h.to(device);  mel_s  = mel_s.to(device)
        has_s  = has_s.to(device);  sym    = sym.to(device)
        labels = labels.to(device); domain = domain.to(device)

        disease_logits, domain_logits, _ = model(
            mel_h, mel_s, has_s, sym, lam=lam)

        loss_cls    = focal_loss(disease_logits, labels,
                                 class_weights.to(device))
        loss_domain = F.cross_entropy(domain_logits, domain)
        loss        = loss_cls + 0.3 * lam * loss_domain

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',        default='ml_service/v5_data.csv')
    p.add_argument('--output_dir', default='checkpoints_v5')
    p.add_argument('--epochs',     type=int,   default=80)
    p.add_argument('--batch_size', type=int,   default=16)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--folds',      type=int,   default=5)
    p.add_argument('--seed',       type=int,   default=42)
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

    # Class weights: uniform — the balanced sampler already ensures each class
    # appears ~equally in every batch. Stacking heavy loss weights on top causes
    # the model to collapse to the minority class. Keep weights equal so the
    # loss landscape treats all 4 classes symmetrically.
    class_weights = torch.ones(N_CLASSES, dtype=torch.float32)
    print(f"\nClass weights: uniform {class_weights.tolist()}")

    labels_arr = np.array([CLASS_NAMES.index(s['label']) for s in all_samples])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                          random_state=args.seed)

    fold_results = []
    log_path = out_dir / 'training_log_v5.txt'
    log = open(log_path, 'w')

    for fold, (tr_idx, val_idx) in enumerate(skf.split(all_samples, labels_arr)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.folds}")

        tr_samples  = [all_samples[i] for i in tr_idx]
        val_samples = [all_samples[i] for i in val_idx]

        # Check asthma count in training fold
        tr_counts = Counter(s['label'] for s in tr_samples)
        print(f"  Train: {len(tr_samples)}  Val: {len(val_samples)}")
        print(f"  Train asthma: {tr_counts.get('asthma',0)} samples")

        tr_ds  = CoughDatasetV5(tr_samples,  training=True)
        val_ds = CoughDatasetV5(val_samples, training=False)

        sampler   = make_weighted_sampler(tr_samples)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                               sampler=sampler, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

        model     = CoughSenseV5(n_classes=N_CLASSES,
                                 sym_dim=len(SYMPTOM_COLS)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.epochs, T_mult=1)

        best_f1, best_bal_acc, best_auc = 0.0, 0.0, 0.0
        best_per_class = {}
        best_path = out_dir / f'fold{fold+1}_best.pt'

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, tr_loader, optimizer, scheduler,
                                   class_weights, device, epoch, args.epochs)
            bal_acc, f1, auc, per_class = evaluate(model, val_loader, device)

            line = (f"Fold {fold+1} Ep {epoch:3d}/{args.epochs} | "
                    f"loss={loss:.3f} bal_acc={bal_acc:.4f} "
                    f"f1={f1:.4f} auc={auc:.4f} | "
                    f"healthy={per_class.get('healthy',0):.3f} "
                    f"asthma={per_class.get('asthma',0):.3f} "
                    f"covid={per_class.get('covid',0):.3f} "
                    f"resp={per_class.get('respiratory_cond',0):.3f}")
            print(line)
            log.write(line + '\n'); log.flush()

            # Save best checkpoint based on balanced accuracy (rewards all 4 classes)
            if bal_acc > best_bal_acc:
                best_f1        = f1
                best_bal_acc   = bal_acc
                best_auc       = auc
                best_per_class = per_class
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Best bal_acc={best_bal_acc:.4f} f1={best_f1:.4f}")

        fold_results.append({
            'fold':       fold + 1,
            'bal_acc':    best_bal_acc,
            'f1_macro':   best_f1,
            'auc':        best_auc,
            'per_class':  best_per_class,
        })
        print(f"\nFold {fold+1} best → bal_acc={best_bal_acc:.4f} "
              f"f1={best_f1:.4f} auc={best_auc:.4f}")
        pcs = '  '.join(f"{k}={v:.3f}" for k,v in best_per_class.items())
        print(f"  Per-class recall: {pcs}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    bal_accs = [r['bal_acc']  for r in fold_results]
    f1s      = [r['f1_macro'] for r in fold_results]
    aucs     = [r['auc']      for r in fold_results]

    summary = {
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

    for k, v in summary.items():
        if k not in ('fold_results', 'config', 'class_names'):
            print(f"  {k:18s}: {v:.4f}")

    summary_path = out_dir / 'cv_summary_v5.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")

    fin = (f"\nFINAL: bal_acc={summary['mean_bal_acc']*100:.1f}% "
           f"±{summary['std_bal_acc']*100:.1f}%  "
           f"F1={summary['mean_f1']:.3f}  AUC={summary['mean_auc']:.3f}")
    print(fin)
    log.write(fin + '\n')
    log.close()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params/1e6:.2f}M parameters")


if __name__ == '__main__':
    main()
