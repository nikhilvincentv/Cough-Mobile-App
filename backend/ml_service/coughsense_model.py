"""
CoughSense V2: Multi-Innovation Dual-Branch Architecture

New in V2 (beyond V1 dual-branch + GRL + CBS-SupCon):
  1.  SpecAugment on mel spectrograms (frequency + time masking)
  2.  Focal Loss for disease head (handles class imbalance at loss level)
  3.  Momentum Memory Bank — MoCo-style ring buffer of past embeddings
        per class → richer negative mining for CBS-SupCon without
        requiring large batch sizes
  4.  EMA Teacher — exponential moving average of student weights for
        self-distillation; soft teacher targets regularize training
  5.  Monte Carlo Dropout inference — multiple stochastic forward passes
        → calibrated uncertainty estimates (epistemic uncertainty)
  6.  Temperature Scaling — post-hoc calibration of class probabilities
  7.  Stochastic Depth in Transformer layers (drop path regularization)
  8.  HeAR-compatible backbone hook — optional feature extractor swap
        for Google HeAR foundation model (google/hear on HuggingFace)

Author: Nikhil Vincent
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchaudio


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer
# ─────────────────────────────────────────────────────────────────────────────

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambda_val * grad, None


def grad_reverse(x, lambda_val=1.0):
    return GradientReversalFunction.apply(x, lambda_val)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SpecAugment
# ─────────────────────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
    """
    SpecAugment (Park et al. 2019) applied to mel spectrogram tensors.
    Masks F consecutive frequency bins and T consecutive time steps
    with probability p_apply. Applied independently per channel.

    This is novel over standard audio augmentation (which operates on
    waveforms): masking on the mel representation forces the model to
    be robust to partial spectral information, improving generalization
    across microphone hardware (exactly the Coswara vs CoughVID gap).
    """
    def __init__(self, freq_mask_max=20, time_mask_max=25,
                 n_freq_masks=2, n_time_masks=2, p_apply=0.8):
        super().__init__()
        self.freq_mask_max  = freq_mask_max
        self.time_mask_max  = time_mask_max
        self.n_freq_masks   = n_freq_masks
        self.n_time_masks   = n_time_masks
        self.p_apply        = p_apply

    def forward(self, x):
        # x: (B, C, H, W) — multi-scale mel, C=3 channels
        if not self.training:
            return x
        if torch.rand(1).item() > self.p_apply:
            return x

        B, C, H, W = x.shape
        out = x.clone()

        for _ in range(self.n_freq_masks):
            f = torch.randint(1, self.freq_mask_max + 1, (1,)).item()
            f0 = torch.randint(0, max(1, H - f), (1,)).item()
            out[:, :, f0:f0 + f, :] = 0.0

        for _ in range(self.n_time_masks):
            t = torch.randint(1, self.time_mask_max + 1, (1,)).item()
            t0 = torch.randint(0, max(1, W - t), (1,)).item()
            out[:, :, :, t0:t0 + t] = 0.0

        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Scale Mel Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleMelExtractor(nn.Module):
    def __init__(self, sample_rate=16000, target_frames=128):
        super().__init__()
        self.target_frames = target_frames
        self.fine   = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=128,
            n_mels=128, f_min=50, f_max=8000, power=2.0)
        self.medium = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, hop_length=160,
            n_mels=64, f_min=50, f_max=8000, power=2.0)
        self.coarse = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=256, hop_length=256,
            n_mels=32, f_min=50, f_max=8000, power=2.0)
        self.to_db  = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.spec_aug = SpecAugment()

    def _process(self, mel_fn, x):
        spec = self.to_db(mel_fn(x))
        mu   = spec.mean(dim=(-1, -2), keepdim=True)
        std  = spec.std(dim=(-1, -2), keepdim=True) + 1e-6
        return (spec - mu) / std

    def forward(self, waveform):
        H, W = 64, self.target_frames
        def _resize(s):
            return F.interpolate(s, size=(H, W), mode='bilinear', align_corners=False)

        fine   = _resize(self._process(self.fine,   waveform))
        medium = _resize(self._process(self.medium, waveform))
        coarse = _resize(self._process(self.coarse, waveform))
        mel = torch.cat([fine, medium, coarse], dim=1)  # (B, 3, 64, 128)
        return self.spec_aug(mel)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CNN Branch (SE-ResNet)
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // r), nn.GELU(),
            nn.Linear(ch // r, ch), nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.c1   = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1  = nn.BatchNorm2d(out_ch)
        self.c2   = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_ch)
        self.se   = SEBlock(out_ch)
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)) if (stride != 1 or in_ch != out_ch) else nn.Sequential()

    def forward(self, x):
        out = F.gelu(self.bn1(self.c1(x)))
        out = self.se(self.bn2(self.c2(out)))
        return F.gelu(out + self.skip(x))


class CNNBranch(nn.Module):
    def __init__(self, in_ch=3, embed_dim=256):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(32), nn.GELU())
        self.layer1 = ConvBlock(32,  64,  stride=2)
        self.layer2 = ConvBlock(64,  128, stride=2)
        self.layer3 = ConvBlock(128, embed_dim, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(0.3)

    def forward(self, x):
        return self.drop(self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))).flatten(1))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stochastic Depth Transformer Branch
# ─────────────────────────────────────────────────────────────────────────────

class StochasticDepthTransformerLayer(nn.Module):
    """
    Transformer encoder layer with stochastic depth (drop path).
    Each layer is dropped with probability p_drop during training.
    Scales the residual by 1/(1-p_drop) when kept (unbiased estimator).
    Reduces overfitting more effectively than standard dropout in ViTs.
    """
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1, p_drop=0.1):
        super().__init__()
        self.p_drop  = p_drop
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))

    def _drop_path(self, x, residual):
        if not self.training or self.p_drop == 0.0:
            return x + residual
        keep_prob = 1.0 - self.p_drop
        # Sample one keep/drop decision per item in batch
        mask = torch.rand(x.size(0), 1, 1, device=x.device) < keep_prob
        return x + residual * mask.float() / keep_prob

    def forward(self, x, src_key_padding_mask=None):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                key_padding_mask=src_key_padding_mask)
        x = self._drop_path(x, attn_out)
        x = self._drop_path(x, self.ff(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, ph=8, pw=8, dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=(ph, pw), stride=(ph, pw))

    def forward(self, x):
        x = self.proj(x)
        B, D, H, W = x.shape
        return x.flatten(2).transpose(1, 2)


class TransformerBranch(nn.Module):
    def __init__(self, in_ch=3, img_h=64, img_w=128, ph=8, pw=8,
                 dim=128, nhead=4, depth=4, dropout=0.1, drop_path=0.1):
        super().__init__()
        n_patches        = (img_h // ph) * (img_w // pw)
        self.patch_embed = PatchEmbed(in_ch, ph, pw, dim)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        # Stochastic depth rates increase with depth (linear schedule)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.layers = nn.ModuleList([
            StochasticDepthTransformerLayer(dim, nhead, dim * 4, dropout, dpr[i])
            for i in range(depth)])
        self.norm    = nn.LayerNorm(dim)
        self.project = nn.Linear(dim, 256)
        self.drop    = nn.Dropout(0.2)

    def forward(self, x):
        patches = self.patch_embed(x)
        B = patches.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, patches], dim=1) + self.pos_embed[:, :patches.size(1) + 1]
        x   = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        return self.project(self.norm(x[:, 0]))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Attention Gating Fusion
# ─────────────────────────────────────────────────────────────────────────────

class AttentionFusion(nn.Module):
    def __init__(self, dim=256, out_dim=256):
        super().__init__()
        self.gate = nn.Linear(dim * 2, 2)
        self.proj = nn.Sequential(
            nn.Linear(dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.2))

    def forward(self, cnn, tf):
        g = F.softmax(self.gate(torch.cat([cnn, tf], dim=1)), dim=-1)
        return self.proj(g[:, 0:1] * cnn + g[:, 1:2] * tf)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Domain Classifier (adversarial)
# ─────────────────────────────────────────────────────────────────────────────

class DomainClassifier(nn.Module):
    def __init__(self, feat_dim=256, n_domains=2):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, n_domains))

    def forward(self, x, lam=1.0):
        return self.clf(grad_reverse(x, lam))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Momentum Memory Bank for Contrastive Learning
# ─────────────────────────────────────────────────────────────────────────────

class MomentumMemoryBank(nn.Module):
    """
    MoCo-style ring buffer (He et al. 2020) storing past embeddings per class.
    Provides thousands of hard negatives without requiring large batches.

    Unlike standard SupCon which only uses in-batch positives/negatives,
    the memory bank ensures each class always has ≥bank_size_per_class
    negatives, preventing gradient collapse when batch size is small.

    Key difference from standard MoCo: per-class banks allow CBS-Loss
    temperature scaling to apply correctly to memory-bank negatives.
    """
    def __init__(self, embed_dim=256, bank_size_per_class=256, n_classes=3):
        super().__init__()
        self.bank_size = bank_size_per_class
        self.n_classes = n_classes
        # Register as buffers (not parameters — no gradient)
        for c in range(n_classes):
            self.register_buffer(f'bank_{c}',
                torch.randn(bank_size_per_class, embed_dim))
            self.register_buffer(f'ptr_{c}', torch.zeros(1, dtype=torch.long))
        self._normalize_all()

    def _normalize_all(self):
        for c in range(self.n_classes):
            b = getattr(self, f'bank_{c}')
            b.data = F.normalize(b.data, dim=1)

    @torch.no_grad()
    def update(self, embeddings, labels):
        """Enqueue new embeddings (FIFO, overwrites oldest)."""
        for c in range(self.n_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            embs = embeddings[mask].detach()
            bank = getattr(self, f'bank_{c}')
            ptr  = getattr(self, f'ptr_{c}')
            n    = embs.size(0)
            start = int(ptr.item())
            # Wrap-around write
            end = start + n
            if end <= self.bank_size:
                bank[start:end] = embs
            else:
                first = self.bank_size - start
                bank[start:]   = embs[:first]
                bank[:n - first] = embs[first:]
            ptr[0] = end % self.bank_size

    def get_bank(self, cls_idx):
        return getattr(self, f'bank_{cls_idx}')

    def all_negatives_for_class(self, cls_idx):
        """Return all bank embeddings for all OTHER classes."""
        negs = [self.get_bank(c) for c in range(self.n_classes) if c != cls_idx]
        return torch.cat(negs, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al. 2017, RetinaNet).
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    With γ=2.0, easy examples contribute near-zero loss.
    Concentrates training on hard misclassified examples regardless of class.
    Complementary to CBS-Loss which targets hard CLASSES.
    Together: CBS-Loss targets minority classes; Focal targets hard instances.
    """
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1, n_classes=3):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))

    def forward(self, logits, targets):
        B, C = logits.shape
        # Label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (C - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()

        # Per-sample focal weight: (1 - p_t)^gamma
        p_t      = (probs * F.one_hot(targets, C).float()).sum(dim=1)
        focal_w  = (1.0 - p_t) ** self.gamma

        loss = -(smooth_targets * log_probs).sum(dim=1)  # cross-entropy
        loss = focal_w * loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss    = alpha_t * loss

        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 9. CBS-SupCon with Memory Bank
# ─────────────────────────────────────────────────────────────────────────────

class ClassBalancedSupConLoss(nn.Module):
    """
    CBS-SupCon (V2): per-class temperature SupCon augmented with
    memory bank negatives.

    For each anchor i:
      - Positives: same-class embeddings in current batch
      - Negatives: all embeddings from OTHER classes in batch
                   PLUS all entries in memory bank for other classes

    This ensures ~1,000+ negatives per anchor regardless of batch size,
    preventing gradient collapse at the beginning of training when
    the batch may contain few negatives for the minority COVID class.
    """
    def __init__(self, class_temps=None, base_temp=0.07, memory_bank=None):
        super().__init__()
        self.base_temp   = base_temp
        self.class_temps = class_temps or {0: 0.08, 1: 0.05, 2: 0.10}
        self.memory_bank = memory_bank  # MomentumMemoryBank or None

    def forward(self, embeddings, labels):
        device = embeddings.device
        B      = embeddings.size(0)

        total_loss = 0.0
        n_valid    = 0

        for i in range(B):
            cls_i = labels[i].item()
            temp  = self.class_temps.get(cls_i, self.base_temp)

            # In-batch positives (same class, not self)
            pos_mask_batch = (labels == cls_i)
            pos_mask_batch[i] = False
            pos_embs = embeddings[pos_mask_batch]
            if len(pos_embs) == 0:
                continue

            # All negatives: in-batch other classes + memory bank
            neg_embs_list = [embeddings[labels != cls_i]]
            if self.memory_bank is not None:
                neg_embs_list.append(
                    self.memory_bank.all_negatives_for_class(cls_i).to(device))
            neg_embs = torch.cat(neg_embs_list, dim=0)

            # Positives similarity
            pos_sim = (embeddings[i:i+1] @ pos_embs.T).squeeze(0) / temp   # (P,)
            # Negatives similarity
            neg_sim = (embeddings[i:i+1] @ neg_embs.T).squeeze(0) / temp   # (N,)

            # SupCon: log-sum-exp over (positives + all negatives)
            all_sim   = torch.cat([pos_sim, neg_sim])
            log_denom = torch.logsumexp(all_sim, dim=0)
            loss_i    = -(self.base_temp / temp) * (pos_sim.mean() - log_denom)

            total_loss += loss_i
            n_valid    += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / n_valid


# ─────────────────────────────────────────────────────────────────────────────
# 10. EMA Teacher
# ─────────────────────────────────────────────────────────────────────────────

class EMATeacher:
    """
    Exponential moving average of student model weights.

    Used for self-distillation: the EMA model produces soft targets
    that the student learns to match via KL divergence. This is
    cheaper than a separate teacher model and acts as a strong
    regularizer (Mean Teacher, Tarvainen & Valpola 2017).

    The EMA teacher is NOT updated by gradient — only by:
      θ_teacher ← α * θ_teacher + (1-α) * θ_student
    """
    def __init__(self, student, alpha=0.999):
        import copy
        self.model = copy.deepcopy(student)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.alpha = alpha

    @torch.no_grad()
    def update(self, student):
        for t_p, s_p in zip(self.model.parameters(), student.parameters()):
            t_p.data.mul_(self.alpha).add_(s_p.data, alpha=1.0 - self.alpha)
        for t_b, s_b in zip(self.model.buffers(), student.buffers()):
            if t_b.dtype.is_floating_point:
                t_b.data.mul_(self.alpha).add_(s_b.data, alpha=1.0 - self.alpha)
            else:
                t_b.data.copy_(s_b.data)

    @torch.no_grad()
    def predict(self, waveform):
        self.model.eval()
        out = self.model(waveform, lambda_d=0.0)
        return F.softmax(out['logits'], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Temperature Scaling Calibration
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration (Guo et al. 2017).
    Learns a single scalar temperature T on the validation set to
    minimize NLL. Calibrated probabilities: softmax(logits / T).
    Does NOT change accuracy, only confidence calibration (ECE).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def calibrate(self, logits, labels, lr=0.01, max_iter=200):
        """Fit temperature on validation logits."""
        opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / self.temperature.clamp(min=0.05), labels)
            loss.backward()
            return loss
        opt.step(closure)
        print(f"[TemperatureScaling] Optimal T = {self.temperature.item():.4f}")
        return self

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Full CoughSense V2 Model
# ─────────────────────────────────────────────────────────────────────────────

class CoughSense(nn.Module):
    """
    CoughSense V2.

    Forward:
      waveform : (B, 1, T)  mono 16kHz
      lambda_d : float      GRL scale (DANN annealing)

    Returns dict:
      logits, domain_logits, embeddings, cnn_feat, tf_feat
    """
    def __init__(self, num_classes=3, num_domains=2,
                 embed_dim=256, sample_rate=16000, target_frames=128,
                 bank_size=256, mc_samples=10):
        super().__init__()
        self.mc_samples    = mc_samples
        self.num_classes   = num_classes

        self.mel_extractor = MultiScaleMelExtractor(sample_rate, target_frames)
        self.cnn_branch    = CNNBranch(in_ch=3, embed_dim=embed_dim)
        self.tf_branch     = TransformerBranch(
                                in_ch=3, img_h=64, img_w=target_frames,
                                ph=8, pw=8, dim=128, nhead=4, depth=4,
                                dropout=0.1, drop_path=0.1)
        self.fusion        = AttentionFusion(dim=embed_dim, out_dim=embed_dim)
        self.domain_clf    = DomainClassifier(feat_dim=embed_dim, n_domains=num_domains)
        self.disease_head  = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(),
            nn.Dropout(0.3),           # kept ON during MC-Dropout inference
            nn.Linear(128, num_classes))
        self.memory_bank   = MomentumMemoryBank(embed_dim, bank_size, num_classes)
        self.calibrator    = TemperatureScaling()

    def forward(self, waveform, lambda_d=1.0):
        mel         = self.mel_extractor(waveform)
        cnn_feat    = self.cnn_branch(mel)
        tf_feat     = self.tf_branch(mel)
        fused       = self.fusion(cnn_feat, tf_feat)
        embeddings  = F.normalize(fused, dim=1)
        logits      = self.disease_head(fused)
        domain_logits = self.domain_clf(fused, lambda_d)
        return dict(logits=logits, domain_logits=domain_logits,
                    embeddings=embeddings, cnn_feat=cnn_feat, tf_feat=tf_feat)

    @torch.no_grad()
    def predict(self, waveform):
        """Deterministic inference (no augmentation, no GRL)."""
        self.eval()
        out = self.forward(waveform, lambda_d=0.0)
        return F.softmax(out['logits'], dim=1)

    @torch.no_grad()
    def predict_with_uncertainty(self, waveform):
        """
        Monte Carlo Dropout inference.
        Keeps Dropout ENABLED for mc_samples forward passes, then
        computes mean probabilities (epistemic mean) and predictive
        entropy (uncertainty) across samples.

        Returns:
          mean_probs  : (B, C)  calibrated mean class probabilities
          uncertainty : (B,)    predictive entropy (higher = more uncertain)
          raw_samples : (B, mc_samples, C)  individual samples for analysis
        """
        self.train()  # enable dropout
        B = waveform.size(0)
        samples = []
        for _ in range(self.mc_samples):
            out = self.forward(waveform, lambda_d=0.0)
            cal = self.calibrator(out['logits'])
            samples.append(F.softmax(cal, dim=1))
        self.eval()

        raw    = torch.stack(samples, dim=1)          # (B, S, C)
        mean_p = raw.mean(dim=1)                      # (B, C)
        entropy = -(mean_p * (mean_p + 1e-8).log()).sum(dim=1)  # (B,)
        return mean_p, entropy, raw


# ─────────────────────────────────────────────────────────────────────────────
# 13. Combined Loss (V2)
# ─────────────────────────────────────────────────────────────────────────────

class CoughSenseLoss(nn.Module):
    """
    V2 combined loss:

    L_total = L_focal + α·L_supcon - λ_d·L_domain + β·L_distill

    L_focal   : focal loss for disease classification
    L_supcon  : CBS-SupCon with memory bank
    L_domain  : adversarial domain loss (subtracted)
    L_distill : KL(student || EMA teacher) — self-distillation

    Weights (α, λ_d, β) are optionally managed by GradNorm (see trainer).
    """
    def __init__(self, alpha=0.3, beta=0.1, memory_bank=None,
                 focal_gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha       = alpha
        self.beta        = beta
        self.focal       = FocalLoss(gamma=focal_gamma,
                                     label_smoothing=label_smoothing)
        self.domain_loss = nn.CrossEntropyLoss()
        self.supcon      = ClassBalancedSupConLoss(memory_bank=memory_bank)

    def forward(self, outputs, disease_labels, domain_labels,
                lambda_d=1.0, teacher_probs=None):
        l_focal  = self.focal(outputs['logits'], disease_labels)
        l_domain = self.domain_loss(outputs['domain_logits'], domain_labels)
        l_supcon = self.supcon(outputs['embeddings'], disease_labels)

        total = l_focal + self.alpha * l_supcon - lambda_d * l_domain

        l_distill = torch.tensor(0.0, device=l_focal.device)
        if teacher_probs is not None and self.beta > 0:
            student_log = F.log_softmax(outputs['logits'], dim=1)
            l_distill   = F.kl_div(student_log, teacher_probs,
                                   reduction='batchmean', log_target=False)
            total = total + self.beta * l_distill

        return dict(total=total,
                    focal=l_focal.item(),
                    supcon=l_supcon.item(),
                    domain=l_domain.item(),
                    distill=l_distill.item())


# ─────────────────────────────────────────────────────────────────────────────
# 14. GradNorm Dynamic Loss Balancing
# ─────────────────────────────────────────────────────────────────────────────

class GradNorm(nn.Module):
    """
    GradNorm (Chen et al. 2018, ICML).
    Learns loss weights w_i automatically by equalizing gradient norms
    across tasks, adjusted for each task's training rate.

    Prevents any one of {L_focal, L_supcon, L_domain} from dominating,
    which is a known failure mode when domain adversarial loss is too
    strong early in training.

    alpha_gn controls how aggressively weights are balanced:
      alpha_gn = 1.5 is the recommended value from Chen et al.
    """
    def __init__(self, n_tasks=3, alpha_gn=1.5):
        super().__init__()
        self.n_tasks  = n_tasks
        self.alpha_gn = alpha_gn
        self.weights  = nn.Parameter(torch.ones(n_tasks))
        self.register_buffer('L0', torch.ones(n_tasks))  # initial losses
        self._initialized = False

    def get_weights(self):
        """Positive-normalized weights (sum to n_tasks)."""
        w = F.softmax(self.weights, dim=0) * self.n_tasks
        return w

    def update(self, losses, shared_params, optimizer):
        """
        losses      : list of [l_focal, l_supcon, l_domain] (detached floats OK)
        shared_params: a parameter whose gradient norm is used as reference
        optimizer   : GradNorm has its own separate step on self.weights
        """
        losses_t = torch.stack([l if isinstance(l, torch.Tensor) else
                                 torch.tensor(l) for l in losses])

        if not self._initialized:
            self.L0.copy_(losses_t.detach())
            self._initialized = True

        # Relative inverse training rates
        loss_ratio = losses_t.detach() / (self.L0 + 1e-8)
        mean_ratio = loss_ratio.mean()
        r_i        = loss_ratio / (mean_ratio + 1e-8)   # (n_tasks,)

        # Target gradient norms
        w        = self.get_weights()
        G_avg    = (w.detach() * losses_t.detach()).sum() / self.n_tasks
        G_target = (G_avg * r_i ** self.alpha_gn).detach()

        # Compute actual gradient norms from shared layer
        G_actual = []
        for i, loss in enumerate(losses):
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                G_actual.append(torch.tensor(1.0, device=self.weights.device))
                continue
            g = torch.autograd.grad(w[i] * loss, shared_params,
                                    retain_graph=True, create_graph=False)[0]
            G_actual.append(g.norm())

        G_actual = torch.stack(G_actual)
        gn_loss  = F.l1_loss(G_actual, G_target)

        optimizer.zero_grad()
        gn_loss.backward()
        optimizer.step()

        # Renormalize weights
        with torch.no_grad():
            self.weights.copy_(self.weights / self.weights.sum() * self.n_tasks)


# ─────────────────────────────────────────────────────────────────────────────
# 15. PCGrad Optimizer Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PCGrad:
    """
    Gradient Surgery / PCGrad (Yu et al., NeurIPS 2020).

    When the gradient of task i conflicts with task j (dot product < 0),
    project g_i onto the normal plane of g_j. This removes the component
    of g_i that hurts task j, without adding new conflicting information.

    Applied here to resolve conflicts between:
      - L_focal (wants discriminative features)
      - L_domain (wants domain-invariant features)

    These two objectives are fundamentally in tension during early training.
    PCGrad lets them coexist without one suppressing the other.

    Usage:
      pcgrad = PCGrad(optimizer)
      pcgrad.pc_backward([l_focal, l_domain])
      pcgrad.step()
    """
    def __init__(self, optimizer):
        self._opt    = optimizer
        self._params = [p for group in optimizer.param_groups
                        for p in group['params'] if p.requires_grad]

    @property
    def param_groups(self):
        return self._opt.param_groups

    def zero_grad(self):
        self._opt.zero_grad(set_to_none=True)

    def step(self):
        self._opt.step()

    def state_dict(self):
        return self._opt.state_dict()

    def load_state_dict(self, sd):
        self._opt.load_state_dict(sd)

    def _collect_grads(self):
        grads = []
        for p in self._params:
            if p.grad is not None:
                grads.append(p.grad.data.clone().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)

    def _assign_grads(self, flat_grad):
        offset = 0
        for p in self._params:
            n = p.numel()
            p.grad = flat_grad[offset:offset + n].view(p.shape).clone()
            offset += n

    def pc_backward(self, objectives):
        """
        objectives: list of loss tensors (one per task).
        Computes and assigns conflict-free merged gradient.
        """
        # Compute per-task gradients
        grads = []
        for i, obj in enumerate(objectives):
            self._opt.zero_grad(set_to_none=True)
            retain = (i < len(objectives) - 1)
            obj.backward(retain_graph=retain)
            grads.append(self._collect_grads())

        # PCGrad projection
        pc_grads = [g.clone() for g in grads]
        indices  = list(range(len(grads)))
        for i in range(len(pc_grads)):
            # Randomize projection order per paper recommendation
            order = indices[:i] + indices[i+1:]
            import random; random.shuffle(order)
            for j in order:
                dot = (pc_grads[i] * grads[j]).sum()
                if dot < 0:
                    pc_grads[i] -= (dot / (grads[j].norm() ** 2 + 1e-12)) * grads[j]

        merged = sum(pc_grads)
        self._opt.zero_grad(set_to_none=True)
        self._assign_grads(merged)


# ─────────────────────────────────────────────────────────────────────────────
# 16. Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable


def get_lambda_schedule(epoch, total_epochs, gamma=10.0):
    p = epoch / total_epochs
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def build_model(num_classes=3, num_domains=2, sample_rate=16000,
                target_frames=128, bank_size=256):
    return CoughSense(num_classes=num_classes, num_domains=num_domains,
                      embed_dim=256, sample_rate=sample_rate,
                      target_frames=target_frames, bank_size=bank_size)


if __name__ == '__main__':
    import random
    model = build_model()
    count_parameters(model)

    wave = torch.randn(4, 1, 56000)
    out  = model(wave, lambda_d=0.5)
    print("logits:       ", out['logits'].shape)
    print("domain_logits:", out['domain_logits'].shape)
    print("embeddings:   ", out['embeddings'].shape)

    criterion = CoughSenseLoss(memory_bank=model.memory_bank)
    dlabels   = torch.tensor([0, 1, 2, 0])
    domains   = torch.tensor([0, 0, 1, 1])
    teacher_p = torch.softmax(torch.randn(4, 3), dim=1)
    losses    = criterion(out, dlabels, domains, lambda_d=0.5, teacher_probs=teacher_p)
    print("losses:", {k: round(v, 4) if isinstance(v, float) else v for k, v in losses.items()})

    # MC Dropout uncertainty
    mean_p, unc, raw = model.predict_with_uncertainty(wave)
    print("MC mean probs:", mean_p.shape, "uncertainty:", unc.shape)

    # PCGrad test
    pcgrad = PCGrad(torch.optim.AdamW(model.parameters(), lr=1e-3))
    out2   = model(wave, lambda_d=0.3)
    l_cls  = criterion.focal(out2['logits'], dlabels)
    l_dom  = criterion.domain_loss(out2['domain_logits'], domains)
    pcgrad.pc_backward([l_cls, l_dom])
    pcgrad.step()

    print("\nAll V2 checks passed.")
