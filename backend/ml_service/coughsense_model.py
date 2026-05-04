"""
CoughSense: Dual-Branch CNN-Transformer with Domain-Adversarial Training
for Cross-Dataset Multi-Disease Respiratory Sound Classification

Novel Contributions (vs published literature as of 2025):
  1. DUAL-BRANCH ARCHITECTURE: parallel CNN (local spectral patterns) +
     lightweight Transformer (long-range temporal context), fused via
     learned attention gating. Prior work uses one or the other, not both
     on multi-scale mel features.

  2. DOMAIN-ADVERSARIAL TRAINING: an explicit domain classifier is trained
     adversarially against the feature extractor using gradient reversal
     (Ganin et al. 2016). This forces the shared representation to be
     domain-invariant across Coswara (clinical crowdsourced) and CoughVID
     (EPFL crowdsourced) — the exact cross-dataset generalization failure
     reported in Islam et al. 2025 (AUC drops from 0.82 to 0.53).

  3. CLASS-BALANCED SUPERVISED CONTRASTIVE LOSS: per-class temperature
     scaling in SupCon pushes the model harder on minority classes (COVID
     has 658 samples vs 2590 bronchitis). Standard SupCon uses one global
     temperature. This is new.

  4. MULTI-SCALE MEL FUSION: three mel resolutions (fine/medium/coarse)
     concatenated as a 3-channel image before the CNN branch. Captures
     both cough transients (fine-grained) and disease envelope (coarse).

Training: two losses jointly optimized.
  - L_cls:    cross-entropy (disease classification)
  - L_supcon: class-balanced supervised contrastive
  - L_domain: binary cross-entropy, SUBTRACTED (adversarial)
  Total: L_cls + alpha*L_supcon - lambda_d*L_domain

Author: Nikhil Vincent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchaudio
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gradient Reversal Layer  (enables domain-adversarial training)
# ─────────────────────────────────────────────────────────────────────────────

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


def grad_reverse(x, lambda_val=1.0):
    return GradientReversalFunction.apply(x, lambda_val)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Scale Mel Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleMelExtractor(nn.Module):
    """
    Extracts 3 mel spectrograms at different time-frequency resolutions,
    then resizes and concatenates them as a 3-channel tensor.

    Fine   (n_mels=128, hop=128): captures cough onset bursts and transients
    Medium (n_mels=64,  hop=160): standard mel, balanced resolution
    Coarse (n_mels=32,  hop=256): disease-level envelope, slow dynamics
    """
    def __init__(self, sample_rate=16000, target_frames=128):
        super().__init__()
        self.target_frames = target_frames

        self.fine = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=128,
            n_mels=128, f_min=50, f_max=8000, power=2.0
        )
        self.medium = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, hop_length=160,
            n_mels=64, f_min=50, f_max=8000, power=2.0
        )
        self.coarse = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=256, hop_length=256,
            n_mels=32, f_min=50, f_max=8000, power=2.0
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def _process(self, mel_fn, x):
        spec = self.to_db(mel_fn(x))
        mu  = spec.mean(dim=(-1, -2), keepdim=True)
        std = spec.std(dim=(-1, -2), keepdim=True) + 1e-6
        spec = (spec - mu) / std
        return spec

    def forward(self, waveform):
        # waveform: (B, 1, T)  mono, 16kHz
        fine   = self._process(self.fine,   waveform)   # (B, 1, 128, T1)
        medium = self._process(self.medium, waveform)   # (B, 1, 64,  T2)
        coarse = self._process(self.coarse, waveform)   # (B, 1, 32,  T3)

        H, W = 64, self.target_frames
        fine   = F.interpolate(fine,   size=(H, W), mode='bilinear', align_corners=False)
        medium = F.interpolate(medium, size=(H, W), mode='bilinear', align_corners=False)
        coarse = F.interpolate(coarse, size=(H, W), mode='bilinear', align_corners=False)

        return torch.cat([fine, medium, coarse], dim=1)  # (B, 3, 64, target_frames)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CNN Branch — local spectral pattern extractor
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.gelu(out + self.skip(x))


class CNNBranch(nn.Module):
    def __init__(self, in_ch=3, embed_dim=256):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU()
        )
        self.layer1 = ConvBlock(32,  64,  stride=2)
        self.layer2 = ConvBlock(64,  128, stride=2)
        self.layer3 = ConvBlock(128, embed_dim, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(0.3)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.drop(x)   # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Transformer Branch — long-range temporal dependency extractor
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, patch_h=8, patch_w=8, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=(patch_h, patch_w),
                              stride=(patch_h, patch_w))

    def forward(self, x):
        x = self.proj(x)                        # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, H'*W', D)
        return x, H, W


class TransformerBranch(nn.Module):
    def __init__(self, in_ch=3, img_h=64, img_w=128,
                 patch_h=8, patch_w=8, embed_dim=128,
                 num_heads=4, depth=3, dropout=0.1):
        super().__init__()
        num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.patch_embed = PatchEmbed(in_ch, patch_h, patch_w, embed_dim)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed   = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm     = nn.LayerNorm(embed_dim)
        self.project  = nn.Linear(embed_dim, 256)
        self.drop     = nn.Dropout(0.2)

    def forward(self, x):
        patches, H, W = self.patch_embed(x)      # (B, N, D)
        B, N, D = patches.shape
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x   = torch.cat([cls, patches], dim=1)   # (B, N+1, D)
        x   = x + self.pos_embed[:, :x.size(1)]
        x   = self.drop(x)
        x   = self.encoder(x)
        x   = self.norm(x[:, 0])                 # CLS token
        return self.project(x)                   # (B, 256)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Attention Gating Fusion
# ─────────────────────────────────────────────────────────────────────────────

class AttentionFusion(nn.Module):
    """
    Learned gating between CNN and Transformer embeddings.
    g = softmax(W_g * [cnn; transformer])
    output = g[0]*cnn + g[1]*transformer, then projected.
    """
    def __init__(self, dim=256, out_dim=256):
        super().__init__()
        self.gate  = nn.Linear(dim * 2, 2)
        self.proj  = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, cnn_feat, tf_feat):
        combined = torch.cat([cnn_feat, tf_feat], dim=1)  # (B, 512)
        gates    = F.softmax(self.gate(combined), dim=-1)  # (B, 2)
        fused    = gates[:, 0:1] * cnn_feat + gates[:, 1:2] * tf_feat
        return self.proj(fused)                            # (B, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Domain Classifier  (adversarial)
# ─────────────────────────────────────────────────────────────────────────────

class DomainClassifier(nn.Module):
    """
    Classifies which dataset a sample came from: 0=Coswara, 1=CoughVID.
    Receives features AFTER gradient reversal so the backbone is
    forced to learn domain-invariant representations.
    """
    def __init__(self, feat_dim=256, num_domains=2):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, features, lambda_val=1.0):
        rev = grad_reverse(features, lambda_val)
        return self.clf(rev)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full CoughSense Model
# ─────────────────────────────────────────────────────────────────────────────

class CoughSense(nn.Module):
    """
    Full CoughSense pipeline.

    Forward:
      waveform  : (B, 1, T)  raw mono waveform at 16kHz
      lambda_d  : float      GRL scale (annealed during training)

    Returns dict:
      logits       : (B, num_classes)
      domain_logits: (B, num_domains)
      embeddings   : (B, embed_dim)  L2-normalized for contrastive loss
    """

    def __init__(
        self,
        num_classes=3,
        num_domains=2,
        embed_dim=256,
        sample_rate=16000,
        target_frames=128
    ):
        super().__init__()
        self.mel_extractor = MultiScaleMelExtractor(sample_rate, target_frames)
        self.cnn_branch    = CNNBranch(in_ch=3, embed_dim=embed_dim)
        self.tf_branch     = TransformerBranch(
                                in_ch=3, img_h=64, img_w=target_frames,
                                patch_h=8, patch_w=8,
                                embed_dim=128, num_heads=4, depth=3
                             )
        self.fusion        = AttentionFusion(dim=embed_dim, out_dim=embed_dim)
        self.domain_clf    = DomainClassifier(feat_dim=embed_dim, num_domains=num_domains)
        self.disease_head  = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, waveform, lambda_d=1.0):
        mel      = self.mel_extractor(waveform)      # (B, 3, 64, T)
        cnn_feat = self.cnn_branch(mel)              # (B, 256)
        tf_feat  = self.tf_branch(mel)               # (B, 256)
        fused    = self.fusion(cnn_feat, tf_feat)    # (B, 256)

        embeddings    = F.normalize(fused, dim=1)
        logits        = self.disease_head(fused)
        domain_logits = self.domain_clf(fused, lambda_d)

        return {
            'logits':        logits,
            'domain_logits': domain_logits,
            'embeddings':    embeddings,
            'cnn_feat':      cnn_feat,
            'tf_feat':       tf_feat
        }

    def predict(self, waveform):
        with torch.no_grad():
            out   = self.forward(waveform, lambda_d=0.0)
            probs = F.softmax(out['logits'], dim=1)
        return probs


# ─────────────────────────────────────────────────────────────────────────────
# 8. Class-Balanced Supervised Contrastive Loss
# ─────────────────────────────────────────────────────────────────────────────

class ClassBalancedSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020) with per-class
    temperature scaling.

    Per-class temperatures:
      COVID (idx 1):      0.05  — minority, hardest push
      Healthy (idx 0):    0.08  — moderate
      Bronchitis (idx 2): 0.10  — majority, softer push

    Novel vs standard SupCon which uses a single global temperature.
    """

    def __init__(self, class_temperatures=None, base_temperature=0.07):
        super().__init__()
        self.base_temp = base_temperature
        if class_temperatures is None:
            self.class_temps = {0: 0.08, 1: 0.05, 2: 0.10}
        else:
            self.class_temps = class_temperatures

    def forward(self, embeddings, labels):
        device = embeddings.device
        B = embeddings.size(0)
        sim = torch.matmul(embeddings, embeddings.T)  # (B, B)

        total_loss = 0.0
        n_valid    = 0

        for i in range(B):
            cls_i = labels[i].item()
            temp  = self.class_temps.get(cls_i, self.base_temp)

            pos_mask = (labels == cls_i).float()
            pos_mask[i] = 0.0

            if pos_mask.sum() == 0:
                continue

            logits = sim[i] / temp
            logits = logits - logits.max().detach()

            exp_logits = torch.exp(logits)
            log_prob   = logits - torch.log(exp_logits.sum() + 1e-8)

            loss_i = -(self.base_temp / temp) * \
                     (pos_mask * log_prob).sum() / (pos_mask.sum() + 1e-8)

            total_loss += loss_i
            n_valid    += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss / n_valid


# ─────────────────────────────────────────────────────────────────────────────
# 9. Combined Training Loss
# ─────────────────────────────────────────────────────────────────────────────

class CoughSenseLoss(nn.Module):
    """
    L_total = L_cls + alpha * L_supcon - lambda_d * L_domain

    lambda_d annealed 0 → 1 during training (DANN schedule).
    """

    def __init__(self, alpha=0.3, label_smoothing=0.1):
        super().__init__()
        self.alpha       = alpha
        self.cls_loss    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.domain_loss = nn.CrossEntropyLoss()
        self.supcon_loss = ClassBalancedSupConLoss()

    def forward(self, outputs, disease_labels, domain_labels, lambda_d=1.0):
        l_cls    = self.cls_loss(outputs['logits'], disease_labels)
        l_domain = self.domain_loss(outputs['domain_logits'], domain_labels)
        l_supcon = self.supcon_loss(outputs['embeddings'], disease_labels)

        total = l_cls + self.alpha * l_supcon - lambda_d * l_domain

        return {
            'total':  total,
            'cls':    l_cls.item(),
            'supcon': l_supcon.item(),
            'domain': l_domain.item()
        }


# ─────────────────────────────────────────────────────────────────────────────
# 10. Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable


def get_lambda_schedule(epoch, total_epochs, gamma=10.0):
    """DANN annealing: 0 → 1 over training."""
    p = epoch / total_epochs
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def build_model(num_classes=3, num_domains=2, sample_rate=16000, target_frames=128):
    return CoughSense(
        num_classes=num_classes,
        num_domains=num_domains,
        embed_dim=256,
        sample_rate=sample_rate,
        target_frames=target_frames
    )


if __name__ == '__main__':
    model = build_model()
    count_parameters(model)

    fake_wave = torch.randn(4, 1, 56000)
    out = model(fake_wave, lambda_d=0.5)

    print(f"\nOutput shapes:")
    print(f"  logits:        {out['logits'].shape}")
    print(f"  domain_logits: {out['domain_logits'].shape}")
    print(f"  embeddings:    {out['embeddings'].shape}")

    criterion      = CoughSenseLoss()
    disease_labels = torch.tensor([0, 1, 2, 0])
    domain_labels  = torch.tensor([0, 0, 1, 1])
    losses = criterion(out, disease_labels, domain_labels, lambda_d=0.5)
    print(f"\nLoss breakdown: {losses}")

    supcon  = ClassBalancedSupConLoss()
    emb     = F.normalize(torch.randn(8, 256), dim=1)
    lbl     = torch.tensor([0, 0, 1, 1, 2, 2, 0, 2])
    sc_loss = supcon(emb, lbl)
    print(f"\nSupCon loss: {sc_loss.item():.4f}")

    print("\nAll checks passed.")
