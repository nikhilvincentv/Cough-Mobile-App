# CoughSense — Research Overview

## Novel ML Contributions

This project introduces **CoughSense**, a dual-branch CNN-Transformer architecture for
cross-dataset respiratory disease classification from cough audio. Three contributions
distinguish it from all published work as of 2025:

### 1. Dual-Branch CNN + Transformer with Attention Gating
Prior work uses either a CNN (local patterns) or a Transformer (global context) — never
both in parallel on multi-scale mel features. CoughSense runs both branches simultaneously
on three mel resolutions (fine/medium/coarse) and fuses them via learned attention gating.

### 2. Domain-Adversarial Training (Gradient Reversal)
Cross-dataset generalization is the open problem: AUC drops from 0.82 to 0.53 when a
model trained on Coswara is tested on CoughVID (Islam et al. 2025). CoughSense adds a
domain classifier trained adversarially via gradient reversal (Ganin et al. 2016),
forcing the feature extractor to learn domain-invariant representations. This is the
first application of DANN to multi-class multi-dataset cough audio.

### 3. Class-Balanced Supervised Contrastive Loss (CBS-SupCon)
Standard SupCon (Khosla et al. 2020) uses one global temperature for all classes.
CoughSense uses per-class temperatures:
- COVID-19 (658 samples, minority): τ = 0.05 — hardest push
- Healthy (1,414 samples):          τ = 0.08
- Bronchitis (2,590 samples):       τ = 0.10 — softest push

This is novel. No prior SupCon paper varies temperature per class.

---

## Target Journals (non-high-school, high-impact)

| Journal | Impact Factor | Review Time | Fit |
|---------|--------------|-------------|-----|
| **IEEE JBHI** (Journal of Biomedical and Health Informatics) | 7.7 | ~2.5 months | Primary target |
| **Computers in Biology and Medicine** (Elsevier) | 7.0 | ~6 weeks | Strong fit |
| **Biomedical Signal Processing and Control** (Elsevier) | 5.1 | ~8 weeks | Good fit |
| **PLOS ONE** | 3.7 | ~4 weeks | Fallback, open access |

---

## Data

| Dataset | Source | Healthy | COVID-19 | Bronchitis |
|---------|--------|---------|----------|------------|
| Coswara | IISc Bangalore (GitHub) | 1,414 | 658 | — |
| CoughVID | EPFL / Zenodo | — | ~1,010 | 2,590 |

**Total: ~5,672 samples across 2 datasets, 3 diseases.**

After targeted augmentation (COVID 6×, others 2×): ~12,000 training samples.

---

## Files

```
backend/ml_service/
  coughsense_model.py      # Architecture: dual-branch CNN+Transformer
  train_coughsense.py      # Training pipeline with 5-fold CV
  coughsense_service.py    # Flask inference service (drop-in for app.py)
```

---

## Quick Start

### Train (using existing CSVs — no dataset download needed)
```bash
cd backend
npm run train:coughsense
```

### Train (with raw audio datasets)
```bash
cd backend/ml_service
python3 train_coughsense.py \
  --coswara_dir /path/to/Coswara-Data/all_audio \
  --coughvid_dir /path/to/coughvid \
  --epochs 60
```

### Run the new ML service
```bash
cd backend
npm run ml
# or directly:
python3 backend/ml_service/coughsense_service.py
```

Checkpoints are saved to `backend/ml_service/checkpoints/fold{N}_best.pt`.
The service auto-loads the best checkpoint at startup.

---

## Architecture Summary

```
Waveform (16kHz mono)
       │
       ▼
MultiScaleMelExtractor
  ├── Fine mel   (hop=128, n_mels=128)
  ├── Medium mel (hop=160, n_mels=64)
  └── Coarse mel (hop=256, n_mels=32)
  → Stack as 3-channel image (B, 3, 64, 128)
       │
   ┌───┴───┐
   ▼       ▼
CNNBranch  TransformerBranch
(ResNet    (ViT-style patch
 + SE)      embed + 3-layer
            encoder)
   │              │
   └──────┬───────┘
          ▼
    AttentionFusion (learned gate)
          │
    ┌─────┴─────┐
    ▼           ▼
DiseaseHead  DomainClassifier
(3-class)    (2-class, adversarial
              via GRL)
```

**Loss:** `L_total = L_cls + 0.3·L_supcon − λ_d·L_domain`

λ_d annealed 0→1 using DANN schedule over training epochs.
