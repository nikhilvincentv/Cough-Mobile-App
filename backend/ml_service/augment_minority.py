"""
augment_minority.py — Offline waveform augmentation for CoughSense minority classes.

Paper Section IV-A: augment bronchitis/pneumonia/covid/respiratory_cond to ≥5,000
samples using time-stretch, pitch-shift, and noise injection (waveform domain,
before mel computation). 8× factor for smallest classes.

Augmentation variants per sample:
  0. time_stretch 0.85 (slow down 15%)
  1. time_stretch 0.90
  2. time_stretch 1.10 (speed up 10%)
  3. time_stretch 1.15
  4. pitch_shift  -2.0 semitones
  5. pitch_shift  -1.0 semitone
  6. pitch_shift  +1.0 semitone
  7. gaussian noise (SNR ~25 dB)

Outputs:
  ml_service/augmented_audio/   — saved WAV files for each augmented sample
  ml_service/whisper_mels/      — precomputed whisper mels (reuses existing cache)
  ml_service/v7_aug_data.csv    — original + augmented rows

Usage:
  cd backend
  python3 ml_service/augment_minority.py --csv ml_service/v7_data.csv --out_csv ml_service/v7_aug_data.csv
"""

import os, csv, random, hashlib, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import librosa
import soundfile as sf

# ─── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES   = ['healthy', 'covid', 'respiratory_cond', 'bronchitis', 'pneumonia']
DISEASE_REMAP = {'asthma': 'respiratory_cond'}
SYMPTOM_COLS  = ['sym_fever','sym_cold','sym_cough',
                 'sym_diarrhoea','sym_loss_of_smell','sym_ftg','sym_st']

SR            = 16000
WHISPER_FRAMES = 3000       # 30s at 10ms hop
SAMPLES        = WHISPER_FRAMES * 160  # 480 000 samples = 30s

WHISPER_MEL_DIR = Path(__file__).parent / 'whisper_mels'
AUG_AUDIO_DIR   = Path(__file__).parent / 'augmented_audio'

# Target counts: bring all minority classes to ≥5 000 augmented samples
TARGET = {
    'covid':            5_500,
    'respiratory_cond': 5_500,
    'bronchitis':       5_500,
    'pneumonia':        5_500,
    'healthy':          0,        # no augmentation needed
}

# Augmentation variants (8 types)
AUG_VARIANTS = [
    ('stretch085', lambda y: librosa.effects.time_stretch(y, rate=0.85)),
    ('stretch090', lambda y: librosa.effects.time_stretch(y, rate=0.90)),
    ('stretch110', lambda y: librosa.effects.time_stretch(y, rate=1.10)),
    ('stretch115', lambda y: librosa.effects.time_stretch(y, rate=1.15)),
    ('pitch_m2',   lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=-2.0)),
    ('pitch_m1',   lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=-1.0)),
    ('pitch_p1',   lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=+1.0)),
    ('noise25db',  lambda y: _add_noise(y, snr_db=25)),
]


def _add_noise(y: np.ndarray, snr_db: float = 25) -> np.ndarray:
    sig_power  = np.mean(y ** 2) + 1e-10
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(y)) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


def _whisper_mel_path(audio_path: str) -> Path:
    h = hashlib.md5(audio_path.encode()).hexdigest()[:16]
    return WHISPER_MEL_DIR / f'{h}_whisper.npy'


def _compute_whisper_mel(y: np.ndarray) -> np.ndarray:
    """Compute (80, 3000) whisper-compatible mel spectrogram."""
    T = len(y)
    if T >= SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, SAMPLES - T))

    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=80, n_fft=400, hop_length=160,
        win_length=400, window='hann', center=True, pad_mode='reflect',
        power=2.0, norm='slaney', htk=True)
    mel = mel[:, :WHISPER_FRAMES]
    if mel.shape[1] < WHISPER_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, WHISPER_FRAMES - mel.shape[1])))
    mel = np.log(mel + 1e-6)
    log_max = mel.max()
    mel = np.clip(mel, a_min=log_max - 8.0, a_max=None)
    mel = (mel + 4.0) / 4.0
    return mel.astype(np.float16)


def load_audio(path: str) -> np.ndarray:
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        peak = abs(y).max()
        if peak > 0:
            y = y / (peak + 1e-8)
        return y.astype(np.float32)
    except Exception as e:
        print(f"  WARN: failed to load {path}: {e}")
        return None


def augment_sample(row: dict, variant_name: str, aug_fn, aug_dir: Path) -> dict | None:
    """Apply one augmentation to a sample, save WAV + mel, return new CSV row."""
    src_path = row['audio_path']
    y = load_audio(src_path)
    if y is None:
        return None

    try:
        y_aug = aug_fn(y).astype(np.float32)
        # Clip to avoid distortion
        peak = abs(y_aug).max()
        if peak > 0:
            y_aug = y_aug / (peak + 1e-8)
    except Exception as e:
        print(f"  WARN: aug {variant_name} failed on {src_path}: {e}")
        return None

    # Save WAV
    stem = Path(src_path).stem
    aug_path = aug_dir / f"{stem}_{variant_name}.wav"
    try:
        sf.write(str(aug_path), y_aug, SR)
    except Exception as e:
        print(f"  WARN: could not write {aug_path}: {e}")
        return None

    # Compute + cache whisper mel
    mel_path = _whisper_mel_path(str(aug_path))
    if not mel_path.exists():
        mel = _compute_whisper_mel(y_aug)
        np.save(mel_path, mel)

    new_row = dict(row)
    new_row['audio_path'] = str(aug_path)
    return new_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',     default='ml_service/v7_data.csv')
    ap.add_argument('--out_csv', default='ml_service/v7_aug_data.csv')
    ap.add_argument('--seed',    type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    WHISPER_MEL_DIR.mkdir(exist_ok=True)
    AUG_AUDIO_DIR.mkdir(exist_ok=True)

    # Load original CSV
    all_rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            disease = row.get('disease', '').strip().lower()
            disease = DISEASE_REMAP.get(disease, disease)
            if disease not in CLASS_NAMES:
                continue
            if not Path(row.get('audio_path', '')).exists():
                continue
            row['disease'] = disease
            all_rows.append(row)

    counts = Counter(r['disease'] for r in all_rows)
    print(f"Original dataset: {len(all_rows)} samples")
    for cls in CLASS_NAMES:
        print(f"  {cls:25s}: {counts.get(cls, 0)}")

    # Group by class
    by_class = {cls: [r for r in all_rows if r['disease'] == cls] for cls in CLASS_NAMES}

    aug_rows = []
    for cls, target in TARGET.items():
        if target == 0:
            continue
        current = counts.get(cls, 0)
        need    = max(0, target - current)
        if need == 0:
            print(f"\n{cls}: already has {current} ≥ {target}, skipping")
            continue
        print(f"\n{cls}: {current} → target {target}  (need {need} augmented samples)")

        pool   = by_class[cls]
        done   = 0
        # Cycle through variants and source samples
        variant_cycle = 0
        while done < need:
            src_row     = random.choice(pool)
            vname, vfn  = AUG_VARIANTS[variant_cycle % len(AUG_VARIANTS)]
            variant_cycle += 1

            new_row = augment_sample(src_row, vname, vfn, AUG_AUDIO_DIR)
            if new_row is not None:
                aug_rows.append(new_row)
                done += 1
                if done % 200 == 0:
                    print(f"  {cls}: {done}/{need} done...")

        print(f"  {cls}: generated {done} augmented samples ✓")

    # Write combined CSV
    all_out = all_rows + aug_rows
    random.shuffle(all_out)

    # Ensure fieldnames include all original columns
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_out)

    final_counts = Counter(r['disease'] for r in all_out)
    print(f"\n{'='*50}")
    print(f"Augmented dataset written to: {args.out_csv}")
    print(f"Total samples: {len(all_out)}")
    for cls in CLASS_NAMES:
        print(f"  {cls:25s}: {final_counts.get(cls, 0)}")


if __name__ == '__main__':
    main()
