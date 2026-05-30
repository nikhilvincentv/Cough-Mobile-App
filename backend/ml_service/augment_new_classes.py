"""
augment_new_classes.py — Augment bronchitis & pneumonia MP3s to meet 500+ threshold
Outputs WAV files + appended CSV rows for v6_data.csv

Augmentation ops (7 per file × N files → 8× total):
  1. original (as WAV)
  2. Gaussian noise (SNR ~15dB)
  3. time stretch 0.88
  4. time stretch 1.12
  5. pitch shift -1.5 semitones
  6. pitch shift +1.5 semitones
  7. time shift +15%
  8. noise + pitch shift +1 (combo)

Usage:
    python3 ml_service/augment_new_classes.py \
        --bronchitis_dir bronchitis_pneumonia_data/bronchitis \
        --pneumonia_dir  bronchitis_pneumonia_data/pneumonia \
        --out_dir        bronchitis_pneumonia_data/augmented \
        --csv_in         ml_service/v5_data.csv \
        --csv_out        ml_service/v6_data.csv
"""

import os, argparse, csv, warnings
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings('ignore')

SR = 16_000


def load_mp3(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / (peak + 1e-8)
    return y.astype(np.float32)


def add_noise(y: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    noise_rms = rms / (10 ** (snr_db / 20))
    noise = np.random.randn(len(y)).astype(np.float32) * noise_rms
    return np.clip(y + noise, -1.0, 1.0)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, semitones: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=semitones)


def time_shift(y: np.ndarray, shift_frac: float = 0.15) -> np.ndarray:
    shift = int(len(y) * shift_frac)
    return np.roll(y, shift)


def normalize(y: np.ndarray) -> np.ndarray:
    peak = np.abs(y).max()
    return y / (peak + 1e-8) if peak > 0 else y


AUGMENTS = [
    ('orig',  lambda y: y),
    ('noisy', lambda y: add_noise(y, snr_db=15.0)),
    ('ts088', lambda y: time_stretch(y, 0.88)),
    ('ts112', lambda y: time_stretch(y, 1.12)),
    ('psm15', lambda y: pitch_shift(y, -1.5)),
    ('psp15', lambda y: pitch_shift(y, +1.5)),
    ('tshft', lambda y: time_shift(y, 0.15)),
    ('combo', lambda y: add_noise(pitch_shift(y, +1.0), snr_db=18.0)),
]


def augment_class(src_dir: Path, out_dir: Path, disease: str, domain: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    mp3s = sorted(src_dir.glob('*.mp3'))
    print(f"  {disease}: {len(mp3s)} source files → {len(mp3s) * len(AUGMENTS)} augmented")

    for mp3 in mp3s:
        try:
            y = load_mp3(str(mp3))
        except Exception as e:
            print(f"    SKIP {mp3.name}: {e}")
            continue

        for tag, fn in AUGMENTS:
            try:
                aug = normalize(fn(y))
            except Exception as e:
                print(f"    AUG FAIL {mp3.stem}_{tag}: {e}")
                aug = y

            out_name = f"{mp3.stem}_{tag}.wav"
            out_path = out_dir / out_name
            sf.write(str(out_path), aug, SR)

            rows.append({
                'disease': disease,
                'audio_path': str(out_path),
                'audio_path_shallow': '',
                'source': 'figshare_west_china',
                'domain': domain,
                'age': '',
                'gender': '',
                'sym_fever': 0,
                'sym_cold': 0,
                'sym_cough': 1,
                'sym_diarrhoea': 0,
                'sym_loss_of_smell': 0,
                'sym_ftg': 0,
                'sym_st': 0,
            })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bronchitis_dir', default='bronchitis_pneumonia_data/bronchitis')
    ap.add_argument('--pneumonia_dir',  default='bronchitis_pneumonia_data/pneumonia')
    ap.add_argument('--out_dir',        default='bronchitis_pneumonia_data/augmented')
    ap.add_argument('--csv_in',         default='ml_service/v5_data.csv')
    ap.add_argument('--csv_out',        default='ml_service/v6_data.csv')
    args = ap.parse_args()

    np.random.seed(42)

    base = Path(__file__).parent.parent
    bronch_dir = base / args.bronchitis_dir
    pneumo_dir = base / args.pneumonia_dir
    out_dir    = base / args.out_dir
    csv_in     = base / args.csv_in
    csv_out    = base / args.csv_out

    print("Augmenting bronchitis...")
    bronch_rows = augment_class(bronch_dir, out_dir / 'bronchitis', 'bronchitis', domain=2)

    print("Augmenting pneumonia...")
    pneumo_rows = augment_class(pneumo_dir, out_dir / 'pneumonia', 'pneumonia', domain=2)

    # Read existing CSV
    with open(csv_in) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing = list(reader)

    # Write v6 CSV
    all_rows = existing + bronch_rows + pneumo_rows
    with open(csv_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nv6_data.csv written: {len(all_rows)} rows")
    print(f"  bronchitis: {len(bronch_rows)}")
    print(f"  pneumonia:  {len(pneumo_rows)}")
    print(f"  existing:   {len(existing)}")
    print(f"  output:     {csv_out}")

    # Class summary
    from collections import Counter
    counts = Counter(r['disease'] for r in all_rows)
    for cls, n in sorted(counts.items()):
        print(f"    {cls}: {n}")


if __name__ == '__main__':
    main()
