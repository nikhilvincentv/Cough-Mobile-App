"""
precompute_whisper_mels.py — Precompute 80-mel Whisper-format spectrograms

Produces (80, 3000) float16 .npy files for each audio row in v6_data.csv.
3000 frames = 30 seconds at 10ms hop (Whisper's standard input size).
For 3.5-second cough clips, the remaining ~2650 frames are zero-padded.

Uses librosa (not torchaudio) to avoid version mismatch issues.

Storage: ~480KB/file × 18,164 files ≈ 8GB total.
Run time: ~15-20 minutes on MacBook Air.

Usage:
    python3 ml_service/precompute_whisper_mels.py \
        --csv ml_service/v6_data.csv \
        --out_dir ml_service/whisper_mels
"""

import os, csv, argparse, hashlib as _hashlib, warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings('ignore')

SAMPLE_RATE    = 16_000
WHISPER_N_MELS = 80
WHISPER_N_FFT  = 400   # 25 ms window
WHISPER_HOP    = 160   # 10 ms hop
WHISPER_FRAMES = 3000  # 30 seconds
WHISPER_SAMPLES = WHISPER_FRAMES * WHISPER_HOP  # 480,000 samples = 30s


def load_audio(path: str) -> np.ndarray:
    """Load audio → (WHISPER_SAMPLES,) float32, padded/trimmed to 30s."""
    if not path or not Path(path).exists():
        return np.zeros(WHISPER_SAMPLES, dtype=np.float32)
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        peak = np.abs(y).max()
        if peak > 0:
            y = y / (peak + 1e-8)
        T = len(y)
        if T >= WHISPER_SAMPLES:
            y = y[:WHISPER_SAMPLES]
        else:
            y = np.pad(y, (0, WHISPER_SAMPLES - T))
        return y.astype(np.float32)
    except Exception:
        return np.zeros(WHISPER_SAMPLES, dtype=np.float32)


def wav_to_whisper_mel(y: np.ndarray) -> np.ndarray:
    """(WHISPER_SAMPLES,) waveform → (80, 3000) float16 log-mel array."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=WHISPER_N_MELS,
        n_fft=WHISPER_N_FFT,
        hop_length=WHISPER_HOP,
        win_length=WHISPER_N_FFT,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=2.0,
        norm='slaney',
        htk=True,
    )  # (80, T)
    mel = mel[:, :WHISPER_FRAMES]
    if mel.shape[1] < WHISPER_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, WHISPER_FRAMES - mel.shape[1])))
    mel = np.log(mel + 1e-6)
    # Whisper-style normalisation: clip to [max-8, max], then scale
    log_max = mel.max()
    mel = np.clip(mel, a_min=log_max - 8.0, a_max=None)
    mel = (mel + 4.0) / 4.0
    return mel.astype(np.float16)


def out_path(audio_path: str, out_dir: Path) -> Path:
    h = _hashlib.md5(audio_path.encode()).hexdigest()[:16]
    return out_dir / f'{h}_whisper.npy'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',     default='ml_service/v6_data.csv')
    ap.add_argument('--out_dir', default='ml_service/whisper_mels')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            p = row.get('audio_path', '').strip()
            if p and Path(p).exists():
                rows.append(p)
            p2 = row.get('audio_path_shallow', '').strip()
            if p2 and Path(p2).exists():
                rows.append(p2)

    # Deduplicate
    rows = list(dict.fromkeys(rows))
    print(f"Processing {len(rows)} audio files → {out_dir}")

    done = errors = skipped = 0
    for i, path in enumerate(rows):
        dst = out_path(path, out_dir)
        if dst.exists():
            skipped += 1
            continue
        try:
            y   = load_audio(path)
            arr = wav_to_whisper_mel(y)
            np.save(dst, arr)
            done += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR {path}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)}  done={done} skip={skipped} err={errors}")

    print(f"\nDone. Processed={done}  Skipped={skipped}  Errors={errors}")
    print(f"Output dir: {out_dir}  ({sum(1 for _ in out_dir.glob('*.npy'))} files)")


if __name__ == '__main__':
    main()
