"""
precompute_mels.py — Convert all audio in v5_data.csv to fixed-size mel tensors.
Saves each as a .npy file next to the audio. Handles wav/ogg/webm via soundfile+ffmpeg.
Run once before training. Takes ~20-40 min for 16k files.
"""
import csv, os, subprocess, tempfile, sys
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
from pathlib import Path

SAMPLE_RATE  = 16_000
CLIP_SAMPLES = int(SAMPLE_RATE * 3.5)   # 56000
N_MELS       = 64
HOP          = 160
N_FFT        = 512
MEL_H        = 64
MEL_W        = 128

CSV_PATH = Path(__file__).parent / 'v5_data.csv'
MEL_DIR  = Path(__file__).parent / 'precomputed_mels'
MEL_DIR.mkdir(exist_ok=True)

mel_fn = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP, n_fft=N_FFT)

def ffmpeg_to_wav(path):
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-ar', str(SAMPLE_RATE),
             '-ac', '1', '-f', 'wav', tmp.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30, check=True)
        return tmp.name
    except Exception:
        try: os.unlink(tmp.name)
        except: pass
        return ''

def load_audio(path):
    if not path or not Path(path).exists():
        return torch.zeros(1, CLIP_SAMPLES)
    tmp_path = ''
    try:
        try:
            data, sr = sf.read(path, always_2d=True)
            w = torch.from_numpy(data.T).float()
        except Exception:
            tmp_path = ffmpeg_to_wav(path)
            if not tmp_path:
                return torch.zeros(1, CLIP_SAMPLES)
            data, sr = sf.read(tmp_path, always_2d=True)
            w = torch.from_numpy(data.T).float()
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
            try: os.unlink(tmp_path)
            except: pass

def audio_to_mel(path):
    w   = load_audio(path)
    spec = mel_fn(w)
    spec = (spec + 1e-6).log()
    mu, sigma = spec.mean(), spec.std() + 1e-8
    spec = (spec - mu) / sigma
    spec = F.interpolate(spec.unsqueeze(0), size=(MEL_H, MEL_W),
                         mode='bilinear', align_corners=False).squeeze(0)
    return spec.numpy().astype(np.float16)   # float16 saves ~half disk space

def mel_out_path(audio_path, suffix='_heavy'):
    h = abs(hash(audio_path)) % 10**9
    return MEL_DIR / f'{h}{suffix}.npy'

# ── Collect all unique audio paths ───────────────────────────────────────────
rows = list(csv.DictReader(open(CSV_PATH)))
paths = {}   # audio_path → out_path
for r in rows:
    for col, suf in [('audio_path', '_heavy'), ('audio_path_shallow', '_shallow')]:
        p = r.get(col, '').strip()
        if p and Path(p).exists():
            out = mel_out_path(p, suf)
            if not out.exists():
                paths[p] = out

total = len(paths)
print(f"Need to precompute {total} mel spectrograms → {MEL_DIR}")
print("(Already computed files are skipped)")

done = 0
errors = 0
for audio_path, out_path in paths.items():
    try:
        mel = audio_to_mel(audio_path)
        np.save(out_path, mel)
        done += 1
        if done % 200 == 0:
            pct = done / total * 100
            print(f"  {done}/{total} ({pct:.0f}%) done, {errors} errors", flush=True)
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"  ERROR {audio_path}: {e}")

print(f"\nDone! {done} mels saved, {errors} errors.")
print(f"Mel dir: {MEL_DIR}  ({sum(1 for _ in MEL_DIR.glob('*.npy'))} files)")
