"""
build_full_dataset.py — Build full_data.csv from all Coswara TRANSFER folders.

Scans:
  ~/Downloads/TRANSFER/
  ~/Downloads/TRANSFER 2/
  ~/Downloads/TRANSFER 3/
  ~/Downloads/TRANSFER 4/

Each must contain:
  cough-ai-expo/Coswara-Data/all_audio/   ← WAV files named <pid>_<modality>.wav
  cough-ai-expo/Coswara-Data/<date>/<date>.csv  ← per-date metadata with symptoms

Output: ml_service/full_data.csv

Author: Nikhil Vincent
"""

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
TRANSFERS    = ['TRANSFER', 'TRANSFER 2', 'TRANSFER 3', 'TRANSFER 4']
BASE_ROOT    = Path(os.environ.get('COSWARA_ROOT', Path.home() / 'Downloads'))
LABEL_MAP    = {
    'healthy':          'healthy',
    'positive_mild':    'covid',
    'positive_moderate':'covid',
    'positive_severe':  'covid',
    'positive_asymp':   'covid',
}
SYMPTOM_KEYS = {
    'sym_fever':         'fever',
    'sym_cold':          'cold',
    'sym_cough':         'cough',
    'sym_diarrhoea':     'diarrhoea',
    'sym_loss_of_smell': 'loss_of_smell',
    'sym_ftg':           'ftg',
    'sym_st':            'st',
}
OUT_PATH = Path(__file__).parent / 'full_data.csv'

# ── Step 1: Collect audio paths ───────────────────────────────────────────────
print("Scanning audio files...")
all_audio: dict[str, dict[str, str]] = defaultdict(dict)
for folder in TRANSFERS:
    audio_dir = BASE_ROOT / folder / 'cough-ai-expo' / 'Coswara-Data' / 'all_audio'
    if not audio_dir.exists():
        continue
    for f in audio_dir.glob('*.wav'):
        parts = f.stem.rsplit('_', 1)
        if len(parts) == 2:
            pid, modality = parts
            if modality not in all_audio[pid]:   # prefer first TRANSFER found
                all_audio[pid][modality] = str(f)

print(f"  Found {len(all_audio)} unique patients with audio")
mod_counts = Counter(m for mods in all_audio.values() for m in mods)
print(f"  Modalities: {dict(mod_counts)}")

# ── Step 2: Collect metadata from per-date CSVs ───────────────────────────────
print("Scanning metadata...")
all_meta: dict[str, dict] = {}
for folder in TRANSFERS:
    base = BASE_ROOT / folder / 'cough-ai-expo' / 'Coswara-Data'
    if not base.exists():
        continue
    for date_dir in base.iterdir():
        if not (date_dir.is_dir() and date_dir.name[:4].isdigit()):
            continue
        for csv_file in date_dir.glob('*.csv'):
            try:
                for row in csv.DictReader(open(csv_file)):
                    pid = row.get('id', '').strip()
                    if pid and pid not in all_meta:
                        all_meta[pid] = row
            except Exception:
                pass

print(f"  Found metadata for {len(all_meta)} patients")

# ── Step 3: Build labeled rows ────────────────────────────────────────────────
rows_out = []
for pid, audio_files in all_audio.items():
    if 'cough-heavy' not in audio_files:
        continue
    meta   = all_meta.get(pid, {})
    status = meta.get('covid_status', '').strip()
    label  = LABEL_MAP.get(status)
    if not label:
        continue

    sym = {}
    for col, src_key in SYMPTOM_KEYS.items():
        val = meta.get(src_key, '')
        sym[col] = 1 if str(val).lower() in ('true', '1', 'yes', 'y') else 0

    rows_out.append({
        'disease':            label,
        'audio_path':         audio_files['cough-heavy'],
        'audio_path_shallow': audio_files.get('cough-shallow', ''),
        'source':             'coswara',
        'age':                meta.get('a', ''),
        'gender':             meta.get('g', ''),
        **sym,
    })

label_counts = Counter(r['disease'] for r in rows_out)
print(f"\nFinal dataset: {len(rows_out)} samples")
print(f"  Labels: {dict(label_counts)}")
print(f"  With shallow cough: {sum(1 for r in rows_out if r['audio_path_shallow'])}")
print(f"  With age metadata:  {sum(1 for r in rows_out if r['age'])}")

# ── Write ─────────────────────────────────────────────────────────────────────
if rows_out:
    fieldnames = list(rows_out[0].keys())
    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nWrote {len(rows_out)} rows → {OUT_PATH}")
else:
    print("No samples found — check TRANSFER folder paths.")
