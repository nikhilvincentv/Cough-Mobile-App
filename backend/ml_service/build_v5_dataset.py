"""
build_v5_dataset.py — Build v5_data.csv for 4-class respiratory disease classifier.

Classes:
  0  healthy             — Coswara healthy (no comorbidities) + CoughVID healthy
  1  covid               — Coswara PCR-confirmed COVID + CoughVID self-reported COVID
  2  asthma              — Coswara resp_illness_not_identified + asthma=True
  3  respiratory_cond    — CoughVID symptomatic (non-COVID respiratory illness)

Sources:
  Coswara   — TRANSFER / TRANSFER 2 / TRANSFER 3 / TRANSFER 4 in ~/Downloads
  CoughVID  — ~/Downloads/coughvid/coughvid_20211012/

Output: ml_service/v5_data.csv

Author: Nikhil Vincent
"""

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
TRANSFERS   = ['TRANSFER', 'TRANSFER 2', 'TRANSFER 3', 'TRANSFER 4']
BASE_ROOT   = Path(os.environ.get('COSWARA_ROOT', Path.home() / 'Downloads'))
COUGHVID_DIR = Path.home() / 'Downloads' / 'coughvid' / 'coughvid_20211012'
OUT_PATH    = Path(__file__).parent / 'v5_data.csv'

COSWARA_LABEL_MAP = {
    'healthy':           'healthy',
    'positive_mild':     'covid',
    'positive_moderate': 'covid',
    'positive_severe':   'covid',
    'positive_asymp':    'covid',
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

# ── Step 1: Coswara audio paths ───────────────────────────────────────────────
print("Scanning Coswara audio files...")
all_audio: dict = defaultdict(dict)
for folder in TRANSFERS:
    audio_dir = BASE_ROOT / folder / 'cough-ai-expo' / 'Coswara-Data' / 'all_audio'
    if not audio_dir.exists():
        continue
    for f in audio_dir.glob('*.wav'):
        parts = f.stem.rsplit('_', 1)
        if len(parts) == 2:
            pid, modality = parts
            if modality not in all_audio[pid]:
                all_audio[pid][modality] = str(f)

print(f"  Found {len(all_audio)} Coswara patients with audio")

# ── Step 2: Coswara metadata ──────────────────────────────────────────────────
print("Scanning Coswara metadata...")
all_meta: dict = {}
for folder in TRANSFERS:
    base = BASE_ROOT / folder / 'cough-ai-expo' / 'Coswara-Data'
    if not base.exists():
        continue
    for date_dir in base.iterdir():
        if not (date_dir.is_dir() and date_dir.name[:4].isdigit()):
            continue
        for csv_file in date_dir.glob('*.csv'):
            try:
                for row in csv.DictReader(open(csv_file, errors='replace')):
                    pid = row.get('id', '').strip()
                    if pid and pid not in all_meta:
                        all_meta[pid] = row
            except Exception:
                pass

print(f"  Found metadata for {len(all_meta)} Coswara patients")

# ── Step 3: Build Coswara rows ────────────────────────────────────────────────
rows_out = []
label_counts = Counter()

for pid, audio_files in all_audio.items():
    if 'cough-heavy' not in audio_files:
        continue
    meta   = all_meta.get(pid, {})
    status = meta.get('covid_status', '').strip()
    asthma_flag = meta.get('asthma', '').strip().lower() in ('true', '1', 'yes')

    # Determine label
    if status in COSWARA_LABEL_MAP:
        label = COSWARA_LABEL_MAP[status]
        # Skip healthy patients who have asthma (they go into asthma class)
        if label == 'healthy' and asthma_flag:
            continue
    elif status == 'resp_illness_not_identified' and asthma_flag:
        label = 'asthma'
    else:
        continue  # skip recovered, under_validation, etc.

    sym = {}
    for col, src_key in SYMPTOM_KEYS.items():
        val = meta.get(src_key, '')
        sym[col] = 1 if str(val).lower() in ('true', '1', 'yes', 'y') else 0

    rows_out.append({
        'disease':            label,
        'audio_path':         audio_files['cough-heavy'],
        'audio_path_shallow': audio_files.get('cough-shallow', ''),
        'source':             'coswara',
        'domain':             0,
        'age':                meta.get('a', ''),
        'gender':             meta.get('g', ''),
        **sym,
    })
    label_counts[label] += 1

print(f"\nCoswara rows: {len(rows_out)}")
print(f"  {dict(label_counts)}")

# ── Step 4: CoughVID rows ─────────────────────────────────────────────────────
print("\nScanning CoughVID...")
coughvid_label_map = {
    'healthy':   'healthy',
    'COVID-19':  'covid',
    'symptomatic': 'respiratory_cond',
}

coughvid_counts = Counter()
coughvid_added  = Counter()

COUGHVID_QUALITY_THRESH = 0.7   # only keep recordings with cough_detected >= threshold

if COUGHVID_DIR.exists():
    meta_path = COUGHVID_DIR / 'metadata_compiled.csv'
    with open(meta_path, errors='replace') as f:
        for row in csv.DictReader(f):
            status = row.get('status', '').strip()
            label  = coughvid_label_map.get(status)
            if not label:
                continue

            # Quality filter
            try:
                cough_det = float(row.get('cough_detected', '0') or '0')
                if cough_det < COUGHVID_QUALITY_THRESH:
                    continue
            except ValueError:
                continue

            uuid = row.get('uuid', '').strip()
            if not uuid:
                continue

            # Find audio file (prefer wav > ogg > webm)
            audio_path = None
            for ext in ('wav', 'ogg', 'webm'):
                p = COUGHVID_DIR / f'{uuid}.{ext}'
                if p.exists():
                    audio_path = str(p)
                    break

            if not audio_path:
                coughvid_counts[label] += 1  # count missing
                continue

            coughvid_added[label] += 1
            rows_out.append({
                'disease':            label,
                'audio_path':         audio_path,
                'audio_path_shallow': '',
                'source':             'coughvid',
                'domain':             1,
                'age':                row.get('age', ''),
                'gender':             row.get('gender', ''),
                'sym_fever':          1 if row.get('fever_muscle_pain','').lower() in ('true','1') else 0,
                'sym_cold':           0,
                'sym_cough':          1,
                'sym_diarrhoea':      0,
                'sym_loss_of_smell':  0,
                'sym_ftg':            0,
                'sym_st':             0,
            })

    print(f"  CoughVID rows added: {dict(coughvid_added)}")
else:
    print(f"  CoughVID not found at {COUGHVID_DIR}")

# ── Step 5: Final stats ───────────────────────────────────────────────────────
final_counts = Counter(r['disease'] for r in rows_out)
print(f"\nFinal dataset: {len(rows_out)} rows")
for cls in ['healthy', 'covid', 'asthma', 'respiratory_cond']:
    print(f"  {cls:25s}: {final_counts.get(cls, 0)}")

# ── Write ─────────────────────────────────────────────────────────────────────
if rows_out:
    fieldnames = list(rows_out[0].keys())
    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nWrote {len(rows_out)} rows → {OUT_PATH}")
else:
    print("ERROR: No rows produced.")
