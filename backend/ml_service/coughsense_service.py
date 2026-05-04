"""
CoughSense ML Inference Service
Drop-in upgrade for backend/ml_service/app.py

Loads the trained CoughSense dual-branch model and serves predictions
through the same /analyze and /health endpoints the existing backend expects.

Backward compatible: returns the same JSON schema as the original app.py
so the Node.js backend requires zero changes.

New additions to response:
  - embedding_norm: quality signal (well-separated = higher)
  - model_version: "coughsense-v1"
  - architecture: description of the model

Author: Nikhil Vincent
"""

from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import os
import sys
import tempfile
import librosa
from pathlib import Path

# Allow importing coughsense_model from same directory
sys.path.insert(0, str(Path(__file__).parent))

try:
    from coughsense_model import build_model
    COUGHSENSE_AVAILABLE = True
except ImportError:
    COUGHSENSE_AVAILABLE = False
    print("[WARN] coughsense_model not found — falling back to legacy model")

from prediction_adjuster import smart_adjust

app = Flask(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE  = 16000
CLIP_SAMPLES = int(SAMPLE_RATE * 3.5)
CLASS_NAMES  = ['Healthy', 'COVID-19', 'Bronchitis']
AUDIO_TYPES  = [
    'breathing-deep', 'breathing-shallow',
    'cough-heavy', 'cough-shallow',
    'vowel-a', 'vowel-e', 'vowel-o',
    'counting-normal', 'counting-fast'
]

# ─── Model Loading ────────────────────────────────────────────────────────────

models = {}


def load_coughsense():
    if not COUGHSENSE_AVAILABLE:
        return False

    search_dirs = [
        Path(__file__).parent.parent / "checkpoints",
        Path(__file__).parent / "checkpoints",
    ]

    ckpt_path = None
    for d in search_dirs:
        if d.is_dir():
            ckpts = sorted(d.glob("fold*_best.pt"))
            if ckpts:
                best, best_f1 = None, 0.0
                for ck in ckpts:
                    try:
                        data = torch.load(ck, map_location='cpu')
                        f1   = data.get('best_val_f1', 0.0)
                        if f1 > best_f1:
                            best_f1, best = f1, ck
                    except Exception:
                        pass
                if best:
                    ckpt_path = best
                    break

    if ckpt_path is None:
        print("[CoughSense] No trained checkpoint found.")
        print("  Train first: python backend/ml_service/train_coughsense.py")
        model = build_model()
        model.eval()
        models['coughsense']         = model
        models['coughsense_trained'] = False
        return True

    try:
        data  = torch.load(ckpt_path, map_location='cpu')
        model = build_model()
        model.load_state_dict(data['model_state'])
        model.eval()
        models['coughsense']         = model
        models['coughsense_trained'] = True
        f1 = data.get('best_val_f1', 0.0)
        print(f"[CoughSense] Loaded {ckpt_path.name} (val F1={f1:.4f})")
        return True
    except Exception as e:
        print(f"[CoughSense] Failed to load checkpoint: {e}")
        return False


def load_legacy_models():
    from app import load_models as legacy_load
    try:
        legacy_load()
        models['legacy_loaded'] = True
        print("[Legacy] Original models loaded as fallback")
    except Exception as e:
        print(f"[Legacy] Could not load original models: {e}")


# ─── Audio Preprocessing ──────────────────────────────────────────────────────

def preprocess_audio(audio_path):
    try:
        wave, sr = torchaudio.load(audio_path)
    except Exception:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        wave  = torch.tensor(y).unsqueeze(0)

    if sr != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)

    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)

    T = wave.shape[-1]
    if T >= CLIP_SAMPLES:
        start = (T - CLIP_SAMPLES) // 2
        wave  = wave[..., start:start + CLIP_SAMPLES]
    else:
        wave = F.pad(wave, (0, CLIP_SAMPLES - T))

    peak = wave.abs().max()
    if peak > 0:
        wave = wave / (peak + 1e-8)

    return wave.unsqueeze(0)   # (1, 1, CLIP_SAMPLES)


def assess_quality(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        if duration < 0.8:
            return 'bad', 'Too short (< 0.8 sec)'
        rms = float(np.mean(librosa.feature.rms(y=y)))
        if rms < 0.01:
            return 'bad', 'Audio too quiet'
        if np.max(np.abs(y)) > 0.99:
            return 'fair', 'Possible clipping'
        sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        if sc < 400:
            return 'fair', 'Low frequency content'
        return 'good', 'Quality OK'
    except Exception as e:
        return 'unknown', str(e)


# ─── Inference ────────────────────────────────────────────────────────────────

def run_coughsense(waveform_tensor):
    model = models.get('coughsense')
    if model is None:
        return None, None

    with torch.no_grad():
        out      = model(waveform_tensor, lambda_d=0.0)
        probs    = F.softmax(out['logits'], dim=1)[0]
        emb_norm = out['embeddings'][0].norm().item()

    results = [[CLASS_NAMES[i], float(probs[i])] for i in range(len(CLASS_NAMES))]
    results.sort(key=lambda x: x[1], reverse=True)
    return results, emb_norm


def run_legacy_demo(audio_type):
    import random
    if 'cough' in audio_type:
        base = {'Healthy': 0.45, 'COVID-19': 0.20, 'Bronchitis': 0.25,
                'Asthma / COPD': 0.07, 'Common Cold': 0.03}
    elif 'breathing' in audio_type:
        base = {'Healthy': 0.55, 'Asthma / COPD': 0.25, 'COVID-19': 0.10,
                'Bronchitis': 0.05, 'Common Cold': 0.05}
    else:
        base = {'Healthy': 0.65, 'COVID-19': 0.10, 'Asthma / COPD': 0.10,
                'Bronchitis': 0.05, 'Common Cold': 0.10}
    j = {k: max(0.02, v + random.uniform(-0.05, 0.05)) for k, v in base.items()}
    total = sum(j.values())
    results = [[k, v / total] for k, v in j.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':               'ok',
        'model_version':        'coughsense-v1' if 'coughsense' in models else 'legacy',
        'model_trained':        models.get('coughsense_trained', False),
        'models_loaded':        [k for k in models if not isinstance(models[k], bool)],
        'audio_types_supported': AUDIO_TYPES,
        'classes':              CLASS_NAMES,
        'architecture':         'Dual-Branch CNN-Transformer + Domain-Adversarial Training'
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_type = request.form.get('audio_type', 'cough-heavy')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_file.save(tmp.name)
        temp_path = tmp.name

    try:
        quality, quality_msg = assess_quality(temp_path)

        if 'coughsense' in models:
            waveform    = preprocess_audio(temp_path)
            predictions, emb_norm = run_coughsense(waveform)
            model_used  = 'coughsense-v1'
        else:
            predictions = run_legacy_demo(audio_type)
            emb_norm    = None
            model_used  = 'legacy-demo'

        try:
            y, sr = librosa.load(temp_path, sr=16000)
            scalar_feats = {
                'rms':               float(np.mean(librosa.feature.rms(y=y))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'zcr':               float(np.mean(librosa.feature.zero_crossing_rate(y))),
            }
        except Exception:
            scalar_feats = {}

        top = predictions[0]
        response = {
            'disease':      top[0],
            'confidence':   round(top[1] * 100, 2),
            'audio_type':   audio_type,
            'predictions':  predictions,
            'quality': {
                'rating':       quality,
                'message':      quality_msg,
                'should_retake': quality == 'bad'
            },
            'model_used':     model_used,
            'model_version':  'coughsense-v1',
            'audio_features': scalar_feats,
        }

        if emb_norm is not None:
            response['embedding_norm'] = round(emb_norm, 4)

        print(f"[/analyze] {audio_type} → {top[0]} ({top[1]*100:.1f}%) [{model_used}]")
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    results = []

    for audio_type in AUDIO_TYPES:
        if audio_type not in request.files:
            continue

        audio_file = request.files[audio_type]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            temp_path = tmp.name

        try:
            quality, quality_msg = assess_quality(temp_path)

            if 'coughsense' in models:
                waveform    = preprocess_audio(temp_path)
                predictions, _ = run_coughsense(waveform)
            else:
                predictions = run_legacy_demo(audio_type)

            results.append({
                'audio_type':  audio_type,
                'predictions': predictions,
                'quality': {
                    'rating':       quality,
                    'message':      quality_msg,
                    'should_retake': quality == 'bad'
                }
            })
        except Exception as e:
            print(f"[batch] Error on {audio_type}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    agg = {}
    for r in results:
        for cls, prob in r['predictions']:
            agg.setdefault(cls, []).append(prob)
    agg_mean  = {k: float(np.mean(v)) for k, v in agg.items()}
    aggregate = sorted([[k, v] for k, v in agg_mean.items()], key=lambda x: x[1], reverse=True)

    return jsonify({
        'individual_results':    results,
        'aggregate_predictions': aggregate,
        'total_analyzed':        len(results),
        'model_version':         'coughsense-v1'
    })


# ─── Startup ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("CoughSense ML Inference Service v1")
    print("=" * 60)

    if not load_coughsense():
        print("[Fallback] Attempting to load legacy models...")
        load_legacy_models()

    if not any(isinstance(v, torch.nn.Module) for v in models.values()):
        print("[WARN] No models loaded — running in demo mode")

    print(f"\nModels: {list(models.keys())}")
    print("Service starting on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=False)
