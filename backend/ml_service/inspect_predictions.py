"""
Quick script to see what the model actually predicts
"""
import torch
import torch.nn as nn
import librosa
import numpy as np
from pathlib import Path

class CoughCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(32, 8)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        features = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
        return features
    except Exception as e:
        print(f"Error: {e}")
        return None

# Load model
model_path = Path('../../backend/cough_model.pth')
model = CoughCNN()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

print("🔍 Testing model predictions on a few samples...\n")

# Find some audio files
audio_dir = Path('../../Coswara-Data/all_audio')
audio_files = list(audio_dir.glob('*.wav'))[:5]

for audio_file in audio_files:
    features = extract_features(audio_file)
    if features is None:
        continue
    
    with torch.no_grad():
        output = model(features)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(output, dim=1).item()
    
    print(f"File: {audio_file.name}")
    print(f"Predicted class: {pred}")
    print(f"Class probabilities:")
    for i, prob in enumerate(probs):
        print(f"  Class {i}: {prob.item():.4f}")
    print()

print("\n💡 INSIGHT:")
print("Your model outputs 8 classes (0-7), not binary (0-1).")
print("The model was likely trained on 8 different disease categories.")
print("We need to map these 8 classes to the test labels properly.")
