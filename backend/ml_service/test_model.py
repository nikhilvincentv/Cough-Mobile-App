"""
Test your models on Coswara dataset
"""
import torch
import torch.nn as nn
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Model architectures (same as app.py)
class CoughCNN(nn.Module):
    """Architecture for cough_model.pth - simpler model with 8 outputs"""
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

class CoswaraCNN(nn.Module):
    """Architecture for coswara_cnn.pth"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((16, 16))
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def extract_features(audio_path):
    """Extract mel spectrogram features"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, fmax=8000
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # Convert to tensor [1, 1, 64, time]
        features = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def test_model(model_path, labels_csv, audio_dir, model_type='coswara'):
    """Test model on labeled data"""
    print(f"\n{'='*60}")
    print(f"Testing {model_path.name}")
    print(f"{'='*60}\n")
    
    # Load model
    if model_type == 'cough':
        model = CoughCNN()
    else:
        model = CoswaraCNN()
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load labels
    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} labeled samples from {labels_csv.name}")
    
    # Test on subset (first 50 samples for speed)
    test_samples = min(50, len(df))
    df = df.head(test_samples)
    
    predictions = []
    true_labels = []
    
    print(f"\nTesting on {test_samples} samples...")
    
    for idx, row in df.iterrows():
        # Get filename - handle different column names
        if 'FILENAME' in row:
            audio_file = row['FILENAME'].strip()
        elif 'id' in row:
            audio_file = row['id'] + '.wav'
        else:
            continue
        
        # Add .wav if not present
        if not audio_file.endswith('.wav'):
            audio_file += '.wav'
            
        audio_path = audio_dir / audio_file
        
        if not audio_path.exists():
            continue
        
        # Extract features
        features = extract_features(audio_path)
        if features is None:
            continue
        
        # Get prediction
        with torch.no_grad():
            output = model(features)
            pred = torch.argmax(output, dim=1).item()
        
        predictions.append(pred)
        
        # Get true label from QUALITY column (good=0, bad=1) or other columns
        if 'QUALITY' in row:
            true_label = 0 if row['QUALITY'].strip().lower() == 'good' else 1
        elif 'label' in row:
            true_label = row['label']
        elif 'covid_status' in row:
            true_label = 1 if 'positive' in str(row['covid_status']).lower() else 0
        else:
            true_label = 0  # default to healthy
        
        true_labels.append(true_label)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{test_samples} samples...")
    
    # Calculate metrics
    if len(predictions) > 0:
        accuracy = accuracy_score(true_labels, predictions)
        print(f"\n✅ Accuracy: {accuracy:.2%}")
        
        print("\n📊 Classification Report:")
        print(classification_report(true_labels, predictions, zero_division=0))
        
        print("\n🔢 Confusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
    else:
        print("❌ No valid predictions made")

def main():
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    coswara_dir = base_dir / 'Coswara-Data'
    audio_dir = coswara_dir / 'all_audio'
    annotations_dir = coswara_dir / 'annotations'
    
    cough_model_path = base_dir / 'backend' / 'cough_model.pth'
    coswara_model_path = base_dir / 'backend' / 'coswara_cnn.pth'
    
    print("🧪 TESTING YOUR MODELS ON COSWARA DATASET")
    print("=" * 60)
    
    # Test cough model
    if cough_model_path.exists():
        cough_labels = annotations_dir / 'cough-heavy_labels.csv'
        if cough_labels.exists():
            test_model(cough_model_path, cough_labels, audio_dir, model_type='cough')
        else:
            print(f"⚠️ Labels not found: {cough_labels}")
    else:
        print(f"⚠️ Model not found: {cough_model_path}")
    
    # Test coswara model
    if coswara_model_path.exists():
        breathing_labels = annotations_dir / 'breathing-deep_labels.csv'
        if breathing_labels.exists():
            test_model(coswara_model_path, breathing_labels, audio_dir, model_type='coswara')
        else:
            print(f"⚠️ Labels not found: {breathing_labels}")
    else:
        print(f"⚠️ Model not found: {coswara_model_path}")
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
