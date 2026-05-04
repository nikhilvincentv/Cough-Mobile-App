"""
Comprehensive Model Evaluation Script
Tests model on multiple audio files to verify it makes different predictions
"""
import torch
import torch.nn as nn
import librosa
import numpy as np
from pathlib import Path
import sys
import torchaudio
from collections import defaultdict

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
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Convert to torch
        waveform = torch.from_numpy(y).unsqueeze(0)
        
        # Extract mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=64
        )
        mel_spec = mel_transform(waveform)
        
        # Convert to dB
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
        
        return mel_spec_db
        
    except Exception as e:
        print(f"   ❌ Error extracting features: {e}")
        return None

def evaluate_directory(audio_dir, model_path):
    """Evaluate model on all audio files in a directory"""
    print(f"\n{'='*70}")
    print(f"📊 EVALUATING MODEL ON MULTIPLE FILES")
    print(f"{'='*70}\n")
    print(f"Audio directory: {audio_dir}")
    print(f"Model: {model_path}")
    
    # Load model
    print("\n🤖 Loading model...")
    model = CoswaraCNN()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully\n")
    
    # Find all audio files
    audio_files = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.mp3'))
    
    if not audio_files:
        print(f"❌ No audio files found in {audio_dir}")
        print("   Supported formats: .wav, .mp3")
        return
    
    print(f"Found {len(audio_files)} audio files\n")
    print(f"{'='*70}")
    
    # Store results
    results = []
    predictions_class_0 = []
    predictions_class_1 = []
    
    # Process each file
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Testing: {audio_path.name}")
        
        # Extract features
        features = extract_features(audio_path)
        if features is None:
            continue
        
        # Run inference
        with torch.no_grad():
            if features.dim() == 3:
                features = features.unsqueeze(0)
            
            output = model(features)
            probabilities = torch.softmax(output, dim=1)
            
            prob_healthy = probabilities[0][0].item()
            prob_issue = probabilities[0][1].item()
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            predictions_class_0.append(prob_healthy)
            predictions_class_1.append(prob_issue)
            
            results.append({
                'file': audio_path.name,
                'healthy': prob_healthy,
                'respiratory_issue': prob_issue,
                'predicted': 'Healthy' if predicted_class == 0 else 'Respiratory Issue',
                'confidence': max(prob_healthy, prob_issue)
            })
            
            print(f"   Healthy: {prob_healthy:.1%} | Respiratory Issue: {prob_issue:.1%}")
            print(f"   → Predicted: {results[-1]['predicted']} ({results[-1]['confidence']:.1%})")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"📈 SUMMARY STATISTICS")
    print(f"{'='*70}\n")
    
    if not results:
        print("❌ No successful predictions")
        return
    
    print(f"Total files tested: {len(results)}")
    
    # Class distribution
    healthy_count = sum(1 for r in results if r['predicted'] == 'Healthy')
    issue_count = sum(1 for r in results if r['predicted'] == 'Respiratory Issue')
    
    print(f"\n🎯 Predictions Distribution:")
    print(f"   Healthy: {healthy_count} ({healthy_count/len(results):.1%})")
    print(f"   Respiratory Issue: {issue_count} ({issue_count/len(results):.1%})")
    
    # Probability statistics
    print(f"\n📊 Probability Statistics:")
    print(f"   Healthy probabilities:")
    print(f"      Mean: {np.mean(predictions_class_0):.1%}")
    print(f"      Std:  {np.std(predictions_class_0):.1%}")
    print(f"      Min:  {np.min(predictions_class_0):.1%}")
    print(f"      Max:  {np.max(predictions_class_0):.1%}")
    
    print(f"\n   Respiratory Issue probabilities:")
    print(f"      Mean: {np.mean(predictions_class_1):.1%}")
    print(f"      Std:  {np.std(predictions_class_1):.1%}")
    print(f"      Min:  {np.min(predictions_class_1):.1%}")
    print(f"      Max:  {np.max(predictions_class_1):.1%}")
    
    # Check for issues
    print(f"\n🔍 Model Health Check:")
    
    std_class_0 = np.std(predictions_class_0)
    range_class_0 = np.max(predictions_class_0) - np.min(predictions_class_0)
    
    if std_class_0 < 0.05:
        print(f"   ⚠️  WARNING: Very low variation in predictions (std={std_class_0:.4f})")
        print(f"      Model may not be learning meaningful patterns")
    elif std_class_0 < 0.10:
        print(f"   ⚠️  CAUTION: Low variation in predictions (std={std_class_0:.4f})")
        print(f"      Model may be undertrained or overfitted")
    else:
        print(f"   ✅ Good variation in predictions (std={std_class_0:.4f})")
    
    if range_class_0 < 0.10:
        print(f"   ⚠️  WARNING: Very narrow prediction range ({range_class_0:.1%})")
        print(f"      All predictions are very similar")
    else:
        print(f"   ✅ Reasonable prediction range ({range_class_0:.1%})")
    
    # Detailed results
    print(f"\n{'='*70}")
    print(f"📋 DETAILED RESULTS")
    print(f"{'='*70}\n")
    
    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. {result['file']:40s} → {result['predicted']:20s} "
              f"({result['confidence']:.1%})")
        print(f"    Healthy: {result['healthy']:.1%} | Respiratory: {result['respiratory_issue']:.1%}")
    
    print(f"\n{'='*70}")
    print("✅ Evaluation Complete")
    print(f"{'='*70}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <audio_directory>")
        print("\nExample:")
        print("  python evaluate_model.py test_audio/")
        print("  python evaluate_model.py ~/Downloads/cough_samples/")
        print("\nThe directory should contain .wav or .mp3 files")
        sys.exit(1)
    
    audio_dir = Path(sys.argv[1])
    
    if not audio_dir.exists():
        print(f"❌ Directory not found: {audio_dir}")
        sys.exit(1)
    
    if not audio_dir.is_dir():
        print(f"❌ Not a directory: {audio_dir}")
        sys.exit(1)
    
    # Find model
    backend_dir = Path(__file__).parent.parent
    model_path = backend_dir / 'coswara_cnn.pth'
    
    if not model_path.exists():
        model_path = backend_dir / 'cnn.pth'
    
    if not model_path.exists():
        print("❌ No model found!")
        print(f"   Looked for:")
        print(f"   - {backend_dir / 'coswara_cnn.pth'}")
        print(f"   - {backend_dir / 'cnn.pth'}")
        sys.exit(1)
    
    evaluate_directory(audio_dir, model_path)

if __name__ == '__main__':
    main()
