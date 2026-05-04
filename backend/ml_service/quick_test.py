"""
Quick Test Script
Analyzes a single audio file step-by-step to debug prediction issues
"""
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torchaudio

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

def extract_features_detailed(audio_path):
    """Extract features with detailed logging"""
    print(f"\n{'='*70}")
    print(f"🎵 EXTRACTING FEATURES FROM: {audio_path.name}")
    print(f"{'='*70}\n")
    
    # Load audio
    print("1️⃣  Loading audio with librosa...")
    y, sr = librosa.load(audio_path, sr=16000)
    print(f"   ✅ Loaded: {len(y)} samples at {sr} Hz")
    print(f"   Duration: {len(y)/sr:.2f} seconds")
    print(f"   Min: {y.min():.4f}, Max: {y.max():.4f}, Mean: {y.mean():.4f}")
    
    # Convert to torch
    print("\n2️⃣  Converting to torch tensor...")
    waveform = torch.from_numpy(y).unsqueeze(0)
    print(f"   ✅ Waveform shape: {waveform.shape}")
    
    # Extract mel spectrogram
    print("\n3️⃣  Extracting mel spectrogram...")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=64
    )
    mel_spec = mel_transform(waveform)
    print(f"   ✅ Mel spectrogram shape: {mel_spec.shape}")
    print(f"   Min: {mel_spec.min():.4f}, Max: {mel_spec.max():.4f}")
    
    # Convert to dB
    print("\n4️⃣  Converting to dB scale...")
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    print(f"   ✅ dB spectrogram shape: {mel_spec_db.shape}")
    print(f"   Min: {mel_spec_db.min():.2f} dB, Max: {mel_spec_db.max():.2f} dB")
    
    # Normalize
    print("\n5️⃣  Normalizing...")
    mean = mel_spec_db.mean()
    std = mel_spec_db.std()
    mel_spec_normalized = (mel_spec_db - mean) / std
    print(f"   Original mean: {mean:.2f}, std: {std:.2f}")
    print(f"   ✅ Normalized mean: {mel_spec_normalized.mean():.6f}, std: {mel_spec_normalized.std():.6f}")
    
    return mel_spec_normalized, mel_spec_db

def visualize_spectrogram(mel_spec_db, save_path=None):
    """Visualize the mel spectrogram"""
    print("\n6️⃣  Visualizing mel spectrogram...")
    
    plt.figure(figsize=(12, 4))
    plt.imshow(mel_spec_db.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Frequency Bins')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"   ✅ Saved visualization to {save_path}")
    else:
        plt.savefig('/tmp/mel_spec.png')
        print(f"   ✅ Saved visualization to /tmp/mel_spec.png")
    
    plt.close()

def test_audio_file(audio_path, model_path):
    """Test a single audio file"""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING AUDIO FILE")
    print(f"{'='*70}")
    print(f"Audio: {audio_path}")
    print(f"Model: {model_path}")
    
    # Extract features
    features, mel_spec_db = extract_features_detailed(audio_path)
    
    # Visualize
    visualize_spectrogram(mel_spec_db)
    
    # Load model
    print(f"\n{'='*70}")
    print(f"🤖 LOADING MODEL")
    print(f"{'='*70}\n")
    
    model = CoswaraCNN()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully")
    
    # Run inference
    print(f"\n{'='*70}")
    print(f"🔮 RUNNING INFERENCE")
    print(f"{'='*70}\n")
    
    with torch.no_grad():
        # Add batch dimension
        if features.dim() == 3:
            features = features.unsqueeze(0)
        
        print(f"Input shape: {features.shape}")
        
        # Forward pass
        output = model(features)
        print(f"\n📊 Raw model output (logits):")
        print(f"   Class 0 (Healthy): {output[0][0].item():.6f}")
        print(f"   Class 1 (Respiratory Issue): {output[0][1].item():.6f}")
        
        # Apply softmax
        probabilities = torch.softmax(output, dim=1)
        print(f"\n📊 Probabilities (after softmax):")
        print(f"   Class 0 (Healthy): {probabilities[0][0].item():.1%}")
        print(f"   Class 1 (Respiratory Issue): {probabilities[0][1].item():.1%}")
        
        # Prediction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        print(f"\n🎯 FINAL PREDICTION:")
        print(f"   Predicted: {'Healthy' if predicted_class == 0 else 'Respiratory Issue'}")
        print(f"   Confidence: {confidence:.1%}")
        
        # Warning if predictions are too close
        prob_diff = abs(probabilities[0][0].item() - probabilities[0][1].item())
        if prob_diff < 0.1:
            print(f"\n⚠️  WARNING: Predictions are very close!")
            print(f"   Difference: {prob_diff:.1%}")
            print(f"   This suggests the model is uncertain or not well-trained")
    
    print(f"\n{'='*70}")
    print("✅ Test Complete")
    print(f"{'='*70}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py <audio_file.wav>")
        print("\nExample:")
        print("  python quick_test.py test_cough.wav")
        sys.exit(1)
    
    audio_path = Path(sys.argv[1])
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
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
    
    test_audio_file(audio_path, model_path)

if __name__ == '__main__':
    main()
