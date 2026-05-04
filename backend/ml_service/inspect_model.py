"""
Model Inspection Tool
Verifies that the saved model has valid weights and correct architecture
"""
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

class CoswaraCNN(nn.Module):
    """Architecture for coswara_cnn.pth - matches training script"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((16, 16))  # fixed output size
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

def inspect_model(model_path):
    """Inspect a saved model file"""
    print(f"\n{'='*70}")
    print(f"🔍 INSPECTING MODEL: {model_path.name}")
    print(f"{'='*70}\n")
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Load state dict
    print("📦 Loading model state dict...")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"✅ Successfully loaded state dict")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return
    
    # Show all layers
    print(f"\n📋 Model Layers ({len(state_dict)} total):")
    print("-" * 70)
    for i, (name, param) in enumerate(state_dict.items(), 1):
        print(f"{i:2d}. {name:40s} Shape: {str(list(param.shape)):20s} "
              f"Mean: {param.mean():.6f}, Std: {param.std():.6f}")
    
    # Check for issues
    print(f"\n🔍 Checking for Common Issues:")
    print("-" * 70)
    
    issues_found = []
    
    # Check for NaN or Inf
    for name, param in state_dict.items():
        if torch.isnan(param).any():
            issues_found.append(f"❌ NaN values in {name}")
        if torch.isinf(param).any():
            issues_found.append(f"❌ Inf values in {name}")
    
    # Check for all zeros
    for name, param in state_dict.items():
        if torch.all(param == 0):
            issues_found.append(f"⚠️  All zeros in {name}")
    
    # Check for suspiciously small/large values
    for name, param in state_dict.items():
        if param.std() < 1e-6:
            issues_found.append(f"⚠️  Very small std ({param.std():.2e}) in {name}")
        if param.std() > 100:
            issues_found.append(f"⚠️  Very large std ({param.std():.2e}) in {name}")
    
    if issues_found:
        for issue in issues_found:
            print(issue)
    else:
        print("✅ No obvious issues found in weights")
    
    # Try loading into model
    print(f"\n🏗️  Loading into Model Architecture:")
    print("-" * 70)
    try:
        model = CoswaraCNN()
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Successfully loaded into CoswaraCNN architecture")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Failed to load into architecture: {e}")
        return
    
    # Test with dummy input
    print(f"\n🧪 Testing with Dummy Input:")
    print("-" * 70)
    try:
        # Create dummy mel-spectrogram: [batch, channels, height, width]
        dummy_input = torch.randn(1, 1, 64, 100)  # 64 mel bins, 100 time frames
        
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {list(dummy_input.shape)}")
        print(f"   Output shape: {list(output.shape)}")
        print(f"   Raw output: {output[0].numpy()}")
        print(f"   Probabilities: {probabilities[0].numpy()}")
        print(f"   Predicted class: {torch.argmax(probabilities, dim=1).item()}")
        
        # Check if predictions are too uniform
        prob_diff = abs(probabilities[0][0].item() - probabilities[0][1].item())
        if prob_diff < 0.1:
            print(f"\n⚠️  WARNING: Predictions are very close ({probabilities[0][0]:.1%} vs {probabilities[0][1]:.1%})")
            print(f"   This suggests the model may not be well-trained")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with multiple random inputs to see variation
    print(f"\n📊 Testing Prediction Variation (10 random inputs):")
    print("-" * 70)
    try:
        predictions_list = []
        with torch.no_grad():
            for i in range(10):
                dummy_input = torch.randn(1, 1, 64, 100)
                output = model(dummy_input)
                probabilities = torch.softmax(output, dim=1)
                pred_class_0 = probabilities[0][0].item()
                predictions_list.append(pred_class_0)
                print(f"   Test {i+1}: Class 0: {pred_class_0:.1%}, Class 1: {probabilities[0][1].item():.1%}")
        
        # Calculate variation
        pred_std = np.std(predictions_list)
        pred_range = max(predictions_list) - min(predictions_list)
        
        print(f"\n   Standard deviation: {pred_std:.4f}")
        print(f"   Range: {pred_range:.4f}")
        
        if pred_std < 0.05:
            print(f"\n⚠️  WARNING: Very low variation in predictions!")
            print(f"   Model may be stuck or not properly trained")
        else:
            print(f"\n✅ Model shows reasonable variation in predictions")
            
    except Exception as e:
        print(f"❌ Variation test failed: {e}")
    
    print(f"\n{'='*70}")
    print("✅ Inspection Complete")
    print(f"{'='*70}\n")

def main():
    # Model paths
    backend_dir = Path(__file__).parent.parent
    coswara_model = backend_dir / 'coswara_cnn.pth'
    user_model = backend_dir / 'cnn.pth'
    
    print("\n🔬 MODEL INSPECTION TOOL")
    print("=" * 70)
    
    # Check which models exist
    if user_model.exists():
        inspect_model(user_model)
    
    if coswara_model.exists():
        inspect_model(coswara_model)
    
    if not user_model.exists() and not coswara_model.exists():
        print("❌ No models found!")
        print(f"   Looked for:")
        print(f"   - {user_model}")
        print(f"   - {coswara_model}")

if __name__ == '__main__':
    main()
