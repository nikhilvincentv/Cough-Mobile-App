"""
Simple Audio Recorder
Records audio samples for testing the model
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import sys

def record_audio(duration=3, sample_rate=16000):
    """Record audio from microphone"""
    print(f"\n🎤 Recording for {duration} seconds...")
    print("   (Make your cough sound now!)")
    
    # Record
    audio = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, 
                   channels=1, 
                   dtype='float32')
    sd.wait()
    
    print("✅ Recording complete!")
    return audio, sample_rate

def save_audio(audio, sample_rate, filename):
    """Save audio to file"""
    sf.write(filename, audio, sample_rate)
    print(f"💾 Saved to: {filename}")

def main():
    print("\n" + "="*70)
    print("🎙️  AUDIO RECORDER FOR MODEL TESTING")
    print("="*70)
    
    # Get output directory
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path('test_audio')
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print("\nThis tool will help you record multiple audio samples to test the model.")
    print("You can record coughs, breathing, or any other sounds.\n")
    
    sample_num = 1
    
    while True:
        print(f"\n{'='*70}")
        print(f"Sample #{sample_num}")
        print(f"{'='*70}")
        
        # Get description
        description = input("\nDescribe this sample (e.g., 'healthy_cough', 'sick_cough'): ").strip()
        if not description:
            description = f"sample_{sample_num}"
        
        # Clean filename
        filename = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in description)
        filename = f"{filename}.wav"
        filepath = output_dir / filename
        
        # Record
        input("\nPress ENTER when ready to record...")
        audio, sr = record_audio(duration=3)
        
        # Save
        save_audio(audio, sr, filepath)
        
        # Continue?
        print(f"\n{'='*70}")
        choice = input("\nRecord another? (y/n): ").strip().lower()
        if choice != 'y':
            break
        
        sample_num += 1
    
    print(f"\n{'='*70}")
    print(f"✅ Recorded {sample_num} samples")
    print(f"📁 Saved to: {output_dir}")
    print(f"\nTo test these samples, run:")
    print(f"   python3 evaluate_model.py {output_dir}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have sounddevice and soundfile installed:")
        print("   pip install sounddevice soundfile")
