import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import requests
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

def load_dataset_info():
    """Load dataset information and paths"""
    base_dir = Path("/Users/nvb/Desktop/cough-ai-expo")
    dataset_info = {
        'healthy': {
            'path': base_dir / "Coswara-Data" / "all_audio",
            'pattern': "*_cough-heavy.wav",
            'count': 0
        },
        'covid': {
            'path': base_dir / "Coswara-Data" / "all_audio",
            'pattern': "*_cough-heavy.wav",  # Adjust pattern if needed
            'count': 0
        },
        'bronchitis': {
            'path': base_dir / "Coswara-Data" / "all_audio",
            'pattern': "*_cough-heavy.wav",  # Adjust pattern if needed
            'count': 0
        }
    }
    return dataset_info

def find_audio_files(dataset_info):
    """Find all audio files for each class"""
    for cls, info in dataset_info.items():
        pattern = str(info['path'] / info['pattern'])
        files = glob.glob(pattern)
        info['files'] = files
        info['count'] = len(files)
    return dataset_info

def evaluate_model(dataset_info, sample_size=50):
    """Evaluate model on test data"""
    results = {
        'true_labels': [],
        'pred_labels': [],
        'files': [],
        'predictions': []
    }
    
    # Process each class
    for cls, info in dataset_info.items():
        print(f"\n🔍 Processing {cls} samples...")
        files = info['files'][:sample_size]  # Limit number of samples per class
        
        for audio_file in tqdm(files, desc=f"Evaluating {cls}"):
            try:
                with open(audio_file, 'rb') as f:
                    files = {
                        'audio': (os.path.basename(audio_file), f, 'audio/wav'),
                        'audio_type': (None, 'cough-heavy')
                    }
                    
                    response = requests.post(
                        'http://localhost:5000/analyze',
                        files=files,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Handle different response formats
                        if isinstance(result, list):
                            # If result is a list, take the first item if it exists
                            if result and isinstance(result[0], dict):
                                # If list contains dictionaries, take the first prediction
                                pred_dict = result[0]
                                if 'predictions' in pred_dict and isinstance(pred_dict['predictions'], dict):
                                    # If predictions is a dictionary, get the class with highest probability
                                    if pred_dict['predictions']:
                                        pred_class = max(
                                            pred_dict['predictions'].items(),
                                            key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0
                                        )[0]
                                    else:
                                        pred_class = 'unknown'
                                    pred_dict = pred_dict['predictions']
                                else:
                                    # If predictions is not a dictionary, use the first key as class
                                    pred_class = next(iter(pred_dict.keys())) if pred_dict else 'unknown'
                                    pred_dict = {pred_class: 1.0}
                            else:
                                # If list contains strings or other types
                                pred_class = str(result[0]) if result else 'unknown'
                                pred_dict = {pred_class: 1.0}
                        elif isinstance(result, dict):
                            # Original dictionary handling
                            if 'predictions' in result and isinstance(result['predictions'], dict):
                                if result['predictions']:
                                    pred_class = max(
                                        result['predictions'].items(),
                                        key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0
                                    )[0]
                                    pred_dict = result['predictions']
                                else:
                                    pred_class = 'unknown'
                                    pred_dict = {}
                            else:
                                # If no 'predictions' key, try to use the dictionary directly
                                if result:
                                    pred_class = max(
                                        result.items(),
                                        key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0
                                    )[0]
                                    pred_dict = result
                                else:
                                    pred_class = 'unknown'
                                    pred_dict = {}
                        else:
                            # Handle any other response type
                            pred_class = str(result) if result is not None else 'unknown'
                            pred_dict = {pred_class: 1.0}
                        
                        results['true_labels'].append(cls)
                        results['pred_labels'].append(pred_class)
                        results['files'].append(audio_file)
                        results['predictions'].append(pred_dict)
                    
            except Exception as e:
                print(f"\n⚠️ Error processing {audio_file}: {str(e)}")
                # Add placeholder for failed predictions
                results['true_labels'].append(cls)
                results['pred_labels'].append('unknown')
                results['files'].append(audio_file)
                results['predictions'].append({})
    
    return results

def calculate_metrics(results):
    """Calculate performance metrics"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Ensure we have predictions
    if not results['true_labels'] or not results['pred_labels']:
        print("⚠️ No predictions to evaluate!")
        return {
            'overall_accuracy': 0.0,
            'per_class_accuracy': {},
            'classification_report': {},
            'confusion_matrix': [],
            'confusion_matrix_labels': [],
            'total_samples': 0
        }
    
    # Get unique classes, ensuring we have at least one class
    all_labels = sorted(set(results['true_labels'] + results['pred_labels']))
    if not all_labels:
        all_labels = ['unknown']
    
    try:
        # Generate classification report
        report = classification_report(
            results['true_labels'],
            results['pred_labels'],
            target_names=all_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(
            results['true_labels'],
            results['pred_labels'],
            labels=all_labels
        )
    except Exception as e:
        print(f"⚠️ Error calculating metrics: {str(e)}")
        # Return empty metrics on error
        return {
            'overall_accuracy': 0.0,
            'per_class_accuracy': {},
            'classification_report': {},
            'confusion_matrix': [],
            'confusion_matrix_labels': all_labels,
            'total_samples': len(results['true_labels'])
        }
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for cls in all_labels:
        total = sum(1 for t in results['true_labels'] if t == cls)
        correct = sum(1 for t, p in zip(results['true_labels'], results['pred_labels']) 
                     if t == cls and p == cls)
        per_class_acc[cls] = {
            'accuracy': correct / total if total > 0 else 0,
            'samples': total
        }
    
    # Overall accuracy
    correct_predictions = sum(1 for t, p in zip(results['true_labels'], results['pred_labels']) if t == p)
    overall_acc = correct_predictions / len(results['true_labels']) if results['true_labels'] else 0.0
    
    return {
        'overall_accuracy': overall_acc,
        'per_class_accuracy': per_class_acc,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': all_labels,
        'total_samples': len(results['true_labels'])
    }

def generate_report(metrics):
    """Generate a markdown report"""
    report = [
        "# 🧪 Model Evaluation Report",
        "",
        "## 📊 Performance Summary",
        "",
        f"### Overall Accuracy: {metrics['overall_accuracy']*100:.1f}%",
        "",
        "### Per-Class Performance:",
        "```"
    ]
    
    # Add per-class accuracy
    max_name_len = max(len(cls) for cls in metrics['per_class_accuracy'])
    for cls, stats in metrics['per_class_accuracy'].items():
        report.append(
            f"{cls.capitalize():<{max_name_len+2}} {stats['accuracy']*100:5.1f}% ({stats['samples']} samples)"
        )
    report.append("```")
    
    # Add classification report
    report.extend([
        "",
        "## 📈 Detailed Metrics",
        "",
        "### Classification Report",
        "```"
    ])
    
    # Check if classification_report is already a formatted string
    if isinstance(metrics['classification_report'], str):
        report.append(metrics['classification_report'])
    else:
        # Format the classification report manually
        report.append("              precision    recall  f1-score   support\n")
        
        # Add each class metrics
        for label in metrics['confusion_matrix_labels']:
            if label in metrics['classification_report'] and isinstance(metrics['classification_report'][label], dict):
                report.append(f"{label:<15} "
                            f"{metrics['classification_report'][label]['precision']:0.4f}    "
                            f"{metrics['classification_report'][label]['recall']:0.4f}    "
                            f"{metrics['classification_report'][label]['f1-score']:0.4f}    "
                            f"{metrics['classification_report'][label]['support']:>6}")
        
        # Add accuracy
        if 'accuracy' in metrics['classification_report']:
            report.extend([
                "",
                "accuracy" + " "*8 + f"{metrics['classification_report']['accuracy']:0.4f}    "
            ])
    
    # Add confusion matrix
    report.extend([
        "",
        "### Confusion Matrix",
        "```",
        " " * (max_name_len + 2) + " " + " ".join(f"{label:>10}" for label in metrics['confusion_matrix_labels']),
    ])
    
    # Add each row of the confusion matrix
    for i, (true_label, row) in enumerate(zip(metrics['confusion_matrix_labels'], metrics['confusion_matrix'])):
        # Format the true label on the left
        row_str = f"{true_label.capitalize():<{max_name_len+2}}"
        # Add each count in the row
        for count in row:
            row_str += f" {count:10d}"
        report.append(row_str)
    
    report.append("```")
    
    # Add timestamps
    report.extend([
        "",
        "## 📅 Timestamps",
        "",
        f"- Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ])
    
    return "\n".join(report)

def main():
    print("🚀 Starting model evaluation...")
    
    # Check if ML service is running
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code != 200:
            print("❌ ML Service is not running. Please start it first.")
            print("   Command: cd backend/ml_service && python app.py")
            return
    except requests.exceptions.RequestException:
        print("❌ Could not connect to ML Service. Is it running?")
        print("   Command: cd backend/ml_service && python app.py")
        return
    
    # Load and prepare dataset
    print("\n📂 Loading dataset information...")
    dataset_info = load_dataset_info()
    dataset_info = find_audio_files(dataset_info)
    
    # Print dataset statistics
    print("\n📊 Dataset Statistics:")
    for cls, info in dataset_info.items():
        print(f"   - {cls.capitalize()}: {info['count']} files")
    
    # Run evaluation
    print("\n🧪 Starting model evaluation...")
    results = evaluate_model(dataset_info, sample_size=50)  # Adjust sample_size as needed
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Generate and save report
    report = generate_report(metrics)
    with open('model_evaluation_report.md', 'w') as f:
        f.write(report)
    
    print("\n✅ Evaluation complete!")
    print(f"📝 Report saved to: model_evaluation_report.md")
    print("\n📊 Results Summary:")
    print(f"   - Overall Accuracy: {metrics['overall_accuracy']*100:.1f}%")
    for cls, stats in metrics['per_class_accuracy'].items():
        print(f"   - {cls.capitalize()}: {stats['accuracy']*100:.1f}% ({stats['samples']} samples)")

if __name__ == "__main__":
    main()