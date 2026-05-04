"""
Simple 5-Class Prediction Adjuster
Just reduces bronchitis over-prediction and boosts the runner-up
"""

def simple_adjust_5class(predictions, bronchitis_reduction=0.7):
    """
    Simple adjustment: Reduce bronchitis, boost second place
    
    Args:
        predictions: dict of {disease: probability}
        bronchitis_reduction: Multiply bronchitis by this (0.7 = reduce to 70%)
    
    Returns:
        Adjusted predictions
    """
    if 'bronchitis' not in predictions:
        return predictions
    
    # Sort by probability
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_disease, top_prob = sorted_preds[0]
    
    # Only adjust if bronchitis is on top
    if top_disease != 'bronchitis':
        return predictions
    
    adjusted = predictions.copy()
    
    # Reduce bronchitis
    adjusted['bronchitis'] = predictions['bronchitis'] * bronchitis_reduction
    
    # Find second place
    if len(sorted_preds) > 1:
        second_disease, second_prob = sorted_preds[1]
        
        # Boost second place with the reduction from bronchitis
        boost = predictions['bronchitis'] - adjusted['bronchitis']
        adjusted[second_disease] = predictions[second_disease] + boost
    
    # Normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    
    return adjusted

# Test it
if __name__ == "__main__":
    print("="*70)
    print("🧪 SIMPLE 5-CLASS ADJUSTMENT TESTING")
    print("="*70)
    
    # Test 1: Bronchitis over-predicted
    test1 = {
        'healthy': 0.24,
        'covid': 0.10,
        'asthma': 0.05,
        'copd': 0.03,
        'bronchitis': 0.58
    }
    
    print("\n📊 Test 1: Bronchitis 58%, Healthy 24%")
    print("Original:", {k: f"{v:.1%}" for k, v in test1.items()})
    adjusted = simple_adjust_5class(test1, bronchitis_reduction=0.7)
    print("Adjusted:", {k: f"{v:.1%}" for k, v in adjusted.items()})
    print(f"Winner: {max(adjusted.items(), key=lambda x: x[1])[0]}")
    
    # Test 2: Healthy should win
    test2 = {
        'healthy': 0.35,
        'covid': 0.08,
        'asthma': 0.04,
        'copd': 0.02,
        'bronchitis': 0.51
    }
    
    print("\n📊 Test 2: Bronchitis 51%, Healthy 35%")
    print("Original:", {k: f"{v:.1%}" for k, v in test2.items()})
    adjusted = simple_adjust_5class(test2, bronchitis_reduction=0.7)
    print("Adjusted:", {k: f"{v:.1%}" for k, v in adjusted.items()})
    print(f"Winner: {max(adjusted.items(), key=lambda x: x[1])[0]}")
    
    # Test 3: COVID should win
    test3 = {
        'healthy': 0.20,
        'covid': 0.25,
        'asthma': 0.05,
        'copd': 0.03,
        'bronchitis': 0.47
    }
    
    print("\n📊 Test 3: Bronchitis 47%, COVID 25%")
    print("Original:", {k: f"{v:.1%}" for k, v in test3.items()})
    adjusted = simple_adjust_5class(test3, bronchitis_reduction=0.7)
    print("Adjusted:", {k: f"{v:.1%}" for k, v in adjusted.items()})
    print(f"Winner: {max(adjusted.items(), key=lambda x: x[1])[0]}")
    
    # Test 4: Bronchitis is actually correct
    test4 = {
        'healthy': 0.10,
        'covid': 0.05,
        'asthma': 0.03,
        'copd': 0.02,
        'bronchitis': 0.80
    }
    
    print("\n📊 Test 4: Bronchitis 80% (should stay)")
    print("Original:", {k: f"{v:.1%}" for k, v in test4.items()})
    adjusted = simple_adjust_5class(test4, bronchitis_reduction=0.7)
    print("Adjusted:", {k: f"{v:.1%}" for k, v in adjusted.items()})
    print(f"Winner: {max(adjusted.items(), key=lambda x: x[1])[0]}")
    
    print("\n" + "="*70)
