"""
5-Class Prediction Adjuster
Corrects for severe class imbalance in training data

Training Distribution:
- Bronchitis: 2,590 samples (53.6%)
- Healthy:    1,414 samples (29.3%)
- COVID:        658 samples (13.6%)
- Asthma:       134 samples ( 2.8%)
- COPD:          25 samples ( 0.5%)

Model learns to over-predict bronchitis. This corrects it.
"""

import numpy as np

# Class frequencies from training data
CLASS_FREQUENCIES = {
    'bronchitis': 0.536,
    'healthy': 0.293,
    'covid': 0.136,
    'asthma': 0.028,
    'copd': 0.005
}

# Inverse frequencies (for rebalancing)
# Higher weight = more boost for under-represented classes
REBALANCE_WEIGHTS = {
    'bronchitis': 1.0 / 0.536,   # 1.87
    'healthy': 1.0 / 0.293,      # 3.41
    'covid': 1.0 / 0.136,        # 7.35
    'asthma': 1.0 / 0.028,       # 35.71
    'copd': 1.0 / 0.005          # 200.00
}

def adjust_for_class_imbalance(predictions, strength=0.5):
    """
    Adjust predictions to correct for training class imbalance
    
    Args:
        predictions: dict of {disease: probability}
        strength: How much to adjust (0=none, 1=full correction)
                 0.5 = moderate correction (recommended)
    
    Returns:
        Adjusted predictions dict
    """
    adjusted = {}
    
    for disease, prob in predictions.items():
        if disease in REBALANCE_WEIGHTS:
            # Apply rebalancing weight
            weight = REBALANCE_WEIGHTS[disease]
            
            # Blend between original and weighted (controlled by strength)
            # strength=0: no change
            # strength=1: full rebalancing
            adjusted_weight = 1.0 + (weight - 1.0) * strength
            
            adjusted[disease] = prob * adjusted_weight
        else:
            adjusted[disease] = prob
    
    # Normalize to sum to 1.0
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    
    return adjusted

def penalize_bronchitis_bias(predictions, penalty=0.7):
    """
    Specifically penalize bronchitis over-prediction
    
    Since model was trained on 54% bronchitis, it over-predicts it.
    This reduces bronchitis confidence and redistributes to other classes.
    
    Args:
        predictions: dict of {disease: probability}
        penalty: How much to reduce bronchitis (0.7 = reduce to 70%)
    
    Returns:
        Adjusted predictions
    """
    if 'bronchitis' not in predictions:
        return predictions
    
    adjusted = predictions.copy()
    bronchitis_prob = predictions['bronchitis']
    
    # Reduce bronchitis probability
    adjusted['bronchitis'] = bronchitis_prob * penalty
    
    # Redistribute the reduction to other classes proportionally
    reduction = bronchitis_prob - adjusted['bronchitis']
    other_diseases = [k for k in predictions.keys() if k != 'bronchitis']
    
    if other_diseases:
        other_total = sum(predictions[k] for k in other_diseases)
        if other_total > 0:
            for disease in other_diseases:
                # Distribute proportionally
                share = predictions[disease] / other_total
                adjusted[disease] = predictions[disease] + (reduction * share)
    
    # Normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    
    return adjusted

def boost_minority_classes(predictions, boost_factor=2.0, threshold=0.05):
    """
    Boost predictions for minority classes (asthma, COPD)
    Only boost if there's already some signal (above threshold)
    
    Args:
        predictions: dict of {disease: probability}
        boost_factor: How much to boost minority classes
        threshold: Minimum probability to boost (default 5%)
    
    Returns:
        Adjusted predictions
    """
    adjusted = predictions.copy()
    
    # Define minority classes
    minority_classes = ['asthma', 'copd']
    
    for disease in minority_classes:
        if disease in adjusted and adjusted[disease] >= threshold:
            # Only boost if there's already some signal
            adjusted[disease] = min(adjusted[disease] * boost_factor, 0.95)
    
    # Normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    
    return adjusted

def smart_adjust_5class(predictions, mode='moderate'):
    """
    Smart adjustment for 5-class predictions
    Combines multiple strategies to correct for class imbalance
    
    Args:
        predictions: dict of {disease: probability}
        mode: 'conservative', 'moderate', or 'aggressive'
    
    Returns:
        dict with adjusted predictions and metadata
    """
    # Configuration for different modes
    configs = {
        'conservative': {
            'imbalance_strength': 0.2,
            'bronchitis_penalty': 0.9,
            'minority_boost': 1.2
        },
        'moderate': {
            'imbalance_strength': 0.3,
            'bronchitis_penalty': 0.8,
            'minority_boost': 1.5
        },
        'aggressive': {
            'imbalance_strength': 0.5,
            'bronchitis_penalty': 0.6,
            'minority_boost': 2.0
        }
    }
    
    config = configs.get(mode, configs['moderate'])
    
    original = predictions.copy()
    adjusted = predictions.copy()
    adjustments_made = []
    
    # Step 1: Correct for overall class imbalance
    adjusted = adjust_for_class_imbalance(
        adjusted, 
        strength=config['imbalance_strength']
    )
    adjustments_made.append('class_imbalance_corrected')
    
    # Step 2: Penalize bronchitis over-prediction
    adjusted = penalize_bronchitis_bias(
        adjusted,
        penalty=config['bronchitis_penalty']
    )
    adjustments_made.append('bronchitis_bias_reduced')
    
    # Step 3: Boost minority classes (asthma, COPD)
    adjusted = boost_minority_classes(
        adjusted,
        boost_factor=config['minority_boost']
    )
    adjustments_made.append('minority_classes_boosted')
    
    # Calculate confidence change
    original_top = max(original.items(), key=lambda x: x[1])
    adjusted_top = max(adjusted.items(), key=lambda x: x[1])
    
    return {
        'original_predictions': original,
        'adjusted_predictions': adjusted,
        'adjustments_made': adjustments_made,
        'original_top': original_top,
        'adjusted_top': adjusted_top,
        'mode': mode
    }

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("🧪 5-CLASS PREDICTION ADJUSTMENT TESTING")
    print("="*70)
    
    # Test case 1: Model predicts bronchitis (typical problem)
    test1 = {
        'healthy': 0.24,
        'covid': 0.10,
        'asthma': 0.05,
        'copd': 0.03,
        'bronchitis': 0.58  # Over-predicted!
    }
    
    print("\n📊 Test 1: Bronchitis Over-Prediction")
    print("Original:", {k: f"{v:.1%}" for k, v in test1.items()})
    
    result = smart_adjust_5class(test1, mode='moderate')
    print("Adjusted:", {k: f"{v:.1%}" for k, v in result['adjusted_predictions'].items()})
    print(f"Changed: {result['original_top'][0]} → {result['adjusted_top'][0]}")
    
    # Test case 2: Healthy sample misclassified as bronchitis
    test2 = {
        'healthy': 0.35,
        'covid': 0.08,
        'asthma': 0.04,
        'copd': 0.02,
        'bronchitis': 0.51
    }
    
    print("\n📊 Test 2: Healthy Misclassified as Bronchitis")
    print("Original:", {k: f"{v:.1%}" for k, v in test2.items()})
    
    result = smart_adjust_5class(test2, mode='moderate')
    print("Adjusted:", {k: f"{v:.1%}" for k, v in result['adjusted_predictions'].items()})
    print(f"Changed: {result['original_top'][0]} → {result['adjusted_top'][0]}")
    
    # Test case 3: COPD sample (minority class)
    test3 = {
        'healthy': 0.15,
        'covid': 0.10,
        'asthma': 0.08,
        'copd': 0.12,  # Small but present
        'bronchitis': 0.55
    }
    
    print("\n📊 Test 3: COPD Sample (Minority Class)")
    print("Original:", {k: f"{v:.1%}" for k, v in test3.items()})
    
    result = smart_adjust_5class(test3, mode='moderate')
    print("Adjusted:", {k: f"{v:.1%}" for k, v in result['adjusted_predictions'].items()})
    print(f"Changed: {result['original_top'][0]} → {result['adjusted_top'][0]}")
    
    # Test case 4: COVID sample
    test4 = {
        'healthy': 0.20,
        'covid': 0.25,
        'asthma': 0.05,
        'copd': 0.03,
        'bronchitis': 0.47
    }
    
    print("\n📊 Test 4: COVID Sample")
    print("Original:", {k: f"{v:.1%}" for k, v in test4.items()})
    
    result = smart_adjust_5class(test4, mode='moderate')
    print("Adjusted:", {k: f"{v:.1%}" for k, v in result['adjusted_predictions'].items()})
    print(f"Changed: {result['original_top'][0]} → {result['adjusted_top'][0]}")
    
    print("\n" + "="*70)
