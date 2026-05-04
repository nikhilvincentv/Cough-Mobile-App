"""
Smart prediction adjustment to amplify disease signals
When disease probability exceeds healthy, increase the confidence
"""

import numpy as np

def adjust_predictions(predictions, mode='conservative'):
    """
    Adjust predictions to amplify disease signals when they exceed healthy
    
    Args:
        predictions: dict of {disease: probability}
        mode: 'conservative', 'moderate', or 'aggressive'
    
    Returns:
        Adjusted predictions dict
    """
    healthy_prob = predictions.get('healthy', 0)
    
    # Find highest disease probability (excluding healthy)
    disease_probs = {k: v for k, v in predictions.items() if k != 'healthy'}
    
    if not disease_probs:
        return predictions
    
    max_disease = max(disease_probs.items(), key=lambda x: x[1])
    max_disease_name, max_disease_prob = max_disease
    
    # Only adjust if disease probability is higher than healthy
    if max_disease_prob <= healthy_prob:
        return predictions  # No adjustment needed
    
    # Calculate the "confidence gap" - how much disease exceeds healthy
    confidence_gap = max_disease_prob - healthy_prob
    
    # Define amplification factors based on mode
    amplification_factors = {
        'conservative': {
            'threshold': 0.05,  # 5% gap needed
            'boost': 1.3        # 30% boost
        },
        'moderate': {
            'threshold': 0.03,  # 3% gap needed
            'boost': 1.5        # 50% boost
        },
        'aggressive': {
            'threshold': 0.01,  # 1% gap needed
            'boost': 2.0        # 100% boost
        }
    }
    
    config = amplification_factors.get(mode, amplification_factors['moderate'])
    
    # Only amplify if gap exceeds threshold
    if confidence_gap < config['threshold']:
        return predictions
    
    # Create adjusted predictions
    adjusted = predictions.copy()
    
    # Amplify the disease signal
    boost_factor = config['boost']
    
    # Apply sigmoid-based boost (stronger for larger gaps)
    # This prevents over-amplification
    sigmoid_boost = 1 + (boost_factor - 1) * (1 / (1 + np.exp(-10 * (confidence_gap - 0.1))))
    
    # Boost disease probability
    adjusted[max_disease_name] = min(max_disease_prob * sigmoid_boost, 0.99)
    
    # Reduce healthy probability proportionally
    reduction_factor = adjusted[max_disease_name] - max_disease_prob
    adjusted['healthy'] = max(healthy_prob - reduction_factor, 0.01)
    
    # Normalize to sum to 1.0
    total = sum(adjusted.values())
    adjusted = {k: v / total for k, v in adjusted.items()}
    
    return adjusted

def apply_confidence_threshold(predictions, threshold=0.6):
    """
    If top prediction exceeds threshold, amplify it further
    
    Args:
        predictions: dict of {disease: probability}
        threshold: confidence threshold (e.g., 0.6 = 60%)
    
    Returns:
        Adjusted predictions
    """
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_disease, top_prob = sorted_preds[0]
    
    if top_prob < threshold:
        return predictions  # Below threshold, no adjustment
    
    # Calculate how much above threshold
    excess = top_prob - threshold
    
    # Amplify based on excess (max 20% boost)
    boost = min(excess * 0.5, 0.2)
    
    adjusted = predictions.copy()
    adjusted[top_disease] = min(top_prob + boost, 0.99)
    
    # Redistribute the boost from other predictions
    other_total = sum(v for k, v in predictions.items() if k != top_disease)
    if other_total > 0:
        reduction_per_class = boost / len([k for k in predictions.keys() if k != top_disease])
        for disease in predictions.keys():
            if disease != top_disease:
                adjusted[disease] = max(predictions[disease] - reduction_per_class, 0.001)
    
    # Normalize
    total = sum(adjusted.values())
    adjusted = {k: v / total for k, v in adjusted.items()}
    
    return adjusted

def smart_adjust(predictions, config=None):
    """
    Smart adjustment combining multiple strategies
    
    Args:
        predictions: dict of {disease: probability}
        config: dict with adjustment settings
            {
                'mode': 'conservative' | 'moderate' | 'aggressive',
                'threshold': float (0-1),
                'enable_disease_boost': bool,
                'enable_confidence_boost': bool
            }
    
    Returns:
        Adjusted predictions with metadata
    """
    if config is None:
        config = {
            'mode': 'moderate',
            'threshold': 0.6,
            'enable_disease_boost': True,
            'enable_confidence_boost': True
        }
    
    original = predictions.copy()
    adjusted = predictions.copy()
    adjustments_made = []
    
    # Step 1: Amplify disease signals
    if config.get('enable_disease_boost', True):
        adjusted = adjust_predictions(adjusted, mode=config.get('mode', 'moderate'))
        if adjusted != original:
            adjustments_made.append('disease_signal_amplified')
    
    # Step 2: Apply confidence threshold boost
    if config.get('enable_confidence_boost', True):
        threshold = config.get('threshold', 0.6)
        adjusted = apply_confidence_threshold(adjusted, threshold=threshold)
        if adjusted != predictions:
            adjustments_made.append('confidence_threshold_applied')
    
    return {
        'original_predictions': original,
        'adjusted_predictions': adjusted,
        'adjustments_made': adjustments_made,
        'adjustment_config': config
    }

# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("🧪 PREDICTION ADJUSTMENT TESTING")
    print("="*60)
    
    # Test case 1: COVID slightly higher than healthy
    test1 = {
        'healthy': 0.48,
        'covid': 0.52,
        'asthma': 0.0,
        'copd': 0.0,
        'bronchitis': 0.0
    }
    
    print("\n📊 Test 1: COVID 52% vs Healthy 48%")
    print("Original:", test1)
    
    result = smart_adjust(test1, {'mode': 'moderate', 'threshold': 0.6})
    print("Adjusted:", result['adjusted_predictions'])
    print("Changes:", result['adjustments_made'])
    
    # Test case 2: COVID much higher (like your real test)
    test2 = {
        'healthy': 0.016,
        'covid': 0.984,
        'asthma': 0.0,
        'copd': 0.0,
        'bronchitis': 0.0
    }
    
    print("\n📊 Test 2: COVID 98.4% vs Healthy 1.6%")
    print("Original:", test2)
    
    result = smart_adjust(test2, {'mode': 'moderate', 'threshold': 0.6})
    print("Adjusted:", result['adjusted_predictions'])
    print("Changes:", result['adjustments_made'])
    
    # Test case 3: Healthy dominant (your recording)
    test3 = {
        'healthy': 0.74,
        'covid': 0.10,
        'asthma': 0.08,
        'copd': 0.05,
        'bronchitis': 0.03
    }
    
    print("\n📊 Test 3: Healthy 74% vs COVID 10%")
    print("Original:", test3)
    
    result = smart_adjust(test3, {'mode': 'moderate', 'threshold': 0.6})
    print("Adjusted:", result['adjusted_predictions'])
    print("Changes:", result['adjustments_made'])
    
    print("\n" + "="*60)
