#!/usr/bin/env python3
"""
Test script for the Voice Confidence Analysis Model
"""

import os
import joblib
from confidence_voice_model import predict_confidence, extract_voice_features

def test_confidence_model():
    """Test the confidence model with sample files"""
    print("🧪 Testing Voice Confidence Analysis Model")
    print("=" * 50)
    
    # Check if model exists
    model_path = "voice_confidence_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please run confidence_voice_model.py first to train the model.")
        return
    
    # Load model info
    try:
        model_data = joblib.load(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"Model size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
        print(f"Features: {len(model_data['feature_names'])}")
        print(f"Confidence classes: {model_data['confidence_classes']}")
        
        # Show feature importance
        print(f"\nTop 5 Most Important Features:")
        reg_importance = sorted(model_data['feature_importance']['regression'].items(), 
                              key=lambda x: x[1], reverse=True)
        for feature, importance in reg_importance[:5]:
            print(f"  {feature}: {importance:.4f}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with sample files from CREMA-D dataset
    dataset_path = r"C:\Users\chris\.cache\kagglehub\datasets\ejlok1\cremad\versions\1\AudioWAV"
    
    if os.path.exists(dataset_path):
        print(f"\n🎵 Testing with CREMA-D dataset files...")
        
        # Get sample files with different emotions
        sample_files = []
        for file in os.listdir(dataset_path):
            if file.endswith('.wav'):
                if any(emotion in file for emotion in ['ANG', 'HAP', 'NEU', 'SAD', 'FEA']):
                    sample_files.append(file)
                if len(sample_files) >= 10:  # Limit to 10 samples
                    break
        
        print(f"Testing {len(sample_files)} sample files:")
        print("-" * 60)
        
        for i, filename in enumerate(sample_files, 1):
            file_path = os.path.join(dataset_path, filename)
            
            # Extract features manually
            features, manual_confidence = extract_voice_features(file_path)
            
            # Predict using the model
            predicted_score, predicted_level = predict_confidence(file_path, model_path)
            
            if predicted_score is not None:
                print(f"{i:2d}. {filename}")
                print(f"    Manual: {manual_confidence:.3f} | Model: {predicted_score:.3f} | Level: {predicted_level}")
                
                # Show emotion analysis
                if features:
                    emotion = "Unknown"
                    for key, value in features.items():
                        if key.startswith('emotion_') and value == 1:
                            emotion = key.replace('emotion_', '').upper()
                            break
                    print(f"    Emotion: {emotion}")
                print()
    
    else:
        print(f"\n❌ Dataset path not found: {dataset_path}")
    
    # Test with a hypothetical file
    print("🎭 Testing with hypothetical scenarios:")
    print("-" * 40)
    
    # Create test scenarios
    test_scenarios = [
        ("confident_speech.wav", "HAP", "HI"),  # Happy, High intensity
        ("nervous_speech.wav", "FEA", "LO"),    # Fear, Low intensity
        ("neutral_speech.wav", "NEU", "MD"),    # Neutral, Medium intensity
        ("angry_speech.wav", "ANG", "HI"),      # Anger, High intensity
        ("sad_speech.wav", "SAD", "LO"),        # Sad, Low intensity
    ]
    
    for scenario_name, emotion, intensity in test_scenarios:
        # Create mock features
        mock_features = {
            'file_size_mb': 0.5,
            'filename_length': len(scenario_name),
            'emotion_ang': 1 if emotion == 'ANG' else 0,
            'emotion_dis': 1 if emotion == 'DIS' else 0,
            'emotion_fea': 1 if emotion == 'FEA' else 0,
            'emotion_hap': 1 if emotion == 'HAP' else 0,
            'emotion_neu': 1 if emotion == 'NEU' else 0,
            'emotion_sad': 1 if emotion == 'SAD' else 0,
            'emotion_sur': 1 if emotion == 'SUR' else 0,
            'intensity_lo': 1 if intensity == 'LO' else 0,
            'intensity_md': 1 if intensity == 'MD' else 0,
            'intensity_hi': 1 if intensity == 'HI' else 0,
            'base_confidence': 0.5,  # Default
            'intensity_modifier': 0.0,  # Default
            'predicted_confidence': 0.5,  # Default
        }
        
        # Calculate expected confidence
        emotion_confidence_map = {
            'ANG': 0.3, 'DIS': 0.4, 'FEA': 0.2, 'HAP': 0.8, 
            'NEU': 0.6, 'SAD': 0.3, 'SUR': 0.5
        }
        intensity_modifiers = {'XX': 0.0, 'LO': -0.1, 'MD': 0.0, 'HI': 0.1}
        
        base_conf = emotion_confidence_map.get(emotion, 0.5)
        intensity_mod = intensity_modifiers.get(intensity, 0.0)
        expected_confidence = max(0.0, min(1.0, base_conf + intensity_mod))
        
        print(f"📝 {scenario_name}")
        print(f"    Emotion: {emotion} | Intensity: {intensity}")
        print(f"    Expected Confidence: {expected_confidence:.3f}")
        
        # Determine confidence level
        if expected_confidence >= 0.8:
            level = "very_high_confidence"
        elif expected_confidence >= 0.6:
            level = "high_confidence"
        elif expected_confidence >= 0.4:
            level = "medium_confidence"
        elif expected_confidence >= 0.2:
            level = "low_confidence"
        else:
            level = "very_low_confidence"
        
        print(f"    Confidence Level: {level}")
        print()

if __name__ == "__main__":
    test_confidence_model()
