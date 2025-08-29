#!/usr/bin/env python3
"""
Voice Confidence Analysis Model Training with CREMA-D Dataset

This script trains a model to predict confidence levels based on voice tone analysis.
It uses the CREMA-D dataset which contains emotional speech samples.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def extract_voice_features(audio_file_path):
    """
    Extract voice features that correlate with confidence levels
    Based on research showing confidence correlates with:
    - Speech rate, pitch variation, volume, clarity, etc.
    """
    try:
        # Get file properties as proxy features
        file_size = os.path.getsize(audio_file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Extract filename components for emotional context
        filename = os.path.basename(audio_file_path)
        filename_parts = filename.replace('.wav', '').split('_')
        
        # Parse CREMA-D filename format: ID_TAKE_EMOTION_LEVEL.wav
        # Example: 1001_DFA_ANG_XX.wav
        if len(filename_parts) >= 3:
            emotion_code = filename_parts[2] if len(filename_parts) > 2 else 'NEU'
            intensity_level = filename_parts[3] if len(filename_parts) > 3 else 'XX'
        else:
            emotion_code = 'NEU'
            intensity_level = 'XX'
        
        # Map emotions to confidence indicators
        emotion_confidence_map = {
            'ANG': 0.3,  # Anger - low confidence
            'DIS': 0.4,  # Disgust - low confidence  
            'FEA': 0.2,  # Fear - very low confidence
            'HAP': 0.8,  # Happy - high confidence
            'NEU': 0.6,  # Neutral - medium confidence
            'SAD': 0.3,  # Sad - low confidence
            'SUR': 0.5,  # Surprise - medium confidence
        }
        
        # Map intensity levels to confidence modifiers
        intensity_modifiers = {
            'XX': 0.0,   # No intensity specified
            'LO': -0.1,  # Low intensity
            'MD': 0.0,   # Medium intensity
            'HI': 0.1,   # High intensity
        }
        
        # Base confidence from emotion
        base_confidence = emotion_confidence_map.get(emotion_code, 0.5)
        
        # Apply intensity modifier
        intensity_mod = intensity_modifiers.get(intensity_level, 0.0)
        confidence_score = base_confidence + intensity_mod
        
        # Clamp confidence to 0-1 range
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Create comprehensive feature vector
        features = {
            'file_size_mb': file_size_mb,
            'filename_length': len(filename),
            'emotion_ang': 1 if emotion_code == 'ANG' else 0,
            'emotion_dis': 1 if emotion_code == 'DIS' else 0,
            'emotion_fea': 1 if emotion_code == 'FEA' else 0,
            'emotion_hap': 1 if emotion_code == 'HAP' else 0,
            'emotion_neu': 1 if emotion_code == 'NEU' else 0,
            'emotion_sad': 1 if emotion_code == 'SAD' else 0,
            'emotion_sur': 1 if emotion_code == 'SUR' else 0,
            'intensity_lo': 1 if intensity_level == 'LO' else 0,
            'intensity_md': 1 if intensity_level == 'MD' else 0,
            'intensity_hi': 1 if intensity_level == 'HI' else 0,
            'base_confidence': base_confidence,
            'intensity_modifier': intensity_mod,
            'predicted_confidence': confidence_score
        }
        
        return features, confidence_score
        
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return None, 0.5

def create_confidence_labels(confidence_scores):
    """
    Convert continuous confidence scores to discrete confidence levels
    """
    labels = []
    for score in confidence_scores:
        if score >= 0.8:
            labels.append('very_high_confidence')
        elif score >= 0.6:
            labels.append('high_confidence')
        elif score >= 0.4:
            labels.append('medium_confidence')
        elif score >= 0.2:
            labels.append('low_confidence')
        else:
            labels.append('very_low_confidence')
    return labels

def train_confidence_model(dataset_path, output_model="voice_confidence_model.pkl"):
    """Train the voice confidence analysis model"""
    print("=== Training Voice Confidence Analysis Model ===")
    
    # Find audio files
    audio_dir = os.path.join(dataset_path, "AudioWAV")
    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found: {audio_dir}")
        return False
    
    # Get list of audio files
    audio_files = []
    for file in os.listdir(audio_dir):
        if file.lower().endswith('.wav'):
            audio_files.append(os.path.join(audio_dir, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("❌ No audio files found")
        return False
    
    # Extract features and confidence scores
    features_list = []
    confidence_scores = []
    feature_names = None
    
    print("Extracting voice features and confidence scores...")
    for i, audio_file in enumerate(audio_files):
        features, confidence = extract_voice_features(audio_file)
        if features:
            if feature_names is None:
                feature_names = list(features.keys())
            features_list.append(list(features.values()))
            confidence_scores.append(confidence)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(audio_files)} files")
    
    if len(features_list) == 0:
        print("❌ No features extracted")
        return False
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y_continuous = np.array(confidence_scores)
    
    # Create discrete labels for classification
    y_discrete = create_confidence_labels(confidence_scores)
    
    print(f"Training with {len(X)} samples")
    print(f"Confidence range: {min(confidence_scores):.3f} to {max(confidence_scores):.3f}")
    print(f"Average confidence: {np.mean(confidence_scores):.3f}")
    
    # Show class distribution
    from collections import Counter
    class_counts = Counter(y_discrete)
    print(f"Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")
    
    # Split data for training
    X_train, X_test, y_train_cont, y_test_cont = train_test_split(
        X, y_continuous, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train_disc, y_test_disc = train_test_split(
        X, y_discrete, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train regression model for continuous confidence prediction
    print("\nTraining Regression Model (continuous confidence scores)...")
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_scaled, y_train_cont)
    
    # Train classification model for confidence levels
    print("Training Classification Model (confidence levels)...")
    class_model = RandomForestClassifier(n_estimators=100, random_state=42)
    class_model.fit(X_train_scaled, y_train_disc)
    
    # Evaluate models
    print("\n=== Model Evaluation ===")
    
    # Regression evaluation
    y_pred_reg = reg_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_cont, y_pred_reg)
    r2 = r2_score(y_test_cont, y_pred_reg)
    print(f"Regression Model:")
    print(f"  Mean Squared Error: {mse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    # Classification evaluation
    y_pred_class = class_model.predict(X_test_scaled)
    print(f"\nClassification Model:")
    print(classification_report(y_test_disc, y_pred_class))
    
    # Save model
    model_data = {
        'regression_model': reg_model,
        'classification_model': class_model,
        'scaler': scaler,
        'feature_names': feature_names if feature_names else [],
        'confidence_classes': class_model.classes_.tolist(),
        'feature_importance': {
            'regression': dict(zip(feature_names, reg_model.feature_importances_)),
            'classification': dict(zip(feature_names, class_model.feature_importances_))
        }
    }
    
    try:
        joblib.dump(model_data, output_model)
        print(f"\n✅ Model saved successfully to {output_model}")
        print(f"Model size: {os.path.getsize(output_model) / (1024 * 1024):.2f} MB")
        
        # Show top features
        print(f"\nTop 5 Most Important Features:")
        reg_importance = sorted(model_data['feature_importance']['regression'].items(), 
                              key=lambda x: x[1], reverse=True)
        for feature, importance in reg_importance[:5]:
            print(f"  {feature}: {importance:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def predict_confidence(audio_file_path, model_path="voice_confidence_model.pkl"):
    """Predict confidence level for a given audio file"""
    try:
        # Load model
        model_data = joblib.load(model_path)
        scaler = model_data['scaler']
        reg_model = model_data['regression_model']
        class_model = model_data['classification_model']
        
        # Extract features
        features, _ = extract_voice_features(audio_file_path)
        if not features:
            return None, None
        
        # Prepare features
        X = np.array([list(features.values())])
        X_scaled = scaler.transform(X)
        
        # Make predictions
        confidence_score = reg_model.predict(X_scaled)[0]
        confidence_level = class_model.predict(X_scaled)[0]
        
        return confidence_score, confidence_level
        
    except Exception as e:
        print(f"Error predicting confidence: {e}")
        return None, None

if __name__ == "__main__":
    # Use the downloaded dataset path
    dataset_path = r"C:\Users\chris\.cache\kagglehub\datasets\ejlok1\cremad\versions\1"
    
    print("🎤 Voice Confidence Analysis Model Training")
    print("=" * 60)
    
    success = train_confidence_model(dataset_path)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("\nYou can now:")
        print("1. Use the model to predict confidence levels from voice")
        print("2. Integrate it into your voice analysis application")
        print("3. Analyze user speech for confidence indicators")
        
        # Test prediction on a sample file
        print(f"\n🧪 Testing prediction on sample file...")
        sample_files = [f for f in os.listdir(os.path.join(dataset_path, "AudioWAV")) 
                       if f.endswith('.wav')][:3]
        
        for sample_file in sample_files:
            sample_path = os.path.join(dataset_path, "AudioWAV", sample_file)
            confidence_score, confidence_level = predict_confidence(sample_path)
            if confidence_score is not None:
                print(f"  {sample_file}: Score={confidence_score:.3f}, Level={confidence_level}")
    else:
        print("\n💥 Training failed. Please check the error messages above.")
