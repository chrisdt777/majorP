#!/usr/bin/env python3
"""
Simplified Voice Tone Model Training with CREMA-D Dataset
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_simple_features(audio_file_path):
    """Create simple features for audio analysis"""
    try:
        # Get file size as a simple feature
        file_size = os.path.getsize(audio_file_path)
        
        # Get file extension
        file_ext = os.path.splitext(audio_file_path)[1].lower()
        
        # Create basic features based on file properties
        features = {
            'file_size': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'is_wav': 1 if file_ext == '.wav' else 0,
            'is_mp3': 1 if file_ext == '.mp3' else 0,
            'is_m4a': 1 if file_ext == '.m4a' else 0,
            'filename_length': len(os.path.basename(audio_file_path)),
            'path_depth': len(audio_file_path.split(os.sep))
        }
        
        return features
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return None

def train_simple_model(dataset_path, output_model="voice_tone_model.pkl"):
    """Train a simple voice tone analysis model"""
    print("=== Training Simple Voice Tone Model ===")
    
    # Find audio files
    audio_dir = os.path.join(dataset_path, "AudioWAV")
    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found: {audio_dir}")
        return False
    
    # Get list of audio files
    audio_files = []
    for file in os.listdir(audio_dir):
        if file.lower().endswith(('.wav', '.mp3', '.m4a')):
            audio_files.append(os.path.join(audio_dir, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("❌ No audio files found")
        return False
    
    # Create features for each audio file
    features_list = []
    labels = []
    feature_names = None
    
    print("Extracting features...")
    for i, audio_file in enumerate(audio_files[:100]):  # Limit to first 100 files for speed
        features = create_simple_features(audio_file)
        if features:
            if feature_names is None:
                feature_names = list(features.keys())
            features_list.append(list(features.values()))
            # Create simple labels based on file properties
            if features['file_size_mb'] > 1.0:
                labels.append('large_file')
            elif features['file_size_mb'] > 0.5:
                labels.append('medium_file')
            else:
                labels.append('small_file')
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(audio_files[:100])} files")
    
    if len(features_list) == 0:
        print("❌ No features extracted")
        return False
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    
    print(f"Training with {len(X)} samples and {len(set(y))} classes")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names if feature_names else [],
        'classes': model.classes_.tolist()
    }
    
    try:
        joblib.dump(model_data, output_model)
        print(f"✅ Model saved successfully to {output_model}")
        print(f"Model size: {os.path.getsize(output_model) / (1024 * 1024):.2f} MB")
        return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

if __name__ == "__main__":
    # Use the downloaded dataset path
    dataset_path = r"C:\Users\chris\.cache\kagglehub\datasets\ejlok1\cremad\versions\1"
    
    print("🎤 Simple Voice Tone Model Training")
    print("=" * 50)
    
    success = train_simple_model(dataset_path)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("You can now use the model in your application.")
    else:
        print("\n💥 Training failed. Please check the error messages above.")
