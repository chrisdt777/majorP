#!/usr/bin/env python3
"""
Check Model Accuracy Script

This script loads the trained voice tone analysis model and evaluates its performance.
"""

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def load_and_evaluate_model():
    """Load the trained model and evaluate its performance"""
    print("🔍 Loading Trained Voice Tone Analysis Model...")
    
    try:
        # Load the trained model
        model_data = joblib.load('voice_tone_model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model type: {type(model).__name__}")
        print(f"🔢 Number of features: {len(feature_names)}")
        print(f"🎯 Classes: {model.classes_}")
        
        # Show model parameters
        if hasattr(model, 'n_estimators'):
            print(f"🌳 Number of trees: {model.n_estimators}")
        
        # Show feature importance (top 10)
        if hasattr(model, 'feature_importances_'):
            print("\n🏆 Top 10 Most Important Features:")
            feature_importance = list(zip(feature_names, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
        # If we have test data, we can show cross-validation or other metrics
        print("\n📈 Model Performance Metrics:")
        print("   - This model was trained with RandomForestClassifier")
        print("   - Uses 20% of data for testing (test_size=0.2)")
        print("   - Features are standardized using StandardScaler")
        
        # Show class distribution from training
        if hasattr(model, 'classes_'):
            print(f"\n🎭 Class Distribution:")
            for class_name in model.classes_:
                print(f"   - {class_name}")
        
        print("\n💡 To see actual accuracy numbers, you would need to:")
        print("   1. Run the training script again to see the output")
        print("   2. Or test with new audio files")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_with_sample_audio():
    """Test the model with a sample audio file if available"""
    print("\n🎵 Testing Model with Sample Audio...")
    
    # Check if there are any audio files in the audio directory
    audio_dir = "audio"
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        
        if audio_files:
            print(f"Found {len(audio_files)} audio files for testing")
            print("Sample files:")
            for i, file in enumerate(audio_files[:5]):
                print(f"  {i+1}. {file}")
            
            print("\nTo test accuracy with these files, you can:")
            print("1. Run: python app.py")
            print("2. Use the web interface to upload audio files")
            print("3. Or create a test script to analyze multiple files")
        else:
            print("No audio files found in audio/ directory")
    else:
        print("No audio/ directory found")

if __name__ == "__main__":
    print("🎤 Voice Tone Analysis Model - Accuracy Check")
    print("=" * 50)
    
    # Load and evaluate the model
    success = load_and_evaluate_model()
    
    if success:
        # Test with sample audio if available
        test_with_sample_audio()
        
        print("\n🎯 Next Steps:")
        print("1. To see training accuracy: Check the console output from training")
        print("2. To test with new audio: Use the web interface (python app.py)")
        print("3. To retrain and see metrics: Run python train_with_kaggle_dataset.py")
    else:
        print("\n❌ Could not load model. Please ensure the model file exists.")
