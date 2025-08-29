#!/usr/bin/env python3
"""
Quick Accuracy Check for Voice Tone Model
"""

import joblib
import numpy as np

def quick_accuracy_check():
    """Quick check of the trained model"""
    print("🔍 Quick Accuracy Check for Voice Tone Model")
    print("=" * 50)
    
    try:
        # Load the trained model
        model_data = joblib.load('voice_tone_model.pkl')
        model = model_data['model']
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model type: {type(model).__name__}")
        print(f"🌳 Number of trees: {model.n_estimators}")
        print(f"🎯 Classes: {list(model.classes_)}")
        
        # Show feature importance (top 5)
        if hasattr(model, 'feature_importances_'):
            print("\n🏆 Top 5 Most Important Features:")
            feature_names = model_data['feature_names']
            feature_importance = list(zip(feature_names, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(feature_importance[:5]):
                print(f"  {i+1}. {feature}: {importance:.4f}")
        
        print("\n📈 Expected Performance:")
        print("   - Random Forest typically achieves 70-90% accuracy")
        print("   - Your model has 45 features and 4 classes")
        print("   - Training on 7,442 audio files from CREMA-D dataset")
        
        print("\n💡 To see EXACT accuracy:")
        print("   1. Wait for current training to complete")
        print("   2. Look for 'Model accuracy: X.XXX' in output")
        print("   3. Or check your previous training console logs")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    quick_accuracy_check()

