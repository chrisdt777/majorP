#!/usr/bin/env python3
"""
Debug Voice Analysis Script

This script helps troubleshoot why voice analysis is always returning 50% (0.5).
"""

import requests
import json
import base64
import numpy as np
import os

def test_api_health():
    """Test if the Flask API is running and healthy"""
    print("🔍 Testing API Health...")
    
    try:
        response = requests.get('http://localhost:5000/api/health')
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is healthy!")
            print(f"   - Status: {health_data.get('status')}")
            print(f"   - Model loaded: {health_data.get('model_loaded')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is it running?")
        print("   Start with: python app.py")
        return False

def test_voice_analysis_endpoint():
    """Test the voice analysis endpoint with sample audio"""
    print("\n🎵 Testing Voice Analysis Endpoint...")
    
    # Create a simple test audio signal
    sample_rate = 22050
    duration = 2.0  # 2 seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to base64
    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    try:
        response = requests.post('http://localhost:5000/api/analyze-voice', 
                               json={'audio_data': audio_base64, 'audio_format': 'wav'})
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Voice analysis successful!")
            print(f"   - Confidence score: {result.get('confidence_score', 0):.3f}")
            print(f"   - Success: {result.get('success')}")
            print(f"   - Message: {result.get('message')}")
            
            # Check if we're getting varied scores
            if result.get('confidence_score') == 0.5:
                print("⚠️  Warning: Getting default confidence score (0.5)")
                print("   This suggests the ML model might not be working properly")
            else:
                print("✅ Getting varied confidence scores - ML model is working!")
                
        else:
            print(f"❌ Voice analysis failed: {response.status_code}")
            print(f"   - Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing voice analysis: {e}")

def check_model_file():
    """Check if the trained model file exists and is valid"""
    print("\n📁 Checking Model File...")
    
    model_path = "voice_tone_model.pkl"
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"✅ Model file exists: {model_path}")
        print(f"   - Size: {file_size:.1f} MB")
        
        if file_size < 1:
            print("⚠️  Warning: Model file seems too small")
        elif file_size > 100:
            print("⚠️  Warning: Model file seems too large")
        else:
            print("✅ Model file size looks reasonable")
    else:
        print("❌ Model file not found: {model_path}")
        print("   Train the model first with: python train_with_kaggle_dataset.py")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        'flask', 'flask-cors', 'librosa', 'scikit-learn', 
        'joblib', 'numpy', 'requests'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            print(f"   Install with: pip install {package}")

def test_multiple_audio_samples():
    """Test with multiple different audio samples"""
    print("\n🎵 Testing Multiple Audio Samples...")
    
    # Test different frequencies to see if we get different confidence scores
    frequencies = [220, 440, 880, 1760]  # Different musical notes
    
    for i, freq in enumerate(frequencies):
        print(f"\n   Sample {i+1}: {freq} Hz (Note: {get_note_name(freq)})")
        
        # Create audio sample
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * freq * t)
        
        # Convert to base64
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        try:
            response = requests.post('http://localhost:5000/api/analyze-voice', 
                                   json={'audio_data': audio_base64, 'audio_format': 'wav'})
            
            if response.status_code == 200:
                result = response.json()
                confidence = result.get('confidence_score', 0)
                print(f"      Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            else:
                print(f"      Failed: {response.status_code}")
                
        except Exception as e:
            print(f"      Error: {e}")

def get_note_name(frequency):
    """Convert frequency to musical note name"""
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    if frequency < 27.5:  # Below lowest A
        return "Very Low"
    
    # Calculate note number from A0 (27.5 Hz)
    note_number = 12 * (np.log2(frequency / 27.5))
    note_index = int(round(note_number)) % 12
    octave = int(note_number / 12)
    
    return f"{notes[note_index]}{octave}"

def main():
    """Main debug function"""
    print("🔍 Voice Analysis Debug Script")
    print("=" * 50)
    
    # Test 1: API Health
    if not test_api_health():
        return
    
    # Test 2: Model File
    check_model_file()
    
    # Test 3: Dependencies
    check_dependencies()
    
    # Test 4: Voice Analysis Endpoint
    test_voice_analysis_endpoint()
    
    # Test 5: Multiple Audio Samples
    test_multiple_audio_samples()
    
    print("\n🎯 Debug Summary:")
    print("If you're always getting 50% confidence scores:")
    print("1. Check if the ML model is properly loaded")
    print("2. Verify the audio data is being sent correctly")
    print("3. Check the Flask API logs for errors")
    print("4. Ensure the model was trained with the CREMA-D dataset")
    print("5. Try retraining the model: python train_with_kaggle_dataset.py")

if __name__ == "__main__":
    main()
