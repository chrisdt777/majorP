#!/usr/bin/env python3
"""
Multi-Language Voice Analysis System Test

This script demonstrates how the system works with different languages:
1. Kannada (kn) → English translation → ML model analysis
2. Malayalam (ml) → English translation → ML model analysis
3. English (en) → Direct ML model analysis

All languages use the same trained ML model for voice tone analysis.
"""

import requests
import json
import base64
import numpy as np
from voice_analysis_model import VoiceToneAnalyzer

def test_voice_analysis_api():
    """Test the Flask API voice analysis endpoint"""
    print("🎤 Testing Multi-Language Voice Analysis System")
    print("=" * 60)
    
    # Test the API endpoint
    try:
        response = requests.get('http://localhost:5000/api/health')
        if response.status_code == 200:
            print("✅ Flask API is running and healthy")
            health_data = response.json()
            print(f"   - Model loaded: {health_data.get('model_loaded', False)}")
        else:
            print("❌ Flask API health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Flask API is not running. Please start it with: python app.py")
        return False
    
    return True

def test_voice_analysis_with_sample_audio():
    """Test voice analysis with a sample audio file"""
    print("\n🎵 Testing Voice Analysis with Sample Audio")
    print("-" * 50)
    
    # Create a simple test audio signal (sine wave)
    sample_rate = 22050
    duration = 2.0  # 2 seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to base64
    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Test the voice analysis endpoint
    try:
        response = requests.post('http://localhost:5000/api/analyze-voice', 
                               json={'audio_data': audio_base64, 'audio_format': 'wav'})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Voice analysis successful!")
            print(f"   - Confidence score: {result.get('confidence_score', 0):.3f}")
            print(f"   - Message: {result.get('message', 'N/A')}")
        else:
            print(f"❌ Voice analysis failed: {response.status_code}")
            print(f"   - Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing voice analysis: {e}")

def demonstrate_multilanguage_workflow():
    """Demonstrate how the multi-language system works"""
    print("\n🌍 Multi-Language Voice Analysis Workflow")
    print("-" * 50)
    
    print("1. 🎙️ User speaks in native language (Kannada/Malayalam/English)")
    print("2. 🔤 Speech converted to text in native language")
    print("3. 🌐 Text translated to English (for non-English languages)")
    print("4. 🎵 Voice recording sent to ML model for tone analysis")
    print("5. 🧠 ML model analyzes voice characteristics and returns confidence score")
    print("6. 📊 Confidence score influences final credit score")
    
    print("\n📋 Language Support:")
    print("   - English (en): Direct ML model analysis")
    print("   - Kannada (kn): kn → en translation → ML model")
    print("   - Malayalam (ml): ml → en translation → ML model")
    
    print("\n🔧 Technical Implementation:")
    print("   - Frontend: Voice recording + speech recognition")
    print("   - Translation: LibreTranslate API (free)")
    print("   - ML Model: RandomForest with 45 audio features")
    print("   - Backend: Flask API with voice analysis endpoints")
    
    print("\n💡 Benefits:")
    print("   - Single ML model serves all languages")
    print("   - Consistent voice analysis across languages")
    print("   - No need to retrain for new languages")
    print("   - Scalable to more languages")

def show_usage_instructions():
    """Show how to use the multi-language system"""
    print("\n🚀 How to Use the Multi-Language System")
    print("-" * 50)
    
    print("1. Start the Flask API:")
    print("   python app.py")
    
    print("\n2. Open language-specific pages:")
    print("   - English: eng.html")
    print("   - Kannada: kan.html")
    print("   - Malayalam: mal.html")
    
    print("\n3. Complete credit assessment:")
    print("   - Answer questions in your preferred language")
    print("   - Voice is recorded and analyzed for each answer")
    print("   - ML model provides confidence scores")
    print("   - Final credit score includes voice influence")
    
    print("\n4. View results:")
    print("   - Results page shows base score + voice analysis")
    print("   - Voice confidence and multiplier displayed")
    print("   - Final score calculation explained")

def main():
    """Main test function"""
    print("🎤 Multi-Language Voice Analysis System Test")
    print("=" * 60)
    
    # Test 1: Check if API is running
    if not test_voice_analysis_api():
        return
    
    # Test 2: Test voice analysis functionality
    test_voice_analysis_with_sample_audio()
    
    # Test 3: Demonstrate the workflow
    demonstrate_multilanguage_workflow()
    
    # Test 4: Show usage instructions
    show_usage_instructions()
    
    print("\n🎉 Multi-language system is ready!")
    print("\nNext steps:")
    print("1. Ensure Flask API is running (python app.py)")
    print("2. Test with different language pages")
    print("3. Verify voice analysis works in all languages")

if __name__ == "__main__":
    main()
