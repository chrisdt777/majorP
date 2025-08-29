from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import librosa
import io
import base64
import wave
import tempfile
import os
from voice_analysis_model import VoiceToneAnalyzer
import json

app = Flask(__name__)
CORS(app)

# Initialize the voice analyzer
voice_analyzer = VoiceToneAnalyzer()

# Try to load pre-trained model if available
model_path = "voice_tone_model.pkl"
if os.path.exists(model_path):
    voice_analyzer.load_model(model_path)
    print("Pre-trained model loaded successfully")
else:
    print("No pre-trained model found. Please train the model first.")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/analyze-voice', methods=['POST'])
def analyze_voice():
    """Analyze voice tone from audio data"""
    try:
        data = request.get_json()
        
        if not data or 'audio_data' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Decode base64 audio data
        audio_base64 = data['audio_data']
        audio_bytes = base64.b64decode(audio_base64.split(',')[1])
        
        # Convert to numpy array
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Load audio using librosa
            audio_data, sr = librosa.load(temp_file_path, sr=22050)
            
            # Analyze voice tone
            confidence_score = voice_analyzer.analyze_voice_tone(audio_data, sr)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return jsonify({
                'success': True,
                'confidence_score': confidence_score,
                'message': 'Voice analysis completed successfully'
            })
            
        except Exception as e:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'confidence_score': 0.5
        }), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train the voice tone analysis model with provided data"""
    try:
        data = request.get_json()
        
        if not data or 'audio_files' not in data or 'labels' not in data:
            return jsonify({'error': 'Audio files and labels are required'}), 400
        
        audio_files = data['audio_files']
        labels = data['labels']
        
        if len(audio_files) != len(labels):
            return jsonify({'error': 'Number of audio files must match number of labels'}), 400
        
        # Train the model
        success = voice_analyzer.train_model(audio_files, labels)
        
        if success:
            # Save the trained model
            voice_analyzer.save_model("voice_tone_model.pkl")
            
            return jsonify({
                'success': True,
                'message': 'Model trained and saved successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to train model'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/credit-score', methods=['POST'])
def calculate_credit_score():
    """Calculate credit score including voice confidence"""
    try:
        data = request.get_json()
        
        if not data or 'responses' not in data:
            return jsonify({'error': 'Responses data is required'}), 400
        
        responses = data['responses']
        voice_confidence = data.get('voice_confidence', 0.5)
        
        # Calculate base credit score (existing logic)
        score = 0
        
        for index, response in enumerate(responses):
            if index == 0:  # Full name
                score += 10 if len(response) > 0 else 0
            elif index == 1:  # Employment status
                if 'employed' in response.lower() or 'full-time' in response.lower():
                    score += 20
                elif 'part-time' in response.lower() or 'contract' in response.lower():
                    score += 15
                elif 'self-employed' in response.lower():
                    score += 18
                else:
                    score += 5
            elif index == 2:  # Annual income
                import re
                income = int(re.sub(r'[^0-9]', '', response) or '0')
                if income > 1000000:
                    score += 25
                elif income > 500000:
                    score += 20
                elif income > 300000:
                    score += 15
                elif income > 100000:
                    score += 10
                else:
                    score += 5
            elif index == 3:  # Existing loans/debts
                score += 20 if 'no' in response.lower() else 10
            elif index == 4:  # Credit score
                import re
                credit_score = int(re.sub(r'[^0-9]', '', response) or '0')
                if credit_score > 750:
                    score += 25
                elif credit_score > 650:
                    score += 20
                elif credit_score > 550:
                    score += 15
                else:
                    score += 5
            elif index == 5:  # Purpose of credit
                score += 20 if 'business' in response.lower() else 15
            elif index == 6:  # Amount requested
                import re
                amount = int(re.sub(r'[^0-9]', '', response) or '0')
                if amount < 100000:
                    score += 20
                elif amount < 500000:
                    score += 15
                elif amount < 1000000:
                    score += 10
                else:
                    score += 5
            elif index == 7:  # Residence status
                score += 20 if 'own' in response.lower() else 10
            elif index == 8:  # Duration at current job
                duration = int(response) if response.isdigit() else 0
                score += min(10, duration // 2) if duration > 0 else 0
            elif index == 9:  # Additional income
                score += 10 if 'yes' in response.lower() else 5
        
        # Apply voice confidence multiplier
        voice_multiplier = 0.8 + (voice_confidence * 0.4)  # Range: 0.8 to 1.2
        final_score = int(score * voice_multiplier)
        
        # Ensure score is within 0-100 range
        final_score = max(0, min(100, final_score))
        
        return jsonify({
            'success': True,
            'base_score': score,
            'voice_confidence': voice_confidence,
            'voice_multiplier': voice_multiplier,
            'final_score': final_score
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': voice_analyzer.model is not None
    })

@app.route('/<path:filename>')
def serve_file(filename):
    """Serve static files - must be after API routes to avoid conflicts"""
    return send_from_directory('.', filename)

if __name__ == '__main__':
    print("Starting Voice Analysis API Server...")
    print("Make sure to install required dependencies:")
    print("pip install flask flask-cors librosa scikit-learn joblib numpy")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 