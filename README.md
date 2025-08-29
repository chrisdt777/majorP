# Voice Tone Analysis Credit Assessment System

This project combines a credit assessment application with an advanced ML model that analyzes voice tone to provide confidence scores, which are then used to influence credit scoring decisions.

## 🚀 Features

- **Voice Tone Analysis**: ML-powered analysis of voice characteristics
- **Credit Assessment**: Comprehensive credit scoring system
- **Real-time Processing**: Live voice analysis during assessment
- **Multi-language Support**: English, Kannada, and Malayalam
- **Aadhaar Integration**: Secure identity verification
- **Responsive UI**: Modern, user-friendly interface

## 🏗️ Architecture

```
Frontend (HTML/CSS/JS) ←→ Flask API ←→ ML Model (Python)
     ↓                        ↓              ↓
User Interface          Voice Analysis   Credit Scoring
```

## 📋 Prerequisites

- Python 3.8+
- Node.js (optional, for development)
- Microphone access
- Modern web browser (Chrome, Edge, Firefox)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd frontend1
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Additional Dependencies (if needed)
```bash
# For audio processing
pip install soundfile
pip install scipy

# For development
pip install flask-cors
```

## 🎯 Usage

### Starting the ML API Server

1. **Start the Flask server:**
```bash
python app.py
```

2. **Server will run on:** `http://localhost:5000`

3. **API Endpoints:**
   - `GET /` - Main application
   - `POST /api/analyze-voice` - Voice tone analysis
   - `POST /api/train-model` - Train the ML model
   - `POST /api/credit-score` - Calculate credit score
   - `GET /api/health` - Health check

### Training the ML Model

#### Option 1: Using Kaggle Dataset (Recommended)
```bash
# Install kagglehub
pip install kagglehub

# Login to Kaggle
kagglehub login

# Train with the Truth/Deception Detection dataset
python train_with_kaggle_dataset.py
```

#### Option 2: Using the Training Script
```bash
python train_model.py --interactive
```

#### Option 3: Command Line
```bash
python train_model.py --audio_dir /path/to/audio/files --labels_file /path/to/labels.txt
```

#### Option 4: Custom Training
```python
from train_model import train_with_custom_data

audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
labels = [0.8, 0.6, 0.9]  # Confidence scores (0.0 to 1.0)

train_with_custom_data(audio_files, labels)
```

#### Option 5: Test Dataset Download
```bash
# Test the Kaggle dataset download first
python test_dataset.py
```

### Training Data Format

#### Audio Files
- **Supported formats**: WAV, MP3, FLAC, M4A, OGG
- **Recommended**: WAV format, 22050 Hz sample rate
- **Duration**: 1-10 seconds per response

#### Labels
- **Confidence scores**: 0.0 (uncertain) to 1.0 (very confident)
- **Text file format**: One score per line
- **Example:**
```
0.8
0.6
0.9
0.7
0.5
```

## 🔧 Configuration

### ML Model Parameters

The voice analysis model extracts the following features:
- **Spectral features**: Centroid, rolloff, bandwidth
- **MFCC coefficients**: 13 Mel-frequency cepstral coefficients
- **Pitch analysis**: Mean, standard deviation, range
- **Voice quality**: Jitter, energy consistency, silence ratio
- **Temporal features**: Duration, speaking rate

### Credit Scoring Algorithm

1. **Base Score Calculation**: Based on responses to 10 questions
2. **Voice Confidence Multiplier**: Range 0.8x to 1.2x
3. **Final Score**: Base Score × Voice Multiplier

## 📱 Frontend Integration

### Voice Recording
- Automatic recording during speech recognition
- Real-time audio analysis
- Confidence score display

### Credit Assessment Flow
1. User answers questions via voice/text
2. Voice is analyzed for confidence
3. Base credit score calculated
4. Voice confidence applied as multiplier
5. Final score displayed with breakdown

## 🧪 Testing

### Test the API
```bash
# Health check
curl http://localhost:5000/api/health

# Voice analysis (with base64 audio data)
curl -X POST http://localhost:5000/api/analyze-voice \
  -H "Content-Type: application/json" \
  -d '{"audio_data": "data:audio/wav;base64,UklGRi..."}'
```

### Test the Frontend
1. Open `http://localhost:5000` in your browser
2. Complete the Aadhaar verification
3. Answer assessment questions
4. View results with voice confidence scores

## 📊 Model Performance

### Expected Accuracy
- **Voice Confidence**: 75-85% accuracy
- **Feature Extraction**: Robust to noise and variations
- **Real-time Processing**: <2 seconds per analysis

### Model Improvements
- Collect more training data
- Fine-tune hyperparameters
- Use ensemble methods
- Implement data augmentation

## 🔒 Security Considerations

- Audio data is processed locally
- No voice recordings are stored permanently
- Secure API endpoints with CORS protection
- Session-based authentication

## 🚨 Troubleshooting

### Common Issues

1. **Model not loading**
   - Check if `voice_tone_model.pkl` exists
   - Verify Python dependencies are installed

2. **Audio analysis fails**
   - Check microphone permissions
   - Ensure audio format is supported
   - Verify Flask server is running

3. **Low accuracy scores**
   - Retrain model with more data
   - Check audio quality and duration
   - Verify label consistency

### Debug Mode
```bash
# Enable Flask debug mode
export FLASK_ENV=development
python app.py
```

## 📈 Future Enhancements

- **Real-time streaming**: Continuous voice analysis
- **Emotion detection**: Sentiment analysis integration
- **Multi-modal analysis**: Combine voice with facial expressions
- **Advanced ML models**: Deep learning approaches
- **Cloud deployment**: Scalable infrastructure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section

---

**Note**: This system is designed for educational and demonstration purposes. For production use, ensure compliance with relevant regulations and implement appropriate security measures. 