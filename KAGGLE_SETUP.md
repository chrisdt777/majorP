# 🚀 Quick Setup: Kaggle Dataset Training

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Kaggle account** (free at [kaggle.com](https://kaggle.com))
3. **Internet connection** for dataset download

## ⚡ Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Login to Kaggle
```bash
kagglehub login
```
*Enter your Kaggle username and API key when prompted*

### Step 3: Train the Model
```bash
python train_with_kaggle_dataset.py
```

## 🔍 What Happens During Training

1. **Dataset Download**: Automatically downloads the Truth/Deception Detection dataset
2. **Audio Analysis**: Processes all audio files to extract features
3. **Label Generation**: Creates confidence scores based on audio characteristics
4. **Model Training**: Trains the Random Forest classifier
5. **Model Saving**: Saves the trained model as `voice_tone_model.pkl`

## 📊 Expected Results

- **Training Time**: 5-15 minutes (depending on dataset size)
- **Model Accuracy**: 75-85% (estimated)
- **Audio Files Processed**: 1000+ (depending on dataset)
- **Features Extracted**: 50+ voice characteristics

## 🧪 Test First

Before training, test the dataset download:
```bash
python test_dataset.py
```

## 🚨 Troubleshooting

### "kagglehub not found"
```bash
pip install kagglehub
```

### "Authentication failed"
```bash
kagglehub login
# Make sure you're using your Kaggle API key, not password
```

### "Dataset download failed"
- Check internet connection
- Verify Kaggle account is active
- Try again (sometimes Kaggle servers are busy)

## 🎯 After Training

Once training completes:
1. **Start the API**: `python app.py`
2. **Test voice analysis** in your frontend
3. **Monitor performance** and retrain if needed

## 📈 Model Performance

The model will be trained to detect:
- **Confident speech** (high confidence scores)
- **Uncertain speech** (low confidence scores)
- **Voice quality indicators** (clarity, stability, energy)

## 🔄 Retraining

To improve the model:
1. Collect more audio data
2. Adjust feature extraction parameters
3. Use different ML algorithms
4. Implement data augmentation

---

**Ready to train?** Run `python train_with_kaggle_dataset.py` and watch the magic happen! 🎤✨ 