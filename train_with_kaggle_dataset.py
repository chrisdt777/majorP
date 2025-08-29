#!/usr/bin/env python3
"""
Voice Tone Analysis Model Training with Kaggle Dataset

This script downloads and uses the Truth Detection/Deception Detection dataset
from Kaggle to train our voice tone analysis model.

Dataset: https://www.kaggle.com/datasets/thesergiu/truth-detectiondeception-detectionlie-detection
"""

import kagglehub
import os
import pandas as pd
import numpy as np
from voice_analysis_model import VoiceToneAnalyzer
import librosa
import warnings
warnings.filterwarnings('ignore')

def download_kaggle_dataset():
    """Download the CREMA-D emotional speech dataset from Kaggle"""
    print("=== Downloading CREMA-D Dataset ===")
    
    try:
        # Download latest version of the CREMA-D dataset
        path = kagglehub.dataset_download("ejlok1/cremad")
        print(f"✅ CREMA-D dataset downloaded successfully to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please ensure you have kagglehub installed and configured:")
        print("pip install kagglehub")
        print("kagglehub login")
        return None

def explore_dataset(dataset_path):
    """Explore the structure of the downloaded dataset"""
    print("\n=== Exploring Dataset Structure ===")
    
    try:
        # List all files in the dataset
        files = os.listdir(dataset_path)
        print(f"Files found: {len(files)}")
        
        for file in files:
            file_path = os.path.join(dataset_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"📁 {file} ({size:.2f} MB)")
            elif os.path.isdir(file_path):
                subfiles = os.listdir(file_path)
                print(f"📂 {file}/ ({len(subfiles)} items)")
        
        return files
    except Exception as e:
        print(f"❌ Error exploring dataset: {e}")
        return []

def find_audio_files(dataset_path):
    """Find all audio files in the dataset"""
    print("\n=== Finding Audio Files ===")
    
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma']
    audio_files = []
    
    def search_recursively(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
    
    search_recursively(dataset_path)
    
    print(f"🎵 Found {len(audio_files)} audio files")
    
    # Show some examples
    if audio_files:
        print("Sample audio files:")
        for i, file in enumerate(audio_files[:5]):
            print(f"  {i+1}. {os.path.basename(file)}")
        if len(audio_files) > 5:
            print(f"  ... and {len(audio_files) - 5} more")
    
    return audio_files

def find_labels_file(dataset_path):
    """Find the labels/annotations file in the dataset"""
    print("\n=== Finding Labels File ===")
    
    # Common label file names
    label_patterns = ['*.csv', '*.txt', '*.json', 'labels*', 'annotations*', 'metadata*']
    label_files = []
    
    for pattern in label_patterns:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if any(pattern.replace('*', '').lower() in file.lower() for pattern in label_patterns):
                    label_files.append(os.path.join(root, file))
    
    if label_files:
        print(f"📋 Found {len(label_files)} potential label files:")
        for file in label_files:
            print(f"  - {os.path.basename(file)}")
        return label_files
    else:
        print("⚠️  No label files found. We'll need to create labels manually.")
        return []

def analyze_dataset_structure(dataset_path):
    """Analyze the dataset structure to understand how to map audio to labels"""
    print("\n=== Analyzing Dataset Structure ===")
    
    # Look for common dataset structures
    possible_structures = [
        "audio_files_with_labels_in_filename",
        "separate_audio_and_labels_folders", 
        "metadata_file_with_audio_paths",
        "nested_folder_structure"
    ]
    
    print("Possible dataset structures:")
    for structure in possible_structures:
        print(f"  - {structure}")
    
    # Check for common patterns
    files = os.listdir(dataset_path)
    
    # Look for README or documentation
    readme_files = [f for f in files if 'readme' in f.lower() or 'read_me' in f.lower()]
    if readme_files:
        print(f"\n📖 Found README files: {readme_files}")
        print("Please check these files for dataset documentation.")
    
    return possible_structures

def create_training_data(audio_files, dataset_path):
    """Create training data using CREMA-D emotional labels for confidence scoring"""
    print("\n=== Creating Training Data from CREMA-D ===")
    
    # CREMA-D has emotional labels that we can map to confidence scores:
    # - Happy, Sad, Angry, Fear, Disgust, Neutral
    # - We'll map emotions to confidence levels for credit assessment
    
    training_data = []
    
    print("Analyzing audio files and mapping emotions to confidence scores...")
    
    # Emotion to confidence mapping for credit assessment context
    emotion_confidence_map = {
        'HAP': 0.9,  # Happy - high confidence
        'SAD': 0.4,  # Sad - low confidence  
        'ANG': 0.7,  # Angry - moderate-high confidence
        'FEA': 0.3,  # Fear - low confidence
        'DIS': 0.5,  # Disgust - moderate confidence
        'NEU': 0.6   # Neutral - moderate confidence
    }
    
    for i, audio_file in enumerate(audio_files):
        try:
            print(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
            
            # Extract emotion from filename (CREMA-D naming convention)
            filename = os.path.basename(audio_file)
            
            # CREMA-D files typically have format: 1001_DFA_HAP_XX.wav
            # Extract emotion code (HAP, SAD, ANG, FEA, DIS, NEU)
            emotion_code = None
            for code in emotion_confidence_map.keys():
                if code in filename.upper():
                    emotion_code = code
                    break
            
            if not emotion_code:
                # If no emotion found, analyze audio characteristics
                emotion_code = analyze_audio_characteristics(audio_file)
            
            # Get confidence score from emotion
            confidence = emotion_confidence_map.get(emotion_code, 0.5)
            
            # Add some variation to make training more robust
            confidence += np.random.uniform(-0.1, 0.1)
            confidence = max(0.0, min(1.0, confidence))
            
            # Load audio for duration check
            audio_data, sr = librosa.load(audio_file, sr=22050)
            duration = len(audio_data) / sr
            
            # Skip very short or very long audio
            if duration < 0.5 or duration > 30:
                print(f"  ⏭️  Skipping: duration {duration:.2f}s (too short/long)")
                continue
            
            training_data.append({
                'audio_file': audio_file,
                'emotion': emotion_code,
                'confidence': confidence,
                'duration': duration
            })
            
            print(f"  ✅ Emotion: {emotion_code}, Confidence: {confidence:.2f}")
            
        except Exception as e:
            print(f"  ❌ Error processing {audio_file}: {e}")
            continue
    
    print(f"\n🎯 Created training data for {len(training_data)} audio files")
    return training_data

def analyze_audio_characteristics(audio_file):
    """Analyze audio characteristics to infer emotion when filename doesn't contain it"""
    try:
        audio_data, sr = librosa.load(audio_file, sr=22050)
        
        # Extract basic features
        rms_energy = np.sqrt(np.mean(audio_data**2))
        zero_crossings = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Simple heuristic based on audio characteristics
        if rms_energy > 0.15:  # High energy
            return 'ANG'  # Angry
        elif rms_energy < 0.05:  # Low energy
            return 'SAD'  # Sad
        elif 0.01 < zero_crossings < 0.08:  # Moderate variation
            return 'HAP'  # Happy
        else:
            return 'NEU'  # Neutral
            
    except Exception as e:
        print(f"Error analyzing audio characteristics: {e}")
        return 'NEU'  # Default to neutral

def train_model_with_dataset(training_data, output_model="voice_tone_model.pkl"):
    """Train the voice tone analysis model with the created training data"""
    print("\n=== Training Model ===")
    
    if not training_data:
        print("❌ No training data available")
        return False
    
    # Prepare data for training
    audio_files = [item['audio_file'] for item in training_data]
    
    # Convert continuous confidence scores to discrete classes for classification
    confidence_scores = [item['confidence'] for item in training_data]
    
    # Create discrete classes based on confidence ranges
    labels = []
    for confidence in confidence_scores:
        if confidence >= 0.8:
            labels.append('very_confident')
        elif confidence >= 0.6:
            labels.append('confident')
        elif confidence >= 0.4:
            labels.append('moderate')
        else:
            labels.append('uncertain')
    
    print(f"Training with {len(audio_files)} audio files")
    print(f"Confidence range: {min(confidence_scores):.2f} to {max(confidence_scores):.2f}")
    print(f"Class distribution:")
    from collections import Counter
    class_counts = Counter(labels)
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")
    
    # Initialize and train the model
    analyzer = VoiceToneAnalyzer()
    
    print("Starting model training...")
    success = analyzer.train_model(audio_files, labels)
    
    if success:
        # Save the trained model
        analyzer.save_model(output_model)
        print(f"\n✅ Model trained and saved successfully to {output_model}")
        
        # Show training summary
        print("\n📊 Training Summary:")
        print(f"  - Audio files processed: {len(audio_files)}")
        print(f"  - Average confidence: {np.mean(confidence_scores):.2f}")
        print(f"  - Confidence range: {min(confidence_scores):.2f} - {max(confidence_scores):.2f}")
        print(f"  - Model saved to: {output_model}")
        
        return True
    else:
        print("\n❌ Model training failed")
        return False

def main():
    """Main function to download CREMA-D dataset and train voice tone analysis model"""
    print("🎤 Voice Tone Analysis Model Training with CREMA-D Dataset")
    print("=" * 60)
    
    # Step 1: Download dataset
    dataset_path = download_kaggle_dataset()
    if not dataset_path:
        return
    
    # Step 2: Explore dataset structure
    files = explore_dataset(dataset_path)
    
    # Step 3: Find audio files
    audio_files = find_audio_files(dataset_path)
    if not audio_files:
        print("❌ No audio files found in the dataset")
        return
    
    # Step 4: Find labels
    label_files = find_labels_file(dataset_path)
    
    # Step 5: Analyze structure
    analyze_dataset_structure(dataset_path)
    
    # Step 6: Create training data
    training_data = create_training_data(audio_files, dataset_path)
    
    # Step 7: Train model
    if training_data:
        success = train_model_with_dataset(training_data)
        
        if success:
            print("\n🎉 Training completed successfully!")
            print("\nYou can now:")
            print("1. Run the Flask API: python app.py")
            print("2. Use the model in your frontend application")
            print("3. Test the voice analysis functionality")
        else:
            print("\n💥 Training failed. Please check the error messages above.")
    else:
        print("\n❌ Could not create training data from the dataset")

if __name__ == "__main__":
    main() 