#!/usr/bin/env python3
"""
Voice Tone Analysis Model Training Script

This script helps you train the voice tone analysis model with your audio data.
You can provide audio files and corresponding confidence labels to train the model.

Usage:
    python train_model.py --audio_dir /path/to/audio/files --labels_file /path/to/labels.txt
    python train_model.py --help
"""

import argparse
import os
import json
from voice_analysis_model import VoiceToneAnalyzer

def load_labels(labels_file):
    """Load labels from a text file"""
    labels = []
    try:
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Try to parse as float first, then as string
                    try:
                        labels.append(float(line))
                    except ValueError:
                        labels.append(line)
        return labels
    except Exception as e:
        print(f"Error loading labels file: {e}")
        return None

def get_audio_files(audio_dir):
    """Get all audio files from directory"""
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    try:
        for file in os.listdir(audio_dir):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(audio_dir, file))
        return sorted(audio_files)
    except Exception as e:
        print(f"Error reading audio directory: {e}")
        return None

def train_with_directory(audio_dir, labels_file, output_model="voice_tone_model.pkl"):
    """Train model with audio files from directory and labels from file"""
    print("=== Voice Tone Analysis Model Training ===")
    
    # Load labels
    labels = load_labels(labels_file)
    if labels is None:
        return False
    
    # Get audio files
    audio_files = get_audio_files(audio_dir)
    if audio_files is None:
        return False
    
    # Check if number of files matches number of labels
    if len(audio_files) != len(labels):
        print(f"Error: Number of audio files ({len(audio_files)}) doesn't match number of labels ({len(labels)})")
        return False
    
    print(f"Found {len(audio_files)} audio files and {len(labels)} labels")
    
    # Initialize analyzer
    analyzer = VoiceToneAnalyzer()
    
    # Train model
    print("\nStarting model training...")
    success = analyzer.train_model(audio_files, labels)
    
    if success:
        # Save model
        analyzer.save_model(output_model)
        print(f"\n✅ Model trained and saved successfully to {output_model}")
        return True
    else:
        print("\n❌ Model training failed")
        return False

def train_with_custom_data(audio_files, labels, output_model="voice_tone_model.pkl"):
    """Train model with custom audio files and labels"""
    print("=== Voice Tone Analysis Model Training (Custom Data) ===")
    
    if len(audio_files) != len(labels):
        print(f"Error: Number of audio files ({len(audio_files)}) doesn't match number of labels ({len(labels)})")
        return False
    
    print(f"Training with {len(audio_files)} audio files")
    
    # Initialize analyzer
    analyzer = VoiceToneAnalyzer()
    
    # Train model
    print("\nStarting model training...")
    success = analyzer.train_model(audio_files, labels)
    
    if success:
        # Save model
        analyzer.save_model(output_model)
        print(f"\n✅ Model trained and saved successfully to {output_model}")
        return True
    else:
        print("\n❌ Model training failed")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train Voice Tone Analysis Model')
    parser.add_argument('--audio_dir', help='Directory containing audio files')
    parser.add_argument('--labels_file', help='Text file containing labels (one per line)')
    parser.add_argument('--output_model', default='voice_tone_model.pkl', help='Output model file path')
    parser.add_argument('--interactive', action='store_true', help='Interactive training mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        # Interactive mode
        print("=== Interactive Voice Tone Model Training ===")
        
        # Get audio directory
        audio_dir = input("Enter path to audio files directory: ").strip()
        if not os.path.exists(audio_dir):
            print(f"Error: Directory {audio_dir} does not exist")
            return
        
        # Get labels file
        labels_file = input("Enter path to labels file: ").strip()
        if not os.path.exists(labels_file):
            print(f"Error: Labels file {labels_file} does not exist")
            return
        
        # Train model
        success = train_with_directory(audio_dir, labels_file, args.output_model)
        
    elif args.audio_dir and args.labels_file:
        # Command line mode
        success = train_with_directory(args.audio_dir, args.labels_file, args.output_model)
        
    else:
        # Example training with sample data
        print("=== Example Training Mode ===")
        print("No audio directory or labels file provided.")
        print("You can:")
        print("1. Use --audio_dir and --labels_file arguments")
        print("2. Use --interactive for interactive mode")
        print("3. Modify this script to include your training data")
        
        # Example of how to use custom data
        print("\nExample of custom training:")
        print("```python")
        print("from train_model import train_with_custom_data")
        print("")
        print("audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']")
        print("labels = [0.8, 0.6, 0.9]  # Confidence scores")
        print("train_with_custom_data(audio_files, labels)")
        print("```")
        return
    
    if success:
        print("\n🎉 Training completed successfully!")
        print(f"Your model is saved as: {args.output_model}")
        print("\nYou can now:")
        print("1. Run the Flask API: python app.py")
        print("2. Use the model in your frontend application")
        print("3. Test the API endpoints")
    else:
        print("\n💥 Training failed. Please check your data and try again.")

if __name__ == "__main__":
    main() 