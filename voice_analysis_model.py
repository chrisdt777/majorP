import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class VoiceToneAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_audio_features(self, audio_data, sr=22050):
        """Extract comprehensive audio features for voice tone analysis"""
        features = {}
        
        try:
            # Basic audio features
            features['duration'] = len(audio_data) / sr
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # Root mean square energy
            rms = librosa.feature.rms(y=audio_data)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
            pitch_values = pitches[magnitudes > 0.1]
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
            
            # Jitter and shimmer (voice quality measures)
            if len(pitch_values) > 1:
                jitter = np.mean(np.abs(np.diff(pitch_values)))
                features['jitter'] = jitter
            else:
                features['jitter'] = 0
            
            # Energy distribution
            energy = np.sum(audio_data**2)
            features['total_energy'] = energy
            features['energy_per_sample'] = energy / len(audio_data)
            
            # Silence ratio
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(audio_data) < silence_threshold) / len(audio_data)
            features['silence_ratio'] = silence_ratio
            
            # Speaking rate approximation
            features['speaking_rate'] = 1 / (features['duration'] + 1e-6)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default values if extraction fails
            features = {f'feature_{i}': 0 for i in range(50)}
            
        return features
    
    def prepare_features_for_model(self, features_dict):
        """Convert features dictionary to feature vector"""
        feature_vector = []
        self.feature_names = []
        
        for key, value in features_dict.items():
            feature_vector.append(float(value))
            self.feature_names.append(key)
            
        return np.array(feature_vector).reshape(1, -1)
    
    def train_model(self, audio_files, labels):
        """Train the voice tone analysis model"""
        print("Training voice tone analysis model...")
        
        all_features = []
        all_labels = []
        
        for audio_file, label in zip(audio_files, labels):
            try:
                # Load audio file
                audio_data, sr = librosa.load(audio_file, sr=22050)
                
                # Extract features
                features = self.extract_audio_features(audio_data, sr)
                feature_vector = self.prepare_features_for_model(features)
                
                all_features.append(feature_vector.flatten())
                all_labels.append(label)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        if len(all_features) == 0:
            print("No valid audio files found for training")
            return False
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))
        
        return True
    
    def analyze_voice_tone(self, audio_data, sr=22050):
        """Analyze voice tone and return confidence score"""
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return 0.5
        
        try:
            # Extract features
            features = self.extract_audio_features(audio_data, sr)
            feature_vector = self.prepare_features_for_model(features)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get prediction probabilities for each class
            confidence_scores = self.model.predict_proba(feature_vector_scaled)[0]
            
            # Map class probabilities to confidence scores
            class_names = self.model.classes_
            confidence_mapping = {
                'very_confident': 0.9,
                'confident': 0.75,
                'moderate': 0.5,
                'uncertain': 0.25
            }
            
            # Calculate weighted confidence based on class probabilities
            total_confidence = 0.0
            for i, class_name in enumerate(class_names):
                if class_name in confidence_mapping:
                    total_confidence += confidence_scores[i] * confidence_mapping[class_name]
            
            # Additional analysis for voice quality
            voice_quality_score = self._analyze_voice_quality(features)
            
            # Combine confidence with voice quality
            final_score = (total_confidence + voice_quality_score) / 2
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            print(f"Error analyzing voice tone: {e}")
            return 0.5
    
    def _analyze_voice_quality(self, features):
        """Analyze voice quality based on extracted features"""
        quality_score = 0.5  # Base score
        
        try:
            # Pitch stability (lower jitter is better)
            if 'jitter' in features:
                jitter = features['jitter']
                if jitter < 50:  # Good pitch stability
                    quality_score += 0.2
                elif jitter > 200:  # Poor pitch stability
                    quality_score -= 0.2
            
            # Energy consistency
            if 'rms_std' in features:
                rms_std = features['rms_std']
                if rms_std < 0.1:  # Consistent energy
                    quality_score += 0.15
                elif rms_std > 0.3:  # Inconsistent energy
                    quality_score -= 0.15
            
            # Silence ratio (some silence is good, too much is bad)
            if 'silence_ratio' in features:
                silence_ratio = features['silence_ratio']
                if 0.1 < silence_ratio < 0.4:  # Good speaking pattern
                    quality_score += 0.15
                elif silence_ratio > 0.6:  # Too much silence
                    quality_score -= 0.15
            
            # Duration (too short might indicate rushed speech)
            if 'duration' in features:
                duration = features['duration']
                if 1.0 < duration < 5.0:  # Good response length
                    quality_score += 0.1
                elif duration < 0.5:  # Too short
                    quality_score -= 0.1
            
        except Exception as e:
            print(f"Error in voice quality analysis: {e}")
        
        return max(0.0, min(1.0, quality_score))
    
    def train_model(self, audio_files, labels):
        """Train the voice tone analysis model"""
        print("Training voice tone analysis model...")
        
        all_features = []
        all_labels = []
        
        for audio_file, label in zip(audio_files, labels):
            try:
                # Load audio file
                audio_data, sr = librosa.load(audio_file, sr=22050)
                
                # Extract features
                features = self.extract_audio_features(audio_data, sr)
                feature_vector = self.prepare_features_for_model(features)
                
                all_features.append(feature_vector.flatten())
                all_labels.append(label)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        if len(all_features) == 0:
            print("No valid audio files found for training")
            return False
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))
        
        return True
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No trained model to save")
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Example usage and training
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = VoiceToneAnalyzer()
    
    # Example training data structure
    # audio_files = ["path/to/audio1.wav", "path/to/audio2.wav"]
    # labels = ["confident", "uncertain"]  # or numeric labels like [1, 0]
    
    # Train the model (uncomment when you have training data)
    # success = analyzer.train_model(audio_files, labels)
    # if success:
    #     analyzer.save_model("voice_tone_model.pkl")
    
    print("Voice Tone Analyzer initialized. Train the model with your data to start analyzing voice tones.") 