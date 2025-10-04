import numpy as np
import librosa
import logging
from typing import Dict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraAIVoiceDetector:
    """Ultra-enhanced AI voice detection using multiple advanced algorithms"""
    
    def __init__(self):
        self.model_path = "ultra_ai_detector.pkl"
        self.scaler_path = "ultra_ai_scaler.pkl"
        self.feature_scaler_path = "ultra_feature_scaler.pkl"
        
        # Initialize multiple models for ensemble detection
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=200, max_depth=10, random_state=42)
        }
        
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        self.feature_scaler = StandardScaler()
        
        # Load or create models
        self._load_or_create_models()
        
        # Advanced detection parameters
        self.detection_thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
        
        # Feature importance weights
        self.feature_weights = {
            'spectral_consistency': 0.20,
            'mfcc_variation': 0.15,
            'zcr_consistency': 0.15,
            'rms_consistency': 0.15,
            'pitch_stability': 0.10,
            'spectral_centroid_std': 0.10,
            'spectral_rolloff_std': 0.10,
            'chroma_consistency': 0.05
        }
    
    def _load_or_create_models(self):
        """Load existing models or create new ones"""
        try:
            if (os.path.exists(self.model_path) and 
                os.path.exists(self.scaler_path) and 
                os.path.exists(self.feature_scaler_path)):
                
                # Load models
                model_data = joblib.load(self.model_path)
                self.models = model_data['models']
                self.scalers = model_data['scalers']
                self.feature_scaler = joblib.load(self.feature_scaler_path)
                
                logger.info("Loaded existing ultra AI detection models")
            else:
                self._create_models()
                logger.info("Created new ultra AI detection models")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            self._create_models()
    
    def _create_models(self):
        """Create new models with synthetic training data"""
        try:
            # Generate synthetic training data for AI detection
            X_synthetic, y_synthetic = self._generate_synthetic_training_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_synthetic, y_synthetic, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train models
            for name, model in self.models.items():
                model.fit(X_train_scaled, y_train)
                logger.info(f"Trained {name} model")
            
            # Train scalers
            for name, scaler in self.scalers.items():
                scaler.fit(X_train_scaled)
            
            # Save models
            model_data = {
                'models': self.models,
                'scalers': self.scalers
            }
            joblib.dump(model_data, self.model_path)
            joblib.dump(self.feature_scaler, self.feature_scaler_path)
            
            logger.info("Ultra AI detection models created and saved")
            
        except Exception as e:
            logger.error(f"Error creating models: {e}")
    
    def _generate_synthetic_training_data(self, n_samples=1000):
        """Generate synthetic training data for AI voice detection"""
        np.random.seed(42)
        X = []
        y = []
        
        for _ in range(n_samples):
            # Generate features for human voice (0) or AI voice (1)
            is_ai = np.random.choice([0, 1], p=[0.7, 0.3])
            
            if is_ai:
                # AI voice characteristics: more consistent, less variation
                features = [
                    np.random.normal(30, 5),      # spectral_centroid_std (low)
                    np.random.normal(150, 20),    # spectral_rolloff_std (low)
                    np.random.normal(8, 2),       # mfcc_std (low)
                    np.random.normal(0.008, 0.002), # zcr_std (low)
                    np.random.normal(0.008, 0.002), # rms_std (low)
                    np.random.normal(0.9, 0.05),  # pitch_stability (high)
                    np.random.normal(0.05, 0.01), # chroma_std (low)
                    np.random.normal(0.95, 0.03), # spectral_consistency (high)
                    np.random.normal(0.02, 0.005), # energy_variation (low)
                    np.random.normal(0.1, 0.02),  # pitch_variation (low)
                    np.random.normal(0.05, 0.01), # spectral_contrast_std (low)
                    np.random.normal(0.8, 0.1),   # harmonic_ratio (high)
                    np.random.normal(0.02, 0.005), # jitter (low)
                    np.random.normal(0.01, 0.003), # shimmer (low)
                    np.random.normal(0.9, 0.05)   # voice_quality (high)
                ]
            else:
                # Human voice characteristics: more variation, less consistent
                features = [
                    np.random.normal(80, 20),     # spectral_centroid_std (high)
                    np.random.normal(300, 80),    # spectral_rolloff_std (high)
                    np.random.normal(15, 5),      # mfcc_std (high)
                    np.random.normal(0.020, 0.008), # zcr_std (high)
                    np.random.normal(0.020, 0.008), # rms_std (high)
                    np.random.normal(0.7, 0.15),  # pitch_stability (medium)
                    np.random.normal(0.12, 0.03), # chroma_std (high)
                    np.random.normal(0.6, 0.15),  # spectral_consistency (medium)
                    np.random.normal(0.15, 0.05), # energy_variation (high)
                    np.random.normal(0.3, 0.1),   # pitch_variation (high)
                    np.random.normal(0.08, 0.02), # spectral_contrast_std (high)
                    np.random.normal(0.6, 0.2),   # harmonic_ratio (medium)
                    np.random.normal(0.05, 0.02), # jitter (high)
                    np.random.normal(0.03, 0.01), # shimmer (high)
                    np.random.normal(0.7, 0.2)    # voice_quality (medium)
                ]
            
            X.append(features)
            y.append(is_ai)
        
        return np.array(X), np.array(y)
    
    def extract_ultra_features(self, audio_path: str) -> np.ndarray:
        """Extract ultra-comprehensive features for AI voice detection"""
        try:
            y, sr = librosa.load(audio_path, sr=44100)  # Higher sample rate
            
            features = []
            
            # 1. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # 2. MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26)
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 4. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            
            # 5. Root mean square energy
            rms = librosa.feature.rms(y=y)
            
            # 6. Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            # 7. Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            
            # Calculate comprehensive statistics
            features.extend([
                # Spectral consistency (AI voices are more consistent)
                float(np.std(spectral_centroids)),
                float(np.std(spectral_rolloff)),
                float(np.std(spectral_bandwidth)),
                float(np.std(spectral_contrast)),
                
                # MFCC variation (AI voices have less variation)
                float(np.std(mfccs)),
                float(np.mean(np.std(mfccs, axis=1))),
                
                # Zero crossing rate consistency
                float(np.std(zcr)),
                float(np.mean(zcr)),
                
                # RMS energy consistency
                float(np.std(rms)),
                float(np.mean(rms)),
                
                # Pitch stability (AI voices are more stable)
                float(np.std(pitch_values)) if pitch_values else 0.0,
                float(np.mean(pitch_values)) if pitch_values else 0.0,
                
                # Chroma consistency
                float(np.std(chroma)),
                float(np.mean(chroma)),
                
                # Advanced features
                self._calculate_spectral_consistency(spectral_centroids),
                self._calculate_energy_variation(rms),
                self._calculate_pitch_variation(pitch_values),
                self._calculate_spectral_contrast_std(spectral_contrast),
                self._calculate_harmonic_ratio(y, sr),
                self._calculate_jitter(pitch_values),
                self._calculate_shimmer(rms),
                self._calculate_voice_quality(y, sr)
            ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting ultra features: {e}")
            # Return default features
            return np.zeros(20)
    
    def _calculate_spectral_consistency(self, spectral_centroids):
        """Calculate spectral consistency metric"""
        try:
            # AI voices have very consistent spectral centroids
            consistency = 1.0 - (np.std(spectral_centroids) / np.mean(spectral_centroids))
            return float(np.clip(consistency, 0.0, 1.0))
        except:
            return 0.5
    
    def _calculate_energy_variation(self, rms):
        """Calculate energy variation metric"""
        try:
            return float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0.0
        except:
            return 0.1
    
    def _calculate_pitch_variation(self, pitch_values):
        """Calculate pitch variation metric"""
        try:
            if len(pitch_values) > 1:
                return float(np.std(pitch_values) / np.mean(pitch_values)) if np.mean(pitch_values) > 0 else 0.0
            return 0.0
        except:
            return 0.1
    
    def _calculate_spectral_contrast_std(self, spectral_contrast):
        """Calculate spectral contrast standard deviation"""
        try:
            return float(np.std(spectral_contrast))
        except:
            return 0.1
    
    def _calculate_harmonic_ratio(self, y, sr):
        """Calculate harmonic ratio"""
        try:
            # Estimate harmonic ratio using autocorrelation
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks
            peaks, _ = find_peaks(autocorr, height=0.1)
            if len(peaks) > 1:
                # Calculate ratio of first harmonic to fundamental
                fundamental = peaks[0]
                first_harmonic = peaks[1] if len(peaks) > 1 else fundamental * 2
                ratio = autocorr[first_harmonic] / autocorr[fundamental] if autocorr[fundamental] > 0 else 0
                return float(np.clip(ratio, 0.0, 1.0))
            return 0.5
        except:
            return 0.5
    
    def _calculate_jitter(self, pitch_values):
        """Calculate pitch jitter"""
        try:
            if len(pitch_values) > 2:
                periods = np.diff(pitch_values)
                jitter = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else 0
                return float(jitter)
            return 0.0
        except:
            return 0.0
    
    def _calculate_shimmer(self, rms):
        """Calculate amplitude shimmer"""
        try:
            if len(rms) > 2:
                shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
                return float(shimmer)
            return 0.0
        except:
            return 0.0
    
    def _calculate_voice_quality(self, y, sr):
        """Calculate overall voice quality metric"""
        try:
            # Combine multiple quality indicators
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            
            # Quality based on spectral characteristics
            centroid_quality = 1.0 - (np.std(spectral_centroids) / np.mean(spectral_centroids))
            rolloff_quality = 1.0 - (np.std(spectral_rolloff) / np.mean(spectral_rolloff))
            
            quality = (centroid_quality + rolloff_quality) / 2
            return float(np.clip(quality, 0.0, 1.0))
        except:
            return 0.5
    
    def detect_ai_voice_ultra(self, audio_path: str) -> Dict:
        """Ultra-enhanced AI voice detection using ensemble methods"""
        try:
            logger.info(f"Starting ultra AI voice detection for: {audio_path}")
            
            # Extract comprehensive features
            features = self.extract_ultra_features(audio_path)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features)
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(features_scaled)[0]
                    pred_proba = model.predict_proba(features_scaled)[0]
                    predictions[name] = pred
                    confidences[name] = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {e}")
                    predictions[name] = 0
                    confidences[name] = 0.5
            
            # Ensemble prediction
            ensemble_confidence = np.mean(list(confidences.values()))
            ensemble_prediction = 1 if ensemble_confidence > 0.5 else 0
            
            # Enhanced heuristic analysis
            heuristic_confidence = self._advanced_heuristic_detection(features[0])
            
            # Combine ensemble and heuristic results
            final_confidence = (ensemble_confidence * 0.7 + heuristic_confidence * 0.3)
            final_prediction = 1 if final_confidence > 0.5 else 0
            
            # Determine risk level
            if final_confidence > self.detection_thresholds['high_confidence']:
                risk_level = "high"
            elif final_confidence > self.detection_thresholds['medium_confidence']:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # Calculate detection reliability
            reliability = self._calculate_detection_reliability(features[0], final_confidence)
            
            logger.info(f"Ultra AI detection result: {final_prediction}, confidence: {final_confidence:.3f}, risk: {risk_level}")
            
            return {
                "is_ai_generated": bool(final_prediction),
                "confidence_score": float(final_confidence),
                "detection_method": "ultra_ensemble_analysis",
                "risk_level": risk_level,
                "reliability": float(reliability),
                "ensemble_confidence": float(ensemble_confidence),
                "heuristic_confidence": float(heuristic_confidence),
                "model_predictions": predictions,
                "model_confidences": {k: float(v) for k, v in confidences.items()},
                "feature_analysis": self._analyze_features(features[0])
            }
            
        except Exception as e:
            logger.error(f"Ultra AI detection error: {e}")
            return {
                "is_ai_generated": False,
                "confidence_score": 0.0,
                "detection_method": "error",
                "risk_level": "low",
                "reliability": 0.0
            }
    
    def _advanced_heuristic_detection(self, features: np.ndarray) -> float:
        """Advanced heuristic detection using multiple indicators"""
        try:
            if len(features) < 20:
                return 0.0
            
            score = 0.0
            
            # Extract feature values
            spectral_centroid_std = features[0]
            spectral_rolloff_std = features[1]
            mfcc_std = features[4]
            zcr_std = features[6]
            rms_std = features[8]
            pitch_std = features[10]
            chroma_std = features[12]
            spectral_consistency = features[14]
            energy_variation = features[15]
            pitch_variation = features[16]
            harmonic_ratio = features[17]
            jitter = features[18]
            shimmer = features[19]
            
            # 1. Spectral consistency analysis (AI voices are too consistent)
            if spectral_centroid_std < 40:
                score += 0.3
            elif spectral_centroid_std < 80:
                score += 0.15
            
            if spectral_rolloff_std < 200:
                score += 0.25
            elif spectral_rolloff_std < 400:
                score += 0.1
            
            # 2. MFCC variation analysis
            if mfcc_std < 10:
                score += 0.3
            elif mfcc_std < 20:
                score += 0.15
            
            # 3. Zero crossing rate consistency
            if zcr_std < 0.01:
                score += 0.25
            elif zcr_std < 0.02:
                score += 0.1
            
            # 4. RMS energy consistency
            if rms_std < 0.01:
                score += 0.25
            elif rms_std < 0.02:
                score += 0.1
            
            # 5. Pitch stability (AI voices are too stable)
            if pitch_std < 20:
                score += 0.2
            elif pitch_std < 40:
                score += 0.1
            
            # 6. Chroma consistency
            if chroma_std < 0.08:
                score += 0.2
            elif chroma_std < 0.15:
                score += 0.1
            
            # 7. Advanced consistency metrics
            if spectral_consistency > 0.9:
                score += 0.3
            elif spectral_consistency > 0.8:
                score += 0.15
            
            # 8. Energy and pitch variation (AI voices have low variation)
            if energy_variation < 0.1:
                score += 0.2
            if pitch_variation < 0.1:
                score += 0.2
            
            # 9. Harmonic analysis
            if harmonic_ratio > 0.8:
                score += 0.15
            
            # 10. Jitter and shimmer (AI voices have very low jitter/shimmer)
            if jitter < 0.02:
                score += 0.2
            if shimmer < 0.02:
                score += 0.2
            
            # 11. Multiple indicators together (very strong signal)
            perfect_patterns = 0
            if spectral_centroid_std < 50: perfect_patterns += 1
            if mfcc_std < 15: perfect_patterns += 1
            if zcr_std < 0.015: perfect_patterns += 1
            if rms_std < 0.015: perfect_patterns += 1
            if spectral_consistency > 0.85: perfect_patterns += 1
            if energy_variation < 0.12: perfect_patterns += 1
            if pitch_variation < 0.12: perfect_patterns += 1
            if jitter < 0.03: perfect_patterns += 1
            if shimmer < 0.03: perfect_patterns += 1
            
            if perfect_patterns >= 7:
                score += 0.4  # Very likely AI
            elif perfect_patterns >= 5:
                score += 0.3  # Likely AI
            elif perfect_patterns >= 3:
                score += 0.2  # Possibly AI
            
            return float(np.clip(score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Heuristic detection error: {e}")
            return 0.0
    
    def _calculate_detection_reliability(self, features: np.ndarray, confidence: float) -> float:
        """Calculate detection reliability based on feature quality"""
        try:
            # Check if features are within expected ranges
            reliability = 1.0
            
            # Check for extreme values that might indicate poor audio quality
            if np.any(features == 0):
                reliability *= 0.8  # Some features are zero
            
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                reliability *= 0.5  # Invalid features
            
            # Check for reasonable feature ranges
            if features[0] > 500:  # Very high spectral centroid std
                reliability *= 0.9
            
            if features[4] > 100:  # Very high MFCC std
                reliability *= 0.9
            
            # Boost reliability for high confidence
            if confidence > 0.8:
                reliability *= 1.1
            
            return float(np.clip(reliability, 0.0, 1.0))
            
        except:
            return 0.5
    
    def _analyze_features(self, features: np.ndarray) -> Dict:
        """Analyze extracted features for debugging"""
        try:
            return {
                "spectral_centroid_std": float(features[0]),
                "spectral_rolloff_std": float(features[1]),
                "mfcc_std": float(features[4]),
                "zcr_std": float(features[6]),
                "rms_std": float(features[8]),
                "pitch_std": float(features[10]),
                "chroma_std": float(features[12]),
                "spectral_consistency": float(features[14]),
                "energy_variation": float(features[15]),
                "pitch_variation": float(features[16]),
                "harmonic_ratio": float(features[17]),
                "jitter": float(features[18]),
                "shimmer": float(features[19])
            }
        except:
            return {}

