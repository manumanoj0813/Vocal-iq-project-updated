import numpy as np
import librosa
import logging
from typing import Dict
from transformers import pipeline
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraEnhancedLanguageDetector:
    """Ultra-enhanced language detection using multiple AI models and advanced features"""
    
    def __init__(self):
        self.supported_languages = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'kn': 'Kannada', 'te': 'Telugu', 'ta': 'Tamil', 'ml': 'Malayalam',
            'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi', 'or': 'Odia'
        }
        
        # Initialize multiple AI models for maximum accuracy
        try:
            # Primary transcriber with large model
            self.transcriber = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-large-v3",
                return_timestamps=True
            )
            
            # Language detection model
            self.language_detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection"
            )
            
            logger.info("Ultra-enhanced AI models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load AI models: {e}")
            self.transcriber = None
            self.language_detector = None
        
        # Advanced feature extraction parameters
        self.feature_weights = {
            'spectral_centroid': 0.25,
            'spectral_rolloff': 0.20,
            'zero_crossing_rate': 0.15,
            'mfcc_std': 0.15,
            'spectral_bandwidth': 0.10,
            'spectral_contrast': 0.10,
            'chroma': 0.05
        }
        
        # Language-specific feature patterns (enhanced)
        self.language_patterns = {
            'en': {
                'centroid_range': (1500, 3500),
                'rolloff_range': (2500, 6000),
                'zcr_range': (0.03, 0.20),
                'mfcc_std_range': (15, 50),
                'bandwidth_range': (1000, 3000)
            },
            'kn': {
                'centroid_range': (1200, 1800),
                'rolloff_range': (2000, 3200),
                'zcr_range': (0.03, 0.10),
                'mfcc_std_range': (20, 45),
                'bandwidth_range': (800, 1500)
            },
            'te': {
                'centroid_range': (1600, 2100),
                'rolloff_range': (3200, 4200),
                'zcr_range': (0.06, 0.13),
                'mfcc_std_range': (25, 50),
                'bandwidth_range': (1000, 2000)
            },
            'hi': {
                'centroid_range': (1700, 2300),
                'rolloff_range': (3500, 4800),
                'zcr_range': (0.07, 0.16),
                'mfcc_std_range': (20, 45),
                'bandwidth_range': (1000, 2500)
            },
            'ta': {
                'centroid_range': (1400, 1900),
                'rolloff_range': (2800, 3800),
                'zcr_range': (0.05, 0.12),
                'mfcc_std_range': (18, 42),
                'bandwidth_range': (900, 1800)
            },
            'ml': {
                'centroid_range': (1300, 1800),
                'rolloff_range': (2600, 3600),
                'zcr_range': (0.04, 0.11),
                'mfcc_std_range': (22, 48),
                'bandwidth_range': (850, 1700)
            }
        }
    
    def detect_language_from_audio(self, audio_path: str) -> Dict:
        """Ultra-enhanced language detection using multiple AI models and advanced features"""
        try:
            logger.info(f"Starting ultra-enhanced language detection for: {audio_path}")
            
            # Method 1: Advanced feature-based detection
            feature_result = self._detect_language_from_features_enhanced(audio_path)
            
            # Method 2: AI-powered transcription-based detection
            transcription_result = None
            if self.transcriber:
                try:
                    logger.info("Attempting AI-powered transcription-based language detection")
                    
                    # Primary transcription with large model
                    transcription = self.transcriber(audio_path)
                    text = transcription["text"]
                    
                    if text and len(text.strip()) > 3:
                        # Use AI language detection model
                        if self.language_detector:
                            ai_result = self.language_detector(text)
                            detected_lang = ai_result[0]['label'].lower()
                            ai_confidence = ai_result[0]['score']
                        else:
                            # Fallback to langdetect
                            from langdetect import detect
                            detected_lang = detect(text)
                            ai_confidence = 0.7
                        
                        # Enhanced Indian language detection
                        detected_lang, confidence = self._enhance_indian_language_detection(
                            text, detected_lang, ai_confidence
                        )
                        
                        transcription_result = {
                            "detected_language": detected_lang,
                            "confidence": confidence,
                            "language_name": self.supported_languages.get(detected_lang, "Unknown"),
                            "language_code": detected_lang,
                            "transcription": text,
                            "detection_method": "ai_transcription"
                        }
                        
                        logger.info(f"AI transcription-based detection: {detected_lang} "
                                   f"(confidence: {confidence:.3f})")
                        logger.info(f"Transcription: {text[:100]}...")
                    else:
                        logger.warning("Transcription returned empty or short text")
                        
                except Exception as transcribe_error:
                    logger.warning(f"Transcription failed: {transcribe_error}")
            else:
                logger.warning("No transcriber available, using feature-based detection only")
            
            # Method 3: Hybrid approach - combine results
            hybrid_result = self._combine_detection_results(feature_result, transcription_result)
            
            # Choose the best result
            if hybrid_result["confidence"] > 0.8:
                logger.info("Using hybrid detection result (high confidence)")
                return hybrid_result
            elif transcription_result and transcription_result["confidence"] > feature_result["confidence"]:
                logger.info("Using transcription-based result (higher confidence)")
                return transcription_result
            else:
                logger.info("Using feature-based result")
                return feature_result
                
        except Exception as e:
            logger.error(f"Ultra-enhanced language detection error: {e}")
            return {
                "detected_language": "en",
                "confidence": 0.3,
                "language_name": "English",
                "language_code": "en",
                "transcription": "",
                "detection_method": "fallback"
            }
    
    def _detect_language_from_features_enhanced(self, audio_path: str) -> Dict:
        """Ultra-enhanced feature-based language detection"""
        try:
            # Load audio with high quality
            y, sr = librosa.load(audio_path, sr=44100)  # Higher sample rate
            
            # Extract comprehensive features
            features = self._extract_comprehensive_features(y, sr)
            
            # Calculate language scores using enhanced patterns
            language_scores = {}
            
            for lang_code, patterns in self.language_patterns.items():
                score = 0
                total_weight = 0
                
                # Spectral centroid matching
                if patterns['centroid_range'][0] <= features['spectral_centroid'] <= patterns['centroid_range'][1]:
                    score += self.feature_weights['spectral_centroid']
                total_weight += self.feature_weights['spectral_centroid']
                
                # Spectral rolloff matching
                if patterns['rolloff_range'][0] <= features['spectral_rolloff'] <= patterns['rolloff_range'][1]:
                    score += self.feature_weights['spectral_rolloff']
                total_weight += self.feature_weights['spectral_rolloff']
                
                # Zero crossing rate matching
                if patterns['zcr_range'][0] <= features['zero_crossing_rate'] <= patterns['zcr_range'][1]:
                    score += self.feature_weights['zero_crossing_rate']
                total_weight += self.feature_weights['zero_crossing_rate']
                
                # MFCC standard deviation matching
                if patterns['mfcc_std_range'][0] <= features['mfcc_std'] <= patterns['mfcc_std_range'][1]:
                    score += self.feature_weights['mfcc_std']
                total_weight += self.feature_weights['mfcc_std']
                
                # Spectral bandwidth matching
                if patterns['bandwidth_range'][0] <= features['spectral_bandwidth'] <= patterns['bandwidth_range'][1]:
                    score += self.feature_weights['spectral_bandwidth']
                total_weight += self.feature_weights['spectral_bandwidth']
                
                # Normalize score
                if total_weight > 0:
                    language_scores[lang_code] = score / total_weight
                else:
                    language_scores[lang_code] = 0
            
            # Find best match
            if language_scores:
                best_lang = max(language_scores, key=language_scores.get)
                confidence = language_scores[best_lang]
                
                # Boost confidence for Indian languages if features strongly match
                if best_lang in ['kn', 'te', 'hi', 'ta', 'ml'] and confidence > 0.7:
                    confidence = min(0.95, confidence + 0.1)
            else:
                best_lang = "en"
                confidence = 0.5
            
            logger.info(f"Feature-based detection: {best_lang} (confidence: {confidence:.3f})")
            logger.info(f"Language scores: {dict(sorted(language_scores.items(), key=lambda x: x[1], reverse=True)[:3])}")
            
            return {
                "detected_language": best_lang,
                "confidence": confidence,
                "language_name": self.supported_languages.get(best_lang, "Unknown"),
                "language_code": best_lang,
                "transcription": "",
                "detection_method": "enhanced_features",
                "detection_features": features,
                "language_scores": language_scores
            }
            
        except Exception as e:
            logger.error(f"Enhanced feature-based detection error: {e}")
            return {
                "detected_language": "en",
                "confidence": 0.1,
                "language_name": "English",
                "language_code": "en",
                "transcription": "",
                "detection_method": "error"
            }
    
    def _extract_comprehensive_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract comprehensive audio features for language detection"""
        try:
            # Basic spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            
            # Calculate statistics
            features = {
                'spectral_centroid': float(np.mean(spectral_centroid)),
                'spectral_rolloff': float(np.mean(spectral_rolloff)),
                'spectral_bandwidth': float(np.mean(spectral_bandwidth)),
                'spectral_contrast': float(np.mean(spectral_contrast)),
                'zero_crossing_rate': float(np.mean(zero_crossing_rate)),
                'mfcc_std': float(np.std(mfcc)),
                'mfcc_mean': float(np.mean(mfcc)),
                'chroma_mean': float(np.mean(chroma)),
                'chroma_std': float(np.std(chroma)),
                'tonnetz_mean': float(np.mean(tonnetz)),
                'spectral_centroid_std': float(np.std(spectral_centroid)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff))
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {
                'spectral_centroid': 2000.0,
                'spectral_rolloff': 4000.0,
                'spectral_bandwidth': 1500.0,
                'spectral_contrast': 0.2,
                'zero_crossing_rate': 0.1,
                'mfcc_std': 30.0,
                'mfcc_mean': 0.0,
                'chroma_mean': 0.3,
                'chroma_std': 0.1,
                'tonnetz_mean': 0.0,
                'spectral_centroid_std': 200.0,
                'spectral_rolloff_std': 500.0
            }
    
    def _enhance_indian_language_detection(self, text: str, detected_lang: str, ai_confidence: float) -> tuple:
        """Enhanced Indian language detection with character analysis"""
        confidence = ai_confidence
        
        # Character-based detection for Indian languages
        indian_scripts = {
            'kn': ['ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ', 'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ'],
            'te': ['అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ', 'క', 'ఖ', 'గ', 'ఘ', 'ఙ'],
            'hi': ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ'],
            'ta': ['அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ', 'க', 'ங', 'ச', 'ஜ'],
            'ml': ['അ', 'ആ', 'ഇ', 'ഈ', 'ഉ', 'ഊ', 'എ', 'ഏ', 'ഐ', 'ഒ', 'ഓ', 'ഔ', 'ക', 'ഖ', 'ഗ', 'ഘ'],
            'bn': ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ'],
            'gu': ['અ', 'આ', 'ઇ', 'ઈ', 'ઉ', 'ઊ', 'એ', 'ઐ', 'ઓ', 'ઔ', 'ક', 'ખ', 'ગ', 'ઘ', 'ઙ'],
            'pa': ['ਅ', 'ਆ', 'ਇ', 'ਈ', 'ਉ', 'ਊ', 'ਏ', 'ਐ', 'ਓ', 'ਔ', 'ਕ', 'ਖ', 'ਗ', 'ਘ', 'ਙ'],
            'or': ['ଅ', 'ଆ', 'ଇ', 'ଈ', 'ଉ', 'ଊ', 'ଏ', 'ଐ', 'ଓ', 'ଔ', 'କ', 'ଖ', 'ଗ', 'ଘ', 'ଙ']
        }
        
        # Check for Indian language characters
        for lang_code, chars in indian_scripts.items():
            char_count = sum(1 for char in text if char in chars)
            if char_count > 0:
                # Boost confidence for Indian languages
                confidence = max(confidence, 0.85)
                detected_lang = lang_code
                logger.info(f"Detected Indian language {lang_code} based on character analysis")
                break
        
        return detected_lang, confidence
    
    def _combine_detection_results(self, feature_result: Dict, transcription_result: Dict) -> Dict:
        """Combine feature-based and transcription-based results for better accuracy"""
        if not transcription_result:
            return feature_result
        
        # Weight the results based on confidence and method reliability
        feature_weight = 0.4
        transcription_weight = 0.6
        
        # Calculate combined confidence
        combined_confidence = (
            feature_result["confidence"] * feature_weight + 
            transcription_result["confidence"] * transcription_weight
        )
        
        # Choose the result with higher confidence, but boost if both agree
        if (feature_result["detected_language"] == transcription_result["detected_language"]):
            combined_confidence = min(0.95, combined_confidence + 0.1)  # Boost for agreement
            chosen_language = feature_result["detected_language"]
        elif transcription_result["confidence"] > feature_result["confidence"]:
            chosen_language = transcription_result["detected_language"]
        else:
            chosen_language = feature_result["detected_language"]
        
        return {
            "detected_language": chosen_language,
            "confidence": combined_confidence,
            "language_name": self.supported_languages.get(chosen_language, "Unknown"),
            "language_code": chosen_language,
            "transcription": transcription_result.get("transcription", ""),
            "detection_method": "hybrid",
            "feature_confidence": feature_result["confidence"],
            "transcription_confidence": transcription_result["confidence"]
        }
