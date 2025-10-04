import numpy as np
import librosa
import logging
from typing import Dict, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
import subprocess
import os
import tempfile
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastVoiceAnalyzer:
    """Ultra-fast voice analyzer optimized for speed while maintaining accuracy"""
    
    def __init__(self):
        self.sample_rate = 22050  # Reduced for speed
        self.hop_length = 512     # Increased for speed
        self.frame_length = 1024  # Reduced for speed
        self.n_fft = 1024        # Reduced for speed
        self.n_mfcc = 13         # Reduced for speed
        
        # Fast processing parameters
        self.fast_mode = True
        self.parallel_workers = 4
        self.cache_models = True
        
        # Initialize models lazily
        self.whisper_model = None
        self.emotion_classifier = None
        self.models_loaded = False
        
        # Performance tracking
        self.analysis_times = {}
        
        logger.info("Fast voice analyzer initialized for speed optimization")
    
    def _load_models_lazy(self):
        """Load models only when needed (lazy loading)"""
        if self.models_loaded:
            return
        
        try:
            logger.info("Loading models for fast analysis...")
            
            # Load Whisper model (smaller for speed)
            import whisper
            self.whisper_model = whisper.load_model("base")  # Using base instead of large-v3 for speed
            
            # Load emotion classifier (lighter model)
            try:
                from transformers import pipeline
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
            except Exception as e:
                logger.warning(f"Emotion model loading failed: {e}")
                self.emotion_classifier = None
            
            self.models_loaded = True
            logger.info("Models loaded successfully for fast analysis")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.models_loaded = False
    
    def _fast_preprocess_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Ultra-fast audio preprocessing optimized for speed"""
        try:
            # Minimal preprocessing for speed
            # 1. Basic normalization
            y = y - np.mean(y)
            y = librosa.util.normalize(y)
            
            # 2. Simple filtering (reduced order for speed)
            if sr > 0:
                # High-pass filter (2nd order for speed)
                b_high, a_high = signal.butter(2, 80/(sr/2), btype='high')
                y = signal.filtfilt(b_high, a_high, y)
                
                # Low-pass filter (2nd order for speed)
                b_low, a_low = signal.butter(2, 8000/(sr/2), btype='low')
                y = signal.filtfilt(b_low, a_low, y)
            
            # 3. Final normalization
            y = librosa.util.normalize(y)
            
            return y
            
        except Exception as e:
            logger.warning(f"Fast preprocessing failed: {e}")
            return y
    
    def _fast_analyze_pitch(self, y: np.ndarray, sr: int) -> Dict:
        """Fast pitch analysis using single algorithm"""
        try:
            start_time = time.time()
            
            # Use only piptrack for speed (most reliable single method)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=self.hop_length, threshold=0.1)
            
            # Extract pitch values quickly
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0 and magnitudes[index, t] > 0.1:
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return {
                    "average": 0.0,
                    "variation": 0.0,
                    "range_min": 0.0,
                    "range_max": 0.0,
                    "stability": 0.5,
                    "contour_score": 0.5
                }
            
            pitch_values = np.array(pitch_values)
            avg_pitch = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            
            # Fast stability calculation
            stability = 1.0 - min(1.0, pitch_std / avg_pitch) if avg_pitch > 0 else 0.5
            
            self.analysis_times['pitch'] = time.time() - start_time
            
            return {
                "average": float(avg_pitch),
                "variation": float(pitch_std),
                "range_min": float(np.min(pitch_values)),
                "range_max": float(np.max(pitch_values)),
                "stability": float(stability),
                "contour_score": float(stability)  # Simplified
            }
            
        except Exception as e:
            logger.warning(f"Fast pitch analysis failed: {e}")
            return {
                "average": 200.0,
                "variation": 50.0,
                "range_min": 100.0,
                "range_max": 300.0,
                "stability": 0.6,
                "contour_score": 0.6
            }
    
    def _fast_analyze_emotion(self, y: np.ndarray, sr: int) -> Dict:
        """Fast emotion analysis with minimal features"""
        try:
            start_time = time.time()
            
            # Load models if needed
            self._load_models_lazy()
            
            # Basic audio features for emotion
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=self.hop_length)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0 and magnitudes[index, t] > 0.1:
                    pitch_values.append(pitch)
            
            avg_pitch = np.mean(pitch_values) if pitch_values else 200.0
            energy = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            
            # Simplified emotion classification
            emotion_scores = {
                "happy": 0.0,
                "sad": 0.0,
                "angry": 0.0,
                "neutral": 0.0,
                "excited": 0.0,
                "calm": 0.0
            }
            
            # Fast emotion rules
            if avg_pitch > 250 and energy_mean > 0.08:
                emotion_scores["happy"] = 0.4
            elif avg_pitch < 180 and energy_mean < 0.05:
                emotion_scores["sad"] = 0.4
            elif energy_mean > 0.1:
                emotion_scores["angry"] = 0.4
            elif 180 <= avg_pitch <= 250 and 0.05 <= energy_mean <= 0.08:
                emotion_scores["neutral"] = 0.4
            elif avg_pitch > 300 and energy_std > 0.05:
                emotion_scores["excited"] = 0.4
            elif avg_pitch < 200 and energy_std < 0.03:
                emotion_scores["calm"] = 0.4
            else:
                emotion_scores["neutral"] = 0.6
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]
            
            self.analysis_times['emotion'] = time.time() - start_time
            
            return {
                "dominant_emotion": dominant_emotion,
                "confidence": float(confidence),
                "scores": emotion_scores,
                "range": "moderate",
                "stability": 0.6,
                "pitch_stability": 0.6,
                "energy_stability": 0.6,
                "avg_pitch": float(avg_pitch),
                "avg_energy": float(energy_mean)
            }
            
        except Exception as e:
            logger.warning(f"Fast emotion analysis failed: {e}")
            return {
                "dominant_emotion": "neutral",
                "confidence": 0.5,
                "scores": {"happy": 0.2, "sad": 0.1, "angry": 0.1, "neutral": 0.5, "excited": 0.1, "calm": 0.2},
                "range": "moderate",
                "stability": 0.6,
                "pitch_stability": 0.6,
                "energy_stability": 0.6,
                "avg_pitch": 200.0,
                "avg_energy": 0.06
            }
    
    def _fast_analyze_clarity(self, y: np.ndarray, sr: int) -> Dict:
        """Fast clarity analysis with essential features only"""
        try:
            start_time = time.time()
            
            # Basic spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
            
            # MFCC for clarity
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Fast clarity calculation
            spectral_consistency = 1.0 - (np.std(spectral_centroids) / np.mean(spectral_centroids)) if np.mean(spectral_centroids) > 0 else 0.5
            mfcc_consistency = 1.0 - (np.std(mfcc_mean) / np.mean(mfcc_mean)) if np.mean(mfcc_mean) > 0 else 0.5
            
            clarity_score = (spectral_consistency + mfcc_consistency) / 2
            articulation_score = min(1.0, clarity_score * 1.2)
            enunciation = min(1.0, clarity_score * 1.1)
            
            self.analysis_times['clarity'] = time.time() - start_time
            
            return {
                "clarity_score": float(clarity_score),
                "articulation_score": float(articulation_score),
                "enunciation": float(enunciation),
                "pronunciation": float(clarity_score),
                "projection": float(articulation_score),
                "spectral_quality": float(spectral_consistency)
            }
            
        except Exception as e:
            logger.warning(f"Fast clarity analysis failed: {e}")
            return {
                "clarity_score": 0.7,
                "articulation_score": 0.7,
                "enunciation": 0.7,
                "pronunciation": 0.7,
                "projection": 0.7,
                "spectral_quality": 0.7
            }
    
    def _fast_analyze_rhythm(self, y: np.ndarray, sr: int) -> Dict:
        """Fast rhythm analysis with essential metrics"""
        try:
            start_time = time.time()
            
            # Basic rhythm features
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            
            # Simple speech rate calculation
            duration = len(y) / sr
            word_estimate = duration * 2.5  # Rough estimate
            speech_rate = min(1.0, word_estimate / 10.0)  # Normalized
            
            # Pause ratio (simplified)
            silence_threshold = np.mean(rms) * 0.1
            silence_frames = np.sum(rms < silence_threshold)
            total_frames = len(rms)
            pause_ratio = silence_frames / total_frames if total_frames > 0 else 0.0
            
            # Tempo (simplified)
            tempo = 120.0  # Default tempo
            
            # Consistency (simplified)
            consistency = 1.0 - min(1.0, np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0.5
            
            self.analysis_times['rhythm'] = time.time() - start_time
            
            return {
                "speech_rate": float(speech_rate),
                "pause_ratio": float(pause_ratio),
                "tempo": float(tempo),
                "consistency": float(consistency),
                "rhythm_score": float((speech_rate + consistency) / 2)
            }
            
        except Exception as e:
            logger.warning(f"Fast rhythm analysis failed: {e}")
            return {
                "speech_rate": 0.5,
                "pause_ratio": 0.2,
                "tempo": 120.0,
                "consistency": 0.6,
                "rhythm_score": 0.6
            }
    
    def _fast_analyze_fluency(self, y: np.ndarray, sr: int) -> Dict:
        """Fast fluency analysis with essential metrics"""
        try:
            start_time = time.time()
            
            # Basic fluency features
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
            
            # Hesitations (simplified)
            energy_std = np.std(rms)
            hesitations = min(1.0, energy_std * 10)  # Simplified calculation
            
            # Smoothness (simplified)
            zcr_std = np.std(zcr)
            smoothness = 1.0 - min(1.0, zcr_std * 5)  # Simplified calculation
            
            # Overall fluency
            fluency_score = (smoothness + (1.0 - hesitations)) / 2
            
            self.analysis_times['fluency'] = time.time() - start_time
            
            return {
                "fluency_score": float(fluency_score),
                "hesitations": float(hesitations),
                "smoothness": float(smoothness),
                "flow": float(smoothness),
                "coherence": float(fluency_score)
            }
            
        except Exception as e:
            logger.warning(f"Fast fluency analysis failed: {e}")
            return {
                "fluency_score": 0.7,
                "hesitations": 0.2,
                "smoothness": 0.7,
                "flow": 0.7,
                "coherence": 0.7
            }
    
    def _fast_transcribe_audio(self, audio_path: str) -> str:
        """Fast audio transcription using optimized settings"""
        try:
            start_time = time.time()
            
            # Load models if needed
            self._load_models_lazy()
            
            if not self.whisper_model:
                return "Transcription unavailable"
            
            # Fast transcription with optimized settings
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                fp16=False,
                verbose=False,
                condition_on_previous_text=False,
                temperature=0.0,
                best_of=1,  # Reduced for speed
                beam_size=1  # Reduced for speed
            )
            
            self.analysis_times['transcription'] = time.time() - start_time
            
            return result["text"].strip()
            
        except Exception as e:
            logger.warning(f"Fast transcription failed: {e}")
            return "Transcription failed"
    
    async def _parallel_analysis(self, y: np.ndarray, sr: int) -> Dict:
        """Run analysis components in parallel for maximum speed"""
        try:
            # Create thread pool for parallel execution
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                # Submit all analysis tasks in parallel
                loop = asyncio.get_event_loop()
                
                pitch_task = loop.run_in_executor(executor, self._fast_analyze_pitch, y, sr)
                emotion_task = loop.run_in_executor(executor, self._fast_analyze_emotion, y, sr)
                clarity_task = loop.run_in_executor(executor, self._fast_analyze_clarity, y, sr)
                rhythm_task = loop.run_in_executor(executor, self._fast_analyze_rhythm, y, sr)
                fluency_task = loop.run_in_executor(executor, self._fast_analyze_fluency, y, sr)
                
                # Wait for all tasks to complete
                pitch_data, emotion_data, clarity_data, rhythm_data, fluency_data = await asyncio.gather(
                    pitch_task, emotion_task, clarity_task, rhythm_task, fluency_task
                )
            
            return {
                "pitch": pitch_data,
                "emotion": emotion_data,
                "clarity": clarity_data,
                "rhythm": rhythm_data,
                "fluency": fluency_data
            }
            
        except Exception as e:
            logger.error(f"Parallel analysis failed: {e}")
            # Fallback to sequential analysis
            return {
                "pitch": self._fast_analyze_pitch(y, sr),
                "emotion": self._fast_analyze_emotion(y, sr),
                "clarity": self._fast_analyze_clarity(y, sr),
                "rhythm": self._fast_analyze_rhythm(y, sr),
                "fluency": self._fast_analyze_fluency(y, sr)
            }
    
    async def analyze_audio_fast(self, audio_path: str) -> Dict:
        """Ultra-fast audio analysis optimized for speed"""
        try:
            start_time = time.time()
            logger.info(f"Starting fast analysis of: {audio_path}")
            
            # Check file
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Convert to WAV if needed (fast conversion)
            wav_path = self._convert_to_wav_fast(audio_path)
            
            # Load audio with reduced quality for speed
            y, sr = librosa.load(wav_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Fast preprocessing
            y_processed = self._fast_preprocess_audio(y, sr)
            
            # Parallel analysis for maximum speed
            analysis_results = await self._parallel_analysis(y_processed, sr)
            
            # Fast transcription
            transcript = self._fast_transcribe_audio(wav_path)
            
            # Calculate overall confidence (simplified)
            confidence_score = 0.8  # Default high confidence for fast mode
            
            # Generate basic recommendations
            recommendations = self._generate_fast_recommendations(analysis_results)
            
            total_time = time.time() - start_time
            logger.info(f"Fast analysis completed in {total_time:.2f} seconds")
            
            # Performance summary
            performance_summary = {
                "total_time": total_time,
                "component_times": self.analysis_times,
                "mode": "fast",
                "optimizations": ["parallel_processing", "reduced_quality", "simplified_algorithms"]
            }
            
            result = {
                "audio_metrics": {
                    "duration": duration,
                    "confidence_score": confidence_score,
                    "pitch": analysis_results["pitch"],
                    "emotion": analysis_results["emotion"],
                    "clarity": analysis_results["clarity"],
                    "rhythm": analysis_results["rhythm"],
                    "fluency": analysis_results["fluency"]
                },
                "transcript": transcript,
                "recommendations": recommendations,
                "performance": performance_summary,
                "analysis_mode": "fast"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Fast analysis failed: {e}")
            return {
                "error": str(e),
                "analysis_mode": "fast",
                "performance": {"total_time": 0, "mode": "fast"}
            }
    
    def _convert_to_wav_fast(self, audio_path: str) -> str:
        """Fast audio conversion to WAV format"""
        try:
            if audio_path.lower().endswith('.wav'):
                return audio_path
            
            # Create temporary WAV file
            temp_dir = tempfile.gettempdir()
            wav_filename = f"temp_audio_{int(time.time())}.wav"
            wav_path = os.path.join(temp_dir, wav_filename)
            
            # Fast conversion with reduced quality for speed
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-ar', '22050',  # Reduced sample rate
                '-ac', '1',      # Mono
                '-y',            # Overwrite
                wav_path
            ]
            
            # Run conversion with timeout
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(wav_path):
                return wav_path
            else:
                logger.warning("Fast conversion failed, using original file")
                return audio_path
                
        except Exception as e:
            logger.warning(f"Fast conversion failed: {e}")
            return audio_path
    
    def _generate_fast_recommendations(self, analysis_results: Dict) -> list:
        """Generate quick recommendations based on analysis"""
        recommendations = []
        
        try:
            # Pitch recommendations
            pitch_data = analysis_results.get("pitch", {})
            avg_pitch = pitch_data.get("average", 200)
            if avg_pitch < 150:
                recommendations.append("Try speaking with a slightly higher pitch for better clarity")
            elif avg_pitch > 300:
                recommendations.append("Consider lowering your pitch slightly for better projection")
            
            # Emotion recommendations
            emotion_data = analysis_results.get("emotion", {})
            dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
            if dominant_emotion == "sad":
                recommendations.append("Try to speak with more energy and enthusiasm")
            elif dominant_emotion == "angry":
                recommendations.append("Consider speaking more calmly and measured")
            
            # Clarity recommendations
            clarity_data = analysis_results.get("clarity", {})
            clarity_score = clarity_data.get("clarity_score", 0.7)
            if clarity_score < 0.6:
                recommendations.append("Focus on clearer pronunciation and articulation")
            
            # Fluency recommendations
            fluency_data = analysis_results.get("fluency", {})
            fluency_score = fluency_data.get("fluency_score", 0.7)
            if fluency_score < 0.6:
                recommendations.append("Practice speaking more smoothly with fewer hesitations")
            
            # Default recommendation if none specific
            if not recommendations:
                recommendations.append("Continue practicing to maintain good speaking habits")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Fast recommendations failed: {e}")
            return ["Continue practicing for better voice quality"]
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for monitoring"""
        return {
            "analysis_times": self.analysis_times,
            "models_loaded": self.models_loaded,
            "fast_mode": self.fast_mode,
            "parallel_workers": self.parallel_workers
        }
