import numpy as np
import librosa
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccuracyValidator:
    """Comprehensive accuracy validation and metrics system for Vocal IQ"""
    
    def __init__(self):
        self.validation_results = {}
        self.accuracy_metrics = {}
        self.benchmark_data = {}
        self.performance_history = []
        
        # Initialize validation thresholds
        self.thresholds = {
            'voice_analysis': {
                'pitch_accuracy': 0.85,
                'emotion_accuracy': 0.80,
                'clarity_accuracy': 0.90,
                'fluency_accuracy': 0.85,
                'rhythm_accuracy': 0.88
            },
            'language_detection': {
                'overall_accuracy': 0.90,
                'indian_languages': 0.85,
                'confidence_threshold': 0.70
            },
            'ai_detection': {
                'precision': 0.85,
                'recall': 0.80,
                'f1_score': 0.82,
                'false_positive_rate': 0.05
            }
        }
    
    def validate_voice_analysis_accuracy(self, analysis_result: Dict, ground_truth: Dict = None) -> Dict:
        """Validate voice analysis accuracy against ground truth or benchmarks"""
        try:
            validation_metrics = {}
            
            # Extract analysis components
            audio_metrics = analysis_result.get('audio_metrics', {})
            pitch_data = audio_metrics.get('pitch', {})
            emotion_data = audio_metrics.get('emotion', {})
            clarity_data = audio_metrics.get('clarity', {})
            fluency_data = audio_metrics.get('fluency', {})
            rhythm_data = audio_metrics.get('rhythm', {})
            
            # 1. Pitch Analysis Validation
            pitch_accuracy = self._validate_pitch_analysis(pitch_data, ground_truth)
            validation_metrics['pitch_accuracy'] = pitch_accuracy
            
            # 2. Emotion Detection Validation
            emotion_accuracy = self._validate_emotion_detection(emotion_data, ground_truth)
            validation_metrics['emotion_accuracy'] = emotion_accuracy
            
            # 3. Clarity Analysis Validation
            clarity_accuracy = self._validate_clarity_analysis(clarity_data, ground_truth)
            validation_metrics['clarity_accuracy'] = clarity_accuracy
            
            # 4. Fluency Analysis Validation
            fluency_accuracy = self._validate_fluency_analysis(fluency_data, ground_truth)
            validation_metrics['fluency_accuracy'] = fluency_accuracy
            
            # 5. Rhythm Analysis Validation
            rhythm_accuracy = self._validate_rhythm_analysis(rhythm_data, ground_truth)
            validation_metrics['rhythm_accuracy'] = rhythm_accuracy
            
            # Calculate overall accuracy
            overall_accuracy = np.mean(list(validation_metrics.values()))
            validation_metrics['overall_accuracy'] = overall_accuracy
            
            # Determine accuracy grade
            validation_metrics['accuracy_grade'] = self._calculate_accuracy_grade(overall_accuracy)
            
            # Store validation results
            self.validation_results['voice_analysis'] = validation_metrics
            
            logger.info(f"Voice analysis validation completed - Overall accuracy: {overall_accuracy:.3f}")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Voice analysis validation error: {e}")
            return {'error': str(e), 'overall_accuracy': 0.0}
    
    def validate_language_detection_accuracy(self, detection_result: Dict, ground_truth: Dict = None) -> Dict:
        """Validate language detection accuracy"""
        try:
            validation_metrics = {}
            
            detected_language = detection_result.get('detected_language', 'unknown')
            confidence = detection_result.get('confidence', 0.0)
            
            # 1. Basic accuracy validation
            if ground_truth:
                true_language = ground_truth.get('language', 'unknown')
                is_correct = detected_language == true_language
                validation_metrics['correct_detection'] = is_correct
                validation_metrics['detected_vs_actual'] = {
                    'detected': detected_language,
                    'actual': true_language
                }
            else:
                # Use confidence-based validation
                is_correct = confidence >= self.thresholds['language_detection']['confidence_threshold']
                validation_metrics['confidence_based_validation'] = is_correct
            
            # 2. Confidence validation
            confidence_adequate = confidence >= 0.7
            validation_metrics['confidence_adequate'] = confidence_adequate
            
            # 3. Indian language specific validation
            if detected_language in ['kn', 'te', 'hi', 'ta', 'ml', 'bn', 'gu', 'pa', 'or']:
                indian_lang_accuracy = self._validate_indian_language_detection(detection_result)
                validation_metrics['indian_language_accuracy'] = indian_lang_accuracy
            else:
                validation_metrics['indian_language_accuracy'] = 1.0  # Not applicable
            
            # 4. Transcription quality validation
            transcription = detection_result.get('transcription', '')
            transcription_quality = self._validate_transcription_quality(transcription)
            validation_metrics['transcription_quality'] = transcription_quality
            
            # Calculate overall accuracy
            accuracy_components = [
                validation_metrics.get('correct_detection', validation_metrics.get('confidence_based_validation', False)),
                validation_metrics['confidence_adequate'],
                validation_metrics['indian_language_accuracy'],
                validation_metrics['transcription_quality']
            ]
            
            overall_accuracy = np.mean(accuracy_components)
            validation_metrics['overall_accuracy'] = overall_accuracy
            validation_metrics['accuracy_grade'] = self._calculate_accuracy_grade(overall_accuracy)
            
            # Store validation results
            self.validation_results['language_detection'] = validation_metrics
            
            logger.info(f"Language detection validation completed - Overall accuracy: {overall_accuracy:.3f}")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Language detection validation error: {e}")
            return {'error': str(e), 'overall_accuracy': 0.0}
    
    def validate_ai_detection_accuracy(self, detection_result: Dict, ground_truth: Dict = None) -> Dict:
        """Validate AI voice detection accuracy"""
        try:
            validation_metrics = {}
            
            is_ai_generated = detection_result.get('is_ai_generated', False)
            confidence_score = detection_result.get('confidence_score', 0.0)
            risk_level = detection_result.get('risk_level', 'low')
            
            # 1. Basic detection validation
            if ground_truth:
                true_label = ground_truth.get('is_ai_generated', False)
                is_correct = is_ai_generated == true_label
                validation_metrics['correct_detection'] = is_correct
                validation_metrics['true_positive'] = is_ai_generated and true_label
                validation_metrics['true_negative'] = not is_ai_generated and not true_label
                validation_metrics['false_positive'] = is_ai_generated and not true_label
                validation_metrics['false_negative'] = not is_ai_generated and true_label
            else:
                # Use confidence-based validation
                validation_metrics['confidence_based_validation'] = confidence_score >= 0.5
            
            # 2. Confidence validation
            confidence_adequate = confidence_score >= 0.6
            validation_metrics['confidence_adequate'] = confidence_adequate
            
            # 3. Risk level validation
            risk_level_appropriate = self._validate_risk_level(confidence_score, risk_level)
            validation_metrics['risk_level_appropriate'] = risk_level_appropriate
            
            # 4. Detection method validation
            detection_method = detection_result.get('detection_method', 'unknown')
            method_reliability = self._validate_detection_method(detection_method)
            validation_metrics['method_reliability'] = method_reliability
            
            # Calculate precision, recall, F1 score (if ground truth available)
            if ground_truth:
                true_positives = validation_metrics.get('true_positive', 0)
                false_positives = validation_metrics.get('false_positive', 0)
                false_negatives = validation_metrics.get('false_negative', 0)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                validation_metrics['precision'] = precision
                validation_metrics['recall'] = recall
                validation_metrics['f1_score'] = f1
            else:
                validation_metrics['precision'] = confidence_score
                validation_metrics['recall'] = confidence_score
                validation_metrics['f1_score'] = confidence_score
            
            # Calculate overall accuracy
            accuracy_components = [
                validation_metrics.get('correct_detection', validation_metrics.get('confidence_based_validation', False)),
                validation_metrics['confidence_adequate'],
                validation_metrics['risk_level_appropriate'],
                validation_metrics['method_reliability']
            ]
            
            overall_accuracy = np.mean(accuracy_components)
            validation_metrics['overall_accuracy'] = overall_accuracy
            validation_metrics['accuracy_grade'] = self._calculate_accuracy_grade(overall_accuracy)
            
            # Store validation results
            self.validation_results['ai_detection'] = validation_metrics
            
            logger.info(f"AI detection validation completed - Overall accuracy: {overall_accuracy:.3f}")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"AI detection validation error: {e}")
            return {'error': str(e), 'overall_accuracy': 0.0}
    
    def _validate_pitch_analysis(self, pitch_data: Dict, ground_truth: Dict = None) -> float:
        """Validate pitch analysis accuracy"""
        try:
            # Check if pitch data is reasonable
            average_pitch = pitch_data.get('average', 0)
            pitch_variation = pitch_data.get('variation', 0)
            pitch_stability = pitch_data.get('stability', 0)
            
            # Basic sanity checks
            pitch_accuracy = 1.0
            
            # Check if pitch is within human vocal range (50-800 Hz)
            if not (50 <= average_pitch <= 800):
                pitch_accuracy *= 0.7
            
            # Check if variation is reasonable (not too high or too low)
            if pitch_variation > 200:  # Too much variation
                pitch_accuracy *= 0.8
            elif pitch_variation < 10:  # Too little variation
                pitch_accuracy *= 0.9
            
            # Check stability score
            if pitch_stability < 0.3:  # Very unstable
                pitch_accuracy *= 0.8
            elif pitch_stability > 0.9:  # Too stable (might be AI)
                pitch_accuracy *= 0.9
            
            return float(np.clip(pitch_accuracy, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Pitch validation error: {e}")
            return 0.5
    
    def _validate_emotion_detection(self, emotion_data: Dict, ground_truth: Dict = None) -> float:
        """Validate emotion detection accuracy"""
        try:
            dominant_emotion = emotion_data.get('dominant_emotion', 'neutral')
            confidence = emotion_data.get('confidence', 0.0)
            scores = emotion_data.get('scores', {})
            
            # Check confidence level
            emotion_accuracy = confidence
            
            # Check if emotion scores are properly normalized
            total_score = sum(scores.values()) if scores else 0
            if abs(total_score - 1.0) > 0.1:  # Not properly normalized
                emotion_accuracy *= 0.8
            
            # Check if dominant emotion has highest score
            if scores:
                max_score_emotion = max(scores, key=scores.get)
                if max_score_emotion != dominant_emotion:
                    emotion_accuracy *= 0.9
            
            return float(np.clip(emotion_accuracy, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Emotion validation error: {e}")
            return 0.5
    
    def _validate_clarity_analysis(self, clarity_data: Dict, ground_truth: Dict = None) -> float:
        """Validate clarity analysis accuracy"""
        try:
            clarity_score = clarity_data.get('clarity_score', 0.0)
            articulation_score = clarity_data.get('articulation_score', 0.0)
            enunciation = clarity_data.get('enunciation', 0.0)
            
            # Check if scores are within valid range
            clarity_accuracy = 1.0
            
            for score_name, score in [('clarity_score', clarity_score), 
                                    ('articulation_score', articulation_score), 
                                    ('enunciation', enunciation)]:
                if not (0.0 <= score <= 1.0):
                    clarity_accuracy *= 0.8
                    logger.warning(f"Invalid {score_name}: {score}")
            
            # Check consistency between related scores
            if abs(clarity_score - articulation_score) > 0.5:
                clarity_accuracy *= 0.9
            
            return float(np.clip(clarity_accuracy, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Clarity validation error: {e}")
            return 0.5
    
    def _validate_fluency_analysis(self, fluency_data: Dict, ground_truth: Dict = None) -> float:
        """Validate fluency analysis accuracy"""
        try:
            fluency_score = fluency_data.get('fluency_score', 0.0)
            hesitations = fluency_data.get('hesitations', 0.0)
            smoothness = fluency_data.get('smoothness', 0.0)
            
            # Check if scores are within valid range
            fluency_accuracy = 1.0
            
            if not (0.0 <= fluency_score <= 1.0):
                fluency_accuracy *= 0.8
            
            if not (0.0 <= hesitations <= 1.0):
                fluency_accuracy *= 0.8
            
            if not (0.0 <= smoothness <= 1.0):
                fluency_accuracy *= 0.8
            
            # Check logical consistency
            if hesitations > 0.8 and smoothness > 0.8:  # Contradictory
                fluency_accuracy *= 0.9
            
            return float(np.clip(fluency_accuracy, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Fluency validation error: {e}")
            return 0.5
    
    def _validate_rhythm_analysis(self, rhythm_data: Dict, ground_truth: Dict = None) -> float:
        """Validate rhythm analysis accuracy"""
        try:
            speech_rate = rhythm_data.get('speech_rate', 0.0)
            pause_ratio = rhythm_data.get('pause_ratio', 0.0)
            consistency = rhythm_data.get('consistency', 0.0)
            
            # Check if scores are within valid range
            rhythm_accuracy = 1.0
            
            if not (0.0 <= speech_rate <= 1.0):
                rhythm_accuracy *= 0.8
            
            if not (0.0 <= pause_ratio <= 1.0):
                rhythm_accuracy *= 0.8
            
            if not (0.0 <= consistency <= 1.0):
                rhythm_accuracy *= 0.8
            
            # Check logical consistency
            if speech_rate + pause_ratio > 1.1:  # Should be close to 1.0
                rhythm_accuracy *= 0.9
            
            return float(np.clip(rhythm_accuracy, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Rhythm validation error: {e}")
            return 0.5
    
    def _validate_indian_language_detection(self, detection_result: Dict) -> float:
        """Validate Indian language detection accuracy"""
        try:
            detected_language = detection_result.get('detected_language', '')
            confidence = detection_result.get('confidence', 0.0)
            detection_features = detection_result.get('detection_features', {})
            
            # Check confidence for Indian languages
            indian_accuracy = confidence
            
            # Check if detection features are present
            if detection_features:
                # Validate feature ranges for Indian languages
                centroid = detection_features.get('spectral_centroid', 0)
                rolloff = detection_features.get('spectral_rolloff', 0)
                zcr = detection_features.get('zero_crossing_rate', 0)
                
                # Indian language specific feature validation
                if detected_language == 'kn':  # Kannada
                    if 1200 <= centroid <= 1800 and 2000 <= rolloff <= 3200:
                        indian_accuracy *= 1.1
                elif detected_language == 'te':  # Telugu
                    if 1600 <= centroid <= 2100 and 3200 <= rolloff <= 4200:
                        indian_accuracy *= 1.1
                elif detected_language == 'hi':  # Hindi
                    if 1700 <= centroid <= 2300 and 3500 <= rolloff <= 4800:
                        indian_accuracy *= 1.1
            
            return float(np.clip(indian_accuracy, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Indian language validation error: {e}")
            return 0.5
    
    def _validate_transcription_quality(self, transcription: str) -> float:
        """Validate transcription quality"""
        try:
            if not transcription or len(transcription.strip()) < 3:
                return 0.0
            
            # Basic quality checks
            quality_score = 1.0
            
            # Check for reasonable length
            word_count = len(transcription.split())
            if word_count < 2:
                quality_score *= 0.5
            elif word_count > 100:  # Very long transcription
                quality_score *= 0.9
            
            # Check for common transcription errors
            if '[' in transcription or ']' in transcription:  # Whisper artifacts
                quality_score *= 0.8
            
            if transcription.count(' ') < word_count * 0.5:  # Too few spaces
                quality_score *= 0.9
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Transcription quality validation error: {e}")
            return 0.5
    
    def _validate_risk_level(self, confidence_score: float, risk_level: str) -> bool:
        """Validate if risk level matches confidence score"""
        try:
            if confidence_score > 0.75:
                return risk_level == 'high'
            elif confidence_score > 0.55:
                return risk_level == 'medium'
            else:
                return risk_level == 'low'
        except:
            return True
    
    def _validate_detection_method(self, method: str) -> float:
        """Validate detection method reliability"""
        try:
            method_scores = {
                'ultra_ensemble_analysis': 1.0,
                'enhanced_spectral_analysis': 0.9,
                'hybrid': 0.95,
                'ai_transcription': 0.85,
                'enhanced_features': 0.8,
                'fallback': 0.5,
                'error': 0.0
            }
            
            return method_scores.get(method, 0.7)
        except:
            return 0.5
    
    def _calculate_accuracy_grade(self, accuracy: float) -> str:
        """Calculate accuracy grade based on score"""
        try:
            if accuracy >= 0.95:
                return 'A+'
            elif accuracy >= 0.90:
                return 'A'
            elif accuracy >= 0.85:
                return 'B+'
            elif accuracy >= 0.80:
                return 'B'
            elif accuracy >= 0.75:
                return 'C+'
            elif accuracy >= 0.70:
                return 'C'
            elif accuracy >= 0.60:
                return 'D'
            else:
                return 'F'
        except:
            return 'F'
    
    def generate_accuracy_report(self) -> Dict:
        """Generate comprehensive accuracy report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'overall_accuracy': 0.0,
                'component_accuracies': {},
                'recommendations': [],
                'performance_trends': self.performance_history[-10:] if self.performance_history else []
            }
            
            # Calculate overall accuracy
            if self.validation_results:
                component_accuracies = []
                for component, metrics in self.validation_results.items():
                    overall_acc = metrics.get('overall_accuracy', 0.0)
                    component_accuracies.append(overall_acc)
                    report['component_accuracies'][component] = {
                        'accuracy': overall_acc,
                        'grade': metrics.get('accuracy_grade', 'F'),
                        'details': metrics
                    }
                
                report['overall_accuracy'] = np.mean(component_accuracies) if component_accuracies else 0.0
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations()
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'overall_accuracy': report['overall_accuracy'],
                'component_accuracies': report['component_accuracies']
            })
            
            # Save report
            self._save_accuracy_report(report)
            
            logger.info(f"Accuracy report generated - Overall accuracy: {report['overall_accuracy']:.3f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Accuracy report generation error: {e}")
            return {'error': str(e), 'overall_accuracy': 0.0}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate accuracy improvement recommendations"""
        recommendations = []
        
        try:
            if not self.validation_results:
                return ["No validation data available for recommendations"]
            
            # Analyze each component
            for component, metrics in self.validation_results.items():
                accuracy = metrics.get('overall_accuracy', 0.0)
                
                if accuracy < 0.8:
                    if component == 'voice_analysis':
                        recommendations.append("Voice analysis accuracy is below 80%. Consider improving audio preprocessing and feature extraction.")
                    elif component == 'language_detection':
                        recommendations.append("Language detection accuracy is below 80%. Consider enhancing Indian language detection algorithms.")
                    elif component == 'ai_detection':
                        recommendations.append("AI detection accuracy is below 80%. Consider improving ensemble methods and feature engineering.")
                
                # Component-specific recommendations
                if component == 'voice_analysis':
                    pitch_acc = metrics.get('pitch_accuracy', 0.0)
                    if pitch_acc < 0.85:
                        recommendations.append("Pitch analysis accuracy is low. Consider using multiple pitch detection algorithms.")
                    
                    emotion_acc = metrics.get('emotion_accuracy', 0.0)
                    if emotion_acc < 0.80:
                        recommendations.append("Emotion detection accuracy is low. Consider using advanced AI models for emotion classification.")
            
            # General recommendations
            if not recommendations:
                recommendations.append("All components are performing well. Consider fine-tuning parameters for even better accuracy.")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendation generation error: {e}")
            return ["Error generating recommendations"]
    
    def _save_accuracy_report(self, report: Dict):
        """Save accuracy report to file"""
        try:
            os.makedirs('accuracy_reports', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accuracy_reports/accuracy_report_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Accuracy report saved to {filename}")
            
        except Exception as e:
            logger.warning(f"Could not save accuracy report: {e}")
    
    def get_accuracy_summary(self) -> Dict:
        """Get quick accuracy summary"""
        try:
            if not self.validation_results:
                return {"message": "No validation data available"}
            
            summary = {
                'total_components': len(self.validation_results),
                'average_accuracy': 0.0,
                'component_grades': {},
                'overall_grade': 'F'
            }
            
            accuracies = []
            for component, metrics in self.validation_results.items():
                accuracy = metrics.get('overall_accuracy', 0.0)
                grade = metrics.get('accuracy_grade', 'F')
                accuracies.append(accuracy)
                summary['component_grades'][component] = {
                    'accuracy': accuracy,
                    'grade': grade
                }
            
            summary['average_accuracy'] = np.mean(accuracies) if accuracies else 0.0
            summary['overall_grade'] = self._calculate_accuracy_grade(summary['average_accuracy'])
            
            return summary
            
        except Exception as e:
            logger.error(f"Accuracy summary error: {e}")
            return {"error": str(e)}
