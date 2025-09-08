"""
Feature fusion and classification system for highlight detection.
"""

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Represents a fused feature vector for classification."""
    audio_peak: float
    motion_magnitude: float
    voice_activity: float
    prosody: float
    shot_boundary: float
    scoreboard_change: Optional[float] = None
    replay_cue: Optional[float] = None
    laughter: Optional[float] = None
    applause: Optional[float] = None
    excitement: Optional[float] = None
    topic_shift: Optional[float] = None
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        features = [
            self.audio_peak,
            self.motion_magnitude,
            self.voice_activity,
            self.prosody,
            self.shot_boundary,
        ]
        
        # Add optional features
        optional_features = [
            self.scoreboard_change,
            self.replay_cue,
            self.laughter,
            self.applause,
            self.excitement,
            self.topic_shift,
        ]
        
        for feature in optional_features:
            features.append(feature if feature is not None else 0.0)
        
        return np.array(features, dtype=np.float32)


@dataclass
class ClassificationResult:
    """Result of highlight classification."""
    is_highlight: bool
    confidence: float
    logits: List[float]
    top_features: List[str]
    feature_contributions: Dict[str, float]


class FeatureFusion:
    """Fuses audio and visual features for highlight detection."""
    
    def __init__(self, mode: str = "sports"):
        self.mode = mode
        self.feature_weights = self._get_feature_weights(mode)
    
    def _get_feature_weights(self, mode: str) -> Dict[str, float]:
        """Get feature weights based on detection mode."""
        if mode == "sports":
            return {
                "audio_peak": 0.3,
                "motion_magnitude": 0.25,
                "voice_activity": 0.2,
                "prosody": 0.15,
                "shot_boundary": 0.1,
                "scoreboard_change": 0.4,
                "replay_cue": 0.35,
                "laughter": 0.0,
                "applause": 0.0,
                "excitement": 0.0,
                "topic_shift": 0.0,
            }
        elif mode == "podcast":
            return {
                "audio_peak": 0.25,
                "motion_magnitude": 0.15,
                "voice_activity": 0.2,
                "prosody": 0.2,
                "shot_boundary": 0.1,
                "scoreboard_change": 0.0,
                "replay_cue": 0.0,
                "laughter": 0.4,
                "applause": 0.35,
                "excitement": 0.3,
                "topic_shift": 0.25,
            }
        else:
            # Default weights
            return {
                "audio_peak": 0.3,
                "motion_magnitude": 0.25,
                "voice_activity": 0.2,
                "prosody": 0.15,
                "shot_boundary": 0.1,
                "scoreboard_change": 0.0,
                "replay_cue": 0.0,
                "laughter": 0.0,
                "applause": 0.0,
                "excitement": 0.0,
                "topic_shift": 0.0,
            }
    
    def fuse_features(
        self,
        audio_features: Dict[str, np.ndarray],
        vision_features: Dict[str, np.ndarray],
        timestamp: float,
        window_size: float = 5.0
    ) -> FeatureVector:
        """Fuse audio and visual features for a specific timestamp."""
        
        # Extract features around the timestamp
        audio_peak = self._extract_audio_peak(audio_features, timestamp, window_size)
        motion_magnitude = self._extract_motion_magnitude(vision_features, timestamp, window_size)
        voice_activity = self._extract_voice_activity(audio_features, timestamp, window_size)
        prosody = self._extract_prosody(audio_features, timestamp, window_size)
        shot_boundary = self._extract_shot_boundary(vision_features, timestamp, window_size)
        
        # Mode-specific features
        if self.mode == "sports":
            scoreboard_change = self._extract_scoreboard_change(vision_features, timestamp, window_size)
            replay_cue = self._extract_replay_cue(vision_features, timestamp, window_size)
            return FeatureVector(
                audio_peak=audio_peak,
                motion_magnitude=motion_magnitude,
                voice_activity=voice_activity,
                prosody=prosody,
                shot_boundary=shot_boundary,
                scoreboard_change=scoreboard_change,
                replay_cue=replay_cue
            )
        elif self.mode == "podcast":
            laughter = self._extract_laughter(audio_features, timestamp, window_size)
            applause = self._extract_applause(audio_features, timestamp, window_size)
            excitement = self._extract_excitement(audio_features, timestamp, window_size)
            topic_shift = self._extract_topic_shift(audio_features, timestamp, window_size)
            return FeatureVector(
                audio_peak=audio_peak,
                motion_magnitude=motion_magnitude,
                voice_activity=voice_activity,
                prosody=prosody,
                shot_boundary=shot_boundary,
                laughter=laughter,
                applause=applause,
                excitement=excitement,
                topic_shift=topic_shift
            )
        else:
            return FeatureVector(
                audio_peak=audio_peak,
                motion_magnitude=motion_magnitude,
                voice_activity=voice_activity,
                prosody=prosody,
                shot_boundary=shot_boundary
            )
    
    def _extract_audio_peak(self, features: Dict[str, np.ndarray], timestamp: float, window: float) -> float:
        """Extract audio peak strength around timestamp."""
        if 'short_time_energy' not in features:
            return 0.0
        
        energy = features['short_time_energy']
        # Find peak in window around timestamp
        # This is a simplified implementation
        return float(np.max(energy)) if len(energy) > 0 else 0.0
    
    def _extract_motion_magnitude(self, features: Dict[str, np.ndarray], timestamp: float, window: float) -> float:
        """Extract motion magnitude around timestamp."""
        if 'motion' not in features:
            return 0.0
        
        motion = features['motion']
        return float(np.max(motion)) if len(motion) > 0 else 0.0
    
    def _extract_voice_activity(self, features: Dict[str, np.ndarray], timestamp: float, window: float) -> float:
        """Extract voice activity level around timestamp."""
        if 'voice_activity' not in features:
            return 0.0
        
        vad = features['voice_activity']
        return float(np.mean(vad)) if len(vad) > 0 else 0.0
    
    def _extract_prosody(self, features: Dict[str, np.ndarray], timestamp: float, window: float) -> float:
        """Extract prosody features around timestamp."""
        if 'pitch' not in features or 'loudness' not in features:
            return 0.0
        
        pitch = features['pitch']
        loudness = features['loudness']
        
        # Calculate prosody as combination of pitch variation and loudness
        pitch_var = float(np.std(pitch)) if len(pitch) > 0 else 0.0
        loudness_mean = float(np.mean(loudness)) if len(loudness) > 0 else 0.0
        
        return (pitch_var + loudness_mean) / 2.0
    
    def _extract_shot_boundary(self, features: Dict[str, np.ndarray], timestamp: float, window: float) -> float:
        """Extract shot boundary strength around timestamp."""
        if 'shot_boundaries' not in features:
            return 0.0
        
        boundaries = features['shot_boundaries']
        # Check if there's a shot boundary near the timestamp
        if len(boundaries) > 0:
            distances = np.abs(boundaries - timestamp)
            min_distance = float(np.min(distances))
            return 1.0 if min_distance < window else 0.0
        
        return 0.0
    
    def _extract_scoreboard_change(self, features: Dict[str, np.ndarray], timestamp: float, window: float) -> float:
        """Extract scoreboard change indicator around timestamp."""
        if 'scoreboard_changes' not in features:
            return 0.0
        
        changes = features['scoreboard_changes']
        if len(changes) > 0:
            distances = np.abs(changes - timestamp)
            min_distance = float(np.min(distances))
            return 1.0 if min_distance < window else 0.0
        
        return 0.0
    
    def _extract_replay_cue(self, features: Dict[str, np.ndarray], timestamp: float, window: float) -> float:
        """Extract replay cue indicator around timestamp."""
        if 'replay_cues' not in features:
            return 0.0
        
        cues = features['replay_cues']
        if len(cues) > 0:
            distances = np.abs(cues - timestamp)
            min_distance = float(np.min(distances))
            return 1.0 if min_distance < window else 0.0
        
        return 0.0
    
    def _extract_laughter(self, features: Dict[str, np.ndarray], timestamp: float, window: float) -> float:
        """Extract laughter detection around timestamp."""
        # This would use a trained laughter detection model
        # For now, return a mock value based on audio features
        if 'zero_crossing_rate' not in features or 'short_time_energy' not in features:
            return 0.0
        
        zcr = features['zero_crossing_rate']
        energy = features['short_time_energy']
        
        # Simple laughter detection based on high ZCR and energy
        laughter_score = np.mean(zcr) * np.mean(energy)
        return float(laughter_score)
    
    def _extract_applause(self, features: Dict[str, np.ndarray], timestamp: float, window: float) -> float:
        """Extract applause detection around timestamp."""
        if 'percussive_energy' not in features:
            return 0.0
        
        percussive = features['percussive_energy']
        return float(np.max(percussive)) if len(percussive) > 0 else 0.0
    
    def _extract_excitement(self, features: Dict[str, np.ndarray], timestamp: float, window: float) -> float:
        """Extract excitement level around timestamp."""
        if 'pitch' not in features or 'loudness' not in features:
            return 0.0
        
        pitch = features['pitch']
        loudness = features['loudness']
        
        # Excitement based on pitch and loudness variation
        pitch_excitement = float(np.std(pitch)) if len(pitch) > 0 else 0.0
        loudness_excitement = float(np.std(loudness)) if len(loudness) > 0 else 0.0
        
        return (pitch_excitement + loudness_excitement) / 2.0
    
    def _extract_topic_shift(self, features: Dict[str, np.ndarray], timestamp: float, window: float) -> float:
        """Extract topic shift indicator around timestamp."""
        # This would use semantic analysis or topic modeling
        # For now, return a mock value
        return 0.0


class HighlightClassifier:
    """Neural network classifier for highlight detection."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            "audio_peak", "motion_magnitude", "voice_activity", "prosody", "shot_boundary",
            "scoreboard_change", "replay_cue", "laughter", "applause", "excitement", "topic_shift"
        ]
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self._create_default_model()
    
    def _create_default_model(self):
        """Create a default neural network model."""
        class HighlightNet(nn.Module):
            def __init__(self, input_size: int = 11):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 3),  # 3 classes: not_highlight, highlight, strong_highlight
                )
            
            def forward(self, x):
                return self.network(x)
        
        self.model = HighlightNet()
        logger.info("Created default highlight classification model")
    
    def load_model(self, model_path: str):
        """Load a pre-trained model."""
        try:
            if model_path.endswith('.onnx'):
                # Load ONNX model
                self.model = ort.InferenceSession(model_path)
                logger.info(f"Loaded ONNX model from {model_path}")
            else:
                # Load PyTorch model
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                logger.info(f"Loaded PyTorch model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            self._create_default_model()
    
    def predict(self, feature_vector: FeatureVector) -> ClassificationResult:
        """Predict highlight probability for a feature vector."""
        features = feature_vector.to_array()
        
        if isinstance(self.model, ort.InferenceSession):
            # ONNX inference
            input_name = self.model.get_inputs()[0].name
            logits = self.model.run(None, {input_name: features.reshape(1, -1)})[0][0]
        else:
            # PyTorch inference
            with torch.no_grad():
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                logits = self.model(features_tensor).squeeze().numpy()
        
        # Convert logits to probabilities
        probabilities = self._softmax(logits)
        
        # Determine if it's a highlight (class 1 or 2)
        is_highlight = probabilities[1] + probabilities[2] > 0.5
        confidence = float(max(probabilities[1], probabilities[2]))
        
        # Get top contributing features
        top_features = self._get_top_features(features, feature_vector)
        
        # Calculate feature contributions
        feature_contributions = self._calculate_feature_contributions(features)
        
        return ClassificationResult(
            is_highlight=is_highlight,
            confidence=confidence,
            logits=logits.tolist(),
            top_features=top_features,
            feature_contributions=feature_contributions
        )
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to convert logits to probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _get_top_features(self, features: np.ndarray, feature_vector: FeatureVector) -> List[str]:
        """Get top contributing features."""
        # Simple heuristic: features with highest values
        feature_values = []
        for i, name in enumerate(self.feature_names):
            if i < len(features):
                feature_values.append((name, features[i]))
        
        # Sort by value and return top 3
        feature_values.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in feature_values[:3]]
    
    def _calculate_feature_contributions(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate contribution of each feature to the prediction."""
        contributions = {}
        for i, name in enumerate(self.feature_names):
            if i < len(features):
                contributions[name] = float(features[i])
        return contributions


class HighlightDetector:
    """Main highlight detection system that combines fusion and classification."""
    
    def __init__(self, mode: str = "sports", model_path: Optional[str] = None):
        self.mode = mode
        self.fusion = FeatureFusion(mode)
        self.classifier = HighlightClassifier(model_path)
        self.min_confidence = 0.4
        self.non_max_suppression_window = 2.0
    
    def detect_highlights(
        self,
        audio_features: Dict[str, np.ndarray],
        vision_features: Dict[str, np.ndarray],
        timestamps: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect highlights in the given features."""
        highlights = []
        
        for timestamp in timestamps:
            # Fuse features for this timestamp
            feature_vector = self.fusion.fuse_features(
                audio_features, vision_features, timestamp
            )
            
            # Classify
            result = self.classifier.predict(feature_vector)
            
            if result.is_highlight and result.confidence >= self.min_confidence:
                highlight = {
                    'timestamp': timestamp,
                    'confidence': result.confidence,
                    'logits': result.logits,
                    'top_features': result.top_features,
                    'feature_contributions': result.feature_contributions,
                    'feature_vector': feature_vector
                }
                highlights.append(highlight)
        
        # Apply non-maximum suppression
        highlights = self._apply_non_maximum_suppression(highlights)
        
        # Convert to final format
        return self._format_highlights(highlights)
    
    def _apply_non_maximum_suppression(self, highlights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply non-maximum suppression to remove nearby duplicates."""
        if not highlights:
            return highlights
        
        # Sort by confidence
        highlights.sort(key=lambda x: x['confidence'], reverse=True)
        
        suppressed = []
        for highlight in highlights:
            # Check if this highlight is too close to any already selected
            too_close = False
            for selected in suppressed:
                if abs(highlight['timestamp'] - selected['timestamp']) < self.non_max_suppression_window:
                    too_close = True
                    break
            
            if not too_close:
                suppressed.append(highlight)
        
        return suppressed
    
    def _format_highlights(self, highlights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format highlights for output."""
        formatted = []
        
        for highlight in highlights:
            # Generate label based on mode and features
            label = self._generate_label(highlight)
            category = self._generate_category(highlight)
            
            # Estimate start and end times
            start_time = max(0, highlight['timestamp'] - 2.0)
            end_time = highlight['timestamp'] + 4.0
            
            formatted_highlight = {
                'id': f"highlight_{len(formatted)}",
                'start_time': start_time,
                'end_time': end_time,
                'confidence': highlight['confidence'],
                'label': label,
                'category': category,
                'features': highlight['feature_contributions'],
                'evidence': {
                    'classifier_logits': highlight['logits'],
                    'top_features': highlight['top_features']
                }
            }
            formatted.append(formatted_highlight)
        
        return formatted
    
    def _generate_label(self, highlight: Dict[str, Any]) -> str:
        """Generate a descriptive label for the highlight."""
        if self.mode == "sports":
            if highlight['feature_contributions'].get('scoreboard_change', 0) > 0.5:
                return "Score Change"
            elif highlight['feature_contributions'].get('replay_cue', 0) > 0.5:
                return "Replay Moment"
            elif highlight['feature_contributions'].get('audio_peak', 0) > 0.7:
                return "Big Play"
            else:
                return "Key Moment"
        elif self.mode == "podcast":
            if highlight['feature_contributions'].get('laughter', 0) > 0.5:
                return "Funny Moment"
            elif highlight['feature_contributions'].get('applause', 0) > 0.5:
                return "Applause"
            elif highlight['feature_contributions'].get('excitement', 0) > 0.6:
                return "Exciting Point"
            else:
                return "Notable Moment"
        else:
            return "Highlight"
    
    def _generate_category(self, highlight: Dict[str, Any]) -> str:
        """Generate a category for the highlight."""
        if self.mode == "sports":
            return "sports_action"
        elif self.mode == "podcast":
            return "podcast_moment"
        else:
            return "general_highlight"
