"""
Unit tests for feature fusion and classification.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Add packages to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from fusion.classifier import (
    FeatureVector, ClassificationResult, FeatureFusion, 
    HighlightClassifier, HighlightDetector
)


class TestFeatureVector:
    """Test cases for FeatureVector."""
    
    def test_feature_vector_creation(self):
        """Test FeatureVector creation and conversion."""
        vector = FeatureVector(
            audio_peak=0.8,
            motion_magnitude=0.6,
            voice_activity=0.7,
            prosody=0.5,
            shot_boundary=0.3
        )
        
        assert vector.audio_peak == 0.8
        assert vector.motion_magnitude == 0.6
        assert vector.voice_activity == 0.7
        assert vector.prosody == 0.5
        assert vector.shot_boundary == 0.3
    
    def test_feature_vector_to_array(self):
        """Test conversion to numpy array."""
        vector = FeatureVector(
            audio_peak=0.8,
            motion_magnitude=0.6,
            voice_activity=0.7,
            prosody=0.5,
            shot_boundary=0.3,
            scoreboard_change=0.4,
            replay_cue=0.2
        )
        
        array = vector.to_array()
        
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32
        assert len(array) == 11  # All 11 features
        assert array[0] == 0.8  # audio_peak
        assert array[5] == 0.4  # scoreboard_change
        assert array[6] == 0.2  # replay_cue
    
    def test_feature_vector_with_none_values(self):
        """Test FeatureVector with None values."""
        vector = FeatureVector(
            audio_peak=0.8,
            motion_magnitude=0.6,
            voice_activity=0.7,
            prosody=0.5,
            shot_boundary=0.3,
            scoreboard_change=None,
            replay_cue=None
        )
        
        array = vector.to_array()
        
        assert array[5] == 0.0  # scoreboard_change should be 0.0
        assert array[6] == 0.0  # replay_cue should be 0.0


class TestFeatureFusion:
    """Test cases for FeatureFusion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = FeatureFusion("sports")
    
    def test_initialization(self):
        """Test fusion initialization."""
        assert self.fusion.mode == "sports"
        assert "audio_peak" in self.fusion.feature_weights
        assert "scoreboard_change" in self.fusion.feature_weights
    
    def test_sports_mode_weights(self):
        """Test sports mode feature weights."""
        sports_fusion = FeatureFusion("sports")
        weights = sports_fusion.feature_weights
        
        assert weights["scoreboard_change"] > 0
        assert weights["replay_cue"] > 0
        assert weights["laughter"] == 0.0
        assert weights["applause"] == 0.0
    
    def test_podcast_mode_weights(self):
        """Test podcast mode feature weights."""
        podcast_fusion = FeatureFusion("podcast")
        weights = podcast_fusion.feature_weights
        
        assert weights["laughter"] > 0
        assert weights["applause"] > 0
        assert weights["excitement"] > 0
        assert weights["scoreboard_change"] == 0.0
        assert weights["replay_cue"] == 0.0
    
    def test_fuse_features_sports(self):
        """Test feature fusion for sports mode."""
        audio_features = {
            'short_time_energy': np.array([0.1, 0.8, 0.2]),
            'voice_activity': np.array([0.0, 1.0, 0.0]),
            'pitch': np.array([100, 200, 150]),
            'loudness': np.array([0.1, 0.9, 0.2])
        }
        
        vision_features = {
            'motion': np.array([0.1, 0.7, 0.2]),
            'shot_boundaries': np.array([1.5])
        }
        
        vector = self.fusion.fuse_features(
            audio_features, vision_features, timestamp=1.0, window_size=1.0
        )
        
        assert isinstance(vector, FeatureVector)
        assert vector.audio_peak >= 0
        assert vector.motion_magnitude >= 0
        assert vector.voice_activity >= 0
        assert vector.prosody >= 0
        assert vector.shot_boundary >= 0
        assert vector.scoreboard_change is not None
        assert vector.replay_cue is not None
    
    def test_fuse_features_podcast(self):
        """Test feature fusion for podcast mode."""
        podcast_fusion = FeatureFusion("podcast")
        
        audio_features = {
            'short_time_energy': np.array([0.1, 0.8, 0.2]),
            'voice_activity': np.array([0.0, 1.0, 0.0]),
            'pitch': np.array([100, 200, 150]),
            'loudness': np.array([0.1, 0.9, 0.2]),
            'zero_crossing_rate': np.array([0.1, 0.8, 0.2]),
            'percussive_energy': np.array([0.1, 0.7, 0.2])
        }
        
        vision_features = {
            'motion': np.array([0.1, 0.7, 0.2]),
            'shot_boundaries': np.array([1.5])
        }
        
        vector = podcast_fusion.fuse_features(
            audio_features, vision_features, timestamp=1.0, window_size=1.0
        )
        
        assert isinstance(vector, FeatureVector)
        assert vector.laughter is not None
        assert vector.applause is not None
        assert vector.excitement is not None
        assert vector.topic_shift is not None


class TestHighlightClassifier:
    """Test cases for HighlightClassifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = HighlightClassifier()
    
    def test_initialization(self):
        """Test classifier initialization."""
        assert self.classifier.device == "cpu"
        assert self.classifier.model is not None
        assert len(self.classifier.feature_names) == 11
    
    def test_predict(self):
        """Test highlight prediction."""
        feature_vector = FeatureVector(
            audio_peak=0.8,
            motion_magnitude=0.6,
            voice_activity=0.7,
            prosody=0.5,
            shot_boundary=0.3
        )
        
        result = self.classifier.predict(feature_vector)
        
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.is_highlight, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.logits, list)
        assert len(result.logits) == 3  # 3 classes
        assert isinstance(result.top_features, list)
        assert isinstance(result.feature_contributions, dict)
    
    def test_softmax(self):
        """Test softmax function."""
        logits = np.array([1.0, 2.0, 3.0])
        probabilities = self.classifier._softmax(logits)
        
        assert isinstance(probabilities, np.ndarray)
        assert np.isclose(np.sum(probabilities), 1.0)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
    
    def test_get_top_features(self):
        """Test top features extraction."""
        features = np.array([0.1, 0.8, 0.3, 0.9, 0.2])
        vector = FeatureVector(
            audio_peak=0.1,
            motion_magnitude=0.8,
            voice_activity=0.3,
            prosody=0.9,
            shot_boundary=0.2
        )
        
        top_features = self.classifier._get_top_features(features, vector)
        
        assert isinstance(top_features, list)
        assert len(top_features) <= 3
        assert "prosody" in top_features  # Should be highest
        assert "motion_magnitude" in top_features  # Should be second highest


class TestHighlightDetector:
    """Test cases for HighlightDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = HighlightDetector("sports")
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.mode == "sports"
        assert self.detector.fusion is not None
        assert self.detector.classifier is not None
        assert self.detector.min_confidence == 0.4
    
    def test_detect_highlights(self):
        """Test highlight detection."""
        audio_features = {
            'short_time_energy': np.array([0.1, 0.8, 0.2]),
            'voice_activity': np.array([0.0, 1.0, 0.0]),
            'pitch': np.array([100, 200, 150]),
            'loudness': np.array([0.1, 0.9, 0.2])
        }
        
        vision_features = {
            'motion': np.array([0.1, 0.7, 0.2]),
            'shot_boundaries': np.array([1.5])
        }
        
        timestamps = np.array([0.5, 1.0, 1.5])
        
        highlights = self.detector.detect_highlights(
            audio_features, vision_features, timestamps
        )
        
        assert isinstance(highlights, list)
        # Each highlight should have required fields
        for highlight in highlights:
            assert 'id' in highlight
            assert 'start_time' in highlight
            assert 'end_time' in highlight
            assert 'confidence' in highlight
            assert 'label' in highlight
            assert 'category' in highlight
            assert 'features' in highlight
            assert 'evidence' in highlight
    
    def test_non_maximum_suppression(self):
        """Test non-maximum suppression."""
        highlights = [
            {'timestamp': 1.0, 'confidence': 0.9},
            {'timestamp': 1.5, 'confidence': 0.8},
            {'timestamp': 3.0, 'confidence': 0.7},
            {'timestamp': 3.2, 'confidence': 0.6}
        ]
        
        suppressed = self.detector._apply_non_maximum_suppression(highlights)
        
        assert isinstance(suppressed, list)
        assert len(suppressed) <= len(highlights)
        # Should keep the highest confidence ones and remove nearby duplicates
    
    def test_generate_label_sports(self):
        """Test label generation for sports mode."""
        highlight = {
            'feature_contributions': {
                'scoreboard_change': 0.6,
                'audio_peak': 0.3,
                'replay_cue': 0.1
            }
        }
        
        label = self.detector._generate_label(highlight)
        
        assert isinstance(label, str)
        assert label == "Score Change"  # Should prioritize scoreboard change
    
    def test_generate_label_podcast(self):
        """Test label generation for podcast mode."""
        podcast_detector = HighlightDetector("podcast")
        
        highlight = {
            'feature_contributions': {
                'laughter': 0.6,
                'excitement': 0.3,
                'applause': 0.1
            }
        }
        
        label = podcast_detector._generate_label(highlight)
        
        assert isinstance(label, str)
        assert label == "Funny Moment"  # Should prioritize laughter


class TestIntegration:
    """Integration tests for the complete fusion and classification pipeline."""
    
    def test_end_to_end_detection(self):
        """Test complete end-to-end detection pipeline."""
        detector = HighlightDetector("sports")
        
        # Create realistic feature data
        audio_features = {
            'short_time_energy': np.random.randn(100) * 0.5 + 0.5,
            'voice_activity': np.random.randn(100) * 0.3 + 0.5,
            'pitch': np.random.randn(100) * 50 + 150,
            'loudness': np.random.randn(100) * 0.3 + 0.5,
            'spectral_flux': np.random.randn(100) * 0.2 + 0.3
        }
        
        vision_features = {
            'motion': np.random.randn(100) * 0.3 + 0.4,
            'shot_boundaries': np.array([10.0, 25.0, 45.0]),
            'scoreboard_changes': np.array([15.0, 30.0]),
            'replay_cues': np.array([20.0])
        }
        
        timestamps = np.linspace(0, 60, 100)  # 1 minute video
        
        highlights = detector.detect_highlights(
            audio_features, vision_features, timestamps
        )
        
        assert isinstance(highlights, list)
        # Should detect some highlights or return empty list
        for highlight in highlights:
            assert highlight['confidence'] >= detector.min_confidence
            assert highlight['start_time'] < highlight['end_time']
            assert highlight['category'] == "sports_action"
    
    def test_mode_comparison(self):
        """Test that different modes produce different results."""
        sports_detector = HighlightDetector("sports")
        podcast_detector = HighlightDetector("podcast")
        
        # Same input features
        audio_features = {
            'short_time_energy': np.random.randn(50) * 0.5 + 0.5,
            'voice_activity': np.random.randn(50) * 0.3 + 0.5,
            'pitch': np.random.randn(50) * 50 + 150,
            'loudness': np.random.randn(50) * 0.3 + 0.5,
            'zero_crossing_rate': np.random.randn(50) * 0.2 + 0.3,
            'percussive_energy': np.random.randn(50) * 0.3 + 0.4
        }
        
        vision_features = {
            'motion': np.random.randn(50) * 0.3 + 0.4,
            'shot_boundaries': np.array([10.0, 25.0])
        }
        
        timestamps = np.linspace(0, 30, 50)
        
        sports_highlights = sports_detector.detect_highlights(
            audio_features, vision_features, timestamps
        )
        
        podcast_highlights = podcast_detector.detect_highlights(
            audio_features, vision_features, timestamps
        )
        
        # Results should be different (though might be same by chance)
        assert isinstance(sports_highlights, list)
        assert isinstance(podcast_highlights, list)
        
        # Categories should be different
        for highlight in sports_highlights:
            assert highlight['category'] == "sports_action"
        
        for highlight in podcast_highlights:
            assert highlight['category'] == "podcast_moment"


if __name__ == "__main__":
    pytest.main([__file__])
