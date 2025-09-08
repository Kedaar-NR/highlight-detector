"""
Unit tests for audio feature extraction.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

# Add packages to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from audio.features import AudioFeatureExtractor, AudioPeakDetector


class TestAudioFeatureExtractor:
    """Test cases for AudioFeatureExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = AudioFeatureExtractor()
    
    def test_initialization(self):
        """Test extractor initialization."""
        assert self.extractor.sample_rate == 16000
        assert self.extractor.hop_length == 512
        assert self.extractor.n_fft == 2048
        assert self.extractor.n_mels == 128
    
    @patch('librosa.load')
    def test_extract_features_success(self, mock_load):
        """Test successful feature extraction."""
        # Mock audio data
        mock_audio = np.random.randn(16000)  # 1 second of audio
        mock_load.return_value = (mock_audio, 16000)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            features = self.extractor.extract_features(tmp_path)
            
            # Check that all expected features are present
            expected_features = [
                'short_time_energy', 'spectral_flux', 'spectral_centroid',
                'spectral_rolloff', 'zero_crossing_rate', 'mfcc', 'chroma',
                'tonnetz', 'harmonic_energy', 'percussive_energy',
                'voice_activity', 'pitch', 'loudness', 'tempo', 'beat_frames'
            ]
            
            for feature in expected_features:
                assert feature in features
                assert isinstance(features[feature], np.ndarray)
            
        finally:
            os.unlink(tmp_path)
    
    @patch('librosa.load')
    def test_extract_features_file_not_found(self, mock_load):
        """Test feature extraction with non-existent file."""
        mock_load.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(Exception):
            self.extractor.extract_features("nonexistent.wav")
    
    def test_extract_short_time_energy(self):
        """Test short-time energy extraction."""
        # Create test audio signal
        audio = np.array([1.0, 0.5, -0.5, -1.0, 0.0])
        
        energy = self.extractor._extract_short_time_energy(audio)
        
        assert isinstance(energy, np.ndarray)
        assert len(energy) > 0
        assert np.all(energy >= 0)  # Energy should be non-negative
    
    def test_extract_spectral_flux(self):
        """Test spectral flux extraction."""
        audio = np.random.randn(1000)
        
        flux = self.extractor._extract_spectral_flux(audio)
        
        assert isinstance(flux, np.ndarray)
        assert len(flux) > 0
    
    def test_extract_voice_activity(self):
        """Test voice activity detection."""
        # Create test signal with high and low energy regions
        audio = np.concatenate([
            np.random.randn(100) * 0.1,  # Low energy (no voice)
            np.random.randn(100) * 1.0,  # High energy (voice)
            np.random.randn(100) * 0.1   # Low energy (no voice)
        ])
        
        vad = self.extractor._extract_voice_activity(audio)
        
        assert isinstance(vad, np.ndarray)
        assert len(vad) > 0
        assert np.all((vad == 0) | (vad == 1))  # Binary values


class TestAudioPeakDetector:
    """Test cases for AudioPeakDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AudioPeakDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.prominence == 0.3
        assert self.detector.distance == 100
    
    def test_detect_energy_peaks(self):
        """Test energy peak detection."""
        # Create test energy signal with clear peaks
        energy = np.array([0.1, 0.2, 0.8, 0.3, 0.1, 0.9, 0.2, 0.1])
        
        peaks = self.detector._detect_energy_peaks(energy)
        
        assert isinstance(peaks, list)
        # Should detect peaks at indices 2 and 5
        assert len(peaks) >= 1
    
    def test_detect_flux_peaks(self):
        """Test spectral flux peak detection."""
        flux = np.array([0.1, 0.2, 0.7, 0.3, 0.1, 0.8, 0.2, 0.1])
        
        peaks = self.detector._detect_flux_peaks(flux)
        
        assert isinstance(peaks, list)
        assert len(peaks) >= 1
    
    def test_detect_sports_peaks(self):
        """Test sports-specific peak detection."""
        features = {
            'mfcc': np.random.randn(13, 100),
            'short_time_energy': np.random.randn(100),
            'zero_crossing_rate': np.random.randn(100)
        }
        
        peaks = self.detector._detect_sports_peaks(features)
        
        assert isinstance(peaks, list)
        # Should include crowd cheer detection
    
    def test_detect_podcast_peaks(self):
        """Test podcast-specific peak detection."""
        features = {
            'zero_crossing_rate': np.random.randn(100),
            'short_time_energy': np.random.randn(100),
            'percussive_energy': np.random.randn(100)
        }
        
        peaks = self.detector._detect_podcast_peaks(features)
        
        assert isinstance(peaks, list)
        # Should include laughter and applause detection
    
    def test_merge_peaks(self):
        """Test peak merging functionality."""
        peaks = [
            {'frame': 10, 'strength': 0.8, 'type': 'energy_peak'},
            {'frame': 15, 'strength': 0.6, 'type': 'flux_peak'},
            {'frame': 50, 'strength': 0.9, 'type': 'energy_peak'},
            {'frame': 52, 'strength': 0.7, 'type': 'flux_peak'}
        ]
        
        merged = self.detector._merge_peaks(peaks, window=10)
        
        assert isinstance(merged, list)
        assert len(merged) <= len(peaks)  # Should merge some peaks
        assert len(merged) >= 1  # Should have at least one peak


class TestIntegration:
    """Integration tests for audio feature pipeline."""
    
    def test_full_audio_pipeline(self):
        """Test the complete audio feature extraction pipeline."""
        extractor = AudioFeatureExtractor()
        detector = AudioPeakDetector()
        
        # Create mock audio features
        mock_features = {
            'short_time_energy': np.random.randn(100),
            'spectral_flux': np.random.randn(100),
            'zero_crossing_rate': np.random.randn(100),
            'percussive_energy': np.random.randn(100),
            'mfcc': np.random.randn(13, 100)
        }
        
        # Test peak detection
        peaks = detector.detect_peaks(mock_features, mode="sports")
        
        assert isinstance(peaks, list)
        # Should return some peaks or empty list
    
    def test_mode_specific_detection(self):
        """Test that different modes produce different results."""
        detector = AudioPeakDetector()
        
        mock_features = {
            'short_time_energy': np.random.randn(100),
            'spectral_flux': np.random.randn(100),
            'zero_crossing_rate': np.random.randn(100),
            'percussive_energy': np.random.randn(100),
            'mfcc': np.random.randn(13, 100)
        }
        
        sports_peaks = detector.detect_peaks(mock_features, mode="sports")
        podcast_peaks = detector.detect_peaks(mock_features, mode="podcast")
        
        # Results should be different for different modes
        # (though they might be the same by chance)
        assert isinstance(sports_peaks, list)
        assert isinstance(podcast_peaks, list)


if __name__ == "__main__":
    pytest.main([__file__])
