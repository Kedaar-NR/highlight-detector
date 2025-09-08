"""
Audio feature extraction for highlight detection.
"""

import numpy as np
import librosa
import torch
import torchaudio
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import skew, kurtosis
import logging

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extracts audio features for highlight detection."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mels: int = 128
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        
        # Mel filter bank
        self.mel_filter = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
    
    def extract_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Extract all audio features from a file."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            features = {}
            
            # Basic features
            features['short_time_energy'] = self._extract_short_time_energy(y)
            features['spectral_flux'] = self._extract_spectral_flux(y)
            features['spectral_centroid'] = self._extract_spectral_centroid(y)
            features['spectral_rolloff'] = self._extract_spectral_rolloff(y)
            features['zero_crossing_rate'] = self._extract_zero_crossing_rate(y)
            
            # Advanced features
            features['mfcc'] = self._extract_mfcc(y)
            features['chroma'] = self._extract_chroma(y)
            features['tonnetz'] = self._extract_tonnetz(y)
            
            # Harmonic and percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            features['harmonic_energy'] = self._extract_short_time_energy(y_harmonic)
            features['percussive_energy'] = self._extract_short_time_energy(y_percussive)
            
            # Voice activity detection
            features['voice_activity'] = self._extract_voice_activity(y)
            
            # Prosody features
            features['pitch'] = self._extract_pitch(y)
            features['loudness'] = self._extract_loudness(y)
            
            # Rhythm features
            features['tempo'] = self._extract_tempo(y)
            features['beat_frames'] = self._extract_beat_frames(y)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            raise
    
    def _extract_short_time_energy(self, y: np.ndarray) -> np.ndarray:
        """Extract short-time energy."""
        return librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
    
    def _extract_spectral_flux(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral flux."""
        stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.n_fft)
        magnitude = np.abs(stft)
        
        # Calculate spectral flux
        flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
        return flux
    
    def _extract_spectral_centroid(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral centroid."""
        return librosa.feature.spectral_centroid(
            y=y, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
    
    def _extract_spectral_rolloff(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral rolloff."""
        return librosa.feature.spectral_rolloff(
            y=y, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
    
    def _extract_zero_crossing_rate(self, y: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate."""
        return librosa.feature.zero_crossing_rate(
            y, hop_length=self.hop_length
        )[0]
    
    def _extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length
        )
        return mfcc
    
    def _extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """Extract chroma features."""
        return librosa.feature.chroma_stft(
            y=y, sr=self.sample_rate, hop_length=self.hop_length
        )
    
    def _extract_tonnetz(self, y: np.ndarray) -> np.ndarray:
        """Extract tonnetz features."""
        return librosa.feature.tonnetz(
            y=y, sr=self.sample_rate, hop_length=self.hop_length
        )
    
    def _extract_voice_activity(self, y: np.ndarray) -> np.ndarray:
        """Extract voice activity detection."""
        # Simple energy-based VAD
        energy = self._extract_short_time_energy(y)
        threshold = np.mean(energy) + 0.5 * np.std(energy)
        vad = (energy > threshold).astype(float)
        
        # Smooth the VAD signal
        vad_smoothed = signal.medfilt(vad, kernel_size=5)
        return vad_smoothed
    
    def _extract_pitch(self, y: np.ndarray) -> np.ndarray:
        """Extract pitch using YIN algorithm."""
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Get the pitch with highest magnitude in each frame
        pitch = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch.append(pitches[index, t])
        
        return np.array(pitch)
    
    def _extract_loudness(self, y: np.ndarray) -> np.ndarray:
        """Extract loudness using perceptual weighting."""
        # Simple RMS-based loudness
        return self._extract_short_time_energy(y)
    
    def _extract_tempo(self, y: np.ndarray) -> float:
        """Extract tempo."""
        tempo, _ = librosa.beat.beat_track(y=y, sr=self.sample_rate)
        return tempo
    
    def _extract_beat_frames(self, y: np.ndarray) -> np.ndarray:
        """Extract beat frames."""
        _, beats = librosa.beat.beat_track(y=y, sr=self.sample_rate)
        return beats


class AudioPeakDetector:
    """Detects peaks in audio features for highlight moments."""
    
    def __init__(self, prominence: float = 0.3, distance: int = 100):
        self.prominence = prominence
        self.distance = distance
    
    def detect_peaks(
        self,
        features: Dict[str, np.ndarray],
        mode: str = "sports"
    ) -> List[Dict[str, any]]:
        """Detect peaks in audio features."""
        peaks = []
        
        # Energy peaks
        energy_peaks = self._detect_energy_peaks(features['short_time_energy'])
        peaks.extend(energy_peaks)
        
        # Spectral flux peaks
        flux_peaks = self._detect_flux_peaks(features['spectral_flux'])
        peaks.extend(flux_peaks)
        
        # Mode-specific peaks
        if mode == "sports":
            peaks.extend(self._detect_sports_peaks(features))
        elif mode == "podcast":
            peaks.extend(self._detect_podcast_peaks(features))
        
        # Merge nearby peaks
        merged_peaks = self._merge_peaks(peaks)
        
        return merged_peaks
    
    def _detect_energy_peaks(self, energy: np.ndarray) -> List[Dict[str, any]]:
        """Detect energy peaks."""
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(
            energy,
            prominence=self.prominence,
            distance=self.distance
        )
        
        peak_data = []
        for i, peak in enumerate(peaks):
            peak_data.append({
                'type': 'energy_peak',
                'frame': peak,
                'strength': energy[peak],
                'prominence': properties['prominences'][i] if 'prominences' in properties else 0
            })
        
        return peak_data
    
    def _detect_flux_peaks(self, flux: np.ndarray) -> List[Dict[str, any]]:
        """Detect spectral flux peaks."""
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(
            flux,
            prominence=self.prominence * 0.5,
            distance=self.distance
        )
        
        peak_data = []
        for i, peak in enumerate(peaks):
            peak_data.append({
                'type': 'flux_peak',
                'frame': peak,
                'strength': flux[peak],
                'prominence': properties['prominences'][i] if 'prominences' in properties else 0
            })
        
        return peak_data
    
    def _detect_sports_peaks(self, features: Dict[str, np.ndarray]) -> List[Dict[str, any]]:
        """Detect sports-specific audio peaks."""
        peaks = []
        
        # Crowd cheer detection (high energy in mid frequencies)
        if 'mfcc' in features:
            mfcc = features['mfcc']
            # Use MFCC coefficients to detect crowd sounds
            crowd_energy = np.mean(mfcc[1:4, :], axis=0)  # Mid-frequency components
            crowd_peaks = self._detect_energy_peaks(crowd_energy)
            for peak in crowd_peaks:
                peak['type'] = 'crowd_cheer'
            peaks.extend(crowd_peaks)
        
        return peaks
    
    def _detect_podcast_peaks(self, features: Dict[str, np.ndarray]) -> List[Dict[str, any]]:
        """Detect podcast-specific audio peaks."""
        peaks = []
        
        # Laughter detection (high zero crossing rate + energy)
        if 'zero_crossing_rate' in features and 'short_time_energy' in features:
            zcr = features['zero_crossing_rate']
            energy = features['short_time_energy']
            
            # Combine ZCR and energy for laughter detection
            laughter_score = zcr * energy
            laughter_peaks = self._detect_energy_peaks(laughter_score)
            for peak in laughter_peaks:
                peak['type'] = 'laughter'
            peaks.extend(laughter_peaks)
        
        # Applause detection (percussive energy)
        if 'percussive_energy' in features:
            applause_peaks = self._detect_energy_peaks(features['percussive_energy'])
            for peak in applause_peaks:
                peak['type'] = 'applause'
            peaks.extend(applause_peaks)
        
        return peaks
    
    def _merge_peaks(self, peaks: List[Dict[str, any]], window: int = 50) -> List[Dict[str, any]]:
        """Merge nearby peaks."""
        if not peaks:
            return peaks
        
        # Sort by frame
        peaks.sort(key=lambda x: x['frame'])
        
        merged = []
        current_peak = peaks[0].copy()
        
        for peak in peaks[1:]:
            if peak['frame'] - current_peak['frame'] <= window:
                # Merge peaks
                current_peak['strength'] = max(current_peak['strength'], peak['strength'])
                current_peak['prominence'] = max(current_peak['prominence'], peak['prominence'])
                if peak['type'] != current_peak['type']:
                    current_peak['type'] = f"{current_peak['type']}+{peak['type']}"
            else:
                merged.append(current_peak)
                current_peak = peak.copy()
        
        merged.append(current_peak)
        return merged
