"""Audio feature extraction utilities"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import requests
from tqdm import tqdm
import hashlib


class AudioFeatureExtractor:
    """Extract audio features from audio files"""
    
    def __init__(self, cache_dir: str = "data/audio_cache", sample_rate: int = 22050):
        """
        Initialize audio feature extractor
        
        Args:
            cache_dir: Directory to cache downloaded audio and features
            sample_rate: Sample rate for audio processing (22050 is standard)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_cache_dir = self.cache_dir / "audio"
        self.feature_cache_dir = self.cache_dir / "features"
        self.audio_cache_dir.mkdir(exist_ok=True)
        self.feature_cache_dir.mkdir(exist_ok=True)
        self.sample_rate = sample_rate
    
    def _get_cache_key(self, audio_url: str) -> str:
        """Generate cache key from audio URL"""
        return hashlib.md5(audio_url.encode()).hexdigest()
    
    def _download_audio(self, audio_url: str, song_id: str) -> Optional[Path]:
        """Download audio file and cache it"""
        cache_key = self._get_cache_key(audio_url)
        audio_path = self.audio_cache_dir / f"{song_id}_{cache_key}.mp3"
        
        if audio_path.exists():
            return audio_path
        
        try:
            response = requests.get(audio_url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return audio_path
        except Exception as e:
            print(f"Warning: Failed to download audio from {audio_url}: {e}")
            return None
    
    def extract_basic_features(
        self,
        audio_path: Path,
        include_mfcc: bool = True,
        include_chroma: bool = True,
        include_spectral: bool = True,
        include_rhythm: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Extract basic audio features using librosa
        
        Args:
            audio_path: Path to audio file
            include_mfcc: Extract MFCC features (timbre)
            include_chroma: Extract chroma features (harmonic content)
            include_spectral: Extract spectral features (brightness, rolloff)
            include_rhythm: Extract rhythm features (tempo, beat)
        
        Returns:
            Dictionary of feature arrays
        """
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate, duration=180)  # Max 3 minutes
            
            features = {}
            
            # MFCC (Mel-Frequency Cepstral Coefficients) - Timbre
            if include_mfcc:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features['mfcc_mean'] = np.mean(mfcc, axis=1)  # 13-dim
                features['mfcc_std'] = np.std(mfcc, axis=1)  # 13-dim
            
            # Chroma - Harmonic content
            if include_chroma:
                chroma = librosa.feature.chroma(y=y, sr=sr)
                features['chroma_mean'] = np.mean(chroma, axis=1)  # 12-dim
                features['chroma_std'] = np.std(chroma, axis=1)  # 12-dim
            
            # Spectral features
            if include_spectral:
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
                
                features['spectral_centroid'] = np.mean(spectral_centroid)
                features['spectral_rolloff'] = np.mean(spectral_rolloff)
                features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
                features['zero_crossing_rate'] = np.mean(zero_crossing_rate)
            
            # Rhythm features
            if include_rhythm:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = tempo
                features['beat_frames'] = len(beats)
            
            # Energy and dynamics
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # Tonnetz - Harmonic relationships
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz, axis=1)  # 6-dim
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return {}
    
    def extract_features_from_url(
        self,
        audio_url: str,
        song_id: str,
        use_cache: bool = True,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract features from audio URL (downloads and processes)
        
        Args:
            audio_url: URL to audio file
            song_id: Song ID for caching
            use_cache: Whether to use cached features
        
        Returns:
            Dictionary of features or None if extraction fails
        """
        # Check feature cache
        if use_cache:
            feature_cache_path = self.feature_cache_dir / f"{song_id}.pkl"
            if feature_cache_path.exists():
                try:
                    return joblib.load(feature_cache_path)
                except Exception as e:
                    print(f"Warning: Failed to load cached features: {e}")
        
        # Download audio
        audio_path = self._download_audio(audio_url, song_id)
        if audio_path is None:
            return None
        
        # Extract features
        features = self.extract_basic_features(audio_path)
        
        # Cache features
        if use_cache and features:
            feature_cache_path = self.feature_cache_dir / f"{song_id}.pkl"
            try:
                joblib.dump(features, feature_cache_path)
            except Exception as e:
                print(f"Warning: Failed to cache features: {e}")
        
        return features
    
    def extract_features_batch(
        self,
        audio_urls: List[str],
        song_ids: List[str],
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract features for multiple audio files
        
        Args:
            audio_urls: List of audio URLs
            song_ids: List of song IDs
            show_progress: Show progress bar
        
        Returns:
            Dictionary mapping song_id to features
        """
        results = {}
        
        iterator = tqdm(zip(audio_urls, song_ids), total=len(audio_urls), desc="Extracting audio features") if show_progress else zip(audio_urls, song_ids)
        
        for audio_url, song_id in iterator:
            features = self.extract_features_from_url(audio_url, song_id)
            if features:
                results[song_id] = features
        
        return results
    
    def features_to_vector(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert feature dictionary to a single vector
        
        Args:
            features: Dictionary of features
        
        Returns:
            Concatenated feature vector
        """
        vectors = []
        
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, np.ndarray):
                vectors.append(value.flatten())
            else:
                vectors.append(np.array([value]))
        
        if vectors:
            return np.concatenate(vectors)
        else:
            return np.array([])
    
    def get_feature_dimension(self) -> int:
        """
        Get the dimension of the feature vector
        
        Returns:
            Feature dimension
        """
        # MFCC: 13 mean + 13 std = 26
        # Chroma: 12 mean + 12 std = 24
        # Spectral: 4 scalars
        # Rhythm: 2 scalars
        # RMS: 2 scalars
        # Tonnetz: 6 mean
        # Total: 26 + 24 + 4 + 2 + 2 + 6 = 64
        
        return 64


class AudioEmbedder:
    """Audio embedding using deep learning models (future enhancement)"""
    
    def __init__(self, model_name: str = 'clap'):
        """
        Initialize audio embedder
        
        Args:
            model_name: Model to use ('clap', 'musiclm', etc.)
        """
        self.model_name = model_name
        self.model = None
        # TODO: Load pre-trained model
        # For now, this is a placeholder
    
    def embed_audio(self, audio_path: Path) -> np.ndarray:
        """Generate embedding for audio file"""
        # TODO: Implement with actual model
        raise NotImplementedError("Audio embedding models not yet implemented")
    
    def embed_audio_batch(self, audio_paths: List[Path]) -> np.ndarray:
        """Generate embeddings for multiple audio files"""
        # TODO: Implement batch processing
        raise NotImplementedError("Audio embedding models not yet implemented")

