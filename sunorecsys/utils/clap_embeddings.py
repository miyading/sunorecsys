"""CLAP audio embedding utilities for content-based similarity"""

import os
import json
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
import joblib

# Try to import laion_clap - it's optional
try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
    print("Warning: laion_clap not available. Install it to use CLAP embeddings.")


# Default CLAP model path
DEFAULT_CLAP_MODEL_PATH = 'sunorecsys/load/clap_score/music_audioset_epoch_15_esc_90.14.pt'

# Module-level cache for CLAP models (shared across all embedders)
_CLAP_MODEL_CACHE = {}


class AudioFileDataset(Dataset):
    """Dataset for loading audio files for CLAP embedding"""
    
    def __init__(self, audio_paths: List[str]):
        """
        Initialize dataset
        
        Args:
            audio_paths: List of paths to audio files
        """
        self.audio_paths = audio_paths
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        return self.audio_paths[idx]


def download_clap_model(model_path: str = DEFAULT_CLAP_MODEL_PATH, force: bool = False) -> None:
    """
    Downloads the CLAP model if it doesn't exist locally.
    
    Args:
        model_path: Path where the model should be saved
        force: If True, re-download even if file exists
    """
    if os.path.exists(model_path) and not force:
        print(f'CLAP model already exists at {model_path}. Skipping download.')
        return
    
    # URL for the CLAP model
    # Note: This URL might need to be updated based on actual HuggingFace structure
    url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
    
    print(f'Downloading CLAP model from {url}...')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as progress_bar:
                for data in response.iter_content(chunk_size=8192):
                    file.write(data)
                    progress_bar.update(len(data))
        
        print(f'✅ CLAP model downloaded to {model_path}')
    except Exception as e:
        print(f'❌ Failed to download CLAP model: {e}')
        print('Please download manually or check the URL.')
        raise


def initialize_clap_model(model_path: str = DEFAULT_CLAP_MODEL_PATH, device: Optional[str] = None) -> Optional[object]:
    """
    Initializes and loads the CLAP model with the correct state dictionary.
    
    Args:
        model_path: Path to the CLAP model file
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        CLAP_Module object with pretrained weights loaded, using CUDA if available, otherwise CPU.
        Returns None if laion_clap is not available.
    """
    if not CLAP_AVAILABLE:
        print("❌ laion_clap is not available. Please install it to use CLAP embeddings.")
        return None
    
    # Check if CUDA is available
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Download the model if needed
    if not os.path.exists(model_path):
        try:
            download_clap_model(model_path)
        except Exception as e:
            print(f"❌ Failed to download CLAP model: {e}")
            return None
    
    try:
        # Initialize the CLAP model
        # Using HTSAT-base model as in the provided script
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=device)
        
        # Load model state dictionary
        # Note: The actual loading method may vary depending on laion_clap version
        # This is a simplified version - you may need to adjust based on actual API
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Try common keys
                state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            else:
                state_dict = checkpoint
            
            # Remove problematic keys if present
            if isinstance(state_dict, dict):
                state_dict.pop('text_branch.embeddings.position_ids', None)
            
            # Load state dict into model
            if hasattr(model, 'model'):
                model.model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
        
        except Exception as e:
            print(f"Warning: Could not load state dict directly: {e}")
            print("Attempting to load using model's built-in method...")
            # Some versions of laion_clap may have a different loading API
            if hasattr(model, 'load_ckpt'):
                model.load_ckpt(model_path)
        
        # Put the model in evaluation mode
        model.eval()
        
        print(f"✅ CLAP model initialized successfully on {device}")
        return model
        
    except Exception as e:
        print(f"❌ Failed to initialize CLAP model: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_audio_embeddings_clap(
    audio_paths: List[str],
    model: object,
    batch_size: int = 16,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute CLAP audio embeddings for a list of audio files.
    
    Args:
        audio_paths: List of paths to audio files
        model: Initialized CLAP model
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar
    
    Returns:
        Dictionary mapping audio file paths to embeddings (numpy arrays)
    """
    if model is None:
        raise ValueError("CLAP model is not initialized")
    
    embeddings = {}
    
    # Filter out non-existent files
    valid_paths = [path for path in audio_paths if os.path.isfile(path)]
    
    if not valid_paths:
        print("No valid audio files found")
        return embeddings
    
    if len(valid_paths) < len(audio_paths):
        print(f"Warning: {len(audio_paths) - len(valid_paths)} files not found")
    
    # Create dataset and dataloader
    dataset = AudioFileDataset(valid_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x  # Return list as-is
    )
    
    iterator = tqdm(dataloader, desc="Computing CLAP embeddings") if show_progress else dataloader
    
    with torch.no_grad():
        for batch_paths in iterator:
            try:
                # Compute audio embeddings using the model
                # The exact API may vary - adjust based on laion_clap version
                if hasattr(model, 'get_audio_embedding_from_filelist'):
                    batch_embeddings = model.get_audio_embedding_from_filelist(
                        x=batch_paths,
                        use_tensor=True
                    )
                elif hasattr(model, 'get_audio_embedding'):
                    # Alternative API - process each file
                    batch_embeddings = []
                    for path in batch_paths:
                        emb = model.get_audio_embedding([path], use_tensor=True)
                        batch_embeddings.append(emb[0] if len(emb) > 0 else None)
                    batch_embeddings = [e for e in batch_embeddings if e is not None]
                    if batch_embeddings:
                        batch_embeddings = torch.stack(batch_embeddings)
                else:
                    print("❌ Model does not have expected audio embedding methods")
                    continue
                
                # Convert to numpy and store
                if isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = batch_embeddings.cpu().numpy()
                
                # Map embeddings to file paths
                for path, emb in zip(batch_paths, batch_embeddings):
                    if emb is not None:
                        embeddings[path] = emb.flatten()  # Ensure 1D array
                        
            except Exception as e:
                print(f"Warning: Failed to process batch: {e}")
                continue
    
    return embeddings


def _get_clap_model(model_path: str = DEFAULT_CLAP_MODEL_PATH, device: Optional[str] = None):
    """
    Get or initialize CLAP model (shared cache across all embedders).
    
    Args:
        model_path: Path to CLAP model file
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        CLAP model instance or None if initialization failed
    """
    cache_key = (model_path, device)
    
    if cache_key not in _CLAP_MODEL_CACHE:
        model = initialize_clap_model(model_path, device)
        if model is not None:
            _CLAP_MODEL_CACHE[cache_key] = model
        else:
            return None
    
    return _CLAP_MODEL_CACHE.get(cache_key)


class CLAPAudioEmbedder:
    """
    CLAP-based audio embedder for the RecSys.
    
    CLAP (Contrastive Language-Audio Pretraining) provides audio embeddings
    in a shared embedding space with text, enabling cross-modal similarity search.
    
    Integrates with existing audio caching infrastructure.
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_CLAP_MODEL_PATH,
        cache_dir: str = "runtime_data/audio_cache",
        device: Optional[str] = None,
        batch_size: int = 16
    ):
        """
        Initialize CLAP audio embedder.
        
        Args:
            model_path: Path to CLAP model file (defaults to DEFAULT_CLAP_MODEL_PATH)
            cache_dir: Directory for audio cache and embeddings
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for processing
        """
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.audio_cache_dir = self.cache_dir / "audio"
        self.embedding_cache_dir = self.cache_dir / "clap_embeddings"
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.batch_size = batch_size
        self.model = None
        
        # Download audio if needed (reuse from AudioFeatureExtractor)
        try:
            from ..utils.audio_features import AudioFeatureExtractor
            self.audio_extractor = AudioFeatureExtractor(cache_dir=cache_dir)
        except ImportError:
            self.audio_extractor = None
            print("Warning: AudioFeatureExtractor not available. Audio downloading disabled.")
    
    def initialize_model(self) -> bool:
        """
        Initialize the CLAP model (uses shared module-level cache).
        
        Returns:
            True if initialization successful, False otherwise
        """
        self.model = _get_clap_model(self.model_path, self.device)
        return self.model is not None
    
    def _get_audio_path(self, audio_url: str, song_id: str) -> Optional[Path]:
        """
        Get path to audio file, downloading if necessary.
        
        Args:
            audio_url: URL to audio file
            song_id: Song ID for caching
        
        Returns:
            Path to audio file, or None if download failed
        """
        if self.audio_extractor:
            # Use AudioFeatureExtractor's download method
            audio_path = self.audio_extractor._download_audio(audio_url, song_id)
            if audio_path and audio_path.exists():
                return audio_path
        
        # Fallback: simple download to cache
        cache_key = hashlib.md5(audio_url.encode()).hexdigest() if audio_url else song_id
        audio_path = self.audio_cache_dir / f"{song_id}_{cache_key}.mp3"
        
        if audio_path.exists():
            return audio_path
        
        if not audio_url:
            return None
        
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
    
    def _get_embedding_cache_path(self, song_id: str) -> Path:
        """Get path to cached embedding file"""
        return self.embedding_cache_dir / f"{song_id}.pkl"
    
    def embed_audio(
        self,
        audio_url: str,
        song_id: str,
        use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """
        Compute CLAP embedding for a single audio file.
        
        Args:
            audio_url: URL to audio file
            song_id: Song ID
            use_cache: Whether to use cached embedding
        
        Returns:
            CLAP embedding as numpy array, or None if failed
        """
        if self.model is None:
            if not self.initialize_model():
                return None
        
        # Check cache
        if use_cache:
            cache_path = self._get_embedding_cache_path(song_id)
            if cache_path.exists():
                try:
                    return joblib.load(cache_path)
                except Exception as e:
                    print(f"Warning: Failed to load cached embedding: {e}")
        
        # Get audio file path
        audio_path = self._get_audio_path(audio_url, song_id)
        if audio_path is None:
            return None
        
        # Compute embedding
        embeddings_dict = compute_audio_embeddings_clap(
            [str(audio_path)],
            self.model,
            batch_size=1,
            show_progress=False
        )
        
        embedding = embeddings_dict.get(str(audio_path))
        if embedding is None:
            return None
        
        # Cache embedding
        if use_cache:
            cache_path = self._get_embedding_cache_path(song_id)
            try:
                joblib.dump(embedding, cache_path)
            except Exception as e:
                print(f"Warning: Failed to cache embedding: {e}")
        
        return embedding
    
    def embed_audio_batch(
        self,
        audio_urls: List[str],
        song_ids: List[str],
        use_cache: bool = True,
        show_progress: bool = True,
        skip_download: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Compute CLAP embeddings for multiple audio files.
        
        Args:
            audio_urls: List of audio URLs
            song_ids: List of song IDs
            use_cache: Whether to use cached embeddings
            show_progress: Whether to show progress bar
        
        Returns:
            Dictionary mapping song_id to embedding
        """
        if self.model is None:
            if not self.initialize_model():
                return {}
        
        if len(audio_urls) != len(song_ids):
            raise ValueError("audio_urls and song_ids must have the same length")
        
        # Check cache and filter already-computed embeddings
        embeddings = {}
        to_process = []
        to_process_ids = []
        to_process_paths = []
        to_download = []
        
        # Use tqdm if available for progress
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        iterator = tqdm(zip(audio_urls, song_ids), total=len(audio_urls), desc="Checking cache") if (use_tqdm and show_progress) else zip(audio_urls, song_ids)
        
        for audio_url, song_id in iterator:
            if use_cache:
                cache_path = self._get_embedding_cache_path(song_id)
                if cache_path.exists():
                    try:
                        embeddings[song_id] = joblib.load(cache_path)
                        continue
                    except Exception:
                        pass
            
            # Check if audio file exists first (don't download yet)
            cache_key = hashlib.md5(audio_url.encode()).hexdigest() if audio_url else song_id
            audio_path = self.audio_cache_dir / f"{song_id}_{cache_key}.mp3"
            
            if audio_path.exists():
                to_process.append((song_id, str(audio_path)))
                to_process_ids.append(song_id)
                to_process_paths.append(str(audio_path))
            elif audio_url:
                # Need to download later
                to_download.append((audio_url, song_id))
        
        # Download missing audio files with progress
        if to_download:
            if skip_download:
                if show_progress:
                    print(f"\n⚠️  Skipping download of {len(to_download)} missing audio files (--skip-download enabled)")
            else:
                if show_progress:
                    print(f"\n📥 Downloading {len(to_download)} missing audio files...")
                download_iterator = tqdm(to_download, desc="Downloading") if use_tqdm else to_download
                for audio_url, song_id in download_iterator:
                    audio_path = self._get_audio_path(audio_url, song_id)
            if audio_path and audio_path.exists():
                to_process.append((song_id, str(audio_path)))
                to_process_ids.append(song_id)
                to_process_paths.append(str(audio_path))
        
        if not to_process:
            return embeddings
        
        if show_progress:
            print(f"\n🎵 Computing CLAP embeddings for {len(to_process)} audio files...")
        
        # Compute embeddings in batches
        batch_embeddings = compute_audio_embeddings_clap(
            to_process_paths,
            self.model,
            batch_size=self.batch_size,
            show_progress=show_progress
        )
        
        # Map embeddings to song_ids
        for song_id, audio_path in to_process:
            embedding = batch_embeddings.get(audio_path)
            if embedding is not None:
                embeddings[song_id] = embedding
                
                # Cache embedding
                if use_cache:
                    cache_path = self._get_embedding_cache_path(song_id)
                    try:
                        joblib.dump(embedding, cache_path)
                    except Exception as e:
                        print(f"Warning: Failed to cache embedding for {song_id}: {e}")
        
        return embeddings
    
    def load_embeddings_from_songs(
        self,
        songs: List[Dict],
        use_cache: bool = True,
        show_progress: bool = True,
        skip_download: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Load CLAP embeddings from a list of song dictionaries.
        
        Args:
            songs: List of song dictionaries (must have 'id' and 'audio_url' keys)
            use_cache: Whether to use cached embeddings
            show_progress: Whether to show progress bar
        
        Returns:
            Dictionary mapping song_id to embedding
        """
        audio_urls = []
        song_ids = []
        
        for song in songs:
            song_id = song.get('id') or song.get('song_id')
            audio_url = song.get('audio_url', '')
            
            if song_id and audio_url:
                song_ids.append(song_id)
                audio_urls.append(audio_url)
        
        return self.embed_audio_batch(
            audio_urls,
            song_ids,
            use_cache=use_cache,
            show_progress=show_progress,
            skip_download=skip_download
        )
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], file_path: str) -> None:
        """
        Save embeddings to a JSON file.
        
        Args:
            embeddings: Dictionary mapping song_id to embedding
            file_path: Path to save JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        embeddings_json = {
            song_id: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            for song_id, embedding in embeddings.items()
        }
        
        with open(file_path, 'w') as f:
            json.dump(embeddings_json, f, indent=2)
        
        print(f"✅ Saved {len(embeddings)} embeddings to {file_path}")
    
    
    def load_embeddings(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings from a JSON file.
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            Dictionary mapping song_id to embedding
        """
        with open(file_path, 'r') as f:
            embeddings_json = json.load(f)
        
        # Convert lists back to numpy arrays
        embeddings = {
            song_id: np.array(embedding)
            for song_id, embedding in embeddings_json.items()
        }
        
        print(f"✅ Loaded {len(embeddings)} embeddings from {file_path}")
        return embeddings


class CLAPTextEmbedder:
    """
    CLAP-based text embedder for the RecSys.
    
    CLAP (Contrastive Language-Audio Pretraining) provides text embeddings
    in a shared embedding space with audio, enabling cross-modal similarity search
    between prompts and audio tracks.
    
    Uses the same CLAP model checkpoint as CLAPAudioEmbedder for aligned embeddings.
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_CLAP_MODEL_PATH,
        cache_dir: str = "runtime_data/audio_cache",
        device: Optional[str] = None,
        batch_size: int = 16
    ):
        """
        Initialize CLAP text embedder.
        
        Args:
            model_path: Path to CLAP model file (defaults to DEFAULT_CLAP_MODEL_PATH, same as CLAPAudioEmbedder)
            cache_dir: Directory for text embedding cache
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for processing
        """
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.text_embedding_cache_dir = self.cache_dir / "clap_text_embeddings"
        self.text_embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.batch_size = batch_size
        self.model = None
    
    def initialize_model(self) -> bool:
        """
        Initialize the CLAP model (uses shared module-level cache).
        
        Returns:
            True if initialization successful, False otherwise
        """
        self.model = _get_clap_model(self.model_path, self.device)
        return self.model is not None
    
    def _get_text_embedding_cache_path(self, text: str) -> Path:
        """Get cache path for text embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.text_embedding_cache_dir / f"{text_hash}.pkl"
    
    def embed_text(
        self,
        text: str,
        use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """
        Compute CLAP text embedding for a single text string.
        
        CLAP provides aligned text and audio embeddings in a shared space,
        enabling direct similarity search between prompts and audio tracks.
        
        Uses the official CLAP API: model.get_text_embedding([text], use_tensor=True)
        
        Args:
            text: Text string (e.g., generation prompt)
            use_cache: Whether to use cached embedding
        
        Returns:
            CLAP text embedding as numpy array, or None if failed
        """
        if self.model is None:
            if not self.initialize_model():
                return None
        
        # Check cache
        if use_cache:
            cache_path = self._get_text_embedding_cache_path(text)
            if cache_path.exists():
                try:
                    return joblib.load(cache_path)
                except Exception as e:
                    print(f"Warning: Failed to load cached text embedding: {e}")
        
        try:
            # Use official CLAP API: model.get_text_embedding(text_data, use_tensor=True)
            # Reference: https://github.com/LAION-AI/CLAP
            text_embed = self.model.get_text_embedding([text], use_tensor=True)
            
            # Convert to numpy
            if isinstance(text_embed, torch.Tensor):
                text_embed = text_embed.cpu().numpy()
            
            # Extract single embedding (batch size 1)
            if len(text_embed) > 0:
                embedding = text_embed[0].flatten()
            else:
                return None
            
            # Cache embedding
            if use_cache:
                cache_path = self._get_text_embedding_cache_path(text)
                try:
                    joblib.dump(embedding, cache_path)
                except Exception as e:
                    print(f"Warning: Failed to cache text embedding: {e}")
            
            return embedding
        except Exception as e:
            print(f"Warning: Failed to compute CLAP text embedding: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def embed_texts(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute CLAP text embeddings for multiple texts.
        
        Uses the official CLAP API: model.get_text_embedding(text_data, use_tensor=True)
        
        Args:
            texts: List of text strings
            use_cache: Whether to use cached embeddings
            show_progress: Whether to show progress bar
        
        Returns:
            Dictionary mapping text to embedding
        """
        if self.model is None:
            if not self.initialize_model():
                return {}
        
        embeddings = {}
        texts_to_process = []
        
        # Check cache and collect texts to process
        for text in texts:
            if use_cache:
                cache_path = self._get_text_embedding_cache_path(text)
                if cache_path.exists():
                    try:
                        embeddings[text] = joblib.load(cache_path)
                        continue
                    except Exception:
                        pass
            
            texts_to_process.append(text)
        
        if not texts_to_process:
            return embeddings
        
        # Process in batches
        try:
            use_tqdm = show_progress
            if show_progress:
                try:
                    from tqdm import tqdm
                except ImportError:
                    use_tqdm = False
            
            iterator = tqdm(range(0, len(texts_to_process), self.batch_size), desc="Computing CLAP text embeddings") if use_tqdm else range(0, len(texts_to_process), self.batch_size)
            
            with torch.no_grad():
                for batch_start in iterator:
                    batch_end = min(batch_start + self.batch_size, len(texts_to_process))
                    batch_texts = texts_to_process[batch_start:batch_end]
                    
                    try:
                        # Use official CLAP API: model.get_text_embedding(text_data, use_tensor=True)
                        # Reference: https://github.com/LAION-AI/CLAP
                        text_embed = self.model.get_text_embedding(batch_texts, use_tensor=True)
                        
                        # Convert to numpy
                        if isinstance(text_embed, torch.Tensor):
                            text_embed = text_embed.cpu().numpy()
                        
                        # Store embeddings
                        for text, emb in zip(batch_texts, text_embed):
                            if emb is not None:
                                emb = emb.flatten()
                                embeddings[text] = emb
                                
                                # Cache
                                if use_cache:
                                    cache_path = self._get_text_embedding_cache_path(text)
                                    try:
                                        joblib.dump(emb, cache_path)
                                    except Exception as e:
                                        print(f"Warning: Failed to cache text embedding: {e}")
                    except Exception as e:
                        print(f"Warning: Failed to process text batch: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        except Exception as e:
            print(f"Warning: Failed to compute CLAP text embeddings: {e}")
            import traceback
            traceback.print_exc()
        
        return embeddings


def main():
    """CLI entry point for computing CLAP embeddings"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute CLAP audio embeddings')
    parser.add_argument('--input', required=True, help='Input JSON file with songs')
    parser.add_argument('--output', required=True, help='Output JSON file for embeddings')
    parser.add_argument('--model-path', default=DEFAULT_CLAP_MODEL_PATH, help='Path to CLAP model')
    parser.add_argument('--cache-dir', default='runtime_data/audio_cache', help='Audio cache directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--device', default=None, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load songs
    print(f"Loading songs from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'songs' in data:
        songs = data['songs']
    elif isinstance(data, list):
        songs = data
    else:
        raise ValueError(f"Unexpected format in {args.input}")
    
    print(f"Found {len(songs)} songs")
    
    # Initialize embedder
    embedder = CLAPAudioEmbedder(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    if not embedder.initialize_model():
        print("❌ Failed to initialize CLAP model")
        return
    
    # Compute embeddings
    embeddings = embedder.load_embeddings_from_songs(
        songs,
        use_cache=not args.no_cache,
        show_progress=True
    )
    
    # Save embeddings
    embedder.save_embeddings(embeddings, args.output)
    
    print(f"\n✅ Computed {len(embeddings)} CLAP embeddings")
    print(f"✅ Saved to {args.output}")


if __name__ == '__main__':
    main()

