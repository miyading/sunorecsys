"""Music Flamingo quality scoring for Suno RecSys"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import torch

# Try to import transformers - Music Flamingo uses it
try:
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install it to use Music Flamingo.")


# Quality scoring prompt template
QUALITY_PROMPT_TEMPLATE = """Describe this Suno AI generated track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.

Then score the track on four dimensions [0,1]: 
1. Content Usefulness: How useful/meaningful is the content? (0.0 to 1.0)
2. Production Quality: Technical production quality? (0.0 to 1.0)
3. Content Enjoyment: How enjoyable is the content? (0.0 to 1.0)
4. Production Complexity: Complexity of production? (0.0 to 1.0)

Finally, give an overall recommendation score from 0.0 to 1.0 on whether this track should be recommended to other listeners.

IMPORTANT: Format your response as JSON with these exact keys:
{{
  "genre": "string",
  "tempo": "string",
  "key": "string",
  "instruments": "string",
  "production_style": "string",
  "mood": "string",
  "content_usefulness": 0.0,
  "production_quality": 0.0,
  "content_enjoyment": 0.0,
  "production_complexity": 0.0,
  "overall_recommendation_score": 0.0
}}"""


class MusicFlamingoQualityScorer:
    """
    Quality scorer using NVIDIA Music Flamingo model.
    
    Scores songs on multiple dimensions for quality filtering.
    """
    
    def __init__(
        self,
        model_id: str = "nvidia/music-flamingo-hf",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Music Flamingo quality scorer.
        
        Args:
            model_id: HuggingFace model ID for Music Flamingo
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_new_tokens: Maximum tokens to generate
            cache_dir: Directory to cache models
        """
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        
        self.model = None
        self.processor = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the Music Flamingo model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not TRANSFORMERS_AVAILABLE:
            print("❌ transformers not available. Please install it.")
            return False
        
        if self.is_initialized:
            return True
        
        try:
            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"🔧 Loading Music Flamingo model on {self.device}...")
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            
            self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto" if self.device == "cuda" else None,
                cache_dir=self.cache_dir
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.is_initialized = True
            
            print(f"✅ Music Flamingo model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Music Flamingo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _parse_quality_response(self, text: str) -> Optional[Dict[str, float]]:
        """
        Parse quality scores from Music Flamingo response.
        
        Args:
            text: Response text from model
        
        Returns:
            Dictionary with quality scores or None if parsing fails
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                scores = {
                    'content_usefulness': float(parsed.get('content_usefulness', 0.5)),
                    'production_quality': float(parsed.get('production_quality', 0.5)),
                    'content_enjoyment': float(parsed.get('content_enjoyment', 0.5)),
                    'production_complexity': float(parsed.get('production_complexity', 0.5)),
                    'overall_recommendation_score': float(parsed.get('overall_recommendation_score', 0.5)),
                }
                # Clamp values to [0, 1]
                scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}
                return scores
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Warning: Failed to parse JSON from response: {e}")
        
        # Fallback: try to extract numeric scores using regex
        patterns = {
            'content_usefulness': r'content\s*usefulness[:\s]+([\d.]+)',
            'production_quality': r'production\s*quality[:\s]+([\d.]+)',
            'content_enjoyment': r'content\s*enjoyment[:\s]+([\d.]+)',
            'production_complexity': r'production\s*complexity[:\s]+([\d.]+)',
            'overall_recommendation_score': r'(?:overall|recommendation)[\s\w]*score[:\s]+([\d.]+)',
        }
        
        scores = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    scores[key] = max(0.0, min(1.0, float(match.group(1))))
                except ValueError:
                    pass
        
        # If we found at least some scores, use them
        if scores:
            # Fill missing with 0.5
            for key in ['content_usefulness', 'production_quality', 'content_enjoyment', 
                       'production_complexity', 'overall_recommendation_score']:
                if key not in scores:
                    scores[key] = 0.5
            return scores
        
        # Last resort: return default scores
        print(f"Warning: Could not parse scores from response. Using defaults.")
        return {
            'content_usefulness': 0.5,
            'production_quality': 0.5,
            'content_enjoyment': 0.5,
            'production_complexity': 0.5,
            'overall_recommendation_score': 0.5,
        }
    
    def score_audio(
        self,
        audio_path: str,
        use_cache: bool = True,
        cache_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Score a single audio file.
        
        Args:
            audio_path: Path to audio file or URL
            use_cache: Whether to use cached scores
            cache_path: Path to cache file
        
        Returns:
            Dictionary with quality scores
        """
        if not self.is_initialized:
            if not self.initialize():
                return self._get_default_scores()
        
        # Check cache
        if use_cache and cache_path and Path(cache_path).exists():
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                    if isinstance(cached, dict) and 'overall_recommendation_score' in cached:
                        return cached
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
        
        try:
            # Prepare conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": QUALITY_PROMPT_TEMPLATE},
                        {"type": "audio", "path": audio_path},
                    ],
                }
            ]
            
            # Process
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Deterministic for consistency
                )
            
            # Decode
            decoded = self.processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            response_text = decoded[0] if decoded else ""
            
            # Parse scores
            scores = self._parse_quality_response(response_text)
            
            # Cache results
            if use_cache and cache_path:
                try:
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, 'w') as f:
                        json.dump(scores, f, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to cache scores: {e}")
            
            return scores
            
        except Exception as e:
            print(f"Warning: Failed to score audio {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_scores()
    
    def _get_default_scores(self) -> Dict[str, float]:
        """Get default quality scores"""
        return {
            'content_usefulness': 0.5,
            'production_quality': 0.5,
            'content_enjoyment': 0.5,
            'production_complexity': 0.5,
            'overall_recommendation_score': 0.5,
        }
    
    def score_audio_batch(
        self,
        audio_paths: List[str],
        audio_ids: List[str],
        cache_dir: Optional[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Score multiple audio files.
        
        Args:
            audio_paths: List of audio file paths or URLs
            audio_ids: List of audio IDs (for caching)
            cache_dir: Directory to cache scores
            show_progress: Whether to show progress bar
        
        Returns:
            Dictionary mapping audio_id to quality scores
        """
        if len(audio_paths) != len(audio_ids):
            raise ValueError("audio_paths and audio_ids must have the same length")
        
        results = {}
        
        iterator = tqdm(zip(audio_paths, audio_ids), total=len(audio_paths), desc="Scoring audio") if show_progress else zip(audio_paths, audio_ids)
        
        for audio_path, audio_id in iterator:
            cache_path = None
            if cache_dir:
                cache_path = Path(cache_dir) / f"{audio_id}.json"
            
            scores = self.score_audio(
                audio_path,
                use_cache=True,
                cache_path=str(cache_path) if cache_path else None,
            )
            results[audio_id] = scores
        
        return results
    
    def get_overall_quality_score(self, scores: Dict[str, float]) -> float:
        """
        Get a single overall quality score from multi-dimensional scores.
        
        Args:
            scores: Dictionary with quality scores
        
        Returns:
            Single quality score in [0, 1]
        """
        # Weighted combination
        weights = {
            'content_usefulness': 0.15,
            'production_quality': 0.30,
            'content_enjoyment': 0.25,
            'production_complexity': 0.10,
            'overall_recommendation_score': 0.20,
        }
        
        overall = sum(
            weights.get(key, 0.0) * scores.get(key, 0.5)
            for key in weights.keys()
        )
        
        return max(0.0, min(1.0, overall))


def create_quality_scorer_function(scorer: MusicFlamingoQualityScorer, cache_dir: str = "runtime_data/music_flamingo_scores") -> callable:
    """
    Create a quality scorer function compatible with QualityFilter.
    
    Args:
        scorer: MusicFlamingoQualityScorer instance
        cache_dir: Directory to cache scores
    
    Returns:
        Function that takes a song row and returns a quality score
    """
    def quality_scorer(row) -> float:
        """Quality scorer function for QualityFilter"""
        audio_url = row.get('audio_url', '')
        song_id = row.get('song_id') or row.get('id', '')
        
        if not audio_url or not song_id:
            return 0.5
        
        cache_path = Path(cache_dir) / f"{song_id}.json"
        
        scores = scorer.score_audio(
            audio_url,
            use_cache=True,
            cache_path=str(cache_path),
        )
        
        return scorer.get_overall_quality_score(scores)
    
    return quality_scorer

