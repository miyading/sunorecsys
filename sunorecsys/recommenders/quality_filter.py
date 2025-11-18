"""Quality filtering using music understanding models"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
import joblib
import json
from pathlib import Path

from .base import BaseRecommender


class QualityFilter:
    """
    Quality filter for songs using music understanding model scores.
    
    This can use:
    - Music Flamingo (NVIDIA) - Multi-dimensional quality scoring
    - Meta Audiobox Aesthetics metrics
    - Engagement-based heuristics (fallback)
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.5,
        quality_scorer: Optional[Callable] = None,
        quality_scores_file: Optional[str] = None,
        use_music_flamingo: bool = False,
        music_flamingo_model_id: str = "nvidia/music-flamingo-hf",
        device: Optional[str] = None,
    ):
        """
        Initialize quality filter
        
        Args:
            quality_threshold: Minimum quality score (0-1)
            quality_scorer: Optional function to compute quality scores.
                          If None, uses engagement-based heuristic (or external precomputed scores).
            quality_scores_file: Path to pre-computed quality scores JSON file
            use_music_flamingo: (Deprecated) Previously toggled Music Flamingo 内部调用。
                                 现在已拆分到精排 / 重排阶段，此参数保留仅为兼容旧代码，实际不会在此处触发 Music Flamingo。
            music_flamingo_model_id: (Deprecated) 保留以兼容旧调用签名
            device: (Deprecated) 保留以兼容旧调用签名
        """
        self.quality_threshold = quality_threshold
        self.quality_scorer = quality_scorer
        self.quality_scores_file = quality_scores_file
        # Music Flamingo 已迁移到精排/重排模块，这里仅保留配置以兼容旧代码
        self.use_music_flamingo = use_music_flamingo
        self.music_flamingo_model_id = music_flamingo_model_id
        self.device = device
        
        self.songs_df = None
        self.precomputed_scores = None
        
        # Load precomputed scores if provided
        if self.quality_scores_file and Path(self.quality_scores_file).exists():
            self._load_precomputed_scores()
    
    def _load_precomputed_scores(self):
        """Load precomputed quality scores from file"""
        try:
            with open(self.quality_scores_file, 'r') as f:
                data = json.load(f)
            
            # Handle different formats
            if isinstance(data, dict):
                # Format: {song_id: {scores: {...}, overall_quality_score: ...}}
                self.precomputed_scores = {}
                for song_id, song_data in data.items():
                    if isinstance(song_data, dict):
                        if 'overall_quality_score' in song_data:
                            self.precomputed_scores[song_id] = song_data['overall_quality_score']
                        elif 'scores' in song_data and 'overall_quality_score' in song_data['scores']:
                            self.precomputed_scores[song_id] = song_data['scores']['overall_quality_score']
                        elif 'overall_recommendation_score' in song_data:
                            self.precomputed_scores[song_id] = song_data['overall_recommendation_score']
                        elif 'scores' in song_data:
                            # Try to extract overall from nested scores
                            scores = song_data['scores']
                            if 'overall_quality_score' in scores:
                                self.precomputed_scores[song_id] = scores['overall_quality_score']
                            elif 'overall_recommendation_score' in scores:
                                self.precomputed_scores[song_id] = scores['overall_recommendation_score']
            
            print(f"✅ Loaded {len(self.precomputed_scores)} precomputed quality scores")
        except Exception as e:
            print(f"Warning: Failed to load precomputed scores: {e}")
            self.precomputed_scores = None
    
    def fit(self, songs_df: pd.DataFrame, **kwargs):
        """Fit the quality filter"""
        self.songs_df = songs_df.copy()
        
        # Use precomputed scores if available
        if self.precomputed_scores:
            self.songs_df['quality_score'] = self.songs_df['song_id'].map(
                self.precomputed_scores
            ).fillna(0.5)  # Default for missing
        
        # Otherwise, compute quality scores
        elif self.quality_scorer:
            self.songs_df['quality_score'] = songs_df.apply(
                lambda row: self.quality_scorer(row), axis=1
            )
        # Fallback to engagement-based heuristic
        else:
            self.songs_df['quality_score'] = self._compute_quality_scores(songs_df)
        
        # Normalize to [0, 1] (only if not already normalized)
        min_score = self.songs_df['quality_score'].min()
        max_score = self.songs_df['quality_score'].max()
        
        # Only normalize if scores aren't already in [0, 1] range
        if max_score > 1.0 or min_score < 0.0:
            if max_score > min_score:
                self.songs_df['quality_score'] = (
                    (self.songs_df['quality_score'] - min_score) / (max_score - min_score)
                )
            else:
                self.songs_df['quality_score'] = 0.5
    
    def _compute_quality_scores(self, songs_df: pd.DataFrame) -> pd.Series:
        """
        Compute quality scores using engagement metrics.
        
        In production, this would use Meta Audiobox Aesthetics or similar.
        For now, we use a heuristic based on engagement.
        """
        # Combine multiple signals
        engagement_rate = songs_df['engagement_rate'].fillna(0)
        popularity = np.log1p(songs_df['play_count'].fillna(0))
        upvote_ratio = songs_df['upvote_count'] / (songs_df['play_count'] + 1)
        
        # Normalize each component
        engagement_norm = (engagement_rate - engagement_rate.min()) / (engagement_rate.max() - engagement_rate.min() + 1e-8)
        popularity_norm = (popularity - popularity.min()) / (popularity.max() - popularity.min() + 1e-8)
        upvote_norm = (upvote_ratio - upvote_ratio.min()) / (upvote_ratio.max() - upvote_ratio.min() + 1e-8)
        
        # Weighted combination
        quality = (
            0.4 * engagement_norm +
            0.3 * popularity_norm +
            0.3 * upvote_norm
        )
        
        return quality
    
    def filter(self, song_ids: List[str]) -> List[str]:
        """Filter songs by quality threshold"""
        if self.songs_df is None:
            return song_ids
        
        filtered = self.songs_df[
            (self.songs_df['song_id'].isin(song_ids)) &
            (self.songs_df['quality_score'] >= self.quality_threshold)
        ]['song_id'].tolist()
        
        return filtered
    
    def score_songs(self, song_ids: List[str]) -> Dict[str, float]:
        """Get quality scores for songs"""
        if self.songs_df is None:
            return {sid: 0.5 for sid in song_ids}
        
        scores = self.songs_df[
            self.songs_df['song_id'].isin(song_ids)
        ][['song_id', 'quality_score']].set_index('song_id')['quality_score'].to_dict()
        
        # Fill missing with threshold
        return {sid: scores.get(sid, self.quality_threshold) for sid in song_ids}
    
    def apply_to_recommendations(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply quality filter to recommendations and adjust scores"""
        if self.songs_df is None:
            return recommendations
        
        filtered = []
        for rec in recommendations:
            song_id = rec['song_id']
            quality_score = self.songs_df[
                self.songs_df['song_id'] == song_id
            ]['quality_score'].values
            
            if len(quality_score) > 0 and quality_score[0] >= self.quality_threshold:
                # Boost score by quality
                rec['score'] = rec['score'] * (0.7 + 0.3 * quality_score[0])
                rec['quality_score'] = float(quality_score[0])
                filtered.append(rec)
        
        return filtered
    
    def save(self, path: str):
        """Save the quality filter"""
        joblib.dump({
            'quality_threshold': self.quality_threshold,
            'songs_df': self.songs_df,
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load the quality filter"""
        data = joblib.load(path)
        
        filter_obj = cls(quality_threshold=data['quality_threshold'])
        filter_obj.songs_df = data['songs_df']
        
        return filter_obj

