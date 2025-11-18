"""Base recommender interface"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Base class for all recommenders"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, songs_df: pd.DataFrame, **kwargs):
        """Fit the recommender on song data"""
        pass
    
    @abstractmethod
    def recommend(
        self,
        user_id: Optional[str] = None,
        song_ids: Optional[List[str]] = None,
        n: int = 10,
        return_details: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations
        
        Args:
            user_id: User ID (if user-based)
            song_ids: List of seed song IDs (if item-based)
            n: Number of recommendations to return
            return_details: If True, return detailed information for debugging
        
        Returns:
            List of recommendation dicts with 'song_id' and 'score'.
            If return_details=True, includes additional debugging information.
        """
        pass
    
    @abstractmethod
    def get_similar_songs(
        self,
        song_id: str,
        n: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get similar songs to a given song"""
        pass
    
    def _format_recommendations(
        self,
        song_ids: List[str],
        scores: np.ndarray,
        songs_df: pd.DataFrame,
        return_details: bool = False,
        details: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Format recommendations with metadata
        
        Args:
            song_ids: List of song IDs
            scores: Array of scores
            songs_df: DataFrame with song metadata
            return_details: If True, include detailed debugging info
            details: Optional dict with channel-specific details (e.g., similarity scores, features used)
        
        Returns:
            List of recommendation dicts
        """
        recommendations = []
        
        for idx, (song_id, score) in enumerate(zip(song_ids, scores)):
            song_info = songs_df[songs_df['song_id'] == song_id].iloc[0].to_dict()
            
            rec = {
                'song_id': song_id,
                'score': float(score),
                'title': song_info.get('title', ''),
                'genre': song_info.get('genre', ''),
                'tags': song_info.get('tags', []),
                'audio_url': song_info.get('audio_url', ''),
                'prompt': song_info.get('prompt', ''),
                'suno_url': f"https://suno.com/song/{song_id}",  # Add Suno URL
                'rank': idx + 1,
            }
            
            # Add detailed information if requested
            if return_details:
                rec['details'] = {
                    'channel': self.name,
                    'raw_score': float(score),
                    'normalized_score': float(score),  # May be overridden by subclasses
                }
                
                # Add channel-specific details if provided
                if details and song_id in details:
                    rec['details'].update(details[song_id])
            
            recommendations.append(rec)
        
        return recommendations

