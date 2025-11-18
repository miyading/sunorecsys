"""Prompt-based similarity recommender"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from annoy import AnnoyIndex
import joblib
from pathlib import Path
import hashlib

from .base import BaseRecommender
from ..utils.embeddings import TextEmbedder, PromptEmbedder


class PromptBasedRecommender(BaseRecommender):
    """
    Recommender based on generation prompt similarity.
    
    This is unique to music generation models - users who like songs
    generated from similar prompts may enjoy similar music.
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        n_trees: int = 50,
    ):
        super().__init__("PromptBased")
        self.embedding_model = embedding_model
        self.n_trees = n_trees
        
        self.prompt_embedder = None
        self.songs_df = None
        self.prompt_embeddings = None
        self.song_index = None
        self.song_id_to_idx = {}
        self.idx_to_song_id = {}
    
    def fit(self, songs_df: pd.DataFrame, user_history: Optional[Dict[str, List[str]]] = None, **kwargs):
        """
        Fit the prompt-based recommender
        
        Args:
            user_history: Optional dict mapping user_id to list of song_ids (for last-n support)
        """
        self.songs_df = songs_df.copy()
        self.user_history = user_history or {}  # Store for last-n support
        
        print(f"  → Initializing prompt embedder (model: {self.embedding_model})...")
        # Initialize embedder
        base_embedder = TextEmbedder(self.embedding_model)
        self.prompt_embedder = PromptEmbedder(base_embedder)
        
        embedding_dim = base_embedder.embedding_dim
        
        # Check for cached prompt embeddings
        cache_dir = kwargs.get('cache_dir', 'data/cache')
        use_cache = kwargs.get('use_cache', True)
        cache_path = None
        song_ids_sorted = sorted(songs_df['song_id'].tolist())
        
        if use_cache:
            # Generate cache key from song_ids and prompts
            prompts_str = '|'.join(songs_df['prompt'].fillna('').tolist())
            cache_key = hashlib.md5(
                (','.join(song_ids_sorted) + '|' + prompts_str + '|' + self.embedding_model).encode()
            ).hexdigest()
            
            cache_dir_path = Path(cache_dir)
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir_path / f"prompt_embeddings_{cache_key}.pkl"
            
            # Try to load cached embeddings
            if cache_path.exists():
                try:
                    print(f"  → Loading cached prompt embeddings from {cache_path}...")
                    cached_data = joblib.load(cache_path)
                    cached_embeddings = cached_data.get('prompt_embeddings')
                    # Verify embeddings match current songs
                    if (cached_embeddings is not None and 
                        cached_embeddings.shape[0] == len(songs_df) and 
                        cached_data.get('song_ids') == song_ids_sorted):
                        self.prompt_embeddings = cached_embeddings
                        print(f"  ✅ Loaded {len(songs_df)} cached prompt embeddings")
                    else:
                        print(f"  ⚠️  Cache mismatch, regenerating embeddings...")
                        self.prompt_embeddings = None  # Force regeneration
                except Exception as e:
                    print(f"  ⚠️  Failed to load cache: {e}, regenerating embeddings...")
                    self.prompt_embeddings = None
        
        # Generate prompt embeddings if not cached
        if self.prompt_embeddings is None:
            print(f"  → Generating prompt embeddings for {len(songs_df)} songs...")
            prompts = songs_df['prompt'].fillna('').tolist()
            self.prompt_embeddings = self.prompt_embedder.embed_prompts(prompts, show_progress=True)
            
            # Save to cache
            if use_cache and cache_path:
                try:
                    print(f"  → Saving prompt embeddings to cache: {cache_path}")
                    joblib.dump({
                        'prompt_embeddings': self.prompt_embeddings,
                        'song_ids': song_ids_sorted,
                        'embedding_model': self.embedding_model,
                        'num_songs': len(songs_df),
                    }, cache_path)
                    print(f"  ✅ Cached prompt embeddings saved")
                except Exception as e:
                    print(f"  ⚠️  Failed to save cache: {e}")
        
        # Build Annoy index
        print(f"  → Building Annoy prompt similarity index (n_trees={self.n_trees})...")
        print(f"     Adding {len(songs_df)} items to index...")
        self.song_index = AnnoyIndex(embedding_dim, 'angular')
        
        # Try to use tqdm if available for progress
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(songs_df['song_id']), total=len(songs_df), desc="     Adding items")
        except ImportError:
            iterator = enumerate(songs_df['song_id'])
            if len(songs_df) > 1000:
                print(f"     (Consider installing tqdm for progress bar: pip install tqdm)")
        
        for idx, song_id in iterator:
            self.song_index.add_item(idx, self.prompt_embeddings[idx])
            self.song_id_to_idx[song_id] = idx
            self.idx_to_song_id[idx] = song_id
        
        print(f"     Building index with {self.n_trees} trees (this may take a moment)...")
        self.song_index.build(self.n_trees)
        self.is_fitted = True
        
        print(f"  ✅ Prompt-Based recommender fitted on {len(songs_df)} songs")
    
    def recommend(
        self,
        user_id: Optional[str] = None,
        song_ids: Optional[List[str]] = None,
        n: int = 10,
        exclude_song_ids: Optional[List[str]] = None,
        return_details: bool = False,
        use_last_n: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Recommend songs based on prompt similarity to seed songs
        
        Args:
            use_last_n: If True and user_id provided, use last-n interactions from user history
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        # If user_id provided and use_last_n, get last-n interactions
        if user_id and use_last_n and hasattr(self, 'user_history') and self.user_history:
            if user_id in self.user_history:
                last_n_interactions = self.user_history[user_id][-n:] if len(self.user_history[user_id]) > n else self.user_history[user_id]
                if last_n_interactions:
                    song_ids = last_n_interactions
        
        if not song_ids:
            return self._get_popular_songs(n)
        
        # Get average prompt embedding of seed songs
        seed_indices = [self.song_id_to_idx[sid] for sid in song_ids if sid in self.song_id_to_idx]
        
        if not seed_indices:
            return self._get_popular_songs(n)
        
        seed_embeddings = self.prompt_embeddings[seed_indices]
        avg_embedding = seed_embeddings.mean(axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        # Find similar songs
        exclude_indices = set()
        if exclude_song_ids:
            exclude_indices = {self.song_id_to_idx[sid] for sid in exclude_song_ids if sid in self.song_id_to_idx}
        
        # Create temporary index for query
        query_idx = len(self.prompt_embeddings)
        temp_index = AnnoyIndex(self.prompt_embeddings.shape[1], 'angular')
        for i in range(len(self.prompt_embeddings)):
            temp_index.add_item(i, self.prompt_embeddings[i])
        temp_index.add_item(query_idx, avg_embedding)
        temp_index.build(self.n_trees)
        
        candidates = temp_index.get_nns_by_item(query_idx, n * 3, include_distances=True)
        
        # Filter and format results
        results = []
        seen = set(song_ids)
        
        for idx, distance in zip(candidates[0], candidates[1]):
            if idx == query_idx or idx in exclude_indices:
                continue
            
            song_id = self.idx_to_song_id[idx]
            if song_id in seen:
                continue
            
            score = 1.0 - distance
            results.append((song_id, score))
            seen.add(song_id)
            
            if len(results) >= n:
                break
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare details if requested
        details = None
        if return_details:
            details = {}
            for song_id, score in results:
                song_idx = self.song_id_to_idx.get(song_id)
                if song_idx is not None:
                    details[song_id] = {
                        'prompt_similarity': float(score),
                        'annoy_distance': float(1.0 - score),
                        'seed_songs_count': len(seed_indices),
                    }
        
        return self._format_recommendations(
            [r[0] for r in results],
            np.array([r[1] for r in results]),
            self.songs_df,
            return_details=return_details,
            details=details
        )
    
    def get_similar_songs(
        self,
        song_id: str,
        n: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get songs with similar prompts"""
        if song_id not in self.song_id_to_idx:
            return []
        
        idx = self.song_id_to_idx[song_id]
        similar_indices, distances = self.song_index.get_nns_by_item(idx, n + 1, include_distances=True)
        
        similar_indices = [i for i in similar_indices if i != idx][:n]
        distances = distances[1:n+1]
        
        song_ids = [self.idx_to_song_id[i] for i in similar_indices]
        scores = 1.0 - np.array(distances)
        
        return self._format_recommendations(song_ids, scores, self.songs_df)
    
    def _get_popular_songs(self, n: int) -> List[Dict[str, Any]]:
        """Get popular songs as fallback"""
        popular = self.songs_df.nlargest(n, 'popularity_score')
        scores = popular['popularity_score'].values
        scores = scores / scores.max()
        
        return self._format_recommendations(
            popular['song_id'].tolist(),
            scores,
            self.songs_df
        )
    
    def save(self, path: str):
        """Save the recommender"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        index_path = str(Path(path).with_suffix('.annoy'))
        if self.song_index:
            self.song_index.save(index_path)
        
        joblib.dump({
            'name': self.name,
            'embedding_model': self.embedding_model,
            'n_trees': self.n_trees,
            'song_id_to_idx': self.song_id_to_idx,
            'idx_to_song_id': self.idx_to_song_id,
            'songs_df': self.songs_df,
            'prompt_embeddings': self.prompt_embeddings,
            'index_path': index_path,
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load the recommender"""
        data = joblib.load(path)
        
        recommender = cls(
            embedding_model=data['embedding_model'],
            n_trees=data['n_trees'],
        )
        
        recommender.song_id_to_idx = data['song_id_to_idx']
        recommender.idx_to_song_id = data['idx_to_song_id']
        recommender.songs_df = data['songs_df']
        recommender.prompt_embeddings = data['prompt_embeddings']
        
        if Path(data['index_path']).exists():
            embedding_dim = recommender.prompt_embeddings.shape[1]
            recommender.song_index = AnnoyIndex(embedding_dim, 'angular')
            recommender.song_index.load(data['index_path'])
        
        base_embedder = TextEmbedder(recommender.embedding_model)
        recommender.prompt_embedder = PromptEmbedder(base_embedder)
        
        recommender.is_fitted = True
        
        return recommender

