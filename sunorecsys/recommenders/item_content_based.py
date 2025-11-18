"""Item content-based recommender combining embeddings and genre/metadata"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import Counter
from annoy import AnnoyIndex
import joblib
from pathlib import Path

from .base import BaseRecommender
from ..utils.embeddings import TextEmbedder, TagEmbedder, PromptEmbedder


class ItemContentBasedRecommender(BaseRecommender):
    """
    Item content-based recommender combining:
    - Embeddings (tags, prompts, metadata)
    - Genre matching
    - Tag matching
    - Popularity matching
    
    This is a unified content-based approach that uses both
    semantic similarity (embeddings) and explicit feature matching (genre/tags).
    """
    
    def __init__(
        self,
        embedding_weight: float = 0.6,
        genre_weight: float = 0.2,
        tag_weight: float = 0.15,
        popularity_weight: float = 0.05,
        embedding_model: str = 'all-MiniLM-L6-v2',
        n_trees: int = 50,
    ):
        super().__init__("ItemContentBased")
        self.embedding_weight = embedding_weight
        self.genre_weight = genre_weight
        self.tag_weight = tag_weight
        self.popularity_weight = popularity_weight
        self.embedding_model = embedding_model
        self.n_trees = n_trees
        
        self.tag_embedder = None
        self.songs_df = None
        self.song_embeddings = None
        self.song_index = None
        self.song_id_to_idx = {}
        self.idx_to_song_id = {}
        self.embedding_dim = None
        self.genre_stats = None
        self.tag_stats = None
    
    def fit(self, songs_df: pd.DataFrame, user_history: Optional[Dict[str, List[str]]] = None, **kwargs):
        """
        Fit the item content-based recommender
        
        Args:
            user_history: Optional dict mapping user_id to list of song_ids (for last-n support)
        """
        self.songs_df = songs_df.copy()
        self.user_history = user_history or {}  # Store for last-n support
        
        print(f"  → Initializing embedders (model: {self.embedding_model})...")
        # Initialize embedders (prompt embedder not needed here, handled by Channel 3)
        base_embedder = TextEmbedder(self.embedding_model)
        self.tag_embedder = TagEmbedder(base_embedder)
        self.embedding_dim = base_embedder.embedding_dim
        
        print(f"  → Generating embeddings for {len(songs_df)} songs...")
        print(f"     Generating tag embeddings (prompt embeddings handled separately in Channel 3)...")
        
        # Generate tag embeddings only (prompt embeddings are handled by PromptBasedRecommender)
        tag_embeddings = self.tag_embedder.embed_tag_list(
            songs_df['tags'].tolist(),
            show_progress=True
        )
        
        print(f"  → Extracting metadata features...")
        # Normalize metadata features
        metadata_features = self._extract_metadata_features(songs_df)
        
        print(f"  → Combining embeddings and metadata...")
        # Combine embeddings (without prompt embeddings)
        combined_embeddings = self._combine_embeddings(
            tag_embeddings,
            metadata_features
        )
        
        self.song_embeddings = combined_embeddings
        
        print(f"  → Computing genre and tag statistics...")
        # Compute genre and tag statistics
        self.genre_stats = songs_df['genre'].value_counts().to_dict()
        self.tag_stats = self._compute_tag_stats(songs_df)
        
        # Store user history if provided
        self.user_history = user_history or {}
        
        # Build Annoy index for fast similarity search
        print(f"  → Building Annoy similarity index (n_trees={self.n_trees})...")
        print(f"     Adding {len(songs_df)} items to index...")
        self.song_index = AnnoyIndex(combined_embeddings.shape[1], 'angular')
        
        # Try to use tqdm if available for progress
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(songs_df['song_id']), total=len(songs_df), desc="     Adding items")
        except ImportError:
            iterator = enumerate(songs_df['song_id'])
            if len(songs_df) > 1000:
                print(f"     (Consider installing tqdm for progress bar: pip install tqdm)")
        
        for idx, song_id in iterator:
            self.song_index.add_item(idx, combined_embeddings[idx])
            self.song_id_to_idx[song_id] = idx
            self.idx_to_song_id[idx] = song_id
        
        print(f"     Building index with {self.n_trees} trees (this may take a moment)...")
        self.song_index.build(self.n_trees)
        self.is_fitted = True
        
        print(f"  ✅ Item Content-Based recommender fitted on {len(songs_df)} songs")
    
    def _extract_metadata_features(self, songs_df: pd.DataFrame) -> np.ndarray:
        """Extract and normalize metadata features"""
        features = []
        
        # Normalize numerical features - handle missing columns gracefully
        duration = songs_df.get('duration', pd.Series([0] * len(songs_df))).fillna(0).values
        duration_norm = (duration - duration.mean()) / (duration.std() + 1e-8)
        
        popularity = songs_df.get('popularity_score', pd.Series([0] * len(songs_df))).fillna(0).values
        popularity_norm = (popularity - popularity.mean()) / (popularity.std() + 1e-8)
        
        # engagement_rate may not exist, calculate if needed
        if 'engagement_rate' in songs_df.columns:
            engagement = songs_df['engagement_rate'].fillna(0).values
        else:
            # Calculate engagement_rate: upvotes / plays
            play_count = songs_df.get('play_count', pd.Series([0] * len(songs_df))).fillna(0).values
            upvote_count = songs_df.get('upvote_count', pd.Series([0] * len(songs_df))).fillna(0).values
            # Avoid division by zero
            engagement = np.where(play_count > 0, upvote_count / play_count, 0.0)
        
        engagement_norm = (engagement - engagement.mean()) / (engagement.std() + 1e-8)
        
        # One-hot encode genre - handle missing column
        genres = songs_df.get('genre', pd.Series(['unknown'] * len(songs_df))).fillna('unknown').values
        unique_genres = pd.unique(genres)
        genre_onehot = np.zeros((len(songs_df), len(unique_genres)))
        for i, genre in enumerate(genres):
            if genre in unique_genres:
                genre_onehot[i, np.where(unique_genres == genre)[0][0]] = 1.0
        
        # Combine features
        metadata = np.column_stack([
            duration_norm,
            popularity_norm,
            engagement_norm,
            genre_onehot,
        ])
        
        return metadata
    
    def _combine_embeddings(
        self,
        tag_embeddings: np.ndarray,
        metadata_features: np.ndarray
    ) -> np.ndarray:
        """Combine tag embeddings and metadata (prompt embeddings handled separately in Channel 3)"""
        # Normalize metadata to match embedding dimension
        if metadata_features.shape[1] > 0:
            target_dim = tag_embeddings.shape[1]
            if metadata_features.shape[1] < target_dim:
                padding = np.zeros((metadata_features.shape[0], target_dim - metadata_features.shape[1]))
                metadata_padded = np.hstack([metadata_features, padding])
            else:
                metadata_padded = metadata_features[:, :target_dim]
            
            metadata_norm = metadata_padded / (np.linalg.norm(metadata_padded, axis=1, keepdims=True) + 1e-8)
        else:
            metadata_norm = np.zeros_like(tag_embeddings)
        
        # Use tag embeddings only (prompt embeddings are in Channel 3)
        # Metadata handled separately in scoring
        combined = tag_embeddings + 0.0 * metadata_norm
        
        # Normalize
        combined = combined / (np.linalg.norm(combined, axis=1, keepdims=True) + 1e-8)
        
        return combined
    
    def _compute_tag_stats(self, songs_df: pd.DataFrame) -> Dict[str, float]:
        """Compute tag frequency statistics"""
        all_tags = []
        for tags in songs_df['tags']:
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        tag_counts = Counter(all_tags)
        total = sum(tag_counts.values())
        
        return {tag: count / total for tag, count in tag_counts.items()}
    
    def _get_user_preferences(self, user_id: Optional[str], song_ids: Optional[List[str]]) -> Dict[str, Any]:
        """Extract user preferences from history or seed songs"""
        preferences = {
            'genres': Counter(),
            'tags': Counter(),
            'popularity_range': None,
        }
        
        # Get songs from user history or seed songs
        if user_id and user_id in self.user_history:
            user_song_ids = self.user_history[user_id]
        elif song_ids:
            user_song_ids = song_ids
        else:
            return preferences
        
        # Get song data
        user_songs = self.songs_df[self.songs_df['song_id'].isin(user_song_ids)]
        
        if len(user_songs) == 0:
            return preferences
        
        # Extract genre preferences
        genres = user_songs['genre'].dropna()
        preferences['genres'] = Counter(genres)
        
        # Extract tag preferences
        all_tags = []
        for tags in user_songs['tags']:
            if isinstance(tags, list):
                all_tags.extend(tags)
        preferences['tags'] = Counter(all_tags)
        
        # Extract popularity preferences
        popularity_scores = user_songs['popularity_score'].values
        if len(popularity_scores) > 0:
            preferences['popularity_range'] = (
                popularity_scores.min(),
                popularity_scores.max(),
                popularity_scores.mean()
            )
        
        return preferences
    
    def _score_genre_tag_match(self, song: pd.Series, preferences: Dict[str, Any]) -> float:
        """Score a song based on genre and tag matches"""
        score = 0.0
        
        # Genre match
        if preferences['genres']:
            song_genre = song.get('genre')
            if song_genre and song_genre in preferences['genres']:
                genre_score = preferences['genres'][song_genre] / sum(preferences['genres'].values())
                score += self.genre_weight * genre_score
        
        # Tag match
        if preferences['tags']:
            song_tags = song.get('tags', [])
            if isinstance(song_tags, list):
                tag_matches = sum(preferences['tags'].get(tag, 0) for tag in song_tags)
                if tag_matches > 0:
                    tag_score = tag_matches / sum(preferences['tags'].values())
                    score += self.tag_weight * tag_score
        
        # Popularity match
        if preferences['popularity_range']:
            min_pop, max_pop, mean_pop = preferences['popularity_range']
            song_pop = song.get('popularity_score', 0)
            
            if min_pop <= song_pop <= max_pop:
                pop_score = 1.0
            else:
                distance = min(abs(song_pop - min_pop), abs(song_pop - max_pop))
                pop_score = 1.0 / (1.0 + distance / (mean_pop + 1e-8))
            
            score += self.popularity_weight * pop_score
        
        return score
    
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
        Recommend songs using combined content-based approach
        
        Args:
            use_last_n: If True and user_id provided, use last-n interactions from user history
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        # If user_id provided and use_last_n, get last-n interactions
        if user_id and use_last_n and self.user_history:
            if user_id in self.user_history:
                # Use last-n interactions from history
                last_n_interactions = self.user_history[user_id][-n:] if len(self.user_history[user_id]) > n else self.user_history[user_id]
                if last_n_interactions:
                    song_ids = last_n_interactions
        
        if not song_ids:
            return self._get_popular_songs(n)
        
        # Get user preferences for genre/tag matching
        preferences = self._get_user_preferences(user_id, song_ids)
        
        # Get average embedding of seed songs
        seed_indices = [self.song_id_to_idx[sid] for sid in song_ids if sid in self.song_id_to_idx]
        
        if not seed_indices:
            return self._get_popular_songs(n)
        
        seed_embeddings = self.song_embeddings[seed_indices]
        avg_embedding = seed_embeddings.mean(axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        # Find similar songs using embeddings
        exclude_indices = set()
        if exclude_song_ids:
            exclude_indices = {self.song_id_to_idx[sid] for sid in exclude_song_ids if sid in self.song_id_to_idx}
        
        # Use Annoy to find nearest neighbors
        query_idx = len(self.song_embeddings)
        temp_index = AnnoyIndex(self.song_embeddings.shape[1], 'angular')
        for i in range(len(self.song_embeddings)):
            temp_index.add_item(i, self.song_embeddings[i])
        temp_index.add_item(query_idx, avg_embedding)
        temp_index.build(self.n_trees)
        
        candidates = temp_index.get_nns_by_item(query_idx, n * 3, include_distances=True)
        
        # Combine embedding similarity with genre/tag matching
        results = []
        seen = set(song_ids)
        
        for idx, distance in zip(candidates[0], candidates[1]):
            if idx == query_idx or idx in exclude_indices:
                continue
            
            song_id = self.idx_to_song_id[idx]
            if song_id in seen:
                continue
            
            # Get embedding similarity score
            embedding_score = 1.0 - distance  # Convert distance to similarity
            
            # Get genre/tag matching score
            song = self.songs_df[self.songs_df['song_id'] == song_id].iloc[0]
            genre_tag_score = self._score_genre_tag_match(song, preferences)
            
            # Combine scores
            combined_score = (
                self.embedding_weight * embedding_score +
                (1.0 - self.embedding_weight) * genre_tag_score
            )
            
            results.append((song_id, combined_score, embedding_score, genre_tag_score))
            seen.add(song_id)
            
            if len(results) >= n:
                break
        
        # Sort by combined score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare details if requested
        details = None
        if return_details:
            details = {}
            for song_id, combined_score, embedding_score, genre_tag_score in results:
                details[song_id] = {
                    'combined_score': float(combined_score),
                    'embedding_similarity': float(embedding_score),
                    'genre_tag_match': float(genre_tag_score),
                    'component_weights': {
                        'embedding_weight': self.embedding_weight,
                        'genre_weight': self.genre_weight,
                        'tag_weight': self.tag_weight,
                        'popularity_weight': self.popularity_weight,
                    },
                    'seed_songs_count': len(seed_indices),
                }
        
        song_ids_list = [r[0] for r in results]
        scores = np.array([r[1] for r in results])
        
        return self._format_recommendations(
            song_ids_list,
            scores,
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
        """Get similar songs using content-based similarity"""
        if song_id not in self.song_id_to_idx:
            return []
        
        idx = self.song_id_to_idx[song_id]
        similar_indices, distances = self.song_index.get_nns_by_item(idx, n + 1, include_distances=True)
        
        # Remove the song itself
        similar_indices = [i for i in similar_indices if i != idx][:n]
        distances = distances[1:n+1]
        
        song_ids = [self.idx_to_song_id[i] for i in similar_indices]
        scores = 1.0 - np.array(distances)  # Convert to similarity
        
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
            'embedding_weight': self.embedding_weight,
            'genre_weight': self.genre_weight,
            'tag_weight': self.tag_weight,
            'popularity_weight': self.popularity_weight,
            'embedding_model': self.embedding_model,
            'n_trees': self.n_trees,
            'song_id_to_idx': self.song_id_to_idx,
            'idx_to_song_id': self.idx_to_song_id,
            'embedding_dim': self.embedding_dim,
            'songs_df': self.songs_df,
            'song_embeddings': self.song_embeddings,
            'genre_stats': self.genre_stats,
            'tag_stats': self.tag_stats,
            'user_history': getattr(self, 'user_history', {}),
            'index_path': index_path,
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load the recommender"""
        data = joblib.load(path)
        
        recommender = cls(
            embedding_weight=data['embedding_weight'],
            genre_weight=data['genre_weight'],
            tag_weight=data['tag_weight'],
            popularity_weight=data['popularity_weight'],
            embedding_model=data['embedding_model'],
            n_trees=data['n_trees'],
        )
        
        recommender.song_id_to_idx = data['song_id_to_idx']
        recommender.idx_to_song_id = data['idx_to_song_id']
        recommender.embedding_dim = data['embedding_dim']
        recommender.songs_df = data['songs_df']
        recommender.song_embeddings = data['song_embeddings']
        recommender.genre_stats = data['genre_stats']
        recommender.tag_stats = data['tag_stats']
        recommender.user_history = data.get('user_history', {})
        
        if Path(data['index_path']).exists():
            recommender.song_index = AnnoyIndex(recommender.embedding_dim, 'angular')
            recommender.song_index.load(data['index_path'])
        
        base_embedder = TextEmbedder(recommender.embedding_model)
        recommender.tag_embedder = TagEmbedder(base_embedder)
        # prompt_embedder removed - handled by Channel 3 (PromptBasedRecommender)
        
        recommender.is_fitted = True
        
        return recommender

