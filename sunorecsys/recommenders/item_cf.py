"""Item-based Collaborative Filtering using user-item interaction matrix"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from .base import BaseRecommender
from ..data.simulate_interactions import get_user_item_matrix


class ItemBasedCFRecommender(BaseRecommender):
    """
    Item-based Collaborative Filtering.
    
    Recommends items based on item-item similarity computed from user-item
    interaction matrix. Two items are similar if the same group of users
    have interacted with both items.
    """
    
    def __init__(self):
        super().__init__("ItemBasedCF")
        self.songs_df = None
        self.user_item_matrix = None
        self.item_item_similarity = None
        self.song_id_to_idx = {}
        self.idx_to_song_id = {}
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
    
    def fit(
        self,
        songs_df: pd.DataFrame,
        playlists: Optional[List[Dict[str, Any]]] = None,
        use_simulated_interactions: bool = True,
        **kwargs
    ):
        """
        Fit the item-based CF recommender
        
        Args:
            songs_df: DataFrame with song data
            playlists: Optional list of playlists (dict with 'id' and 'song_ids')
                      If not provided, will use simulated interactions
            use_simulated_interactions: If True and no playlists, use simulated interactions
                                       If False, fall back to user_id-based approach
        """
        self.songs_df = songs_df.copy()
        
        # Build user-item matrix
        if playlists:
            print("  → Building matrix from playlist data...")
            self._build_matrix_from_playlists(playlists)
        elif use_simulated_interactions:
            # Use simulated interactions (better for CF)
            # If playlists provided, use them as real interaction data first, then simulate additional
            print("  → using simulated interactions...")
            if playlists:
                print("  → Will use playlists as real interaction data, then simulate additional interactions")
            print("  → This may take a moment for large datasets...")
            (
                self.user_item_matrix,
                self.user_id_to_idx,
                self.idx_to_user_id,
                self.song_id_to_idx,
                self.idx_to_song_id,
            ) = get_user_item_matrix(
                songs_df,
                playlists=playlists,  # Pass playlists if provided
                use_simulation=True,
                interaction_rate=kwargs.get('interaction_rate', 0.15),
                item_cold_start_rate=kwargs.get('item_cold_start_rate', 0.05),
                single_user_item_rate=kwargs.get('single_user_item_rate', 0.15),
                random_seed=kwargs.get('random_seed', 42),
                cache_dir=kwargs.get('cache_dir', 'data/cache'),
                use_cache=kwargs.get('use_cache', True),
            )
        else:
            # Fallback: simulate playlists from user_id (each user's songs = a playlist)
            print("  → Building matrix from user_id groups...")
            self._build_matrix_from_users()
        
        print(f"  → Matrix built: {self.user_item_matrix.shape[0]} users × {self.user_item_matrix.shape[1]} items")
        print(f"  → Computing item-item similarity matrix...")
        print(f"     (This may take a while for large item sets...)")
        
        # Compute item-item similarity
        # If matrix has weights (play counts, etc.), use weighted similarity
        # Otherwise, use binary cosine similarity
        item_vectors = self.user_item_matrix.T  # Transpose: items x users
        
        # Check if matrix has weights (non-binary values)
        has_weights = (self.user_item_matrix.data != 1.0).any() if hasattr(self.user_item_matrix, 'data') else False
        
        if has_weights:
            # Weighted cosine similarity: normalize by item weights
            # For each item pair, compute weighted dot product
            print("     → Using weighted item-item similarity (play counts/interaction weights)")
            # Normalize item vectors by their L2 norm (weighted)
            item_norms = np.sqrt(np.array(item_vectors.power(2).sum(axis=1)).flatten())
            item_norms = np.where(item_norms > 0, item_norms, 1.0)  # Avoid division by zero
            
            # Normalize and compute similarity
            item_vectors_normalized = item_vectors.multiply(1.0 / item_norms.reshape(-1, 1))
            self.item_item_similarity = item_vectors_normalized @ item_vectors_normalized.T
        else:
            # Binary cosine similarity (standard CF)
            print("     → Using binary item-item similarity")
            self.item_item_similarity = cosine_similarity(item_vectors, dense_output=False)
        
        self.is_fitted = True
        print(f"  ✅ Item-item similarity matrix computed: {self.item_item_similarity.shape}")
        print(f"  ✅ Fitted item-based CF on {len(self.song_id_to_idx)} items")
    
    def _build_matrix_from_playlists(self, playlists: List[Dict[str, Any]]):
        """Build user-item matrix from playlist data"""
        rows = []
        cols = []
        data = []
        
        all_song_ids = set(self.songs_df['song_id'].unique())
        all_user_ids = set()
        
        for playlist in playlists:
            playlist_id = playlist.get('id', f"playlist_{len(all_user_ids)}")
            song_ids = playlist.get('song_ids', [])
            
            song_ids = [sid for sid in song_ids if sid in all_song_ids]
            
            if len(song_ids) == 0:
                continue
            
            all_user_ids.add(playlist_id)
            
            for song_id in song_ids:
                if song_id not in self.song_id_to_idx:
                    idx = len(self.song_id_to_idx)
                    self.song_id_to_idx[song_id] = idx
                    self.idx_to_song_id[idx] = song_id
                
                rows.append(len(all_user_ids) - 1)
                cols.append(self.song_id_to_idx[song_id])
                data.append(1.0)
        
        for user_id in all_user_ids:
            if user_id not in self.user_id_to_idx:
                idx = len(self.user_id_to_idx)
                self.user_id_to_idx[user_id] = idx
                self.idx_to_user_id[idx] = user_id
        
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_id_to_idx), len(self.song_id_to_idx))
        )
    
    def _build_matrix_from_users(self):
        """Build user-item matrix from user_id in songs (simulate playlists)"""
        user_songs = self.songs_df.groupby('user_id')['song_id'].apply(list).to_dict()
        
        rows = []
        cols = []
        data = []
        
        for song_id in self.songs_df['song_id'].unique():
            if song_id not in self.song_id_to_idx:
                idx = len(self.song_id_to_idx)
                self.song_id_to_idx[song_id] = idx
                self.idx_to_song_id[idx] = song_id
        
        for user_id, song_ids in user_songs.items():
            if user_id not in self.user_id_to_idx:
                idx = len(self.user_id_to_idx)
                self.user_id_to_idx[user_id] = idx
                self.idx_to_user_id[idx] = user_id
            
            user_idx = self.user_id_to_idx[user_id]
            
            for song_id in song_ids:
                if song_id in self.song_id_to_idx:
                    song_idx = self.song_id_to_idx[song_id]
                    rows.append(user_idx)
                    cols.append(song_idx)
                    data.append(1.0)
        
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_id_to_idx), len(self.song_id_to_idx))
        )
    
    def _is_cold_start_item(self, song_id: str) -> bool:
        """Check if an item is a cold-start item (no interactions in training)"""
        if song_id not in self.song_id_to_idx:
            return True  # New item not in training data
        
        song_idx = self.song_id_to_idx[song_id]
        if song_idx >= self.user_item_matrix.shape[1]:
            return True
        
        # Check if item has any interactions
        item_interactions = self.user_item_matrix[:, song_idx].sum()
        return item_interactions == 0
    
    def recommend(
        self,
        user_id: Optional[str] = None,
        song_ids: Optional[List[str]] = None,
        n: int = 10,
        exclude_song_ids: Optional[List[str]] = None,
        return_details: bool = False,
        content_fallback: Optional[Any] = None,
        top_k_per_seed: int = 20,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Recommend songs using item-based CF (Discover Weekly style)
        
        Args:
            user_id: User ID (for user history lookup)
            song_ids: List of seed song IDs (last-n interactions)
            n: Number of recommendations to return
            exclude_song_ids: Songs to exclude from recommendations
            return_details: If True, return detailed information
            content_fallback: Optional content-based recommender for cold-start items
            top_k_per_seed: For each seed song, find top-k most similar (then dedup)
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        if not song_ids:
            return self._get_popular_songs(n)
        
        # Check for cold-start items in seed songs
        cold_start_seeds = [sid for sid in song_ids if self._is_cold_start_item(sid)]
        if cold_start_seeds and content_fallback:
            # If all seed songs are cold-start, use content-based fallback
            if len(cold_start_seeds) == len(song_ids):
                if return_details:
                    print(f"⚠️  All seed songs are cold-start items. Using content-based fallback.")
                return content_fallback.recommend(
                    song_ids=song_ids,
                    n=n,
                    exclude_song_ids=exclude_song_ids,
                    return_details=return_details,
                    **kwargs
                )
            # If some are cold-start, filter them out and proceed with CF
            song_ids = [sid for sid in song_ids if sid not in cold_start_seeds]
            if not song_ids:
                return self._get_popular_songs(n)
        
        # Get seed song indices
        seed_indices = [self.song_id_to_idx[sid] for sid in song_ids if sid in self.song_id_to_idx]
        
        if not seed_indices:
            return self._get_popular_songs(n)
        
        # Discover Weekly style: For each seed song, find top-k most similar
        # Then aggregate and deduplicate
        all_candidates = {}  # song_idx -> (similarity, seed_idx)
        
        for seed_idx in seed_indices:
            if seed_idx >= self.item_item_similarity.shape[0]:
                continue
            
            # Get similarities to this seed song
            similarities = self.item_item_similarity[seed_idx]
            
            # Convert to dense array for top-k selection
            if hasattr(similarities, 'toarray'):
                similarities_array = similarities.toarray().flatten()
            else:
                similarities_array = np.array(similarities).flatten()
            
            # Get top-k most similar items (excluding seed itself)
            similarities_array[seed_idx] = -np.inf  # Exclude seed
            
            top_k_indices = np.argsort(similarities_array)[::-1][:top_k_per_seed]
            
            # Store candidates with their similarity scores and which seed song found them
            for item_idx in top_k_indices:
                similarity = float(similarities_array[item_idx])
                if similarity > 0:  # Only positive similarities
                    if item_idx not in all_candidates:
                        all_candidates[item_idx] = []
                    # Store: (similarity, seed_idx) - seed_idx tells us which seed song found this
                    all_candidates[item_idx].append((similarity, seed_idx))
        
        # Aggregate scores: weighted sum according to Item CF formula
        # Formula: Predicted user interest = Σ like(user, item_j) × sim(item_j, item_new)
        # For last-n interacted items, like(user, item_j) = 1, so:
        # score = Σ sim(item_j, item_new) for all seed songs j that recommend this item
        song_scores = {}
        for item_idx, similarities_list in all_candidates.items():
            # Sum of similarities (not average) - this is the correct formula
            weighted_sum = sum([s[0] for s in similarities_list])
            song_scores[item_idx] = [weighted_sum]  # Keep as list for compatibility
        
        # Average scores for items recommended by multiple seed songs
        results = []
        cold_start_results = []
        exclude_set = set(exclude_song_ids or [])
        exclude_set.update(song_ids)
        exclude_indices = {self.song_id_to_idx[sid] for sid in exclude_set if sid in self.song_id_to_idx}
        
        for item_idx, scores in song_scores.items():
            if item_idx in exclude_indices:
                continue
            
            # scores is a list with one element (weighted_sum)
            # This is the final score according to Item CF formula
            final_score = scores[0] if scores else 0.0
            song_id = self.idx_to_song_id[item_idx]
            
            # Check if this is a cold-start item
            if self._is_cold_start_item(song_id):
                # Store for potential content-based fallback
                cold_start_results.append((song_id, final_score))
            else:
                results.append((song_id, final_score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        cold_start_results.sort(key=lambda x: x[1], reverse=True)
        
        # If we don't have enough results and have content fallback, use it for cold-start items
        if len(results) < n and cold_start_results and content_fallback:
            # Get content-based recommendations for cold-start items
            cold_start_song_ids = [r[0] for r in cold_start_results[:min(5, len(cold_start_results))]]
            if cold_start_song_ids:
                content_recs = content_fallback.recommend(
                    song_ids=cold_start_song_ids,
                    n=n - len(results),
                    exclude_song_ids=list(exclude_set),
                    return_details=return_details,
                    **kwargs
                )
                # Merge with CF results (with lower weight for fallback)
                for rec in content_recs:
                    if rec['song_id'] not in exclude_set:
                        results.append((rec['song_id'], rec['score'] * 0.5))  # Lower weight for fallback
        
        # Take top N
        results = results[:n]
        
        # Prepare details if requested
        details = None
        if return_details:
            details = {}
            # Map item_idx to the specific seed song that found it (with highest similarity)
            # If multiple seed songs found it, we'll show the one with highest similarity
            item_to_primary_seed = {}
            for item_idx, similarities_list in all_candidates.items():
                # Find the seed song with highest similarity for this item
                best_similarity = -1
                best_seed_idx = None
                for similarity, seed_idx in similarities_list:
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_seed_idx = seed_idx
                if best_seed_idx is not None:
                    seed_song_id = self.idx_to_song_id.get(best_seed_idx)
                    if seed_song_id:
                        item_to_primary_seed[item_idx] = seed_song_id
            
            for song_id, score in results:
                song_idx = self.song_id_to_idx.get(song_id)
                # Get the primary seed song that found this recommendation
                primary_seed_song_id = item_to_primary_seed.get(song_idx)
                details[song_id] = {
                    'item_cf_similarity': float(score),
                    'seed_songs_count': len(seed_indices),
                    'primary_seed_song_id': primary_seed_song_id,  # The specific last-n item that found this
                    'method': 'item_item_cosine_similarity',
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
        """Get similar songs using item-item CF similarity"""
        if song_id not in self.song_id_to_idx:
            return []
        
        song_idx = self.song_id_to_idx[song_id]
        
        if song_idx >= self.item_item_similarity.shape[0]:
            return []
        
        # Get similarities
        similarities = self.item_item_similarity[song_idx].toarray().flatten()
        
        # Get top N similar items (excluding the song itself)
        top_indices = np.argsort(similarities)[::-1]
        top_indices = [idx for idx in top_indices if idx != song_idx][:n]
        
        song_ids = [self.idx_to_song_id[idx] for idx in top_indices]
        scores = similarities[top_indices]
        
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
        joblib.dump({
            'name': self.name,
            'songs_df': self.songs_df,
            'song_id_to_idx': self.song_id_to_idx,
            'idx_to_song_id': self.idx_to_song_id,
            'user_id_to_idx': self.user_id_to_idx,
            'idx_to_user_id': self.idx_to_user_id,
            'user_item_matrix': self.user_item_matrix,
            'item_item_similarity': self.item_item_similarity,
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load the recommender"""
        data = joblib.load(path)
        
        recommender = cls()
        recommender.songs_df = data['songs_df']
        recommender.song_id_to_idx = data['song_id_to_idx']
        recommender.idx_to_song_id = data['idx_to_song_id']
        recommender.user_id_to_idx = data['user_id_to_idx']
        recommender.idx_to_user_id = data['idx_to_user_id']
        recommender.user_item_matrix = data['user_item_matrix']
        recommender.item_item_similarity = data['item_item_similarity']
        recommender.is_fitted = True
        
        return recommender

