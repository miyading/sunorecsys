"""User-based collaborative filtering using user-user similarity matrix"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from .base import BaseRecommender
from ..data.simulate_interactions import get_user_item_matrix


class UserBasedRecommender(BaseRecommender):
    """
    User-based collaborative filtering using precomputed user-user similarity matrix.
    
    Discover Weekly style:
    1. Find top-k similar users
    2. For each similar user, find top-k items they liked
    3. Aggregate and deduplicate
    4. Return top-N recommendations
    
    Note: ALS matrix factorization is moved to future work for scalability.
    """
    
    def __init__(
        self,
        top_k_similar_users: int = 20,
        top_k_per_similar_user: int = 5,
        last_n_per_user: int = 50,  # Last n items per similar user
    ):
        super().__init__("UserBased")
        self.top_k_similar_users = top_k_similar_users
        self.top_k_per_similar_user = top_k_per_similar_user
        self.last_n_per_user = last_n_per_user  # Last n interacted items per similar user
        
        self.songs_df = None
        self.user_item_matrix = None
        self.user_user_similarity = None  # Precomputed user-user similarity matrix
        self.song_id_to_idx = {}
        self.idx_to_song_id = {}
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.user_history = None  # Optional: user history for temporal ordering
    
    def fit(
        self,
        songs_df: pd.DataFrame,
        playlists: Optional[List[Dict[str, Any]]] = None,
        use_simulated_interactions: bool = True,
        **kwargs
    ):
        """
        Fit the user-based recommender
        
        Args:
            songs_df: DataFrame with song data
            playlists: Optional list of playlists (dict with 'id' and 'song_ids')
                      If not provided, will use simulated interactions
            use_simulated_interactions: If True and no playlists, use simulated interactions
                                       If False, fall back to user_id-based approach
        """
        self.songs_df = songs_df.copy()
        
        # Store user_history if provided
        self.user_history = kwargs.get('user_history', None)
        
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
        
        # Compute user-user similarity matrix
        print(f"  → Matrix built: {self.user_item_matrix.shape[0]} users × {self.user_item_matrix.shape[1]} items")
        print(f"  → Computing user-user similarity matrix...")
        print(f"     (This may take a while for large user sets...)")
        
        # Compute cosine similarity between all user pairs
        # Each user is represented by a vector of item interactions
        # Users with similar item interaction patterns are similar
        self.user_user_similarity = cosine_similarity(
            self.user_item_matrix, 
            dense_output=False
        )
        
        self.is_fitted = True
        print(f"  ✅ User-user similarity matrix computed: {self.user_user_similarity.shape}")
        print(f"  ✅ Fitted user-based CF on {len(self.user_id_to_idx)} users")
    
    def _build_matrix_from_playlists(self, playlists: List[Dict[str, Any]]):
        """Build user-item matrix from playlist data"""
        # Treat each playlist as a "user"
        rows = []
        cols = []
        data = []
        
        # Create mappings
        all_song_ids = set(self.songs_df['song_id'].unique())
        all_user_ids = set()
        
        for playlist in playlists:
            playlist_id = playlist.get('id', f"playlist_{len(all_user_ids)}")
            song_ids = playlist.get('song_ids', [])
            
            # Filter to songs we know about
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
                data.append(1.0)  # Binary interaction
        
        # Create mappings for users
        for user_id in all_user_ids:
            if user_id not in self.user_id_to_idx:
                idx = len(self.user_id_to_idx)
                self.user_id_to_idx[user_id] = idx
                self.idx_to_user_id[idx] = user_id
        
        # Build sparse matrix
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_id_to_idx), len(self.song_id_to_idx))
        )
    
    def _build_matrix_from_users(self):
        """Build user-item matrix from user_id in songs (simulate playlists)"""
        # Group songs by user_id
        user_songs = self.songs_df.groupby('user_id')['song_id'].apply(list).to_dict()
        
        rows = []
        cols = []
        data = []
        
        # Create song mappings
        for song_id in self.songs_df['song_id'].unique():
            if song_id not in self.song_id_to_idx:
                idx = len(self.song_id_to_idx)
                self.song_id_to_idx[song_id] = idx
                self.idx_to_song_id[idx] = song_id
        
        # Create user mappings and interactions
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
        
        # Build sparse matrix
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_id_to_idx), len(self.song_id_to_idx))
        )
    
    def _get_user_last_n_items(self, similar_user_id: str, n: int) -> Tuple[List[int], List[str]]:
        """
        Get last n interacted items for a similar user.
        
        Priority:
        1. If user_history available, use it (temporal order)
        2. If songs_df has created_at, sort by time
        3. Otherwise, return all items (no temporal ordering)
        
        Returns:
            Tuple of (item_indices, song_ids) - both lists in the same order
            This allows us to map item_position back to the actual song_id
        """
        # Try user_history first (best: has temporal order)
        if self.user_history and similar_user_id in self.user_history:
            history = self.user_history[similar_user_id]
            # Handle both dict format (with 'song_id' key) and list format (just song_ids)
            if isinstance(history, list) and len(history) > 0:
                if isinstance(history[0], dict):
                    # Format: [{'song_id': '...', 'timestamp': '...'}, ...]
                    last_n_song_ids = [item['song_id'] for item in history[-n:]]
                else:
                    # Format: ['song_id1', 'song_id2', ...] (from history_manager.get_user_interactions)
                    last_n_song_ids = history[-n:]
                # Convert to indices, preserving order
                item_indices = []
                valid_song_ids = []
                for sid in last_n_song_ids:
                    if sid in self.song_id_to_idx:
                        item_indices.append(self.song_id_to_idx[sid])
                        valid_song_ids.append(sid)
                if item_indices:
                    return item_indices, valid_song_ids
        
        # Fallback: use songs_df with created_at if available
        if self.songs_df is not None and 'created_at' in self.songs_df.columns:
            user_songs = self.songs_df[self.songs_df['user_id'] == similar_user_id].copy()
            if len(user_songs) > 0:
                # Sort by created_at (most recent first)
                user_songs = user_songs.sort_values('created_at', ascending=False)
                last_n_song_ids = user_songs.head(n)['song_id'].tolist()
                item_indices = []
                valid_song_ids = []
                for sid in last_n_song_ids:
                    if sid in self.song_id_to_idx:
                        item_indices.append(self.song_id_to_idx[sid])
                        valid_song_ids.append(sid)
                if item_indices:
                    return item_indices, valid_song_ids
        
        # Last resort: get all items from matrix (no temporal ordering)
        if similar_user_id in self.user_id_to_idx:
            user_idx = self.user_id_to_idx[similar_user_id]
            user_items = self.user_item_matrix[user_idx].toarray().flatten()
            liked_item_indices = np.where(user_items > 0)[0].tolist()
            # Get song_ids for these indices
            liked_song_ids = [self.idx_to_song_id[idx] for idx in liked_item_indices if idx in self.idx_to_song_id]
            # Return last n (or all if less than n)
            result_indices = liked_item_indices[-n:] if len(liked_item_indices) > n else liked_item_indices
            result_song_ids = liked_song_ids[-n:] if len(liked_song_ids) > n else liked_song_ids
            return result_indices, result_song_ids
        
        return [], []
    
    def recommend(
        self,
        user_id: Optional[str] = None,
        song_ids: Optional[List[str]] = None,
        n: int = 10,
        exclude_song_ids: Optional[List[str]] = None,
        return_details: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Recommend songs using user-based CF (Discover Weekly style)
        
        Process:
        1. Find top-k similar users
        2. For each similar user, find top-k items they liked
        3. Aggregate scores and deduplicate
        4. Return top-N recommendations
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        # If user_id provided, use it directly
        if user_id and user_id in self.user_id_to_idx:
            user_idx = self.user_id_to_idx[user_id]
            
            # Check bounds
            if user_idx >= self.user_user_similarity.shape[0]:
                return self._get_popular_songs(n)
            
            # Get user's own interactions (to exclude)
            user_items = self.user_item_matrix[user_idx].toarray().flatten()
            user_liked_items = set(np.where(user_items > 0)[0])
            
            # Find top-k similar users
            similar_users = self.user_user_similarity[user_idx].toarray().flatten()
            similar_users[user_idx] = -np.inf  # Exclude self
            top_user_indices = np.argsort(similar_users)[::-1][:self.top_k_similar_users]
            
            # Get similarity scores for top-k similar users
            similar_user_indices = []
            sims = []
            all_candidates = set()  # Collect all candidate item indices (n*k items)
            user_item_mapping = {}  # Store mapping for display: similar_user_idx -> liked_song_ids
            item_to_source_user = {}  # Track which similar user's last-n each item came from (for display)
            
            for similar_user_idx in top_user_indices:
                user_similarity = float(similar_users[similar_user_idx])
                if user_similarity <= 0:
                    continue
                
                similar_user_indices.append(similar_user_idx)
                sims.append(user_similarity)
                
                # Get similar user's ID
                similar_user_id = self.idx_to_user_id[similar_user_idx]
                
                # Get last n interacted items for this similar user
                liked_item_indices, liked_song_ids = self._get_user_last_n_items(
                    similar_user_id, 
                    self.last_n_per_user
                )
                # Store mapping for later lookup
                user_item_mapping[similar_user_idx] = liked_song_ids
                
                # Exclude items the target user already liked
                filtered_item_indices = [idx for idx in liked_item_indices if idx not in user_liked_items]
                
                if len(filtered_item_indices) == 0:
                    continue
                
                # Take top-k items from last n (most recent first)
                top_items = filtered_item_indices[:self.top_k_per_similar_user]
                
                # Record which similar user each item came from (for display)
                for item_idx in top_items:
                    all_candidates.add(item_idx)
                    # If this item hasn't been seen yet, or if this user has higher similarity, record it
                    if item_idx not in item_to_source_user:
                        # Find the position of this item in the last-n list
                        item_position = None
                        source_song_id = None
                        song_id = self.idx_to_song_id.get(item_idx)
                        if song_id and song_id in liked_song_ids:
                            item_position = liked_song_ids.index(song_id)
                            source_song_id = song_id
                        
                        item_to_source_user[item_idx] = {
                            'similar_user_idx': similar_user_idx,
                            'similar_user_id': similar_user_id,
                            'item_position': item_position,
                            'source_song_id': source_song_id,
                        }
            
            # Convert to numpy arrays for efficient computation
            similar_user_indices = np.array(similar_user_indices)
            sims = np.array(sims)
            candidates = np.array(list(all_candidates))
            
            # Calculate scores using the correct formula: score = Σ sim(u, userj) × like(userj, item)
            item_scores = {}
            item_to_source = {}  # Track which similar user's which last-n item recommended this (for display)
            
            for item_idx in candidates:
                # Get binary like values for this item from all similar users
                # user_item_matrix: (num_users, num_items) CSR
                # Get the column for this item_idx, rows for similar_user_indices
                likes = self.user_item_matrix[similar_user_indices, item_idx].toarray().flatten()  # 0/1
                
                # Calculate score: Σ sim(u, userj) × like(userj, item)
                score = float((sims * likes).sum())
                item_scores[item_idx] = score
                
                # For display: use the source user from recall stage (the user whose last-n actually contributed this item)
                if item_idx in item_to_source_user:
                    source_info = item_to_source_user[item_idx]
                    item_to_source[item_idx] = {
                        'similar_user_id': source_info['similar_user_id'],
                        'item_position': source_info['item_position'],
                        'source_song_id': source_info['source_song_id'],
                    }
                elif likes.sum() > 0:  # Fallback: if not found in recall tracking, use highest similarity
                    # This shouldn't happen if recall logic is correct, but keep as fallback
                    liked_mask = likes > 0
                    if liked_mask.any():
                        liked_sims = sims[liked_mask]
                        liked_user_indices = similar_user_indices[liked_mask]
                        best_idx = np.argmax(liked_sims)
                        best_similar_user_idx = liked_user_indices[best_idx]
                        
                        # Find which last-n item this was (for display)
                        best_item_position = None
                        source_song_id = None
                        if best_similar_user_idx in user_item_mapping:
                            liked_song_ids = user_item_mapping[best_similar_user_idx]
                            # Get the item's song_id to find its position
                            song_id = self.idx_to_song_id.get(item_idx)
                            if song_id and song_id in liked_song_ids:
                                best_item_position = liked_song_ids.index(song_id)
                                source_song_id = song_id
                        
                        similar_user_id = self.idx_to_user_id.get(best_similar_user_idx)
                        item_to_source[item_idx] = {
                            'similar_user_id': similar_user_id,
                            'item_position': best_item_position,
                            'source_song_id': source_song_id,
                        }
            
            # Convert to recommendations
            recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            # Store item_to_source for details
            self._temp_item_to_source = item_to_source
            
        elif song_ids:
            # Create virtual user from seed songs
            # Build virtual user vector
            print("  → building virtual user...")
            num_items = len(self.song_id_to_idx)
            virtual_user = np.zeros(num_items)
            valid_seed_count = 0
            
            for song_id in song_ids:
                if song_id in self.song_id_to_idx:
                    song_idx = self.song_id_to_idx[song_id]
                    if song_idx < num_items:
                        virtual_user[song_idx] = 1.0
                        valid_seed_count += 1
            
            if valid_seed_count == 0:
                return self._get_popular_songs(n)
            
            # Find similar users to this virtual user
            # Compute similarity: cosine similarity between virtual_user and all users
            virtual_user_vector = csr_matrix(virtual_user.reshape(1, -1))
            user_similarities = cosine_similarity(
                virtual_user_vector,
                self.user_item_matrix,
                dense_output=True
            ).flatten()
            
            # Find top-k similar users
            top_user_indices = np.argsort(user_similarities)[::-1][:self.top_k_similar_users]
            
            # Get similarity scores for top-k similar users
            similar_user_indices = []
            sims = []
            all_candidates = set()  # Collect all candidate item indices (n*k items)
            user_item_mapping = {}  # Store mapping for display: similar_user_idx -> liked_song_ids
            item_to_source_user = {}  # Track which similar user's last-n each item came from (for display)
            seed_indices = set(np.where(virtual_user > 0)[0])
            
            for similar_user_idx in top_user_indices:
                user_similarity = float(user_similarities[similar_user_idx])
                if user_similarity <= 0:
                    continue
                
                similar_user_indices.append(similar_user_idx)
                sims.append(user_similarity)
                
                # Get similar user's ID
                similar_user_id = self.idx_to_user_id[similar_user_idx]
                
                # Get last n interacted items for this similar user
                liked_item_indices, liked_song_ids = self._get_user_last_n_items(
                    similar_user_id,
                    self.last_n_per_user
                )
                # Store mapping for later lookup
                user_item_mapping[similar_user_idx] = liked_song_ids
                
                # Exclude seed songs
                filtered_item_indices = [idx for idx in liked_item_indices if idx not in seed_indices]
                
                if len(filtered_item_indices) == 0:
                    continue
                
                # Take top-k items from last n (most recent first)
                top_items = filtered_item_indices[:self.top_k_per_similar_user]
                
                # Record which similar user each item came from (for display)
                for item_idx in top_items:
                    all_candidates.add(item_idx)
                    # If this item hasn't been seen yet, record it
                    if item_idx not in item_to_source_user:
                        # Find the position of this item in the last-n list
                        item_position = None
                        source_song_id = None
                        song_id = self.idx_to_song_id.get(item_idx)
                        if song_id and song_id in liked_song_ids:
                            item_position = liked_song_ids.index(song_id)
                            source_song_id = song_id
                        
                        item_to_source_user[item_idx] = {
                            'similar_user_idx': similar_user_idx,
                            'similar_user_id': similar_user_id,
                            'item_position': item_position,
                            'source_song_id': source_song_id,
                        }
            
            # Convert to numpy arrays for efficient computation
            similar_user_indices = np.array(similar_user_indices)
            sims = np.array(sims)
            candidates = np.array(list(all_candidates))
            
            # Calculate scores using the correct formula: score = Σ sim(u, userj) × like(userj, item)
            item_scores = {}
            item_to_source = {}  # Track which similar user's which last-n item recommended this (for display)
            
            for item_idx in candidates:
                # Get binary like values for this item from all similar users
                # user_item_matrix: (num_users, num_items) CSR
                # Get the column for this item_idx, rows for similar_user_indices
                likes = self.user_item_matrix[similar_user_indices, item_idx].toarray().flatten()  # 0/1
                
                # Calculate score: Σ sim(u, userj) × like(userj, item)
                score = float((sims * likes).sum())
                item_scores[item_idx] = score
                
                # For display: use the source user from recall stage (the user whose last-n actually contributed this item)
                if item_idx in item_to_source_user:
                    source_info = item_to_source_user[item_idx]
                    item_to_source[item_idx] = {
                        'similar_user_id': source_info['similar_user_id'],
                        'item_position': source_info['item_position'],
                        'source_song_id': source_info['source_song_id'],
                    }
                elif likes.sum() > 0:  # Fallback: if not found in recall tracking, use highest similarity
                    # This shouldn't happen if recall logic is correct, but keep as fallback
                    liked_mask = likes > 0
                    if liked_mask.any():
                        liked_sims = sims[liked_mask]
                        liked_user_indices = similar_user_indices[liked_mask]
                        best_idx = np.argmax(liked_sims)
                        best_similar_user_idx = liked_user_indices[best_idx]
                        
                        # Find which last-n item this was (for display)
                        best_item_position = None
                        source_song_id = None
                        if best_similar_user_idx in user_item_mapping:
                            liked_song_ids = user_item_mapping[best_similar_user_idx]
                            # Get the item's song_id to find its position
                            song_id = self.idx_to_song_id.get(item_idx)
                            if song_id and song_id in liked_song_ids:
                                best_item_position = liked_song_ids.index(song_id)
                                source_song_id = song_id
                        
                        similar_user_id = self.idx_to_user_id.get(best_similar_user_idx)
                        item_to_source[item_idx] = {
                            'similar_user_id': similar_user_id,
                            'item_position': best_item_position,
                            'source_song_id': source_song_id,
                        }
            
            recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            # Store item_to_source for details
            self._temp_item_to_source_virtual = item_to_source
        else:
            return self._get_popular_songs(n)
        
        # Filter excluded songs
        exclude_set = set(exclude_song_ids or [])
        if song_ids:
            exclude_set.update(song_ids)
        
        # Convert to results
        results = []
        for item_idx, score in recommendations:
            if item_idx < len(self.idx_to_song_id) and item_idx in self.idx_to_song_id:
                song_id = self.idx_to_song_id[item_idx]
                if song_id not in exclude_set:
                    results.append((song_id, score))
                    if len(results) >= n:
                        break
        
        song_ids_list = [r[0] for r in results]
        scores = np.array([r[1] for r in results])
        
        # Prepare details if requested
        details = None
        if return_details:
            details = {}
            # Check which temp storage was used
            temp_storage = None
            if hasattr(self, '_temp_item_to_source'):
                temp_storage = self._temp_item_to_source
            elif hasattr(self, '_temp_item_to_source_virtual'):
                temp_storage = self._temp_item_to_source_virtual
            
            for song_id, score in zip(song_ids_list, scores):
                song_idx = self.song_id_to_idx.get(song_id)
                source_info = None
                if temp_storage and song_idx in temp_storage:
                    source_info = temp_storage[song_idx]
                
                # Get the actual song_id from the similar user's last n items
                # source_song_id is now stored directly in item_to_source
                source_song_id = source_info.get('source_song_id') if source_info else None
                
                details[song_id] = {
                    'user_cf_score': float(score),
                    'recommendation_method': 'user_based' if user_id else 'virtual_user',
                    'top_k_similar_users': self.top_k_similar_users,
                    'top_k_per_similar_user': self.top_k_per_similar_user,
                    'source_similar_user_id': source_info.get('similar_user_id') if source_info else None,
                    'source_item_position': source_info.get('item_position') if source_info else None,
                    'source_song_id': source_song_id,  # The actual last-n item from similar user
                }
            # Clean up temp storage
            if hasattr(self, '_temp_item_to_source'):
                delattr(self, '_temp_item_to_source')
            if hasattr(self, '_temp_item_to_source_virtual'):
                delattr(self, '_temp_item_to_source_virtual')
        
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
        """
        Get similar songs using user-based CF.
        
        Finds users who liked this song, then recommends other songs they liked.
        """
        if song_id not in self.song_id_to_idx:
            return []
        
        song_idx = self.song_id_to_idx[song_id]
        num_items = len(self.song_id_to_idx)
        
        if song_idx >= num_items:
            return []
        
        # Find users who liked this song
        users_who_liked = self.user_item_matrix[:, song_idx].toarray().flatten()
        user_indices = np.where(users_who_liked > 0)[0]
        
        if len(user_indices) == 0:
            return []
        
        # Aggregate items liked by these users
        candidate_items = {}
        for user_idx in user_indices:
            user_items = self.user_item_matrix[user_idx].toarray().flatten()
            liked_items = np.where(user_items > 0)[0]
            
            for item_idx in liked_items:
                if item_idx != song_idx:  # Exclude the seed song
                    if item_idx not in candidate_items:
                        candidate_items[item_idx] = 0
                    candidate_items[item_idx] += 1  # Count how many users liked it
        
        # Sort by frequency (more users = more similar)
        sorted_items = sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)[:n]
        
        if not sorted_items:
            return []
        
        song_ids_list = [self.idx_to_song_id[idx] for idx, _ in sorted_items]
        scores = np.array([count for _, count in sorted_items], dtype=float)
        scores = scores / scores.max()  # Normalize to [0, 1]
        
        return self._format_recommendations(song_ids_list, scores, self.songs_df)
    
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
            'top_k_similar_users': self.top_k_similar_users,
            'top_k_per_similar_user': self.top_k_per_similar_user,
            'songs_df': self.songs_df,
            'song_id_to_idx': self.song_id_to_idx,
            'idx_to_song_id': self.idx_to_song_id,
            'user_id_to_idx': self.user_id_to_idx,
            'idx_to_user_id': self.idx_to_user_id,
            'user_item_matrix': self.user_item_matrix,
            'user_user_similarity': self.user_user_similarity,
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load the recommender"""
        data = joblib.load(path)
        
        # Handle backward compatibility
        if 'top_k_similar_users' in data:
            recommender = cls(
                top_k_similar_users=data['top_k_similar_users'],
                top_k_per_similar_user=data['top_k_per_similar_user'],
            )
        else:
            # Old format - use defaults
            recommender = cls()
        
        recommender.songs_df = data['songs_df']
        recommender.song_id_to_idx = data['song_id_to_idx']
        recommender.idx_to_song_id = data['idx_to_song_id']
        recommender.user_id_to_idx = data['user_id_to_idx']
        recommender.idx_to_user_id = data['idx_to_user_id']
        recommender.user_item_matrix = data['user_item_matrix']
        recommender.user_user_similarity = data.get('user_user_similarity')
        recommender.is_fitted = True
        
        return recommender

