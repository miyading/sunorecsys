"""Simulate realistic user-item interactions for collaborative filtering

This module generates synthetic user-item interaction data that mimics real-world patterns:
- Popular songs get more interactions (power-law distribution)
- Users have varying activity levels
- Genre preferences create user clusters
- Temporal effects (recent songs more popular)

This is designed to be easily swappable with real interaction data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy.sparse import csr_matrix
from collections import Counter
import random
import json
import hashlib
import joblib
from pathlib import Path


class InteractionSimulator:
    """Simulates realistic user-item interactions"""
    
    def __init__(
        self,
        interaction_rate: float = 0.15,
        power_law_exponent: float = 2.0,
        genre_clustering: float = 0.6,
        temporal_decay: float = 0.1,
        min_user_interactions: int = 3,
        max_user_interactions: int = 50,
        item_cold_start_rate: float = 0.05,
        single_user_item_rate: float = 0.15,
        random_seed: Optional[int] = 42,
    ):
        """
        Initialize the interaction simulator
        
        Args:
            interaction_rate: Overall density of interactions (0.0 to 1.0)
            power_law_exponent: Exponent for popularity distribution (higher = more concentrated)
            genre_clustering: How much users cluster by genre (0.0 to 1.0)
            temporal_decay: How much recent songs are favored (0.0 to 1.0)
            min_user_interactions: Minimum interactions per user (ensures no user cold-start)
            max_user_interactions: Maximum interactions per user
            item_cold_start_rate: Fraction of items with zero interactions (cold start items)
            single_user_item_rate: Fraction of items with only one user interaction
            random_seed: Random seed for reproducibility
        """
        self.interaction_rate = interaction_rate
        self.power_law_exponent = power_law_exponent
        self.genre_clustering = genre_clustering
        self.temporal_decay = temporal_decay
        self.min_user_interactions = min_user_interactions
        self.max_user_interactions = max_user_interactions
        self.item_cold_start_rate = item_cold_start_rate
        self.single_user_item_rate = single_user_item_rate
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
    
    def simulate_interactions(
        self,
        songs_df: pd.DataFrame,
        user_ids: List[str],
        user_id_field: str = 'user_id',
        return_events: bool = False,
        num_negatives_per_user: int = 50,
        impressions_size: int = 50,  # Size of impression set M for each user
        clap_embeddings: Optional[Dict[str, np.ndarray]] = None,  # CLAP embeddings for hard negative mining
        hard_negative_top_p: float = 0.2,  # Top p% of impressions by similarity to use as hard negatives
        last_n: int = 8,  # Number of recent interactions for user feature computation
    ) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
        """
        Simulate user-item interactions
        
        Args:
            songs_df: DataFrame with song data (must have 'song_id', 'genre', 'popularity_score', etc.)
            user_ids: List of user IDs to simulate interactions for
            user_id_field: Field name for user ID in songs_df (used for filtering)
        
        Returns:
            If return_events=False (默认，兼容旧接口)，返回 5 元组：
            - user_item_matrix: Sparse matrix of interactions (users x items)
            - user_id_to_idx: Mapping from user_id to matrix index
            - idx_to_user_id: Mapping from matrix index to user_id
            - song_id_to_idx: Mapping from song_id to matrix index
            - idx_to_song_id: Mapping from matrix index to song_id

            如果 return_events=True，则额外返回一个 DataFrame：
            - events_df: pd.DataFrame[user_id, song_id, label]，label=1 表示有互动，0 表示“看到了但没互动”的负样本
        """
        print(f"🎲 Simulating user-item interactions...")
        print(f"   Users: {len(user_ids)}")
        print(f"   Songs: {len(songs_df)}")
        
        # Create mappings
        song_id_to_idx = {}
        idx_to_song_id = {}
        for idx, song_id in enumerate(songs_df['song_id'].unique()):
            song_id_to_idx[song_id] = idx
            idx_to_song_id[idx] = song_id
        
        user_id_to_idx = {}
        idx_to_user_id = {}
        for idx, user_id in enumerate(user_ids):
            user_id_to_idx[user_id] = idx
            idx_to_user_id[idx] = user_id
        
        num_users = len(user_ids)
        num_items = len(song_id_to_idx)
        
        # Compute song popularity scores (for power-law distribution)
        song_popularity = self._compute_song_popularity(songs_df, song_id_to_idx)
        
        # Compute user activity levels
        user_activity = self._compute_user_activity(num_users)
        
        # Compute genre preferences for each user
        user_genre_prefs = self._compute_genre_preferences(songs_df, user_ids, user_id_field)
        
        # Create song_id to row mapping for efficient lookup (CRITICAL for performance)
        # This avoids scanning the entire DataFrame for each song lookup (O(n) -> O(1))
        # Use iloc for faster access after setting index
        songs_df_indexed = songs_df.set_index('song_id', drop=False)
        song_id_to_row = {}
        for song_id in song_id_to_idx.keys():
            try:
                song_id_to_row[song_id] = songs_df_indexed.loc[song_id]
            except KeyError:
                # Song ID not found in DataFrame (shouldn't happen, but handle gracefully)
                continue
        
        # Identify items for cold start (no interactions) and single-user items
        num_items = len(song_id_to_idx)
        num_cold_start_items = int(num_items * self.item_cold_start_rate)
        num_single_user_items = int(num_items * self.single_user_item_rate)
        
        # Randomly select items for cold start and single-user
        all_item_indices = np.arange(num_items)
        np.random.shuffle(all_item_indices)
        
        cold_start_items = set(all_item_indices[:num_cold_start_items])
        single_user_items = set(all_item_indices[num_cold_start_items:num_cold_start_items + num_single_user_items])
        multi_user_items = set(all_item_indices[num_cold_start_items + num_single_user_items:])
        
        print(f"   Item distribution:")
        print(f"     Cold start (0 interactions): {len(cold_start_items)} ({len(cold_start_items)/num_items*100:.1f}%)")
        print(f"     Single-user items: {len(single_user_items)} ({len(single_user_items)/num_items*100:.1f}%)")
        print(f"     Multi-user items: {len(multi_user_items)} ({len(multi_user_items)/num_items*100:.1f}%)")
        
        # Generate interactions
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        
        total_interactions = 0
        item_interaction_counts = np.zeros(num_items)
        
        # First pass: ensure all users have minimum interactions (no user cold-start)
        # Try to use tqdm if available for progress
        try:
            from tqdm import tqdm
            user_iterator = tqdm(enumerate(user_ids), total=len(user_ids), desc="  Generating interactions")
        except ImportError:
            user_iterator = enumerate(user_ids)
            if len(user_ids) > 100:
                print(f"  → Processing {len(user_ids)} users...")
                print(f"     (Consider installing tqdm for progress bar: pip install tqdm)")
        
        for user_idx, user_id in user_iterator:
            # Determine how many interactions this user will have
            activity_level = user_activity[user_idx]
            num_interactions = int(
                self.min_user_interactions + 
                activity_level * (self.max_user_interactions - self.min_user_interactions)
            )
            
            # Get user's genre preferences
            genre_prefs = user_genre_prefs.get(user_id, {})
            
            # Sample songs for this user (exclude cold-start items for now)
            available_items = multi_user_items | single_user_items
            song_indices = self._sample_songs_for_user(
                song_id_to_row,  # Use pre-indexed mapping instead of DataFrame
                song_id_to_idx,
                song_popularity,
                genre_prefs,
                num_interactions,
                available_items=available_items,
            )
            
            # Add interactions
            for song_idx in song_indices:
                rows.append(user_idx)
                cols.append(song_idx)
                data.append(1.0)
                item_interaction_counts[song_idx] += 1
                total_interactions += 1
        
        # Second pass: assign single-user items (each to exactly one random user)
        for song_idx in single_user_items:
            if item_interaction_counts[song_idx] == 0:  # Not yet assigned
                # Randomly assign to one user
                user_idx = np.random.randint(0, num_users)
                rows.append(user_idx)
                cols.append(song_idx)
                data.append(1.0)
                item_interaction_counts[song_idx] += 1
                total_interactions += 1
        
        # Note: Cold-start items remain with 0 interactions (will use content-based fallback)
        
        # Build sparse matrix
        user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(num_users, num_items)
        )
        
        sparsity = 1.0 - (total_interactions / (num_users * num_items))
        print(f"✅ Generated {total_interactions:,} interactions")
        print(f"   Sparsity: {sparsity:.2%}")
        print(f"   Avg interactions per user: {total_interactions/num_users:.1f}")
        print(f"   Avg interactions per item: {total_interactions/num_items:.1f}")
        
        # 如果不需要事件表，直接保持旧接口行为
        if not return_events:
            return user_item_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id

        # ------------------------------------------------------------------
        # 构造 two-tower 训练用的正负样本事件表
        # ------------------------------------------------------------------
        # 正样本：所有 user-item 交互
        events = []
        from collections import defaultdict

        user_pos_items: Dict[int, set] = defaultdict(set)
        for u_idx, i_idx in zip(rows, cols):
            user_pos_items[u_idx].add(i_idx)
            user_id = idx_to_user_id[u_idx]
            song_id = idx_to_song_id[i_idx]
            events.append(
                {
                    "user_id": user_id,
                    "song_id": song_id,
                    "label": 1,
                }
            )

        # 负样本：基于曝光集合（impressions）采样，支持 hard negative mining
        # 对每个用户，先从 song_probs 采样 M=impressions_size 个"曝光"的歌曲
        # 如果提供了 CLAP embeddings，使用 hard negative mining：
        #   1. 计算 user_feature（last-n 正样本的 CLAP embedding 平均）
        #   2. 计算曝光集合中每首歌与 user_feature 的余弦相似度
        #   3. 从 top p% 相似度的曝光但未点中采样 negatives
        # 否则，从曝光集合中随机采样
        use_hard_negatives = clap_embeddings is not None and len(clap_embeddings) > 0
        if use_hard_negatives:
            print(f"   Generating impressions (M={impressions_size}) with HARD NEGATIVE mining (top {hard_negative_top_p*100:.0f}%)...")
        else:
            print(f"   Generating impressions (M={impressions_size}) and negative samples (random sampling)...")
        
        # Re-compute song_id_to_row for efficient lookup (needed for impression generation)
        songs_df_indexed = songs_df.set_index('song_id', drop=False)
        song_id_to_row = {}
        for song_id in song_id_to_idx.keys():
            try:
                song_id_to_row[song_id] = songs_df_indexed.loc[song_id]
            except KeyError:
                continue
        
        all_item_indices = np.arange(num_items)
        
        for u_idx in range(num_users):
            user_id = idx_to_user_id[u_idx]
            pos_items = user_pos_items.get(u_idx, set())
            
            if len(pos_items) >= num_items:
                continue

            # Generate impression set M for this user based on song_probs
            # Use the same sampling logic as positive interactions but for impressions
            genre_prefs = user_genre_prefs.get(user_id, {})
            
            # Compute impression probabilities (similar to _sample_songs_for_user)
            impression_probs = np.zeros(num_items)
            for song_id, song_idx in song_id_to_idx.items():
                if song_idx in pos_items:
                    continue  # Skip positive items
                
                if song_id not in song_id_to_row:
                    continue
                
                song = song_id_to_row[song_id]
                
                # Base popularity
                prob = song_popularity[song_idx]
                
                # Genre preference boost
                song_genre = song.get('genre', 'unknown')
                if song_genre in genre_prefs:
                    prob *= (1.0 + genre_prefs[song_genre] * 2.0)
                
                # Temporal effect
                if 'days_since_creation' in song:
                    days = song.get('days_since_creation', 0)
                    if days is not None and not pd.isna(days):
                        recency_boost = np.exp(-self.temporal_decay * days / 365.0)
                        prob *= (1.0 + recency_boost * 0.3)
                
                impression_probs[song_idx] = prob
            
            # Normalize to probabilities
            total_prob = impression_probs.sum()
            if total_prob > 0:
                impression_probs = impression_probs / total_prob
            else:
                # Fallback: uniform over non-positive items
                if pos_items:
                    pos_array = np.fromiter(pos_items, dtype=int)
                    candidate_impressions = np.setdiff1d(all_item_indices, pos_array, assume_unique=True)
                else:
                    candidate_impressions = all_item_indices
                if len(candidate_impressions) > 0:
                    impression_probs[candidate_impressions] = 1.0 / len(candidate_impressions)
            
            # Sample M impressions (with replacement, but we'll deduplicate)
            available_for_impressions = np.where(impression_probs > 0)[0]
            if len(available_for_impressions) == 0:
                continue
            
            # Sample impressions
            M = min(impressions_size, len(available_for_impressions))
            if M == 0:
                continue
            
            # Sample with probabilities
            impression_probs_available = impression_probs[available_for_impressions]
            impression_probs_available = impression_probs_available / impression_probs_available.sum()
            
            # Sample M impressions (may have duplicates, we'll deduplicate)
            impression_indices = np.random.choice(
                available_for_impressions,
                size=M,
                replace=True,  # Allow replacement to get exactly M
                p=impression_probs_available
            )
            impression_set = set(impression_indices)
            
            # Remove positive items from impression set
            impression_set = impression_set - pos_items
            
            if len(impression_set) == 0:
                continue
            
            # Sample negatives from impression set
            k = min(num_negatives_per_user, len(impression_set))
            
            if use_hard_negatives:
                # Hard negative mining: select from top p% by cosine similarity
                neg_items = self._sample_hard_negatives(
                    user_id=user_id,
                    pos_items=pos_items,
                    impression_set=impression_set,
                    song_id_to_idx=song_id_to_idx,
                    idx_to_song_id=idx_to_song_id,
                    clap_embeddings=clap_embeddings,
                    k=k,
                    top_p=hard_negative_top_p,
                    last_n=last_n,
                )
            else:
                # Random sampling from impression set
                impression_list = list(impression_set)
                neg_items = np.random.choice(impression_list, size=k, replace=False)

            for i_idx in neg_items:
                song_id = idx_to_song_id[i_idx]
                events.append(
                    {
                        "user_id": user_id,
                        "song_id": song_id,
                        "label": 0,
                    }
                )

        events_df = pd.DataFrame(events)

        return (
            user_item_matrix,
            user_id_to_idx,
            idx_to_user_id,
            song_id_to_idx,
            idx_to_song_id,
            events_df,
        )
    
    def _compute_song_popularity(
        self,
        songs_df: pd.DataFrame,
        song_id_to_idx: Dict[str, int]
    ) -> np.ndarray:
        """Compute popularity scores for songs"""
        popularity_scores = np.ones(len(song_id_to_idx))
        
        # Use popularity_score if available
        if 'popularity_score' in songs_df.columns:
            for song_id, idx in song_id_to_idx.items():
                song_data = songs_df[songs_df['song_id'] == song_id]
                if len(song_data) > 0:
                    popularity_scores[idx] = song_data.iloc[0]['popularity_score']
        
        # Apply power-law transformation
        # Normalize to [0, 1] and apply power-law
        popularity_scores = popularity_scores / (popularity_scores.max() + 1e-8)
        popularity_scores = np.power(popularity_scores + 0.1, self.power_law_exponent)
        
        # Add some randomness
        popularity_scores = popularity_scores * np.random.lognormal(0, 0.3, len(popularity_scores))
        
        return popularity_scores / (popularity_scores.sum() + 1e-8)  # Normalize to probabilities
    
    def _compute_user_activity(self, num_users: int) -> np.ndarray:
        """Compute activity levels for users (some users are more active)"""
        # Beta distribution: most users are moderately active, few are very active
        activity = np.random.beta(2, 5, num_users)
        return activity
    
    def _compute_genre_preferences(
        self,
        songs_df: pd.DataFrame,
        user_ids: List[str],
        user_id_field: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compute genre preferences for each user based on their created songs"""
        user_genre_prefs = {}
        
        # Get genre distribution for each user's created songs
        for user_id in user_ids:
            user_songs = songs_df[songs_df[user_id_field] == user_id]
            
            if len(user_songs) > 0:
                # Count genres in user's created songs
                genres = user_songs['genre'].dropna().tolist()
                genre_counts = Counter(genres)
                total = sum(genre_counts.values())
                
                # Normalize to probabilities
                genre_prefs = {g: c / total for g, c in genre_counts.items()}
                
                # Add some exploration (users might like other genres too)
                all_genres = set(songs_df['genre'].dropna().unique())
                for genre in all_genres:
                    if genre not in genre_prefs:
                        genre_prefs[genre] = (1 - self.genre_clustering) / len(all_genres)
                    else:
                        genre_prefs[genre] = (
                            self.genre_clustering * genre_prefs[genre] +
                            (1 - self.genre_clustering) / len(all_genres)
                        )
            else:
                # No songs created by this user - uniform preferences
                all_genres = set(songs_df['genre'].dropna().unique())
                genre_prefs = {g: 1.0 / len(all_genres) for g in all_genres}
            
            user_genre_prefs[user_id] = genre_prefs
        
        return user_genre_prefs
    
    def _sample_songs_for_user(
        self,
        song_id_to_row: Dict[str, pd.Series],  # Changed: use pre-indexed mapping instead of DataFrame
        song_id_to_idx: Dict[str, int],
        song_popularity: np.ndarray,
        genre_prefs: Dict[str, float],
        num_interactions: int,
        available_items: Optional[set] = None,
    ) -> List[int]:
        """Sample songs for a user based on popularity and genre preferences
        
        Args:
            song_id_to_row: Pre-computed mapping from song_id to row data (for performance)
            song_id_to_idx: Mapping from song_id to matrix index
            song_popularity: Popularity scores for each song
            genre_prefs: Genre preferences for the user
            num_interactions: Number of interactions to sample
            available_items: Set of available item indices (None = all items)
        """
        # Compute combined probability for each song
        song_probs = np.zeros(len(song_id_to_idx))
        
        for song_id, song_idx in song_id_to_idx.items():
            # Skip if not in available items
            if available_items is not None and song_idx not in available_items:
                continue
            
            # Use pre-indexed mapping for O(1) lookup instead of O(n) DataFrame scan
            if song_id not in song_id_to_row:
                continue
                
            song = song_id_to_row[song_id]
            
            # Base popularity
            prob = song_popularity[song_idx]
            
            # Genre preference boost
            song_genre = song.get('genre', 'unknown')
            if song_genre in genre_prefs:
                prob *= (1.0 + genre_prefs[song_genre] * 2.0)  # Boost preferred genres
            
            # Temporal effect (recent songs slightly more popular)
            if 'days_since_creation' in song:
                days = song.get('days_since_creation', 0)
                if days is not None and not pd.isna(days):
                    recency_boost = np.exp(-self.temporal_decay * days / 365.0)
                    prob *= (1.0 + recency_boost * 0.3)
            
            song_probs[song_idx] = prob
        
        # Normalize to probabilities
        total_prob = song_probs.sum()
        if total_prob > 0:
            song_probs = song_probs / total_prob
        else:
            # Fallback: uniform distribution over available items
            if available_items:
                for idx in available_items:
                    song_probs[idx] = 1.0 / len(available_items)
        
        # Get available indices (non-zero probabilities)
        available_indices = np.where(song_probs > 0)[0]
        
        if len(available_indices) == 0:
            return []
        
        # Sample without replacement (user doesn't interact with same song twice)
        num_to_sample = min(num_interactions, len(available_indices))
        if num_to_sample == 0:
            return []
        
        # Normalize probabilities for available items only
        available_probs = song_probs[available_indices]
        available_probs = available_probs / available_probs.sum()
        
        sampled_indices = np.random.choice(
            available_indices,
            size=num_to_sample,
            replace=False,
            p=available_probs
        )
        
        return sampled_indices.tolist()
    
    def _sample_hard_negatives(
        self,
        user_id: str,
        pos_items: set,
        impression_set: set,
        song_id_to_idx: Dict[str, int],
        idx_to_song_id: Dict[int, str],
        clap_embeddings: Dict[str, np.ndarray],
        k: int,
        top_p: float = 0.2,
        last_n: int = 8,
    ) -> np.ndarray:
        """
        Sample hard negatives from impression set based on cosine similarity.
        
        Args:
            user_id: User ID
            pos_items: Set of positive item indices
            impression_set: Set of impression item indices
            song_id_to_idx: Mapping from song_id to index
            idx_to_song_id: Mapping from index to song_id
            clap_embeddings: Dict mapping song_id to CLAP embedding (512-dim)
            k: Number of negatives to sample
            top_p: Top p% of impressions by similarity to use as hard negatives
            last_n: Number of recent interactions for user feature computation
        
        Returns:
            Array of negative item indices
        """
        if len(impression_set) == 0:
            return np.array([], dtype=int)
        
        # 1. Compute user_feature: average of last-n positive items' CLAP embeddings
        pos_item_list = sorted(list(pos_items))  # Sort for consistent ordering
        n_recent = min(last_n, len(pos_item_list))
        recent_pos_items = pos_item_list[-n_recent:] if n_recent > 0 else pos_item_list
        
        user_feature = None
        valid_pos_count = 0
        
        for pos_idx in recent_pos_items:
            song_id = idx_to_song_id.get(pos_idx)
            if song_id and song_id in clap_embeddings:
                emb = clap_embeddings[song_id]
                if user_feature is None:
                    user_feature = emb.copy().astype(np.float32)
                else:
                    user_feature += emb.astype(np.float32)
                valid_pos_count += 1
        
        if user_feature is None or valid_pos_count == 0:
            # Fallback: random sampling if no valid CLAP embeddings
            impression_list = list(impression_set)
            return np.random.choice(impression_list, size=min(k, len(impression_list)), replace=False)
        
        # Normalize user_feature
        user_feature = user_feature / valid_pos_count
        user_feature_norm = np.linalg.norm(user_feature)
        if user_feature_norm > 0:
            user_feature = user_feature / user_feature_norm
        
        # 2. Compute cosine similarity for each impression item
        impression_list = list(impression_set)
        similarities = []
        valid_impressions = []
        
        for imp_idx in impression_list:
            song_id = idx_to_song_id.get(imp_idx)
            if song_id and song_id in clap_embeddings:
                item_emb = clap_embeddings[song_id].astype(np.float32)
                item_emb_norm = np.linalg.norm(item_emb)
                if item_emb_norm > 0:
                    item_emb = item_emb / item_emb_norm
                    # Cosine similarity
                    sim = np.dot(user_feature, item_emb)
                    similarities.append(sim)
                    valid_impressions.append(imp_idx)
        
        if len(valid_impressions) == 0:
            # Fallback: random sampling
            return np.random.choice(impression_list, size=min(k, len(impression_list)), replace=False)
        
        # 3. Select top p% by similarity
        similarities = np.array(similarities)
        valid_impressions = np.array(valid_impressions)
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_impressions = valid_impressions[sorted_indices]
        
        # Select top p%
        top_n = max(1, int(len(sorted_impressions) * top_p))
        top_impressions = sorted_impressions[:top_n]
        
        # 4. Randomly sample k from top impressions
        k_final = min(k, len(top_impressions))
        if k_final == 0:
            return np.array([], dtype=int)
        
        sampled_indices = np.random.choice(top_impressions, size=k_final, replace=False)
        return sampled_indices


def build_matrix_from_playlists(
    songs_df: pd.DataFrame,
    playlists: List[Dict[str, Any]],
) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    """
    Build user-item matrix from real playlist data.
    Each playlist's creator (user_handle or user_id) is treated as a user,
    and songs in the playlist are treated as real interactions.
    
    Args:
        songs_df: DataFrame with song data
        playlists: List of playlist dicts with 'id', 'song_ids', 'user_handle' or 'user_id'
    
    Returns:
        Tuple of (user_item_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id)
    """
    print("📊 Building user-item matrix from REAL playlist data...")
    
    # Create mappings
    song_id_to_idx = {}
    idx_to_song_id = {}
    all_song_ids = set(songs_df['song_id'].unique())
    
    for idx, song_id in enumerate(all_song_ids):
        song_id_to_idx[song_id] = idx
        idx_to_song_id[idx] = song_id
    
    user_id_to_idx = {}
    idx_to_user_id = {}
    
    # Build interactions from playlists
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    
    for playlist in playlists:
        # Get playlist creator (user)
        # Priority: user_handle > user_id > user_display_name > playlist_id (fallback)
        creator_id = (
            playlist.get('user_handle') or 
            playlist.get('user_id') or
            playlist.get('user_display_name') or 
            f"playlist_{playlist.get('id', len(user_id_to_idx))}"
        )
        
        # Get songs in playlist
        song_ids = playlist.get('song_ids', [])
        # Filter to known songs
        song_ids = [sid for sid in song_ids if sid in all_song_ids]
        
        if len(song_ids) == 0:
            continue
        
        # Create user mapping if needed
        if creator_id not in user_id_to_idx:
            user_idx = len(user_id_to_idx)
            user_id_to_idx[creator_id] = user_idx
            idx_to_user_id[user_idx] = creator_id
        
        user_idx = user_id_to_idx[creator_id]
        
        # Add interactions for songs in this playlist
        for song_id in song_ids:
            song_idx = song_id_to_idx[song_id]
            rows.append(user_idx)
            cols.append(song_idx)
            data.append(1.0)  # Binary interaction
    
    # Build sparse matrix
    num_users = len(user_id_to_idx)
    num_items = len(song_id_to_idx)
    user_item_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(num_users, num_items)
    )
    
    print(f"  ✅ Built matrix from {len(playlists)} playlists")
    print(f"     Users (playlist creators): {num_users}")
    print(f"     Songs: {num_items}")
    print(f"     Total interactions: {len(data)}")
    
    return user_item_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id


def simulate_playlist_interactions(
    songs_df: pd.DataFrame,
    user_ids: Optional[List[str]] = None,
    playlists: Optional[List[Dict[str, Any]]] = None,
    interaction_rate: float = 0.15,
    item_cold_start_rate: float = 0.05,
    single_user_item_rate: float = 0.15,
    random_seed: int = 42,
    return_events: bool = False,
    num_negatives_per_user: int = 50,
    impressions_size: int = 50,  # Size of impression set M for each user
    clap_embeddings: Optional[Dict[str, np.ndarray]] = None,  # CLAP embeddings for hard negative mining
    hard_negative_top_p: float = 0.2,  # Top p% of impressions by similarity to use as hard negatives
    last_n: int = 8,  # Number of recent interactions for user feature computation
) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    """
    Simulate interactions, optionally starting from real playlist data.
    
    If playlists are provided:
    1. First build real interactions from playlists (playlist creator = user, songs in playlist = interactions)
    2. Then simulate additional interactions based on the real data
    
    Args:
        songs_df: DataFrame with song data
        user_ids: Optional list of user IDs. If None and no playlists, uses unique user_ids from songs_df
        playlists: Optional list of playlist dicts. If provided, used as real interaction data
        interaction_rate: Overall density of additional simulated interactions
        item_cold_start_rate: Fraction of items with zero interactions (cold start items)
        single_user_item_rate: Fraction of items with only one user interaction
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (user_item_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id)
    """
    # If playlists provided, start with real data
    if playlists:
        print("📊 Starting with REAL playlist data, then simulating additional interactions...")
        # Build real matrix from playlists
        real_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id = build_matrix_from_playlists(
            songs_df, playlists
        )
        
        # Get user IDs from real data
        user_ids = list(user_id_to_idx.keys())
        
        # Create simulator
        simulator = InteractionSimulator(
            interaction_rate=interaction_rate,
            item_cold_start_rate=item_cold_start_rate,
            single_user_item_rate=single_user_item_rate,
            random_seed=random_seed,
        )
        
        # Simulate additional interactions based on real data
        # This will add more interactions to the existing real matrix
        print("  → Simulating additional interactions based on real playlist data...")
        simulated_matrix, _, _, _, _ = simulator.simulate_interactions(
            songs_df,
            user_ids,
            return_events=False,
            num_negatives_per_user=0,  # Don't generate negative samples here
            impressions_size=impressions_size,
            clap_embeddings=clap_embeddings,
            hard_negative_top_p=hard_negative_top_p,
            last_n=last_n,
        )
        
        # Combine real and simulated interactions
        # Use maximum (union) - if either has interaction, keep it
        # Note: real_matrix and simulated_matrix may have different shapes if they use different song sets
        # We need to ensure they have the same dimensions
        if real_matrix.shape != simulated_matrix.shape:
            # If shapes differ, we need to align them
            # This shouldn't happen if both use the same songs_df, but handle it gracefully
            print(f"  ⚠️  Shape mismatch: real={real_matrix.shape}, simulated={simulated_matrix.shape}")
            print(f"  → Using real matrix shape and aligning simulated matrix...")
            # For now, just use real matrix if shapes don't match
            # In practice, both should use the same songs_df so this shouldn't happen
            combined_matrix = real_matrix
        else:
            combined_matrix = real_matrix + simulated_matrix
            combined_matrix.data = np.minimum(combined_matrix.data, 1.0)  # Ensure binary
        
        print(f"  ✅ Combined matrix: {combined_matrix.shape}")
        print(f"     Real interactions: {real_matrix.nnz}")
        print(f"     Simulated interactions: {simulated_matrix.nnz}")
        print(f"     Combined interactions: {combined_matrix.nnz}")
        
        if return_events:
            # Generate events DataFrame if needed
            events = []
            for user_idx in range(combined_matrix.shape[0]):
                user_id = idx_to_user_id[user_idx]
                user_items = combined_matrix[user_idx].toarray().flatten()
                interacted_items = np.where(user_items > 0)[0]
                for item_idx in interacted_items:
                    song_id = idx_to_song_id[item_idx]
                    events.append({
                        'user_id': user_id,
                        'song_id': song_id,
                        'label': 1
                    })
            
            events_df = pd.DataFrame(events)
            return combined_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id, events_df
        else:
            return combined_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id
    
    # No playlists provided - pure simulation
    # Get user IDs if not provided
    if user_ids is None:
        if 'user_id' in songs_df.columns:
            user_ids = songs_df['user_id'].unique().tolist()
        else:
            raise ValueError("No user_ids provided and 'user_id' column not found in songs_df")
    
    # Create simulator
    simulator = InteractionSimulator(
        interaction_rate=interaction_rate,
        item_cold_start_rate=item_cold_start_rate,
        single_user_item_rate=single_user_item_rate,
        random_seed=random_seed,
    )
    
    # Simulate interactions (+ 可选事件表)
    return simulator.simulate_interactions(
        songs_df,
        user_ids,
        return_events=return_events,
        num_negatives_per_user=num_negatives_per_user,
        impressions_size=impressions_size,
        clap_embeddings=clap_embeddings,
        hard_negative_top_p=hard_negative_top_p,
        last_n=last_n,
    )


def load_songs_from_aggregated_file(
    aggregated_file: str,
    max_songs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load songs from aggregated playlist songs JSON file into DataFrame
    
    Args:
        aggregated_file: Path to all_playlist_songs.json
        max_songs: Optional limit on number of songs to load (for testing)
    
    Returns:
        DataFrame with columns: song_id, user_id, and other metadata
        Compatible with simulate_interactions functions
    """
    file_path = Path(aggregated_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {aggregated_file}")
    
    print(f"📂 Loading songs from {aggregated_file}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        songs_data = json.load(f)
    
    if max_songs:
        songs_data = songs_data[:max_songs]
        print(f"   Limited to {max_songs} songs for testing")
    
    # Extract relevant fields for simulation
    songs_list = []
    for song in songs_data:
        song_id = song.get('id')
        user_id = song.get('user_id')
        
        # Skip songs without required fields
        if not song_id:
            continue  # Skip songs without ID
        if not user_id:
            # Try alternative user ID fields
            user_id = song.get('handle') or song.get('display_name') or f"unknown_user_{len(songs_list)}"
        
        # Build song record compatible with simulate_interactions
        song_record = {
            'song_id': song_id,
            'user_id': user_id,  # Creator of the song
            'title': song.get('title', ''),
            'play_count': song.get('play_count', 0),
            'upvote_count': song.get('upvote_count', 0),
        }
        
        # Extract metadata for genre/popularity/tags/prompt
        metadata = song.get('metadata', {})
        if metadata:
            # Extract tags (string, convert to list)
            tags_str = metadata.get('tags', '')
            if tags_str:
                # Tags can be comma-separated or period-separated
                # Try comma first, then period, then just use the whole string
                if ',' in tags_str:
                    tags_list = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                elif '.' in tags_str:
                    # Split by period but keep meaningful parts
                    tags_list = [tag.strip() for tag in tags_str.split('.') if tag.strip() and len(tag.strip()) > 2]
                else:
                    # Single tag or space-separated
                    tags_list = [tag.strip() for tag in tags_str.split() if tag.strip()]
                
                # If still empty or single long string, use the original
                if not tags_list or (len(tags_list) == 1 and len(tags_list[0]) > 50):
                    # Treat as single tag
                    tags_list = [tags_str.strip()]
                
                song_record['tags'] = tags_list
                # Extract genre from first tag (limit length)
                first_tag = tags_list[0] if tags_list else 'unknown'
                song_record['genre'] = first_tag[:50] if len(first_tag) > 50 else first_tag
            else:
                song_record['tags'] = []
                song_record['genre'] = 'unknown'
            
            # Extract prompt
            song_record['prompt'] = metadata.get('prompt', '')
            
            song_record['duration'] = metadata.get('duration', 0)
            song_record['popularity_score'] = song.get('play_count', 0)  # Use play_count as popularity
        else:
            # No metadata - set defaults
            song_record['tags'] = []
            song_record['genre'] = 'unknown'
            song_record['prompt'] = ''
            song_record['duration'] = 0
            song_record['popularity_score'] = song.get('play_count', 0)
        
        # Calculate engagement_rate (upvotes / plays, avoid division by zero)
        play_count = song.get('play_count', 0) or 0
        upvote_count = song.get('upvote_count', 0) or 0
        if play_count > 0:
            song_record['engagement_rate'] = upvote_count / play_count
        else:
            song_record['engagement_rate'] = 0.0
        
        songs_list.append(song_record)
    
    songs_df = pd.DataFrame(songs_list)
    
    # Fill missing values and ensure correct types
    if len(songs_df) == 0:
        print("⚠️  No songs loaded!")
        return pd.DataFrame()
    
    # Handle genre
    if 'genre' not in songs_df.columns:
        songs_df['genre'] = 'unknown'
    else:
        songs_df['genre'] = songs_df['genre'].fillna('unknown')
    
    # Handle tags - must be lists
    if 'tags' not in songs_df.columns:
        songs_df['tags'] = [[] for _ in range(len(songs_df))]
    else:
        # Ensure all tags are lists, handle NaN/None
        def ensure_tag_list(x):
            # Check for None first (before pd.isna which might fail on lists)
            if x is None:
                return []
            # Check for NaN (but skip if it's a list, as pd.isna might have issues)
            if not isinstance(x, list):
                try:
                    if pd.isna(x):
                        return []
                except (ValueError, TypeError):
                    pass  # If pd.isna fails, continue to type checks
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                return [x] if x.strip() else []
            return []
        songs_df['tags'] = songs_df['tags'].apply(ensure_tag_list)
    
    # Handle prompt - must be strings
    if 'prompt' not in songs_df.columns:
        songs_df['prompt'] = ''
    else:
        songs_df['prompt'] = songs_df['prompt'].fillna('').astype(str)
    
    # Handle popularity_score
    if 'popularity_score' not in songs_df.columns:
        if 'play_count' in songs_df.columns:
            songs_df['popularity_score'] = songs_df['play_count'].fillna(0)
        else:
            songs_df['popularity_score'] = 0
    else:
        songs_df['popularity_score'] = songs_df['popularity_score'].fillna(0)
    
    # Handle other numeric fields
    for col in ['play_count', 'upvote_count', 'duration']:
        if col in songs_df.columns:
            songs_df[col] = songs_df[col].fillna(0)
    
    # Calculate engagement_rate if not present or has NaN values
    if 'engagement_rate' not in songs_df.columns:
        # Calculate from play_count and upvote_count
        play_count = songs_df.get('play_count', pd.Series([0] * len(songs_df))).fillna(0)
        upvote_count = songs_df.get('upvote_count', pd.Series([0] * len(songs_df))).fillna(0)
        songs_df['engagement_rate'] = np.where(play_count > 0, upvote_count / play_count, 0.0)
    else:
        songs_df['engagement_rate'] = songs_df['engagement_rate'].fillna(0)
    
    # Handle string fields
    for col in ['title', 'user_id']:
        if col in songs_df.columns:
            songs_df[col] = songs_df[col].fillna('')
    
    print(f"✅ Loaded {len(songs_df)} songs")
    print(f"   Unique users (creators): {songs_df['user_id'].nunique()}")
    print(f"   Unique songs: {songs_df['song_id'].nunique()}")
    
    return songs_df


def _generate_cache_key(
    songs_df: pd.DataFrame,
    interaction_rate: float,
    item_cold_start_rate: float,
    single_user_item_rate: float,
    random_seed: int,
    max_songs: Optional[int],
    return_events: bool = False,
    num_negatives_per_user: Optional[int] = None,
    playlist_hash: Optional[str] = None,
    impressions_size: Optional[int] = None,
) -> str:
    """Generate a cache key based on parameters and data"""
    # Create a hash from key parameters and data
    key_parts = [
        str(interaction_rate),
        str(item_cold_start_rate),
        str(single_user_item_rate),
        str(random_seed),
        str(max_songs) if max_songs else "None",
        str(len(songs_df)),
        str(return_events),
        str(num_negatives_per_user) if num_negatives_per_user else "None",
        playlist_hash if playlist_hash else "None",  # Include playlist hash if provided
        str(impressions_size) if impressions_size else "None",  # Include impressions_size
        # Hash of song IDs and user IDs (first 1000 for speed)
        str(sorted(songs_df['song_id'].unique()[:1000])),
        str(sorted(songs_df['user_id'].unique()[:1000])),
    ]
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def _save_matrix_cache(
    cache_path: Path,
    user_item_matrix: csr_matrix,
    user_id_to_idx: Dict[str, int],
    idx_to_user_id: Dict[int, str],
    song_id_to_idx: Dict[str, int],
    idx_to_song_id: Dict[int, str],
    cache_key: str,
):
    """Save the matrix and mappings to cache file"""
    cache_data = {
        'user_item_matrix': user_item_matrix,
        'user_id_to_idx': user_id_to_idx,
        'idx_to_user_id': idx_to_user_id,
        'song_id_to_idx': song_id_to_idx,
        'idx_to_song_id': idx_to_song_id,
        'cache_key': cache_key,
    }
    joblib.dump(cache_data, cache_path)
    print(f"  💾 Saved matrix cache to {cache_path}")


def _save_events_cache(
    cache_path: Path,
    user_item_matrix: csr_matrix,
    user_id_to_idx: Dict[str, int],
    idx_to_user_id: Dict[int, str],
    song_id_to_idx: Dict[str, int],
    idx_to_song_id: Dict[int, str],
    events_df: pd.DataFrame,
    cache_key: str,
):
    """Save the matrix, mappings, and events DataFrame to cache file"""
    cache_data = {
        'user_item_matrix': user_item_matrix,
        'user_id_to_idx': user_id_to_idx,
        'idx_to_user_id': idx_to_user_id,
        'song_id_to_idx': song_id_to_idx,
        'idx_to_song_id': idx_to_song_id,
        'events_df': events_df,
        'cache_key': cache_key,
    }
    joblib.dump(cache_data, cache_path)
    print(f"  💾 Saved events cache to {cache_path} ({len(events_df):,} events)")


def _load_events_cache(cache_path: Path, cache_key: str, songs_df: Optional[pd.DataFrame] = None, num_negatives_per_user: int = 50) -> Optional[Tuple]:
    """Load the matrix, mappings, and events DataFrame from cache file if cache_key matches
    
    Only checks data/cache directory (not archived/cache).
    """
    # Try primary cache location only (data/cache)
    if cache_path.exists():
        try:
            cache_data = joblib.load(cache_path)
            if cache_data.get('cache_key') == cache_key:
                print(f"  📂 Loading events from cache: {cache_path}")
                events_df = cache_data.get('events_df')
                if events_df is not None:
                    print(f"     Loaded {len(events_df):,} events")
                    return (
                        cache_data['user_item_matrix'],
                        cache_data['user_id_to_idx'],
                        cache_data['idx_to_user_id'],
                        cache_data['song_id_to_idx'],
                        cache_data['idx_to_song_id'],
                        events_df,
                    )
        except Exception as e:
            print(f"  ⚠️  Error loading events cache: {e}")
    
    return None


def _generate_events_from_matrix(
    user_item_matrix: csr_matrix,
    user_id_to_idx: Dict[str, int],
    idx_to_user_id: Dict[int, str],
    song_id_to_idx: Dict[str, int],
    idx_to_song_id: Dict[int, str],
    num_negatives_per_user: int = 50,
) -> pd.DataFrame:
    """Generate events DataFrame from an existing user-item matrix
    
    This is used when we have a cached matrix but need to generate events.
    Much faster than regenerating the entire matrix.
    """
    from collections import defaultdict
    
    events = []
    user_pos_items: Dict[int, set] = defaultdict(set)
    
    # Extract positive interactions from matrix
    rows, cols = user_item_matrix.nonzero()
    for u_idx, i_idx in zip(rows, cols):
        user_pos_items[u_idx].add(i_idx)
        user_id = idx_to_user_id[u_idx]
        song_id = idx_to_song_id[i_idx]
        events.append({
            "user_id": user_id,
            "song_id": song_id,
            "label": 1,
        })
    
    # Generate negative samples
    num_items = len(song_id_to_idx)
    all_item_indices = np.arange(num_items)
    num_users = len(user_id_to_idx)
    
    for u_idx in range(num_users):
        pos_items = user_pos_items.get(u_idx, set())
        if len(pos_items) >= num_items:
            continue
        
        if pos_items:
            pos_array = np.fromiter(pos_items, dtype=int)
            candidate_neg = np.setdiff1d(all_item_indices, pos_array, assume_unique=True)
        else:
            candidate_neg = all_item_indices
        
        if len(candidate_neg) == 0:
            continue
        
        k = min(num_negatives_per_user, len(candidate_neg))
        neg_items = np.random.choice(candidate_neg, size=k, replace=False)
        
        user_id = idx_to_user_id[u_idx]
        for i_idx in neg_items:
            song_id = idx_to_song_id[i_idx]
            events.append({
                "user_id": user_id,
                "song_id": song_id,
                "label": 0,
            })
    
    return pd.DataFrame(events)


def _load_matrix_cache(cache_path: Path, cache_key: str, songs_df: Optional[pd.DataFrame] = None) -> Optional[Tuple]:
    """Load the matrix and mappings from cache file if cache_key matches
    
    Only checks data/cache directory (not archived/cache).
    If cache key doesn't match but matrix shape is compatible, will use it with a warning.
    """
    # Try primary cache location only (data/cache)
    if cache_path.exists():
        try:
            cache_data = joblib.load(cache_path)
            stored_key = cache_data.get('cache_key')
            if stored_key == cache_key:
                print(f"  📂 Loading matrix from cache: {cache_path}")
                return (
                    cache_data['user_item_matrix'],
                    cache_data['user_id_to_idx'],
                    cache_data['idx_to_user_id'],
                    cache_data['song_id_to_idx'],
                    cache_data['idx_to_song_id'],
                )
            else:
                # Check if matrix shape is compatible even if key doesn't match
                matrix = cache_data.get('user_item_matrix')
                if matrix is not None and songs_df is not None:
                    num_users = songs_df['user_id'].nunique()
                    num_songs = songs_df['song_id'].nunique()
                    if matrix.shape == (num_users, num_songs):
                        print(f"  ⚠️  Cache key mismatch (stored: {stored_key[:16] if stored_key else 'None'}, expected: {cache_key[:16]})")
                        print(f"  💡 Using cache anyway - matrix shape matches ({matrix.shape})")
                        return (
                            cache_data['user_item_matrix'],
                            cache_data['user_id_to_idx'],
                            cache_data['idx_to_user_id'],
                            cache_data['song_id_to_idx'],
                            cache_data['idx_to_song_id'],
                        )
        except Exception as e:
            print(f"  ⚠️  Error loading cache: {e}")
    
    return None


# For easy swapping with real data, use this interface:
def get_user_item_matrix(
    songs_df: Optional[pd.DataFrame] = None,
    aggregated_file: Optional[str] = None,
    playlists: Optional[List[Dict[str, Any]]] = None,
    use_simulation: bool = True,
    interaction_rate: float = 0.15,
    item_cold_start_rate: float = 0.05,
    single_user_item_rate: float = 0.15,
    random_seed: int = 42,
    max_songs: Optional[int] = None,
    cache_dir: Optional[str] = "data/cache",
    use_cache: bool = True,
    return_events: bool = False,
    num_negatives_per_user: int = 50,
    impressions_size: int = 50,  # Size of impression set M for each user
    clap_embeddings: Optional[Dict[str, np.ndarray]] = None,  # CLAP embeddings for hard negative mining
    hard_negative_top_p: float = 0.2,  # Top p% of impressions by similarity to use as hard negatives
    last_n: int = 8,  # Number of recent interactions for user feature computation
) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    """
    Get user-item interaction matrix (real or simulated)
    
    This is the main interface - easily swap between real and simulated data.
    Supports caching to avoid regenerating the matrix.
    
    Args:
        songs_df: DataFrame with song data (if None and aggregated_file provided, will load from file)
        aggregated_file: Optional path to all_playlist_songs.json to load real user_ids and song_ids
        playlists: Optional real playlist data (if available and use_simulation=False)
        use_simulation: If True, use simulated interactions. If False, use real playlist data
        interaction_rate: For simulation only - density of interactions
        item_cold_start_rate: For simulation only - fraction of items with zero interactions
        single_user_item_rate: For simulation only - fraction of items with only one user
        random_seed: For simulation only - random seed
        max_songs: Optional limit on number of songs when loading from aggregated_file
        cache_dir: Directory to store cached matrices (default: "data/cache")
        use_cache: If True, try to load from cache and save to cache (default: True)
    
    Returns:
        Tuple of (user_item_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id)
    """
    # Load from aggregated file if provided
    if aggregated_file:
        print("📊 Loading songs from aggregated file and using SIMULATED interactions")
        songs_df = load_songs_from_aggregated_file(aggregated_file, max_songs=max_songs)
    
    if songs_df is None:
        raise ValueError("Must provide either songs_df or aggregated_file")
    
    if use_simulation or playlists is None:
        if playlists:
            print("📊 Using REAL playlist data + SIMULATED additional interactions")
        else:
            print("📊 Using SIMULATED user-item interactions")
        
        cache_path = None
        cache_key = None
        
        # Generate cache key (includes playlists if provided, return_events and num_negatives_per_user)
        if use_cache:
            # Include playlists in cache key if provided
            playlist_hash = None
            if playlists:
                import hashlib
                playlist_str = json.dumps(sorted([p.get('id', '') for p in playlists]), sort_keys=True)
                playlist_hash = hashlib.md5(playlist_str.encode()).hexdigest()[:8]
            
            cache_key = _generate_cache_key(
                songs_df,
                interaction_rate,
                item_cold_start_rate,
                single_user_item_rate,
                random_seed,
                max_songs,
                return_events=return_events,
                num_negatives_per_user=num_negatives_per_user,
                playlist_hash=playlist_hash,
                impressions_size=impressions_size,
            )
            
            cache_dir_path = Path(cache_dir)
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Use different cache file names for events vs matrix-only
            if return_events:
                cache_path = cache_dir_path / f"user_item_events_{cache_key[:16]}.pkl"
                cached_result = _load_events_cache(cache_path, cache_key, songs_df=songs_df, num_negatives_per_user=num_negatives_per_user)
            else:
                cache_path = cache_dir_path / f"user_item_matrix_{cache_key[:16]}.pkl"
                cached_result = _load_matrix_cache(cache_path, cache_key, songs_df=songs_df)
            
            if cached_result is not None:
                return cached_result
        
        # Generate new matrix (with playlists if provided)
        result = simulate_playlist_interactions(
            songs_df,
            playlists=playlists,
            interaction_rate=interaction_rate,
            item_cold_start_rate=item_cold_start_rate,
            single_user_item_rate=single_user_item_rate,
            random_seed=random_seed,
            return_events=return_events,
            num_negatives_per_user=num_negatives_per_user,
            impressions_size=impressions_size,
            clap_embeddings=clap_embeddings,
            hard_negative_top_p=hard_negative_top_p,
            last_n=last_n,
        )
        
        # Save to cache if enabled
        if use_cache and cache_path is not None and cache_key is not None:
            if return_events:
                # Save with events DataFrame
                _save_events_cache(
                    cache_path,
                    result[0],  # user_item_matrix
                    result[1],  # user_id_to_idx
                    result[2],  # idx_to_user_id
                    result[3],  # song_id_to_idx
                    result[4],  # idx_to_song_id
                    result[5],  # events_df
                    cache_key,
                )
            else:
                # Save matrix only
                _save_matrix_cache(
                    cache_path,
                    result[0],  # user_item_matrix
                    result[1],  # user_id_to_idx
                    result[2],  # idx_to_user_id
                    result[3],  # song_id_to_idx
                    result[4],  # idx_to_song_id
                    cache_key,
                )
        
        return result
    else:
        print("📊 Using REAL playlist data for user-item interactions")
        # TODO: Implement real playlist data loading
        # For now, fall back to simulation
        print("⚠️  Real playlist loading not yet implemented, using simulation")
        return simulate_playlist_interactions(
            songs_df,
            interaction_rate=interaction_rate,
            item_cold_start_rate=item_cold_start_rate,
            single_user_item_rate=single_user_item_rate,
            random_seed=random_seed,
            impressions_size=impressions_size,
            clap_embeddings=clap_embeddings,
            hard_negative_top_p=hard_negative_top_p,
            last_n=last_n,
        )

