"""Two-tower model-based recommender for candidate retrieval.

This recommender uses the trained two-tower model to retrieve candidates:
- Item tower: CLAP audio embeddings + metadata
- User tower: Average of last-n interacted items' CLAP embeddings
- Retrieval: Cosine similarity between user and item embeddings
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import BaseRecommender
from .two_tower import TwoTowerModel, TwoTowerConfig, build_item_feature_matrix


class TwoTowerRecommender(BaseRecommender):
    """
    Two-tower model-based recommender for content-based candidate retrieval.
    
    Uses CLAP embeddings for items and user history for user representation.
    """
    
    def __init__(
        self,
        model_path: str = "model_checkpoints/two_tower.pt",
        clap_embeddings_path: str = "runtime_data/clap_embeddings.json",
        top_k: int = 100,
    ):
        """
        Initialize two-tower recommender.
        
        Args:
            model_path: Path to trained two-tower model checkpoint
            clap_embeddings_path: Path to CLAP embeddings JSON file
            top_k: Number of top candidates to retrieve
        """
        super().__init__("TwoTower")
        self.model_path = Path(model_path)
        self.clap_embeddings_path = Path(clap_embeddings_path)
        self.top_k = top_k
        
        self.model = None
        self.config = None
        self.songs_df = None
        self.song_id_to_idx = None
        self.idx_to_song_id = None
        self.item_features = None  # CLAP embeddings matrix
        self.item_embeddings = None  # Encoded item embeddings (D_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.user_history = {}
    
    def fit(
        self,
        songs_df: pd.DataFrame,
        user_history: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        """
        Fit the two-tower recommender.
        
        Args:
            songs_df: DataFrame with song data
            user_history: Optional dict mapping user_id to list of song_ids (for last-n support)
        """
        self.songs_df = songs_df.copy()
        self.user_history = user_history or {}
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Two-tower model not found at {self.model_path}. "
                f"Please train the model first using train_two_tower.py"
            )
        
        print(f"  → Loading two-tower model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load config
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            if isinstance(config_dict, dict):
                self.config = TwoTowerConfig(**config_dict)
            else:
                self.config = config_dict
        else:
            raise ValueError("Checkpoint missing 'config' field")
        
        # Load song_id_to_idx
        if 'song_id_to_idx' in checkpoint:
            self.song_id_to_idx = checkpoint['song_id_to_idx']
            self.idx_to_song_id = {idx: sid for sid, idx in self.song_id_to_idx.items()}
        else:
            raise ValueError("Checkpoint missing 'song_id_to_idx' field")
        
        # Initialize model
        self.model = TwoTowerModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print(f"  ✅ Model loaded: base_dim={self.config.base_dim}, model_dim={self.config.model_dim}")
        
        # Load CLAP embeddings
        print(f"  → Loading CLAP embeddings from {self.clap_embeddings_path}...")
        if not self.clap_embeddings_path.exists():
            raise FileNotFoundError(
                f"CLAP embeddings not found at {self.clap_embeddings_path}. "
                f"Please compute embeddings first using compute_clap_embeddings.py"
            )
        
        with open(self.clap_embeddings_path, 'r') as f:
            clap_embeddings_dict = json.load(f)
        
        # Convert to numpy arrays
        clap_embeddings = {
            sid: np.asarray(vec, dtype=np.float32)
            for sid, vec in clap_embeddings_dict.items()
        }
        
        print(f"  ✅ Loaded {len(clap_embeddings)} CLAP embeddings")
        
        # Build item feature matrix
        print(f"  → Building item feature matrix...")
        self.item_features = build_item_feature_matrix(
            songs_df=songs_df,
            song_id_to_idx=self.song_id_to_idx,
            clap_embeddings=clap_embeddings,
            target_dim=None,  # Keep original CLAP dim
        )
        print(f"  ✅ Item feature matrix: {self.item_features.shape}")
        
        # Pre-compute item embeddings (for fast retrieval)
        print(f"  → Pre-computing item embeddings...")
        with torch.no_grad():
            item_feats_tensor = torch.from_numpy(self.item_features).to(self.device)
            self.item_embeddings = self.model.encode_items(item_feats_tensor).cpu().numpy()
        print(f"  ✅ Item embeddings: {self.item_embeddings.shape}")
        
        self.is_fitted = True
        print(f"  ✅ Two-tower recommender ready")
    
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
        Generate recommendations using two-tower model.
        
        Args:
            user_id: User ID (if provided, uses last-n interactions)
            song_ids: Seed song IDs (if provided, uses these as user history)
            n: Number of recommendations
            exclude_song_ids: Songs to exclude
            return_details: If True, return detailed information
            use_last_n: If True and user_id provided, use last-n interactions
        
        Returns:
            List of recommendation dicts
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        # Determine seed songs
        if user_id and use_last_n and user_id in self.user_history:
            seed_song_ids = self.user_history[user_id][-8:]  # Last 8 interactions (last-n=8)
        elif song_ids:
            seed_song_ids = song_ids
        else:
            # Fallback: return popular songs
            if return_details:
                print(f"  ⚠️  Two-Tower: No seed songs available! Falling back to popular songs")
            return self._get_popular_songs(n)
        
        # Get user embedding: average of seed songs' CLAP embeddings
        seed_indices = [
            self.song_id_to_idx[sid]
            for sid in seed_song_ids
            if sid in self.song_id_to_idx
        ]
        
        if not seed_indices:
            if return_details:
                print(f"  ⚠️  Two-Tower: None of the {len(seed_song_ids)} seed songs found! Falling back to popular songs")
            return self._get_popular_songs(n)
        
        # Compute user embedding as average of seed song features
        seed_features = self.item_features[seed_indices]  # (num_seeds, D_base)
        user_feature = seed_features.mean(axis=0)  # (D_base,)
        
        # Encode user feature
        with torch.no_grad():
            user_feat_tensor = torch.from_numpy(user_feature).unsqueeze(0).to(self.device)  # (1, D_base)
            user_embedding = self.model.encode_users(user_feat_tensor).cpu().numpy()  # (1, D_model)
            user_embedding = user_embedding[0]  # (D_model,)
        
        # Compute similarity with all items
        similarities = np.dot(self.item_embeddings, user_embedding)  # (num_items,)
        
        # Exclude seed songs and specified exclusions
        exclude_set = set(seed_song_ids)
        if exclude_song_ids:
            exclude_set.update(exclude_song_ids)
        
        exclude_indices = {
            self.song_id_to_idx[sid]
            for sid in exclude_set
            if sid in self.song_id_to_idx
        }
        
        # Set similarities of excluded items to -inf
        for idx in exclude_indices:
            similarities[idx] = -np.inf
        
        # Get top-k candidates
        top_k = min(self.top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        # Filter out excluded items and -inf
        valid_mask = np.isfinite(top_similarities)
        top_indices = top_indices[valid_mask]
        top_similarities = top_similarities[valid_mask]
        
        # If no valid recommendations found, return popular songs
        if len(top_indices) == 0:
            if return_details:
                print(f"⚠️  Two-Tower: No valid recommendations found. Falling back to popular songs")
            return self._get_popular_songs(n)
            
        # Format results
        results = []
        seen_song_ids = set()
        
        for idx, sim in zip(top_indices, top_similarities):
            song_id = self.idx_to_song_id[idx]
            if song_id in seen_song_ids:
                continue
            seen_song_ids.add(song_id)
            
            results.append({
                'song_id': song_id,
                'score': float(sim),
            })
            
            if len(results) >= n:
                break
        
        # Prepare details if requested
        details = None
        if return_details:
            details = {}
            # Two-Tower uses average of seed songs as query
            # Store which seed songs were used (for display)
            for r in results:
                song_id = r['song_id']
                details[song_id] = {
                    'two_tower_similarity': float(r['score']),
                    'seed_songs_count': len(seed_song_ids),
                    'seed_song_ids': seed_song_ids[:5],  # Show first 5 seed songs used
                    'method': 'two_tower_average_query',
                }
        
        # Add metadata
        return self._format_recommendations(
            song_ids=[r['song_id'] for r in results],
            scores=np.array([r['score'] for r in results]),
            songs_df=self.songs_df,
            return_details=return_details,
            details=details,
        )
    
    def get_similar_songs(
        self,
        song_id: str,
        n: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get similar songs using item embedding similarity.
        
        Args:
            song_id: Seed song ID
            n: Number of similar songs
        
        Returns:
            List of similar song dicts
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        if song_id not in self.song_id_to_idx:
            return []
        
        seed_idx = self.song_id_to_idx[song_id]
        seed_embedding = self.item_embeddings[seed_idx]  # (D_model,)
        
        # Compute similarity with all items
        similarities = np.dot(self.item_embeddings, seed_embedding)  # (num_items,)
        
        # Exclude seed song
        similarities[seed_idx] = -np.inf
        
        # Get top-n
        top_indices = np.argsort(similarities)[::-1][:n]
        top_similarities = similarities[top_indices]
        
        # Format results
        return self._format_recommendations(
            song_ids=[self.idx_to_song_id[idx] for idx in top_indices],
            scores=top_similarities,
            songs_df=self.songs_df,
            return_details=False,
        )
    
    def _get_popular_songs(self, n: int) -> List[Dict[str, Any]]:
        """Fallback: return popular songs"""
        if self.songs_df is None or len(self.songs_df) == 0:
            return []
        
        # Sort by play_count if available
        if 'play_count' in self.songs_df.columns:
            popular = self.songs_df.nlargest(n, 'play_count')
        else:
            popular = self.songs_df.head(n)
        
        return self._format_recommendations(
            song_ids=popular['song_id'].tolist(),
            scores=np.ones(len(popular)),
            songs_df=self.songs_df,
            return_details=False,
        )


