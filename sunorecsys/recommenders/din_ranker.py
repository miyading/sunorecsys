"""Deep Interest Network (DIN) for CTR prediction in ranking stage.

DIN uses attention mechanisms to model user interests from historical interactions.
This implementation uses attention to aggregate user's historical track embeddings 
to predict click-through rate for candidate tracks.
Also uses simplified MLP inputs: only last-n + candidate track to predict attention scores.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from pathlib import Path
import joblib
import json

from .base import BaseRecommender


class AttentionAggregator(nn.Module):
    """
    Attention mechanism for aggregating user's historical track embeddings.
    
    This is the core component of DIN - it learns which historical tracks
    are most relevant for predicting interest in a candidate track.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        """
        Initialize attention aggregator.
        
        Args:
            embedding_dim: Dimension of track embeddings
            hidden_dim: Hidden dimension for attention network
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Attention network: computes relevance score for each historical track
        # Input: [historical_track_emb, candidate_track_emb] -> attention_score
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self, 
        historical_embs: torch.Tensor,  # (batch_size, num_history, embedding_dim)
        candidate_emb: torch.Tensor,    # (batch_size, embedding_dim)
        mask: Optional[torch.Tensor] = None  # (batch_size, num_history) - 1 for valid, 0 for padded
    ) -> tuple:
        """
        Compute attention-weighted aggregation of historical embeddings.
        
        Args:
            historical_embs: User's historical track embeddings
            candidate_emb: Candidate track embedding to predict CTR for
            mask: Optional mask to exclude padded positions (1 = valid, 0 = padded)
        
        Returns:
            Tuple of (aggregated_embedding, attention_weights)
        """
        batch_size, num_history, emb_dim = historical_embs.shape
        
        # Expand candidate embedding to match historical embeddings
        candidate_expanded = candidate_emb.unsqueeze(1).expand(-1, num_history, -1)  # (batch, num_history, emb_dim)
        
        # Concatenate historical and candidate embeddings
        combined = torch.cat([historical_embs, candidate_expanded], dim=-1)  # (batch, num_history, emb_dim*2)
        
        # Compute attention scores
        attention_scores = self.attention_net(combined).squeeze(-1)  # (batch, num_history)
        
        # Apply mask: set attention scores to -inf for padded positions
        if mask is not None:
            # mask: (batch, num_history) where 1 = valid, 0 = padded
            # Convert to attention mask: 0 = valid (keep), -inf = padded (mask out)
            attention_mask = (1.0 - mask) * -1e9  # -inf for padded positions
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, num_history)
        
        # Weighted aggregation
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * historical_embs, dim=1)  # (batch, emb_dim)
        
        return aggregated, attention_weights


class DINModel(nn.Module):
    """
    Deep Interest Network for CTR prediction.
    
    Architecture:
    1. Attention aggregation of user's historical tracks
    2. MLP to predict CTR from aggregated user interest and candidate track
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2
    ):
        """
        Initialize DIN model.
        
        Args:
            embedding_dim: Dimension of track embeddings
            hidden_dims: Hidden dimensions for MLP layers
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Attention aggregator
        self.attention = AttentionAggregator(embedding_dim, hidden_dim=64)
        
        # MLP for CTR prediction
        # Input: [aggregated_user_interest, candidate_track_emb] -> CTR score
        mlp_layers = []
        input_dim = embedding_dim * 2  # user_interest + candidate_track
        
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final output layer (CTR prediction: 0-1)
        mlp_layers.append(nn.Linear(input_dim, 1))
        mlp_layers.append(nn.Sigmoid())  # Output CTR probability
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(
        self,
        historical_embs: torch.Tensor,  # (batch, num_history, emb_dim)
        candidate_emb: torch.Tensor,    # (batch, emb_dim)
        mask: Optional[torch.Tensor] = None  # (batch, num_history) - 1 for valid, 0 for padded
    ) -> tuple:
        """
        Predict CTR for candidate tracks given user's historical tracks.
        
        Args:
            historical_embs: User's historical track embeddings
            candidate_emb: Candidate track embedding
            mask: Optional mask to exclude padded positions (1 = valid, 0 = padded)
        
        Returns:
            Tuple of (ctr_prediction, attention_weights)
        """
        # Aggregate user interest using attention (with masking)
        user_interest, attention_weights = self.attention(historical_embs, candidate_emb, mask=mask)
        
        # Concatenate user interest and candidate track
        combined = torch.cat([user_interest, candidate_emb], dim=-1)  # (batch, emb_dim*2)
        
        # Predict CTR
        ctr = self.mlp(combined)  # (batch, 1)
        
        return ctr.squeeze(-1), attention_weights  # (batch,), (batch, num_history)


class DINRanker(BaseRecommender):
    """
    DIN-based ranker for CTR prediction in fine ranking stage.
    
    Uses attention to model user interests from historical track interactions
    and predicts click-through rate for candidate tracks.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,  # CLAP embedding dimension
        model_path: Optional[str] = None,
        clap_embeddings_path: str = "runtime_data/clap_embeddings.json",
        device: Optional[str] = None,
    ):
        """
        Initialize DIN ranker.
        
        Args:
            embedding_dim: Dimension of track embeddings (CLAP dimension)
            model_path: Path to trained DIN model (optional, can train later)
            clap_embeddings_path: Path to CLAP embeddings JSON file
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__("DINRanker")
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.clap_embeddings_path = Path(clap_embeddings_path)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.model = None
        self.model_trained = False  # Track if model is actually trained (not just initialized)
        self.clap_embeddings = None
        self.songs_df = None
        self.user_history = {}
        self.song_id_to_idx = {}
        self.idx_to_song_id = {}
    
    def fit(
        self,
        songs_df: pd.DataFrame,
        user_history: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        """
        Fit the DIN ranker.
        
        Args:
            songs_df: DataFrame with song data
            user_history: Optional dict mapping user_id to list of song_ids
        """
        self.songs_df = songs_df.copy()
        self.user_history = user_history or {}
        
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
        self.clap_embeddings = {
            sid: np.asarray(vec, dtype=np.float32)
            for sid, vec in clap_embeddings_dict.items()
        }
        
        print(f"  ✅ Loaded {len(self.clap_embeddings)} CLAP embeddings")
        
        # Build song_id mappings
        unique_song_ids = songs_df['song_id'].unique()
        self.song_id_to_idx = {sid: idx for idx, sid in enumerate(unique_song_ids)}
        self.idx_to_song_id = {idx: sid for sid, idx in self.song_id_to_idx.items()}
        
        # Initialize or load DIN model
        if self.model_path and Path(self.model_path).exists():
            print(f"  → Loading DIN model from {self.model_path}...")
            self._load_model()
            self.model_trained = True
        else:
            print(f"  ⚠️  Initializing DIN model (not trained yet, will use fallback)")
            print(f"     To train the model, use train_din.py")
            self.model = DINModel(embedding_dim=self.embedding_dim).to(self.device)
            self.model.eval()  # Set to eval mode for inference
            self.model_trained = False  # Mark as untrained
        
        self.is_fitted = True
        status = " (trained)" if self.model_trained else " (using fallback - not trained)"
        print(f"  ✅ DIN ranker ready{status}")
    
    def predict_ctr(
        self,
        user_id: Optional[str],
        candidate_song_ids: List[str],
        historical_song_ids: Optional[List[str]] = None,
        max_history: int = 50
    ) -> Dict[str, float]:
        """
        Predict CTR for candidate tracks given user's historical interactions.
        
        Args:
            user_id: User ID (to lookup history)
            candidate_song_ids: List of candidate track IDs to predict CTR for
            historical_song_ids: Optional explicit history (overrides user_id lookup)
            max_history: Maximum number of historical tracks to use
        
        Returns:
            Dictionary mapping song_id to CTR score
        """
        if not self.is_fitted:
            raise ValueError("Ranker not fitted. Call fit() first.")
        
        # If model not trained, return default CTR scores (fallback)
        if not self.model_trained:
            return {sid: 0.5 for sid in candidate_song_ids}
        
        # Get user's historical tracks
        if historical_song_ids is None:
            if user_id and user_id in self.user_history:
                historical_song_ids = self.user_history[user_id][-max_history:]
            else:
                historical_song_ids = []
        
        if not historical_song_ids:
            # No history: return default CTR for all candidates
            return {sid: 0.5 for sid in candidate_song_ids}
        
        # Get embeddings for historical tracks
        historical_embs = []
        for sid in historical_song_ids:
            if sid in self.clap_embeddings:
                historical_embs.append(self.clap_embeddings[sid])
        
        if not historical_embs:
            return {sid: 0.5 for sid in candidate_song_ids}
        
        # Get embeddings for candidate tracks
        candidate_embs = {}
        for sid in candidate_song_ids:
            if sid in self.clap_embeddings:
                candidate_embs[sid] = self.clap_embeddings[sid]
        
        if not candidate_embs:
            return {sid: 0.5 for sid in candidate_song_ids}
        
        # Prepare tensors
        num_history = len(historical_embs)
        historical_embs_array = np.array(historical_embs)  # (num_history, emb_dim)
        historical_tensor = torch.from_numpy(historical_embs_array).unsqueeze(0).to(self.device)  # (1, num_history, emb_dim)
        
        # Create mask: all positions are valid (1) since we only included valid embeddings
        mask_tensor = torch.ones(1, num_history, dtype=torch.float32).to(self.device)  # (1, num_history)
        
        # Predict CTR for each candidate
        ctr_scores = {}
        with torch.no_grad():
            for sid, candidate_emb in candidate_embs.items():
                candidate_tensor = torch.from_numpy(candidate_emb).unsqueeze(0).to(self.device)  # (1, emb_dim)
                
                # Predict CTR (with mask)
                ctr, _ = self.model(historical_tensor, candidate_tensor, mask=mask_tensor)
                ctr_scores[sid] = float(ctr.cpu().item())
        
        # Fill in default for candidates without embeddings
        for sid in candidate_song_ids:
            if sid not in ctr_scores:
                ctr_scores[sid] = 0.5
        
        return ctr_scores
    
    def _load_model(self):
        """Load trained DIN model from checkpoint"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Check if model was trained
        self.model_trained = checkpoint.get('model_trained', True)  # Default to True if key missing (backward compat)
        
        if 'model_state_dict' in checkpoint:
            self.model = DINModel(embedding_dim=self.embedding_dim).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model = DINModel(embedding_dim=self.embedding_dim).to(self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume entire checkpoint is the model state dict
            self.model = DINModel(embedding_dim=self.embedding_dim).to(self.device)
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
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
        Recommend songs using DIN CTR prediction.
        
        Note: This is typically used in fine ranking stage, not for initial recall.
        """
        if not self.is_fitted:
            raise ValueError("Ranker not fitted. Call fit() first.")
        
        # DIN is used for ranking, not recall
        # This method is mainly for compatibility
        if not song_ids:
            return []
        
        # Predict CTR for candidate songs
        ctr_scores = self.predict_ctr(
            user_id=user_id,
            candidate_song_ids=song_ids,
            max_history=kwargs.get('max_history', 50)
        )
        
        # Sort by CTR
        sorted_songs = sorted(ctr_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for song_id, ctr in sorted_songs[:n]:
            if exclude_song_ids and song_id in exclude_song_ids:
                continue
            
            results.append({
                'song_id': song_id,
                'score': ctr,
                'ctr_prediction': ctr,
            })
        
        return self._format_recommendations(
            [r['song_id'] for r in results],
            np.array([r['score'] for r in results]),
            self.songs_df,
            return_details=return_details
        )
    
    def get_similar_songs(
        self,
        song_id: str,
        n: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """DIN is not designed for similarity search"""
        return []
    
    def save(self, path: str):
        """Save the DIN model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'model_trained': True,  # Mark as trained
        }, path)
    
    @classmethod
    def load(cls, path: str, clap_embeddings_path: str = "runtime_data/clap_embeddings.json"):
        """Load a saved DIN ranker"""
        checkpoint = torch.load(path, map_location='cpu')
        embedding_dim = checkpoint.get('embedding_dim', 512)
        
        ranker = cls(
            embedding_dim=embedding_dim,
            model_path=path,
            clap_embeddings_path=clap_embeddings_path
        )
        ranker._load_model()
        ranker.model_trained = checkpoint.get('model_trained', True)  # Assume trained if loaded
        ranker.is_fitted = True
        
        return ranker

