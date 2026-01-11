"""Hybrid recommender combining all channels"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import defaultdict
import joblib

from .base import BaseRecommender
from .item_cf import ItemBasedCFRecommender
from .prompt_based import PromptBasedRecommender
from .user_based import UserBasedRecommender
from .quality_filter import QualityFilter
from .two_tower_recommender import TwoTowerRecommender
from .din_ranker import DINRanker
from ..datasets.user_history import UserHistoryManager
from ..utils.music_flamingo_quality import MusicFlamingoQualityScorer


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender with four-stage architecture:
    
    Stage 1 (Recall): Candidate Retrieval
    - Channel 1: Item-based CF
    - Channel 2: User-based CF
    - Channel 3: Two-tower model (CLAP-based content retrieval - audio)
    - Channel 4: Prompt-based similarity (CLAP text embeddings - creative intent)
    
    Stage 2 (Coarse Ranking): Quality Filter
    - Channel 5: Quality filter
    
    Stage 3 (Fine Ranking): CTR Prediction
    - Channel 6: DIN with attention (CTR prediction using user history)
    
    Stage 4 (Re-ranking): Final Ranking
    - Music Flamingo (optional, computationally intensive)
    
    Note: CLAP provides aligned text-audio embeddings, enabling direct similarity
    between prompts and audio tracks without cross-modal alignment.
    Prompt-based channel uses prompts from user's listening history (songs by
    other artists) to find similar songs in the catalog - this is a recall task,
    not a diversity signal for ranking.
    """
    
    def __init__(
        self,
        # Stage 1 (Recall) weights
        item_cf_weight: float = 0.25,      # Channel 1: Item-based CF
        user_cf_weight: float = 0.25,      # Channel 2: User-based CF
        two_tower_weight: float = 0.35,    # Channel 3: Two-tower content retrieval
        prompt_weight: float = 0.15,       # Channel 4: Prompt-based similarity
        # Stage 2 (Coarse Ranking) - Quality filter (no weight, just filtering)
        quality_threshold: float = 0.3,
        use_quality_filter: bool = True,
        quality_scores_file: Optional[str] = None,
        # Stage 3 (Fine Ranking) weights
        din_weight: float = 1.0,           # Channel 6: DIN with attention (CTR prediction)
        din_model_path: Optional[str] = None,  # Path to trained DIN model (optional)
        # Stage 4 (Re-ranking)
        use_music_flamingo: bool = False,
        music_flamingo_model_id: str = "nvidia/music-flamingo-hf",
        device: Optional[str] = None,
        # Component toggles
        use_user_cf: bool = True,
        use_two_tower: bool = True,
        use_prompt_based: bool = True,     # Toggle for prompt-based channel
        two_tower_model_path: str = "model_checkpoints/two_tower.pt",
        two_tower_clap_path: str = "runtime_data/clap_embeddings.json",
        history_manager: Optional[UserHistoryManager] = None,
        history_file: str = "runtime_data/user_history.json",
        use_last_n: bool = True,
    ):
        super().__init__("Hybrid")
        # Stage 1 (Recall) weights
        self.item_cf_weight = item_cf_weight
        self.user_cf_weight = user_cf_weight
        self.two_tower_weight = two_tower_weight
        self.prompt_weight = prompt_weight
        # Stage 2 (Coarse Ranking) - Quality filter
        self.quality_threshold = quality_threshold
        self.use_quality_filter = use_quality_filter
        self.quality_scores_file = quality_scores_file
        # Stage 3 (Fine Ranking) weights
        self.din_weight = din_weight
        self.din_model_path = din_model_path
        # Stage 4 (Re-ranking)
        self.use_music_flamingo = use_music_flamingo
        self.music_flamingo_model_id = music_flamingo_model_id
        self.device = device
        # Component toggles
        self.use_user_cf = use_user_cf
        self.use_two_tower = use_two_tower
        self.use_prompt_based = use_prompt_based
        self.two_tower_model_path = two_tower_model_path
        self.two_tower_clap_path = two_tower_clap_path
        self.use_last_n = use_last_n
        
        # Initialize user history manager
        if history_manager is not None:
            self.history_manager = history_manager
        else:
            self.history_manager = UserHistoryManager(history_file=history_file)
        
        # Initialize component recommenders
        # Stage 1 (Recall)
        self.item_cf_recommender = None
        self.user_recommender = None
        self.two_tower_recommender = None
        self.prompt_recommender = None
        # Stage 2 (Coarse Ranking)
        self.quality_filter = None
        # Stage 3 (Fine Ranking)
        self.din_ranker = None  # DIN ranker for CTR prediction
        # Stage 4 (Re-ranking)
        self.music_flamingo_scorer = None
        
        self.songs_df = None
    
    def _display_channel_top5(self, recommendations: List[Dict[str, Any]], channel_name: str):
        """Display top 5 recommendations for a channel with artist handle, title, URL, genre, tags"""
        if not recommendations or self.songs_df is None:
            return
        
        for i, rec in enumerate(recommendations[:5], 1):
            song_id = rec.get('song_id', '')
            title = rec.get('title', 'N/A')
            score = rec.get('score', 0.0)
            url = rec.get('suno_url', f"https://suno.com/song/{song_id}")
            
            # Get additional info from rec first, then fallback to songs_df
            genre = rec.get('genre', 'N/A')
            tags = rec.get('tags', [])
            user_id = rec.get('user_id', '')
            artist_handle = rec.get('user_handle', '')  # Try to get handle from rec
            
            # Try to get additional info from songs_df if missing
            if song_id and self.songs_df is not None:
                song_rows = self.songs_df[self.songs_df['song_id'] == song_id]
                if not song_rows.empty:
                    song_data = song_rows.iloc[0]
                    if not genre or genre == 'N/A':
                        genre = song_data.get('genre', 'N/A')
                    if not tags:
                        tags = song_data.get('tags', [])
                        if isinstance(tags, str):
                            tags = [tags] if tags else []
                    if not user_id:
                        user_id = song_data.get('user_id', '')
                    if not artist_handle:
                        artist_handle = song_data.get('user_handle', song_data.get('handle', ''))
            
            # Format tags
            if isinstance(tags, list):
                tags_str = ', '.join(str(t) for t in tags[:5]) if tags else 'N/A'
            else:
                tags_str = str(tags) if tags else 'N/A'
            
            # Display artist as handle if available, otherwise user_id
            artist_display = artist_handle if artist_handle else (user_id if user_id else 'N/A')
            
            print(f"     {i}. {title}")
            print(f"        Artist: {artist_display}")
            if artist_handle and user_id:
                print(f"        User ID: {user_id}")
            print(f"        Score: {score:.4f}")
            print(f"        URL: {url}")
            print(f"        Genre: {genre}")
            print(f"        Tags: {tags_str}")
            
            # Show source information based on channel
            # Check both rec['details'] and rec.get('details') for compatibility
            details = rec.get('details', {})
            if not details and 'metadata' in rec:
                # Fallback: check metadata if details not directly in rec
                metadata = rec.get('metadata', {})
                details = metadata.get('details', {})
            
            if channel_name == "Item-Based CF" and 'primary_seed_song_id' in details:
                primary_seed_id = details['primary_seed_song_id']
                if primary_seed_id:
                    # Get the specific seed song that found this recommendation
                    seed_rows = self.songs_df[self.songs_df['song_id'] == primary_seed_id]
                    if not seed_rows.empty:
                        seed_title = seed_rows.iloc[0].get('title', primary_seed_id)
                        seed_url = f"https://suno.com/song/{primary_seed_id}"
                        print(f"        From user history: [{seed_title}]({seed_url})")
            
            elif channel_name == "User-Based CF" and 'source_song_id' in details and 'source_similar_user_id' in details:
                source_song_id = details.get('source_song_id')
                source_user_id = details.get('source_similar_user_id')
                if source_song_id:
                    # Get the specific song from similar user's last n items
                    source_rows = self.songs_df[self.songs_df['song_id'] == source_song_id]
                    if not source_rows.empty:
                        # source_title = source_rows.iloc[0].get('title', source_song_id)
                        # source_url = f"https://suno.com/song/{source_song_id}"
                        # Show which similar user and which song
                        user_display = source_user_id if source_user_id else 'Unknown'
                        print(f"        From similar user [{user_display}]'s last-n")
            
            elif channel_name == "Two-Tower" and 'seed_song_ids' in details:
                seed_song_ids = details.get('seed_song_ids', [])
                if seed_song_ids:
                    # Show which seed songs were used (average query)
                    seed_info = []
                    for seed_id in seed_song_ids[:3]:  # Show up to 3 seed songs
                        seed_rows = self.songs_df[self.songs_df['song_id'] == seed_id]
                        if not seed_rows.empty:
                            seed_title = seed_rows.iloc[0].get('title', seed_id)
                            seed_url = f"https://suno.com/song/{seed_id}"
                            seed_info.append(f"[{seed_title}]({seed_url})")
                    if seed_info:
                        print(f"        From user history (average query): {', '.join(seed_info)}")
            
            elif channel_name == "Prompt-Based" and 'seed_song_ids' in details:
                seed_song_ids = details.get('seed_song_ids', [])
                
                # Get the recommended song's prompt
                recommended_song_rows = self.songs_df[self.songs_df['song_id'] == song_id]
                recommended_prompt = ''
                if not recommended_song_rows.empty:
                    recommended_prompt = recommended_song_rows.iloc[0].get('prompt', '')
                
                if recommended_prompt:
                    prompt_preview = recommended_prompt[:200] + "..." if len(recommended_prompt) > 200 else recommended_prompt
                    print(f"        Recommended song prompt: {prompt_preview}")
                
                if seed_song_ids:
                    print(f"        From user history seed songs ({len(seed_song_ids)} seed(s) recommended this):")
                    
                    # Show seed songs with their prompts (these are the specific seeds that found this recommendation)
                    for seed_id in seed_song_ids[:5]:  # Show up to 5 seed songs
                        seed_rows = self.songs_df[self.songs_df['song_id'] == seed_id]
                        if not seed_rows.empty:
                            seed_data = seed_rows.iloc[0]
                            seed_title = seed_data.get('title', seed_id)
                            seed_prompt = seed_data.get('prompt', '')
                            seed_url = f"https://suno.com/song/{seed_id}"
                            
                            if seed_prompt:
                                prompt_preview = seed_prompt[:150] + "..." if len(seed_prompt) > 150 else seed_prompt
                                print(f"          - [{seed_title}]({seed_url}): {prompt_preview}")
                            else:
                                print(f"          - [{seed_title}]({seed_url})")
                    
                    if len(seed_song_ids) > 5:
                        print(f"          ... and {len(seed_song_ids) - 5} more seed songs")
    
    def fit(
        self,
        songs_df: pd.DataFrame,
        playlists: Optional[List[Dict[str, Any]]] = None,
        user_history: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        """
        Fit all component recommenders
        
        Args:
            songs_df: DataFrame with song data
            playlists: Optional playlist data
            user_history: Optional user history dict (if None, will use history_manager)
        """
        self.songs_df = songs_df.copy()
        
        # Get user history from history_manager if not provided
        if user_history is None and self.history_manager:
            user_history = {}
            for uid in self.history_manager.get_all_user_ids():
                interactions = self.history_manager.get_user_interactions(
                    uid, 
                    use_weekly_mixing=False  # Get all for model training
                )
                user_history[uid] = interactions
        
        print("\n" + "="*80)
        print("Fitting Hybrid Recommender Components")
        print("="*80)
        # print(f"Total songs: {len(songs_df)}")
        # print(f"Total users: {songs_df['user_id'].nunique() if 'user_id' in songs_df.columns else 'N/A'}")
        
        # ========================================================================
        # STAGE 1: RECALL - Candidate Retrieval
        # ========================================================================
        print("\n" + "="*80)
        print("STAGE 1: RECALL - Candidate Retrieval")
        print("="*80)
        
        # Channel 1: Item-based CF
        print("\n" + "-"*80)
        print("[Recall Channel 1] Item-Based CF Recommender")
        print("-"*80)
        print("  → Building user-item interaction matrix...")
        self.item_cf_recommender = ItemBasedCFRecommender()
        self.item_cf_recommender.fit(songs_df, playlists=playlists, **kwargs)
        print("  ✅ Item-Based CF ready")
        
        # Channel 2: User-based CF
        if self.use_user_cf:
            print("\n" + "-"*80)
            print("[Recall Channel 2] User-Based CF Recommender")
            print("-"*80)
            print("  → Building user-user similarity matrix...")
            self.user_recommender = UserBasedRecommender()
            # Pass user_history to UserBasedRecommender so it can get last n items per similar user
            fit_kwargs = {**kwargs, 'user_history': user_history}
            self.user_recommender.fit(songs_df, playlists=playlists, **fit_kwargs)
            print("  ✅ User-Based CF ready")
        else:
            print("\n" + "-"*80)
            print("[Recall Channel 2] User-Based CF Recommender - SKIPPED")
            print("-"*80)
            self.user_recommender = None
        
        # Channel 3: Two-tower model (CLAP-based content retrieval)
        if self.use_two_tower:
            print("\n" + "-"*80)
            print("[Recall Channel 3] Two-Tower Recommender (CLAP-based)")
            print("-"*80)
            print("  → Loading trained two-tower model and CLAP embeddings...")
            self.two_tower_recommender = TwoTowerRecommender(
                model_path=self.two_tower_model_path,
                clap_embeddings_path=self.two_tower_clap_path,
            )
            self.two_tower_recommender.fit(songs_df, user_history=user_history, **kwargs)
            print("  ✅ Two-Tower Recommender ready")
        else:
            print("\n" + "-"*80)
            print("[Recall Channel 3] Two-Tower Recommender - SKIPPED")
            print("-"*80)
            self.two_tower_recommender = None
        
        # Channel 4: Prompt-based similarity (creative intent matching)
        if self.use_prompt_based:
            print("\n" + "-"*80)
            print("[Recall Channel 4] Prompt-Based Similarity (Creative Intent Matching)")
            print("-"*80)
            print("  → Building prompt similarity index using CLAP text embeddings...")
            print("  → Note: Uses prompts from user's listening history (songs by other artists)")
            self.prompt_recommender = PromptBasedRecommender(
                use_clap=True,  # Use CLAP text embeddings (aligned with audio)
                clap_model_path=kwargs.get('clap_model_path'),
                clap_cache_dir=kwargs.get('clap_cache_dir', 'runtime_data/audio_cache')
            )
            self.prompt_recommender.fit(songs_df, user_history=user_history, **kwargs)
            print("  ✅ Prompt-Based ready (using CLAP aligned embeddings)")
        else:
            print("\n" + "-"*80)
            print("[Recall Channel 4] Prompt-Based Similarity - SKIPPED")
            print("-"*80)
            self.prompt_recommender = None
        
        # ========================================================================
        # STAGE 2: COARSE RANKING - Quality Filter
        # ========================================================================
        print("\n" + "="*80)
        print("STAGE 2: COARSE RANKING - Quality Filter")
        print("="*80)
        print("  → Computing quality scores...")
        self.quality_filter = QualityFilter(
            quality_threshold=self.quality_threshold,
            quality_scores_file=self.quality_scores_file if 'quality_scores_file' not in kwargs else kwargs.get('quality_scores_file'),
            use_music_flamingo=False,  # Music Flamingo moved to Stage 4 (re-ranking)
        )
        self.quality_filter.fit(songs_df, **kwargs)
        print("  ✅ Quality Filter ready")
        
        # ========================================================================
        # STAGE 3: FINE RANKING - CTR Prediction
        # ========================================================================
        print("\n" + "="*80)
        print("STAGE 3: FINE RANKING - CTR Prediction")
        print("="*80)
        
        # Channel 5: DIN with attention (CTR prediction)
        print("\n" + "-"*80)
        print("[Fine Ranking Channel 5] DIN with Attention (CTR Prediction)")
        print("-"*80)
        print("  → Initializing DIN ranker...")
        self.din_ranker = DINRanker(
            model_path=self.din_model_path,
            clap_embeddings_path=self.two_tower_clap_path,
            device=self.device
        )
        self.din_ranker.fit(songs_df, user_history=user_history, **kwargs)
        
        # ========================================================================
        # STAGE 4: RE-RANKING - Music Flamingo
        # ========================================================================
        if self.use_music_flamingo:
            print("\n" + "="*80)
            print("STAGE 4: RE-RANKING - Music Flamingo")
            print("="*80)
            print("  → Initializing Music Flamingo scorer for re-ranking...")
            self.music_flamingo_scorer = MusicFlamingoQualityScorer(
                model_id=self.music_flamingo_model_id,
                device=self.device,
            )
            print("  ✅ Music Flamingo ready")
        
        self.is_fitted = True
        print("\n" + "="*80)
        print("✅ Hybrid Recommender Fitted Successfully!")
        print("="*80)
        print("Architecture:")
        print("  Stage 1 (Recall): Item CF, User CF, Two-Tower (audio), Prompt-Based (text)")
        print("  Stage 2 (Coarse Ranking): Quality Filter")
        print("  Stage 3 (Fine Ranking): DIN (CTR Prediction)")
        print("  Stage 4 (Re-ranking): Music Flamingo" + (" (enabled)" if self.use_music_flamingo else " (disabled)"))
    
    def recommend(
        self,
        user_id: Optional[str] = None,
        song_ids: Optional[List[str]] = None,
        n: int = 10,
        exclude_song_ids: Optional[List[str]] = None,
        return_details: bool = False,
        use_last_n: Optional[bool] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations (Discover Weekly style)
        
        Args:
            user_id: User ID (if provided, will use last-n interactions from history)
            song_ids: Optional seed song IDs (if not provided and user_id given, uses last-n)
            n: Number of recommendations
            exclude_song_ids: Songs to exclude
            return_details: If True, return detailed information
            use_last_n: Override default use_last_n setting (default: self.use_last_n)
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        # Determine if we should use last-n
        use_last_n_flag = use_last_n if use_last_n is not None else self.use_last_n
        
        # If user_id provided and use_last_n, get last-n interactions from history
        if user_id and use_last_n_flag:
            interactions = self.history_manager.get_user_interactions(
                user_id,
                use_weekly_mixing=True  # 50/50 mix if has this week's interactions
            )
            if interactions:
                # Use last-n interactions as seed songs
                song_ids = interactions
                if return_details:
                    print(f"📊 Using {len(interactions)} last-n interactions for user {user_id}")
            elif return_details:
                print(f"⚠️  No interactions found for user {user_id}")
        
        # Exclude seed songs from recommendations (self-similarity = 1 is not informative)
        seed_song_ids = set(song_ids) if song_ids else set()
        if exclude_song_ids:
            seed_song_ids.update(exclude_song_ids)
        
        # Get recommendations from each channel (RECALL STAGE)
        if return_details:
            print("\n" + "="*80)
            print("🔍 STAGE 1: RECALL - Candidate Retrieval")
            print("="*80)
        
        all_recommendations = {}
        
        # ========================================================================
        # STAGE 1: RECALL - Candidate Retrieval
        # ========================================================================
        
        # Channel 1: Item-based CF
        if return_details:
            print("\n[Recall Channel 1] Item-Based CF...")
        item_cf_recs = self.item_cf_recommender.recommend(
            user_id=user_id,
            song_ids=song_ids,
            n=n * 3,  # Get more candidates for recall stage
            exclude_song_ids=list(seed_song_ids),
            return_details=return_details,
            top_k_per_seed=5,  # Find top-5 most similar for each seed song
            use_last_n=False,  # Already handled above
        )
        item_cf_count = 0
        item_cf_valid = []  # Store valid (non-seed) recommendations for sorting
        for rec in item_cf_recs:
            song_id = rec['song_id']
            if song_id in seed_song_ids:
                continue  # Skip seed songs
            if song_id not in all_recommendations:
                all_recommendations[song_id] = {
                    'song_id': song_id,
                    'scores': defaultdict(float),
                    'metadata': rec,
                }
            all_recommendations[song_id]['scores']['item_cf'] = rec['score'] * self.item_cf_weight
            item_cf_count += 1
            item_cf_valid.append(rec)
        
        # Sort by score and get top 5
        item_cf_top5 = sorted(item_cf_valid, key=lambda x: x.get('score', 0), reverse=True)[:5]
        
        if return_details:
            print(f"  ✅ Retrieved {item_cf_count} candidates from Item-Based CF")
            if item_cf_top5:
                print(f"\n  📊 Top 5 Item-Based CF Recommendations:")
                self._display_channel_top5(item_cf_top5, "Item-Based CF")
        
        # Channel 2: User-based CF
        if self.use_user_cf and self.user_recommender:
            if return_details:
                print("\n[Recall Channel 2] User-Based CF...")
            user_recs = self.user_recommender.recommend(
                user_id=user_id,
                song_ids=song_ids,
                n=n * 3,  # Get more candidates for recall stage
                exclude_song_ids=list(seed_song_ids),
                return_details=return_details,
                use_last_n=False,  # User-based CF uses user_id directly
            )
            user_count = 0
            user_cf_valid = []  # Store valid (non-seed) recommendations for sorting
            for rec in user_recs:
                song_id = rec['song_id']
                if song_id in seed_song_ids:
                    continue  # Skip seed songs
                if song_id not in all_recommendations:
                    all_recommendations[song_id] = {
                        'song_id': song_id,
                        'scores': defaultdict(float),
                        'metadata': rec,
                    }
                all_recommendations[song_id]['scores']['user_cf'] = rec['score'] * self.user_cf_weight
                user_count += 1
                user_cf_valid.append(rec)
            
            # Sort by score and get top 5
            user_cf_top5 = sorted(user_cf_valid, key=lambda x: x.get('score', 0), reverse=True)[:5]
            
            if return_details:
                print(f"  ✅ Retrieved {user_count} candidates from User-Based CF")
                if user_cf_top5:
                    print(f"\n  📊 Top 5 User-Based CF Recommendations:")
                    self._display_channel_top5(user_cf_top5, "User-Based CF")
        
        # Channel 3: Two-Tower Content Retrieval (CLAP-based)
        if self.use_two_tower and self.two_tower_recommender:
            if return_details:
                print("\n[Recall Channel 3] Two-Tower Content Retrieval (CLAP-based)...")
            two_tower_recs = self.two_tower_recommender.recommend(
                user_id=user_id,
                song_ids=song_ids,
                n=n * 3,  # Get more candidates for recall stage
                exclude_song_ids=list(seed_song_ids),
                return_details=return_details,
                use_last_n=use_last_n_flag,
            )
            two_tower_count = 0
            two_tower_valid = []  # Store valid (non-seed) recommendations for sorting
            for rec in two_tower_recs:
                song_id = rec['song_id']
                if song_id in seed_song_ids:
                    continue  # Skip seed songs
                if song_id not in all_recommendations:
                    all_recommendations[song_id] = {
                        'song_id': song_id,
                        'scores': defaultdict(float),
                        'metadata': rec,
                    }
                all_recommendations[song_id]['scores']['two_tower'] = rec['score'] * self.two_tower_weight
                two_tower_count += 1
                two_tower_valid.append(rec)
            
            # Sort by score and get top 5
            two_tower_top5 = sorted(two_tower_valid, key=lambda x: x.get('score', 0), reverse=True)[:5]
            
            if return_details:
                print(f"  ✅ Retrieved {two_tower_count} candidates from Two-Tower")
                if two_tower_top5:
                    print(f"\n  📊 Top 5 Two-Tower Recommendations:")
                    self._display_channel_top5(two_tower_top5, "Two-Tower")
        
        # Channel 4: Prompt-based similarity (creative intent matching)
        if self.use_prompt_based and self.prompt_recommender:
            if return_details:
                print("\n[Recall Channel 4] Prompt-Based Similarity (Creative Intent Matching)...")
                print("  → Finding songs with prompts similar to user's listening history...")
            prompt_recs = self.prompt_recommender.recommend(
                user_id=user_id,
                song_ids=song_ids,
                n=n * 3,  # Get more candidates for recall stage
                exclude_song_ids=list(seed_song_ids),
                return_details=return_details,
                use_last_n=use_last_n_flag,
                top_k_per_seed=5,  # Find top-5 most similar for each seed song (matching Item CF)
                exclude_same_artist=kwargs.get('exclude_same_artist', True),  # Default: exclude songs from seed artists
            )
            prompt_count = 0
            prompt_valid = []  # Store valid (non-seed) recommendations for sorting
            for rec in prompt_recs:
                song_id = rec['song_id']
                if song_id in seed_song_ids:
                    continue  # Skip seed songs
                if song_id not in all_recommendations:
                    all_recommendations[song_id] = {
                        'song_id': song_id,
                        'scores': defaultdict(float),
                        'metadata': rec,
                    }
                all_recommendations[song_id]['scores']['prompt_based'] = rec['score'] * self.prompt_weight
                prompt_count += 1
                prompt_valid.append(rec)
            
            # Sort by score and get top 5
            prompt_top5 = sorted(prompt_valid, key=lambda x: x.get('score', 0), reverse=True)[:5]
            
            if return_details:
                print(f"  ✅ Retrieved {prompt_count} candidates from Prompt-Based")
                if prompt_top5:
                    print(f"\n  📊 Top 5 Prompt-Based Recommendations:")
                    self._display_channel_top5(prompt_top5, "Prompt-Based")
        
        if return_details:
            print(f"\n📊 Recall Summary: {len(all_recommendations)} unique candidates retrieved")
        
        # ========================================================================
        # STAGE 2: COARSE RANKING - Quality Filter
        # ========================================================================
        if return_details:
            print("\n" + "="*80)
            print("⚖️  STAGE 2: COARSE RANKING - Quality Filter")
            print("="*80)
        
        # Apply quality filter
        if self.use_quality_filter:
            quality_scores = self.quality_filter.score_songs(list(all_recommendations.keys()))
            filtered_count = 0
            quality_score_list = []
            for song_id, quality_score in quality_scores.items():
                quality_score_list.append(quality_score)
                # Filter out low-quality songs
                if quality_score < self.quality_threshold:
                    # Mark for removal
                    all_recommendations[song_id]['scores']['quality_filtered'] = True
                    filtered_count += 1
            if return_details:
                print(f"  ✅ Applied quality filter: {filtered_count} candidates filtered out")
                if quality_score_list:
                    import numpy as np
                    quality_scores_array = np.array(quality_score_list)
                    print(f"\n  📊 Quality Filter Statistics:")
                    print(f"     Threshold: {self.quality_threshold:.4f}")
                    print(f"     Mean score: {quality_scores_array.mean():.4f}")
                    print(f"     Median score: {np.median(quality_scores_array):.4f}")
                    print(f"     Min score: {quality_scores_array.min():.4f}")
                    print(f"     Max score: {quality_scores_array.max():.4f}")
                    print(f"     Std dev: {quality_scores_array.std():.4f}")
                    passed_count = np.sum(quality_scores_array >= self.quality_threshold)
                    print(f"     Passed: {passed_count}/{len(quality_scores_array)} ({passed_count/len(quality_scores_array)*100:.1f}%)")
                    print(f"     Filtered: {filtered_count}/{len(quality_scores_array)} ({filtered_count/len(quality_scores_array)*100:.1f}%)")
        
        # Combine recall scores and deduplicate
        coarse_scores = []
        seen_song_ids = set()  # Deduplication
        
        for song_id, data in all_recommendations.items():
            # Skip duplicates
            if song_id in seen_song_ids:
                continue
            seen_song_ids.add(song_id)
            
            # Skip seed songs
            if song_id in seed_song_ids:
                continue
            
            # Skip if quality filter rejected
            if self.use_quality_filter and data['scores'].get('quality_filtered', False):
                continue
            
            # Combine recall scores (Stage 1)
            recall_score = sum(data['scores'].values())
            
            coarse_scores.append({
                'song_id': song_id,
                'recall_score': recall_score,
                'channel_scores': dict(data['scores']),
                **data['metadata'],
            })
        
        # Sort by recall score
        coarse_scores.sort(key=lambda x: x['recall_score'], reverse=True)
        
        if return_details:
            print(f"  ✅ Coarse Ranking: {len(coarse_scores)} candidates after quality filtering and deduplication")
        
        # ========================================================================
        # STAGE 3: FINE RANKING - CTR Prediction
        # ========================================================================
        if return_details:
            print("\n" + "="*80)
            print("🎯 STAGE 3: FINE RANKING - CTR Prediction")
            print("="*80)
        
        # Channel 6: DIN with attention (CTR prediction)
        if return_details:
            print("\n[Fine Ranking Channel 6] DIN with Attention (CTR Prediction)...")
        
        # Get top candidates for fine ranking
        top_candidates_for_fine = coarse_scores[:min(len(coarse_scores), n * 5)]  # Top 5n for fine ranking
        top_candidate_ids = [c['song_id'] for c in top_candidates_for_fine]
        
        # Predict CTR using DIN for all candidates
        if self.din_ranker and self.din_ranker.is_fitted:
            if hasattr(self.din_ranker, 'model_trained') and self.din_ranker.model_trained:
                din_ctr_scores = self.din_ranker.predict_ctr(
                    user_id=user_id,
                    candidate_song_ids=top_candidate_ids,
                    max_history=50
                )
            else:
                # DIN not trained: use recall scores as fallback
                if return_details:
                    print(f"  ⚠️  DIN model not trained, using recall scores as fallback")
                din_ctr_scores = {c['song_id']: c['recall_score'] for c in top_candidates_for_fine}
        else:
            # Fallback: use recall scores
            din_ctr_scores = {c['song_id']: c['recall_score'] for c in top_candidates_for_fine}
        
        # Apply fine ranking scores
        fine_scores = []
        for candidate in top_candidates_for_fine:
            song_id = candidate['song_id']
            
            # DIN CTR score (only channel in fine ranking now)
            ctr_score = din_ctr_scores.get(song_id, 0.5)  # Default CTR if not predicted
            din_score = ctr_score * self.din_weight
            
            # Fine score is just DIN score
            fine_score = din_score
            
            # Track which stage and channel contributed
            stage_channel_info = {
                'recall_stage': {
                    'channels': list(candidate.get('channel_scores', {}).keys()),
                    'score': candidate['recall_score'],
                },
                'coarse_ranking_stage': {
                    'passed_quality_filter': True,
                },
                'fine_ranking_stage': {
                    'din_ctr_score': ctr_score,
                    'din_score': din_score,
                    'total_score': fine_score,
                },
            }
            
            fine_scores.append({
                **candidate,
                'fine_score': fine_score,
                'din_ctr_score': ctr_score,
                'din_score': din_score,
                'stage_channel_info': stage_channel_info,
            })
        
        # Sort by fine score
        fine_scores.sort(key=lambda x: x['fine_score'], reverse=True)
        
        if return_details:
            print(f"  ✅ Fine Ranking: {len(fine_scores)} candidates scored with DIN (CTR prediction)")
        
        # ========================================================================
        # STAGE 4: RE-RANKING - Music Flamingo
        # ========================================================================
        if self.use_music_flamingo and self.music_flamingo_scorer is not None:
            if return_details:
                print("\n" + "="*80)
                print("🎨 STAGE 4: RE-RANKING - Music Flamingo")
                print("="*80)
            # Re-rank top candidates
            top_k_for_rerank = min(max(2 * n, 50), len(fine_scores))
            final_scores = self._rerank_with_music_flamingo(
                fine_scores[:top_k_for_rerank],
                top_k=top_k_for_rerank
            )
            if return_details:
                print(f"  ✅ Re-ranked {len(final_scores)} candidates with Music Flamingo")
        else:
            final_scores = fine_scores
        
        # Take top N
        final_scores = final_scores[:n]
        
        # Format final scores (use fine_score as main score) and add stage/channel info
        for rec in final_scores:
            rec['score'] = rec.get('fine_score', rec.get('recall_score', 0.0))
            
            # Add stage and channel information
            if 'stage_channel_info' in rec:
                stage_info = rec['stage_channel_info']
                # Determine primary contributing channel from recall stage
                if 'channel_scores' in rec:
                    channel_scores = rec['channel_scores']
                    # Remove quality_filtered if present
                    channel_scores_clean = {k: v for k, v in channel_scores.items() if k != 'quality_filtered'}
                    if channel_scores_clean:
                        primary_channel = max(channel_scores_clean.items(), key=lambda x: x[1])[0]
                        rec['primary_recall_channel'] = primary_channel
                        rec['stage'] = 'Stage 1 (Recall)'
                        rec['channel'] = f"Channel {self._get_channel_number(primary_channel)}"
            
            # Add Suno URL
            if 'song_id' in rec:
                rec['suno_url'] = f"https://suno.com/song/{rec['song_id']}"
        
        if return_details:
            print(f"\n✅ Final Recommendations: {len(final_scores)} songs selected")
        
        return final_scores
    
    def _get_channel_number(self, channel_name: str) -> str:
        """Map channel name to channel number"""
        channel_map = {
            'item_cf': '1 (Item-Based CF)',
            'user_cf': '2 (User-Based CF)',
            'two_tower': '3 (Two-Tower)',
            'quality': '4 (Quality Filter)',
            'din': '5 (DIN)',
            'prompt': '6 (Prompt-Based)',
        }
        return channel_map.get(channel_name, channel_name)

    def _rerank_with_music_flamingo(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Use Music Flamingo only on a small number of top candidates for reranking.
        
        This avoids running the heavy model on the full catalog and keeps it in the
        late-stage ranking / reranking phase.
        """
        if self.music_flamingo_scorer is None:
            return candidates

        # Lazy initialization in case transformers / model loading fails at fit time
        if not self.music_flamingo_scorer.is_initialized:
            if not self.music_flamingo_scorer.initialize():
                # Fallback: keep original ranking
                return candidates

        # Only rerank the top_k items
        top_candidates = candidates[:top_k]
        remaining = candidates[top_k:]

        # Collect audio URLs and IDs for scoring
        audio_paths = []
        audio_ids = []
        for rec in top_candidates:
            song_id = rec['song_id']
            # 从保存的 songs_df 中拿 audio_url，如果没有则跳过 Flamingo
            song_row = None
            if self.songs_df is not None:
                rows = self.songs_df[self.songs_df['song_id'] == song_id]
                if len(rows) > 0:
                    song_row = rows.iloc[0]

            audio_url = song_row.get('audio_url') if song_row is not None else None
            if audio_url:
                audio_paths.append(audio_url)
                audio_ids.append(song_id)

        if not audio_paths:
            # No audio URLs available; keep original order
            return candidates

        # 批量调用 Music Flamingo 打分
        try:
            scores_dict = self.music_flamingo_scorer.score_audio_batch(
                audio_paths=audio_paths,
                audio_ids=audio_ids,
                cache_dir="runtime_data/music_flamingo_scores",
                show_progress=False,
            )
        except Exception:
            # 出错时不影响主流程，直接返回原排列
            return candidates

        # 映射到候选，并与原有 hybrid score 融合
        score_map = {}
        for sid, score_components in scores_dict.items():
            overall = self.music_flamingo_scorer.get_overall_quality_score(score_components)
            score_map[sid] = overall

        reranked_top = []
        for rec in top_candidates:
            sid = rec['song_id']
            flamingo_score = score_map.get(sid)
            if flamingo_score is not None:
                # 简单线性融合：主要还是用原 hybrid 分数，Flamingo 作为精排微调
                base_score = rec['score']
                final_score = 0.8 * base_score + 0.2 * flamingo_score
                rec = dict(rec)  # avoid mutating in-place unexpectedly
                rec['score'] = final_score
                rec.setdefault('channel_scores', {})
                rec['channel_scores']['music_flamingo'] = flamingo_score
            reranked_top.append(rec)

        # 对 top_k 内部按新的 score 排序，然后和剩余候选拼接
        reranked_top.sort(key=lambda x: x['score'], reverse=True)
        return reranked_top + remaining
    
    def get_similar_songs(
        self,
        song_id: str,
        n: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get similar songs using hybrid approach"""
        # Get recommendations from each recall channel
        all_similar = {}
        
        # Channel 1: Item-based CF
        item_cf_similar = self.item_cf_recommender.get_similar_songs(song_id, n)
        for rec in item_cf_similar:
            song_id_sim = rec['song_id']
            if song_id_sim not in all_similar:
                all_similar[song_id_sim] = {'scores': defaultdict(float), 'metadata': rec}
            all_similar[song_id_sim]['scores']['item_cf'] = rec['score'] * self.item_cf_weight
        
        # Channel 2: User-based CF
        if self.use_user_cf and self.user_recommender:
            user_similar = self.user_recommender.get_similar_songs(song_id, n)
            for rec in user_similar:
                song_id_sim = rec['song_id']
                if song_id_sim not in all_similar:
                    all_similar[song_id_sim] = {'scores': defaultdict(float), 'metadata': rec}
                all_similar[song_id_sim]['scores']['user_cf'] = rec['score'] * self.user_cf_weight
        
        # Channel 3: Two-tower
        if self.use_two_tower and self.two_tower_recommender:
            two_tower_similar = self.two_tower_recommender.get_similar_songs(song_id, n)
            for rec in two_tower_similar:
                song_id_sim = rec['song_id']
                if song_id_sim not in all_similar:
                    all_similar[song_id_sim] = {'scores': defaultdict(float), 'metadata': rec}
                all_similar[song_id_sim]['scores']['two_tower'] = rec['score'] * self.two_tower_weight
        
        # Combine scores
        final_scores = []
        for song_id_sim, data in all_similar.items():
            total_score = sum(data['scores'].values())
            final_scores.append({
                'song_id': song_id_sim,
                'score': total_score,
                'channel_scores': dict(data['scores']),
                **data['metadata'],
            })
        
        final_scores.sort(key=lambda x: x['score'], reverse=True)
        return final_scores[:n]
    
    def save(self, path: str):
        """Save the hybrid recommender"""
        # Save component recommenders
        base_path = path.replace('.pkl', '')
        
        # Stage 1 (Recall)
        self.item_cf_recommender.save(f"{base_path}_item_cf.pkl")
        if self.user_recommender:
            self.user_recommender.save(f"{base_path}_user.pkl")
        if self.two_tower_recommender:
            # Two-tower model is saved separately (model_checkpoints/two_tower.pt)
            pass
        
        # Stage 2 (Coarse Ranking)
        self.quality_filter.save(f"{base_path}_quality.pkl")
        
        # Stage 3 (Fine Ranking)
        self.prompt_recommender.save(f"{base_path}_prompt.pkl")
        # DIN model is saved separately (model_checkpoints/din_ranker.pt)
        
        # Save main config
        joblib.dump({
            'name': self.name,
            # Stage 1 weights
            'item_cf_weight': self.item_cf_weight,
            'user_cf_weight': self.user_cf_weight,
            'two_tower_weight': self.two_tower_weight,
            # Stage 2
            'quality_threshold': self.quality_threshold,
            'use_quality_filter': self.use_quality_filter,
            # Stage 3 weights
            'din_weight': self.din_weight,
            'prompt_weight': self.prompt_weight,
            'din_model_path': self.din_model_path,  # Save DIN model path
            # Component toggles
            'use_user_cf': self.use_user_cf,
            'use_two_tower': self.use_two_tower,
            'use_last_n': self.use_last_n,
            'history_file': str(self.history_manager.history_file) if self.history_manager else None,
            'songs_df': self.songs_df,
            'base_path': base_path,
        }, path)
    
    @classmethod
    def load(cls, path: str, history_manager: Optional[UserHistoryManager] = None):
        """
        Load the hybrid recommender
        
        Args:
            path: Path to saved model
            history_manager: Optional UserHistoryManager (if None, will create from saved config)
        """
        data = joblib.load(path)
        
        # Get history manager
        if history_manager is None:
            history_file = data.get('history_file', 'runtime_data/user_history.json')
            history_manager = UserHistoryManager(history_file=history_file)
        
        # Handle backward compatibility and new format
        if 'two_tower_weight' in data:
            # New four-stage format
            recommender = cls(
                item_cf_weight=data['item_cf_weight'],
                user_cf_weight=data.get('user_cf_weight', 0.30),
                two_tower_weight=data.get('two_tower_weight', 0.40),
                quality_threshold=data['quality_threshold'],
                use_quality_filter=data['use_quality_filter'],
                din_weight=data.get('din_weight', 0.70),
                prompt_weight=data['prompt_weight'],
                din_model_path=data.get('din_model_path', 'model_checkpoints/din_ranker.pt'),  # Load DIN model path
                use_user_cf=data.get('use_user_cf', True),
                use_two_tower=data.get('use_two_tower', True),
                history_manager=history_manager,
                use_last_n=data.get('use_last_n', True),
            )
        elif 'item_content_weight' in data:
            # Old format - backward compatibility
            recommender = cls(
                item_cf_weight=data['item_cf_weight'],
                user_cf_weight=data.get('user_weight', 0.30),
                two_tower_weight=0.40,  # Default for old format
                quality_threshold=data['quality_threshold'],
                use_quality_filter=data['use_quality_filter'],
                din_weight=0.70,  # Default
                prompt_weight=data['prompt_weight'],
                din_model_path=data.get('din_model_path', 'model_checkpoints/din_ranker.pt'),  # Load DIN model path
                use_user_cf=data.get('use_user_cf', True),
                history_manager=history_manager,
                use_last_n=data.get('use_last_n', True),
            )
        else:
            # Very old format - backward compatibility
            recommender = cls(
                item_cf_weight=data.get('item_weight', 0.30),
                user_cf_weight=data.get('user_weight', 0.30),
                din_model_path=data.get('din_model_path', 'model_checkpoints/din_ranker.pt'),  # Load DIN model path
                two_tower_weight=0.40,
                quality_threshold=data['quality_threshold'],
                use_quality_filter=data['use_quality_filter'],
                din_weight=0.70,
                prompt_weight=data['prompt_weight'],
                use_user_cf=True,
                history_manager=history_manager,
                use_last_n=data.get('use_last_n', True),
            )
        
        base_path = data['base_path']
        recommender.songs_df = data['songs_df']
        
        # Load component recommenders
        # Stage 1 (Recall)
        recommender.item_cf_recommender = ItemBasedCFRecommender.load(f"{base_path}_item_cf.pkl")
        if recommender.use_user_cf:
            recommender.user_recommender = UserBasedRecommender.load(f"{base_path}_user.pkl")
        # Two-tower model is loaded separately (model_checkpoints/two_tower.pt)
        
        # Stage 2 (Coarse Ranking)
        recommender.quality_filter = QualityFilter.load(f"{base_path}_quality.pkl")
        
        # Stage 3 (Fine Ranking)
        recommender.prompt_recommender = PromptBasedRecommender.load(f"{base_path}_prompt.pkl")
        
        # Load DIN ranker if model path is provided
        if recommender.din_model_path:
            print(f"  → Loading DIN ranker from {recommender.din_model_path}...")
            recommender.din_ranker = DINRanker(
                model_path=recommender.din_model_path,
                clap_embeddings_path=recommender.two_tower_clap_path,
                device=recommender.device
            )
            # Fit DIN ranker with loaded songs_df and user history from history_manager
            user_history = {}
            if recommender.history_manager:
                # Build user_history in the same format as fit() method
                for uid in recommender.history_manager.get_all_user_ids():
                    interactions = recommender.history_manager.get_user_interactions(
                        uid, 
                        use_weekly_mixing=False  # Get all for model training
                    )
                    user_history[uid] = interactions
            recommender.din_ranker.fit(recommender.songs_df, user_history=user_history)
            print("  ✅ DIN ranker loaded")
        else:
            recommender.din_ranker = None
        
        recommender.is_fitted = True
        
        return recommender

