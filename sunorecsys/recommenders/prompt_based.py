"""Prompt-based similarity recommender using CLAP text embeddings.

Leverages CLAP's aligned text-audio embedding space to enable direct similarity
search between user prompts and audio tracks, capturing creative intent unique
to AI music generation platforms.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from annoy import AnnoyIndex
import joblib
from pathlib import Path
import hashlib

from .base import BaseRecommender
from ..utils.clap_embeddings import CLAPTextEmbedder, DEFAULT_CLAP_MODEL_PATH


class PromptBasedRecommender(BaseRecommender):
    """
    Recommender based on generation prompt similarity using CLAP text embeddings.
    
    This is unique to music generation models - users who like songs
    generated from similar prompts may enjoy similar music.
    
    Uses CLAP's aligned text-audio embedding space, enabling direct similarity
    between prompts and audio content without cross-modal alignment.
    """
    
    def __init__(
        self,
        use_clap: bool = True,  # Use CLAP text embeddings (aligned with audio)
        embedding_model: str = 'all-MiniLM-L6-v2',  # Fallback if CLAP not available
        clap_model_path: Optional[str] = None,
        clap_cache_dir: str = "runtime_data/audio_cache",
        n_trees: int = 50,
    ):
        super().__init__("PromptBased")
        self.use_clap = use_clap
        self.embedding_model = embedding_model
        self.n_trees = n_trees
        
        self.prompt_embedder = None
        self.clap_text_embedder = None
        self.songs_df = None
        self.prompt_embeddings = None
        self.song_index = None
        self.song_id_to_idx = {}
        self.idx_to_song_id = {}
        self.embedding_dim = None
        
        # CLAP settings
        self.clap_model_path = clap_model_path
        self.clap_cache_dir = clap_cache_dir
    
    def fit(self, songs_df: pd.DataFrame, user_history: Optional[Dict[str, List[str]]] = None, **kwargs):
        """
        Fit the prompt-based recommender
        
        Args:
            user_history: Optional dict mapping user_id to list of song_ids (for last-n support)
        """
        self.songs_df = songs_df.copy()
        self.user_history = user_history or {}  # Store for last-n support
        
        # Initialize embedder (CLAP or fallback)
        if self.use_clap:
            print(f"  → Initializing CLAP text embedder (aligned with audio embeddings)...")
            try:
                # Get model path (use default if not provided)
                clap_model_path = self.clap_model_path or kwargs.get('clap_model_path') or DEFAULT_CLAP_MODEL_PATH
                
                # Create CLAP text embedder (shares model via module-level cache)
                self.clap_text_embedder = CLAPTextEmbedder(
                    model_path=clap_model_path,
                    cache_dir=self.clap_cache_dir,
                    device=kwargs.get('device')
                )
                if not self.clap_text_embedder.initialize_model():
                    print(f"  ⚠️  CLAP model initialization failed, falling back to text embeddings")
                    self.use_clap = False
                else:
                    # CLAP embedding dimension (typically 512)
                    # We'll determine this from the first embedding
                    self.embedding_dim = None  # Will be determined from first embedding
                    print(f"  ✅ CLAP text embedder ready (aligned with audio space)")
            except Exception as e:
                print(f"  ⚠️  CLAP not available ({e}), falling back to text embeddings")
                self.use_clap = False
        
        if not self.use_clap:
            # sentence-transformers fallback removed - CLAP is required
            raise RuntimeError(
                "CLAP embeddings are required. sentence-transformers fallback has been removed. "
                "Please ensure CLAP model is available."
            )
        
        # Check for cached prompt embeddings
        cache_dir = kwargs.get('cache_dir', 'runtime_data/cache')
        use_cache = kwargs.get('use_cache', True)
        cache_path = None
        song_ids_sorted = sorted(songs_df['song_id'].tolist())
        
        if use_cache:
            # Generate cache key from song_ids, prompts, and embedder type
            prompts_str = '|'.join(songs_df['prompt'].fillna('').tolist())
            embedder_id = 'CLAP' if self.use_clap else self.embedding_model
            cache_key = hashlib.md5(
                (','.join(song_ids_sorted) + '|' + prompts_str + '|' + embedder_id).encode()
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
            
            if self.use_clap and self.clap_text_embedder:
                # Use CLAP text embeddings (aligned with audio)
                print(f"     Using CLAP text embeddings (aligned with audio space)...")
                clap_text_embeddings = self.clap_text_embedder.embed_texts(
                    prompts,
                    use_cache=True,
                    show_progress=True
                )
                
                # Convert to numpy array (aligned with songs_df order)
                prompt_emb_list = []
                for prompt in prompts:
                    if prompt in clap_text_embeddings:
                        prompt_emb_list.append(clap_text_embeddings[prompt])
                    else:
                        # Fallback: use zero vector if embedding failed
                        if self.embedding_dim is None:
                            # Try to get dimension from first successful embedding
                            if clap_text_embeddings:
                                first_emb = next(iter(clap_text_embeddings.values()))
                                self.embedding_dim = len(first_emb)
                            else:
                                self.embedding_dim = 512  # Default CLAP dimension
                        prompt_emb_list.append(np.zeros(self.embedding_dim))
                
                self.prompt_embeddings = np.array(prompt_emb_list)
                
                # Set embedding_dim if not set
                if self.embedding_dim is None and len(self.prompt_embeddings) > 0:
                    self.embedding_dim = self.prompt_embeddings.shape[1]
            else:
                # Fallback to text embeddings
                self.prompt_embeddings = self.prompt_embedder.embed_prompts(prompts, show_progress=True)
                if self.embedding_dim is None:
                    self.embedding_dim = self.prompt_embeddings.shape[1]
            
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
        if self.embedding_dim is None:
            self.embedding_dim = self.prompt_embeddings.shape[1]
        self.song_index = AnnoyIndex(self.embedding_dim, 'angular')
        
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
        top_k_per_seed: int = 5,
        exclude_same_artist: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Recommend songs based on prompt similarity to seed songs (Discover Weekly style)
        
        Args:
            use_last_n: If True and user_id provided, use last-n interactions from user history
            top_k_per_seed: For each seed song, find top-k most similar (then dedup and aggregate)
            exclude_same_artist: If True, exclude songs from artists that already have a song in the results
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        # If user_id provided and use_last_n, get last-n interactions
        if user_id and use_last_n and hasattr(self, 'user_history') and self.user_history:
            if user_id in self.user_history:
                last_n_interactions = self.user_history[user_id][-n:] if len(self.user_history[user_id]) > n else self.user_history[user_id]
                if last_n_interactions:
                    # Handle both formats: list of song_ids (strings) or list of interaction dicts
                    if isinstance(last_n_interactions[0], dict) and 'song_id' in last_n_interactions[0]:
                        # Extract song_ids from interaction dicts
                        song_ids = [item['song_id'] for item in last_n_interactions]
                    else:
                        # Already a list of song_ids (strings)
                        song_ids = last_n_interactions
        
        if not song_ids:
            return self._get_popular_songs(n)
        
        # Get seed song indices (deduplicate to avoid processing same seed song multiple times)
        # If user listened to same song multiple times, we only process it once as a seed
        unique_song_ids = list(dict.fromkeys(song_ids))  # Preserve order, remove duplicates
        seed_indices = [self.song_id_to_idx[sid] for sid in unique_song_ids if sid in self.song_id_to_idx]
        
        if not seed_indices:
            return self._get_popular_songs(n)
        
        # If exclude_same_artist is True, get artist IDs from seed songs to exclude them
        seed_artist_ids = set()
        if exclude_same_artist:
            for seed_song_id in unique_song_ids:
                seed_rows = self.songs_df[self.songs_df['song_id'] == seed_song_id]
                if not seed_rows.empty:
                    artist_id = seed_rows.iloc[0].get('user_id')  # user_id represents artist/creator in Suno
                    if artist_id:
                        seed_artist_ids.add(artist_id)
        
        # Discover Weekly style: For each seed song, find top-k most similar
        # Then aggregate and deduplicate by prompt (early deduplication)
        all_candidates = {}  # song_idx -> list of (similarity_score, seed_song_id)
        
        # Helper function to normalize prompt for deduplication
        def normalize_prompt_for_dedup(prompt: str) -> str:
            """Normalize prompt for deduplication - removes style/genre variations"""
            if not prompt:
                return ''
            import re
            # Remove leading/trailing whitespace
            normalized = prompt.strip()
            # Normalize all whitespace (multiple spaces/newlines/tabs -> single space)
            normalized = re.sub(r'\s+', ' ', normalized)
            # Remove leading/trailing whitespace again after normalization
            normalized = normalized.strip()
            return normalized
        
        exclude_indices = set()
        if exclude_song_ids:
            exclude_indices = {self.song_id_to_idx[sid] for sid in exclude_song_ids if sid in self.song_id_to_idx}
        
        # Track prompts seen across all candidates (prompt -> (candidate_idx, best_score))
        # Used for early deduplication during aggregation
        prompt_to_best_candidate = {}  # normalized_prompt -> (candidate_idx, best_score)
        
        # For each seed song, find top-k most similar songs
        for seed_idx in seed_indices:
            seed_song_id = self.idx_to_song_id[seed_idx]
            
            # Find top-k most similar songs for this seed song (using Annoy index)
            similar_indices, distances = self.song_index.get_nns_by_item(
                seed_idx, 
                top_k_per_seed + 1,  # +1 to exclude seed itself
                include_distances=True
            )
            
            # Track prompts seen for this seed (to avoid duplicates within same seed's results)
            seen_prompts_for_seed = set()  # Track normalized prompts we've seen for this seed
            
            # Filter out seed song itself and excluded songs
            for similar_idx, distance in zip(similar_indices, distances):
                if similar_idx == seed_idx or similar_idx in exclude_indices:
                    continue
                
                similar_song_id = self.idx_to_song_id[similar_idx]
                if similar_song_id in song_ids:  # Skip other seed songs
                    continue
                
                # Skip if exclude_same_artist is True and this candidate is from a seed artist
                if exclude_same_artist and seed_artist_ids:
                    similar_rows = self.songs_df[self.songs_df['song_id'] == similar_song_id]
                    if not similar_rows.empty:
                        candidate_artist = similar_rows.iloc[0].get('user_id')
                        if candidate_artist and candidate_artist in seed_artist_ids:
                            continue  # Skip songs from seed artists
                
                # Get prompt for this candidate (early deduplication)
                similar_rows = self.songs_df[self.songs_df['song_id'] == similar_song_id]
                if similar_rows.empty:
                    continue
                candidate_prompt = similar_rows.iloc[0].get('prompt', '')
                normalized_prompt = normalize_prompt_for_dedup(candidate_prompt)
                
                # For this seed, skip if we've already seen this prompt
                # (since results are sorted by similarity descending, first occurrence has highest score)
                if normalized_prompt in seen_prompts_for_seed:
                    continue  # Skip duplicate prompt within this seed's results
                
                # Convert distance to similarity score (1 - distance for angular distance)
                similarity_score = 1.0 - distance
                
                if similarity_score > 0:  # Only positive similarities
                    seen_prompts_for_seed.add(normalized_prompt)
                    
                    if similar_idx not in all_candidates:
                        all_candidates[similar_idx] = []
                    # Store: (similarity_score, seed_song_id)
                    all_candidates[similar_idx].append((similarity_score, seed_song_id))
        
        # Check if we found any candidates
        if len(all_candidates) == 0:
            if return_details:
                print(f"⚠️  Prompt-Based: No similar songs found for seed songs. Falling back to popular songs")
            return self._get_popular_songs(n)
        
        # Aggregate scores: sum similarities for items recommended by multiple seed songs
        # Formula: score = Σ similarity(seed_j, candidate) for all seed songs j that recommend this candidate
        # At the same time, deduplicate by prompt - keep only the best-scoring candidate for each prompt
        song_scores = {}
        prompt_to_candidate = {}  # normalized_prompt -> (candidate_idx, total_score)
        
        for candidate_idx, similarities_list in all_candidates.items():
            # Sum of similarities (similar to Item CF formula)
            total_score = sum([s[0] for s in similarities_list])
            
            # Get prompt for deduplication
            candidate_song_id = self.idx_to_song_id[candidate_idx]
            candidate_rows = self.songs_df[self.songs_df['song_id'] == candidate_song_id]
            if candidate_rows.empty:
                continue
            candidate_prompt = candidate_rows.iloc[0].get('prompt', '')
            normalized_prompt = normalize_prompt_for_dedup(candidate_prompt)
            
            # Skip empty prompts
            if not normalized_prompt:
                continue
            
            # If we've seen this prompt before, keep only the candidate with higher score
            # If scores are equal, keep the first one (earlier in iteration)
            if normalized_prompt in prompt_to_candidate:
                existing_idx, existing_score = prompt_to_candidate[normalized_prompt]
                if total_score > existing_score:
                    # Replace with better-scoring candidate
                    if existing_idx in song_scores:
                        del song_scores[existing_idx]
                    song_scores[candidate_idx] = total_score
                    prompt_to_candidate[normalized_prompt] = (candidate_idx, total_score)
                # else: keep existing candidate, skip this one (don't add to song_scores)
            else:
                # First time seeing this prompt
                song_scores[candidate_idx] = total_score
                prompt_to_candidate[normalized_prompt] = (candidate_idx, total_score)
        
        # Sort by score
        sorted_candidates = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter excluded songs and convert to results (similar to Item CF)
        # Also apply final prompt deduplication to ensure no duplicate prompts in final results
        exclude_set = set(exclude_song_ids or [])
        exclude_set.update(song_ids)
        exclude_indices = {self.song_id_to_idx[sid] for sid in exclude_set if sid in self.song_id_to_idx}
        
        results = []
        seen_prompts_final = set()  # Track prompts in final results
        used_seed_songs = set()  # Track seed songs that have already contributed a recommendation
        
        for candidate_idx, score in sorted_candidates:
            if candidate_idx in exclude_indices:
                continue
            
            candidate_song_id = self.idx_to_song_id[candidate_idx]
            
            # Get seed songs that recommended this candidate
            candidate_seed_songs = set()
            if candidate_idx in all_candidates:
                candidate_seed_songs = set([s[1] for s in all_candidates[candidate_idx]])
            
            # Skip if all seed songs that recommended this candidate have already been used
            # (Each seed song can only contribute one recommendation)
            if candidate_seed_songs and candidate_seed_songs.issubset(used_seed_songs):
                continue  # All seed songs for this candidate are already used
            
            # Final prompt deduplication check (safety net in case aggregation dedup missed something)
            candidate_rows = self.songs_df[self.songs_df['song_id'] == candidate_song_id]
            if not candidate_rows.empty:
                candidate_data = candidate_rows.iloc[0]
                candidate_prompt = candidate_data.get('prompt', '')
                normalized_prompt = normalize_prompt_for_dedup(candidate_prompt)
                
                # Skip if we've already added a song with this prompt
                if normalized_prompt and normalized_prompt in seen_prompts_final:
                    continue  # Skip duplicate prompt
                
                # Skip if exclude_same_artist is True and this candidate is from a seed artist
                if exclude_same_artist and seed_artist_ids:
                    candidate_artist = candidate_data.get('user_id')  # In Suno, user_id represents the artist/creator
                    if candidate_artist and candidate_artist in seed_artist_ids:
                        continue  # Skip songs from seed artists
                
                if normalized_prompt:
                    seen_prompts_final.add(normalized_prompt)
            
            # Add this recommendation and mark its seed songs as used
            results.append((candidate_song_id, score))
            used_seed_songs.update(candidate_seed_songs)
            
            # Take top N (matching Item CF and User CF behavior)
            if len(results) >= n:
                break
        
        # Prepare details if requested
        details = None
        if return_details:
            details = {}
            for song_id, score in results:
                song_idx = self.song_id_to_idx.get(song_id)
                if song_idx is not None:
                    # Find which seed songs recommended this candidate (deduplicate seed song IDs)
                    seed_songs_that_recommended = []
                    if song_idx in all_candidates:
                        # Extract unique seed song IDs (in case same seed appears multiple times)
                        seed_songs_that_recommended = list(set([s[1] for s in all_candidates[song_idx]]))
                    
                    details[song_id] = {
                        'prompt_similarity': float(score),
                        'seed_songs_count': len(seed_songs_that_recommended),
                        'seed_song_ids': seed_songs_that_recommended,  # List of unique seed song IDs that recommended this
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
            'use_clap': self.use_clap,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
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
            use_clap=data.get('use_clap', True),
            embedding_model=data.get('embedding_model', 'all-MiniLM-L6-v2'),
            n_trees=data['n_trees'],
        )
        
        recommender.embedding_dim = data.get('embedding_dim', recommender.prompt_embeddings.shape[1] if 'prompt_embeddings' in data else None)
        recommender.song_id_to_idx = data['song_id_to_idx']
        recommender.idx_to_song_id = data['idx_to_song_id']
        recommender.songs_df = data['songs_df']
        recommender.prompt_embeddings = data['prompt_embeddings']
        
        if Path(data['index_path']).exists():
            embedding_dim = data.get('embedding_dim', recommender.prompt_embeddings.shape[1])
            recommender.song_index = AnnoyIndex(embedding_dim, 'angular')
            recommender.song_index.load(data['index_path'])
        
        # Reinitialize embedder if needed
        if not recommender.use_clap:
            # sentence-transformers fallback removed - CLAP is required
            raise RuntimeError(
                "CLAP embeddings are required. sentence-transformers fallback has been removed. "
                "Please ensure CLAP model is available."
            )
        
        recommender.is_fitted = True
        
        return recommender

