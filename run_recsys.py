#!/usr/bin/env python3
"""
Test the complete recommendation system with detailed output.

This script:
1. Loads songs
2. Trains/fits the hybrid recommender (CLAP embeddings are checked during Two-Tower model loading)
3. Generates user-based recommendations (using last-n interactions)
4. Shows detailed output including stage, channel, and scores
5. Saves the trained model
"""

import json
import pandas as pd
from pathlib import Path

from sunorecsys.data.simulate_interactions import load_songs_from_aggregated_file
from sunorecsys.recommenders.hybrid import HybridRecommender


def main():
    print("=" * 80)
    print("🎵 Suno Recommendation System - Full Pipeline")
    print("=" * 80)
    
    # Step 1: Load songs
    print("\n[Step 1] Loading songs...")
    print("-" * 80)
    songs_file = "sunorecsys/data/curl/all_playlist_songs.json"
    if not Path(songs_file).exists():
        print(f"❌ Songs file not found: {songs_file}")
        return
    
    songs_df = load_songs_from_aggregated_file(songs_file, max_songs=None)
    # print(f"✅ Loaded {len(songs_df)} songs")
    # print(f"   - Unique users: {songs_df['user_id'].nunique()}")
    # print(f"   - Unique songs: {songs_df['song_id'].nunique()}")
    
    # Show sample song
    sample_song = songs_df.sample(1).iloc[0]
    print(f"\n📋 Sample song:")
    print(f"   Title: {sample_song.get('title', 'N/A')}")
    print(f"   URL: https://suno.com/song/{sample_song['song_id']}")
    print(f"   Genre: {sample_song.get('genre', 'N/A')}")
    print(f"   Tags: {', '.join(sample_song.get('tags', [])[:5])}")
    
    # Step 2: Fit recommender (CLAP embeddings are checked during Two-Tower model loading)
    print("\n[Step 2] Fitting hybrid recommender...")
    print("-" * 80)
    
    recommender = HybridRecommender(
        # Stage 1 (Recall) weights
        item_cf_weight=0.30,        # Channel 1: Item-based CF
        user_cf_weight=0.30,        # Channel 2: User-based CF
        two_tower_weight=0.40,      # Channel 3: Two-tower content retrieval (CLAP-based)
        # Stage 2 (Coarse Ranking) - Quality filter (no weight, just filtering)
        quality_threshold=0.3,
        use_quality_filter=True,
        # Stage 3 (Fine Ranking) weights
        din_weight=0.70,            # Channel 5: DIN with attention (placeholder)
        prompt_weight=0.30,         # Channel 6: Prompt-based (user exploration)
        # Stage 4 (Re-ranking) - Music Flamingo (optional)
        use_music_flamingo=False,
        # Component toggles
        use_user_cf=True,
        use_two_tower=True,         # Enable two-tower model
    )
    
    recommender.fit(
        songs_df,
        use_simulated_interactions=True,
        interaction_rate=0.15,
        item_cold_start_rate=0.05,
        single_user_item_rate=0.15,
        random_seed=42,
        cache_dir='data/cache',
        use_cache=True,
    )
    
    print("✅ Recommender fitted successfully")
    
    # Initialize user history from user-item matrix (includes both playlist and simulated interactions)
    print("\n📊 Initializing user history from user-item interaction matrix...")
    from datetime import datetime, timedelta
    import numpy as np
    history_manager = recommender.history_manager
    if history_manager:
        # Option 1: Use user-item matrix (includes playlist + simulated interactions)
        # This gives richer user history for recommendations
        use_matrix_interactions = True  # Set to False to use only playlist songs
        
        if use_matrix_interactions and recommender.item_cf_recommender and hasattr(recommender.item_cf_recommender, 'user_item_matrix'):
            # Extract interactions from user-item matrix (includes simulated)
            matrix = recommender.item_cf_recommender.user_item_matrix
            user_id_to_idx = recommender.item_cf_recommender.user_id_to_idx
            idx_to_user_id = recommender.item_cf_recommender.idx_to_user_id
            song_id_to_idx = recommender.item_cf_recommender.song_id_to_idx
            idx_to_song_id = recommender.item_cf_recommender.idx_to_song_id
            
            interactions_batch = []
            for user_idx in range(matrix.shape[0]):
                user_id = idx_to_user_id.get(user_idx)
                if not user_id:
                    continue
                
                # Get all songs this user interacted with (from matrix)
                user_row = matrix[user_idx].toarray().flatten()
                interacted_song_indices = np.where(user_row > 0)[0]
                
                # Add interactions with timestamps spread over the last few days
                for i, song_idx in enumerate(interacted_song_indices):
                    song_id = idx_to_song_id.get(song_idx)
                    if song_id:
                        timestamp = datetime.now() - timedelta(days=len(interacted_song_indices) - i)
                        interactions_batch.append((user_id, song_id, timestamp, 1.0, 'play'))
            
            history_manager.add_interactions_batch(interactions_batch)
            print(f"✅ Initialized history from matrix: {len(user_id_to_idx)} users ({len(interactions_batch)} interactions, includes simulated)")
        else:
            # Option 2: Use only playlist songs (original behavior)
            interactions_batch = []
            user_counts = songs_df.groupby('user_id').size()
            for user_id, count in user_counts.items():
                user_songs = songs_df[songs_df['user_id'] == user_id]['song_id'].tolist()
                # Add interactions with timestamps spread over the last few days
                for i, song_id in enumerate(user_songs):
                    timestamp = datetime.now() - timedelta(days=len(user_songs) - i)
                    interactions_batch.append((user_id, song_id, timestamp, 1.0, 'play'))
            
            history_manager.add_interactions_batch(interactions_batch)
            print(f"✅ Initialized history from playlists only: {len(user_counts)} users ({len(interactions_batch)} interactions)")
    
    # Step 3: User-based recommendations (using last-n interactions)
    print("\n[Step 3] User-based recommendations (using last-n interactions)")
    print("-" * 80)
    
    user_counts = songs_df.groupby('user_id').size()
    active_users = user_counts[user_counts >= 3].index.tolist()
    
    if active_users:
        user_id = active_users[0]
        # user_id = 'a4ba3b12-33df-4e4b-99e1-0d11469aa159' #MusicTek userID
        user_songs = songs_df[songs_df['user_id'] == user_id]['song_id'].tolist()
        print(f"User: {user_id} (has {len(user_songs)} songs)")
        
        user_recommendations = recommender.recommend(
            user_id=user_id,
            n=10,
            return_details=True,
        )
        
        print(f"\n✅ Generated {len(user_recommendations)} recommendations")
        print(f"\nTop 10 Recommendations:")
        for i, rec in enumerate(user_recommendations[:10], 1):
            print(f"\n{i:2d}. {rec.get('title', 'N/A')}")
            print(f"     Song ID: {rec.get('song_id', 'N/A')}")
            print(f"     Score: {rec.get('score', 0):.4f}")
            print(f"     URL: {rec.get('suno_url', 'N/A')}")
            
            # Show stage and channel information
            if 'stage' in rec and 'channel' in rec:
                print(f"     Stage: {rec['stage']}")
                print(f"     Primary Channel: {rec['channel']}")
            elif 'primary_recall_channel' in rec:
                print(f"     Primary Recall Channel: {rec['primary_recall_channel']}")
            
            # Show channel scores from recall stage
            if 'channel_scores' in rec:
                channel_scores = {k: v for k, v in rec['channel_scores'].items() if k != 'quality_filtered'}
                if channel_scores:
                    print(f"     Recall Channel Scores:")
                    for channel, score in sorted(channel_scores.items(), key=lambda x: x[1], reverse=True):
                        print(f"       - {channel}: {score:.4f}")
            
            # Show fine ranking scores
            if 'din_score' in rec or 'prompt_score' in rec:
                print(f"     Fine Ranking Scores:")
                if 'din_score' in rec:
                    print(f"       - DIN (placeholder): {rec['din_score']:.4f}")
                if 'prompt_score' in rec:
                    print(f"       - Prompt-based: {rec['prompt_score']:.4f}")
            
            # Show metadata (no lyrics)
            if 'genre' in rec:
                print(f"     Genre: {rec['genre']}")
            if 'tags' in rec and rec['tags']:
                print(f"     Tags: {', '.join(rec['tags'][:5])}")
    else:
        print("⚠️  No users with enough interactions for user-based recommendations")
    
    # Step 4: Statistics
    print("\n[Step 4] Statistics")
    print("-" * 80)
    
    if user_recommendations:
        scores = [r.get('score', 0) for r in user_recommendations]
        print(f"   Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"   Average score: {sum(scores) / len(scores):.4f}")
        
        # Count by stage and channel
        stage_counts = {}
        channel_counts = {}
        for rec in user_recommendations:
            if 'stage' in rec:
                stage_counts[rec['stage']] = stage_counts.get(rec['stage'], 0) + 1
            if 'primary_recall_channel' in rec:
                channel = rec['primary_recall_channel']
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
        
        if stage_counts:
            print(f"\n   Recommendations by Stage:")
            for stage, count in sorted(stage_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"     - {stage}: {count} recommendations")
        
        if channel_counts:
            print(f"\n   Top contributing recall channels:")
            for channel, count in sorted(channel_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"     - {channel}: {count} recommendations")
    
    # Step 5: Save model (optional)
    print("\n[Step 5] Saving model (optional)")
    print("-" * 80)
    
    model_path = Path("models/hybrid_recommender.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        recommender.save(str(model_path))
        print(f"✅ Model saved to {model_path}")
    except Exception as e:
        print(f"⚠️  Could not save model: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Recommendation system test complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

