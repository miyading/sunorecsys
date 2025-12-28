#!/usr/bin/env python3
"""
Update recommender system models using aggregated playlist songs file

This script:
1. Loads user_ids and song_ids from all_playlist_songs.json
2. Uses existing simulation functions to build user-item interaction matrix
3. Trains/updates CF recommenders with the new matrix
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sunorecsys.datasets.simulate_interactions import get_user_item_matrix, load_songs_from_aggregated_file
from sunorecsys.recommenders.item_cf import ItemBasedCFRecommender
from sunorecsys.recommenders.user_based import UserBasedRecommender


def main():
    print("="*80)
    print("Update Recommender Models from Aggregated File")
    print("="*80)
    
    # Configuration
    aggregated_file = "sunorecsys/datasets/curl/all_playlist_songs.json"
    interaction_rate = 0.15
    item_cold_start_rate = 0.05
    single_user_item_rate = 0.15
    random_seed = 42
    
    # Check if file exists
    if not Path(aggregated_file).exists():
        print(f"❌ Aggregated file not found: {aggregated_file}")
        print("   Please run playlist extraction first.")
        return
    
    # Step 1: Build matrix using existing simulation functions
    print("\n" + "="*80)
    print("Step 1: Building User-Item Matrix from Aggregated File")
    print("="*80)
    
    user_item_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id = get_user_item_matrix(
        aggregated_file=aggregated_file,
        use_simulation=True,
        interaction_rate=interaction_rate,
        item_cold_start_rate=item_cold_start_rate,
        single_user_item_rate=single_user_item_rate,
        random_seed=random_seed,
        max_songs=None,  # Use all songs
    )
    
    # Step 2: Load songs DataFrame for recommenders
    print("\n" + "="*80)
    print("Step 2: Loading Songs DataFrame")
    print("="*80)
    
    songs_df = load_songs_from_aggregated_file(aggregated_file)
    
    # Step 3: Train Item-Based CF
    print("\n" + "="*80)
    print("Step 3: Training Item-Based CF Recommender")
    print("="*80)
    
    item_cf = ItemBasedCFRecommender()
    
    # Set the matrix and mappings directly
    item_cf.songs_df = songs_df
    item_cf.user_item_matrix = user_item_matrix
    item_cf.user_id_to_idx = user_id_to_idx
    item_cf.idx_to_user_id = idx_to_user_id
    item_cf.song_id_to_idx = song_id_to_idx
    item_cf.idx_to_song_id = idx_to_song_id
    
    # Compute item-item similarity
    print("Computing item-item similarity...")
    from sklearn.metrics.pairwise import cosine_similarity
    
    item_vectors = item_cf.user_item_matrix.T  # Transpose: items x users
    item_cf.item_item_similarity = cosine_similarity(item_vectors, dense_output=False)
    
    item_cf.is_fitted = True
    print(f"✅ Item-Based CF trained")
    print(f"   Item-item similarity matrix shape: {item_cf.item_item_similarity.shape}")
    print(f"   Non-zero similarities: {item_cf.item_item_similarity.nnz:,}")
    
    # Step 4: Train User-Based CF
    print("\n" + "="*80)
    print("Step 4: Training User-Based CF Recommender")
    print("="*80)
    
    user_cf = UserBasedRecommender()
    
    # Set the matrix and mappings directly
    user_cf.songs_df = songs_df
    user_cf.user_item_matrix = user_item_matrix
    user_cf.user_id_to_idx = user_id_to_idx
    user_cf.idx_to_user_id = idx_to_user_id
    user_cf.song_id_to_idx = song_id_to_idx
    user_cf.idx_to_song_id = idx_to_song_id
    
    # Compute user-user similarity
    print("Computing user-user similarity...")
    user_cf.user_user_similarity = cosine_similarity(
        user_cf.user_item_matrix,
        dense_output=False
    )
    
    user_cf.is_fitted = True
    print(f"✅ User-Based CF trained")
    print(f"   User-user similarity matrix shape: {user_cf.user_user_similarity.shape}")
    print(f"   Non-zero similarities: {user_cf.user_user_similarity.nnz:,}")
    
    # Step 5: Save models (optional)
    print("\n" + "="*80)
    print("Step 5: Saving Models")
    print("="*80)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    item_cf.save(str(models_dir / "item_cf_aggregated.pkl"))
    user_cf.save(str(models_dir / "user_cf_aggregated.pkl"))
    
    print(f"✅ Models saved to {models_dir}/")
    
    print("\n" + "="*80)
    print("✅ Model Update Complete!")
    print("="*80)
    print(f"\nModels are ready to use with:")
    print(f"  - {len(user_id_to_idx)} users (from aggregated file)")
    print(f"  - {len(song_id_to_idx)} songs (from aggregated file)")
    print(f"  - {user_item_matrix.nnz:,} simulated interactions")
    print(f"\nThe matrix uses:")
    print(f"  - Real user_ids and song_ids from {aggregated_file}")
    print(f"  - Existing simulation functions for realistic interaction patterns")


if __name__ == '__main__':
    main()

