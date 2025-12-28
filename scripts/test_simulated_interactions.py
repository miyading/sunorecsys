"""Test script to verify simulated interactions work correctly"""

import sys
import pandas as pd
from pathlib import Path

# Add scripts directory to path for local imports
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from sunorecsys.datasets.preprocess import SongDataProcessor
from sunorecsys.datasets.simulate_interactions import get_user_item_matrix, InteractionSimulator
from sunorecsys.recommenders.item_cf import ItemBasedCFRecommender
from check_cf_viability import analyze_cf_viability


def main():
    print("="*80)
    print("Testing Simulated User-Item Interactions")
    print("="*80)
    
    # Try loading from aggregated file first, fallback to processed data
    aggregated_file = "sunorecsys/datasets/curl/all_playlist_songs.json"
    songs_df = None
    
    if Path(aggregated_file).exists():
        print("\n📂 Loading from aggregated file...")
        from sunorecsys.datasets.simulate_interactions import load_songs_from_aggregated_file
        try:
            songs_df = load_songs_from_aggregated_file(aggregated_file, max_songs=None)
            print(f"✅ Loaded {len(songs_df)} songs from aggregated file")
        except Exception as e:
            print(f"⚠️  Error loading from aggregated file: {e}")
            songs_df = None
    
    # Fallback to processed data
    if songs_df is None:
        print("\n📂 Loading from processed data...")
        processor = SongDataProcessor()
        
        data_dir = Path("runtime_data/processed")
        if not data_dir.exists():
            print("❌ Processed data not found. Please run preprocessing first.")
            return
        
        try:
            songs_df = processor.load_processed(str(data_dir))
            print(f"✅ Loaded {len(songs_df)} songs from processed data")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
    
    # Test 1: Generate simulated interactions
    print("\n" + "="*80)
    print("Test 1: Generating Simulated Interactions")
    print("="*80)
    
    # Use aggregated_file if available, otherwise use songs_df
    if Path(aggregated_file).exists():
        user_item_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id = \
            get_user_item_matrix(
                aggregated_file=aggregated_file,
                use_simulation=True,
                interaction_rate=0.15,
                random_seed=42,
            )
    else:
        user_item_matrix, user_id_to_idx, idx_to_user_id, song_id_to_idx, idx_to_song_id = \
            get_user_item_matrix(
                songs_df=songs_df,
                use_simulation=True,
                interaction_rate=0.15,
                random_seed=42,
            )
    
    # Test 2: Verify CF viability
    print("\n" + "="*80)
    print("Test 2: Verifying CF Viability with Simulated Data")
    print("="*80)
    
    results = analyze_cf_viability(
        songs_df,
        user_item_matrix,
        user_id_to_idx,
        song_id_to_idx
    )
    
    # Test 3: Train item-based CF with simulated data
    print("\n" + "="*80)
    print("Test 3: Training Item-Based CF with Simulated Interactions")
    print("="*80)
    
    item_cf = ItemBasedCFRecommender()
    item_cf.fit(
        songs_df,
        use_simulated_interactions=True,
        interaction_rate=0.15,
        random_seed=42,
    )
    
    # Verify it worked
    print(f"\n✅ Item-based CF trained successfully!")
    print(f"   Matrix shape: {item_cf.user_item_matrix.shape}")
    print(f"   Item-item similarity shape: {item_cf.item_item_similarity.shape}")
    
    # Check if similarity matrix has non-zero values
    non_zero_similarities = item_cf.item_item_similarity.nnz
    total_possible = item_cf.item_item_similarity.shape[0] * item_cf.item_item_similarity.shape[1]
    print(f"   Non-zero similarities: {non_zero_similarities:,} / {total_possible:,} ({non_zero_similarities/total_possible*100:.2f}%)")
    
    if non_zero_similarities > 0:
        print("   ✅ Item-item similarity matrix has meaningful values!")
        print("   ✅ Item-based CF should work now!")
    else:
        print("   ❌ Item-item similarity matrix is empty")
    
    # Test 4: Show how to use aggregated file
    print("\n" + "="*80)
    print("Test 4: Using Aggregated File")
    print("="*80)
    print("\nTo use aggregated file with real user_ids and song_ids:")
    print("  from sunorecsys.datasets.simulate_interactions import get_user_item_matrix")
    print("  ")
    print("  user_item_matrix, ... = get_user_item_matrix(")
    print("      aggregated_file='sunorecsys/datasets/curl/all_playlist_songs.json',")
    print("      use_simulation=True,  # Uses simulation with real IDs")
    print("      interaction_rate=0.15,")
    print("  )")
    print("\nThis will:")
    print("  - Extract real user_ids and song_ids from aggregated file")
    print("  - Use existing simulation functions for realistic interactions")
    print("  - Build user-item matrix ready for CF recommenders")
    
    print("\n" + "="*80)
    print("✅ All tests complete!")
    print("="*80)


if __name__ == '__main__':
    main()

