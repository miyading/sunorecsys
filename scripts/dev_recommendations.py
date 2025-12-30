"""Development script to show modular recommendation outputs from each channel"""

import json
import pandas as pd
from pathlib import Path

from sunorecsys.datasets.preprocess import SongDataProcessor
from sunorecsys.recommenders.item_cf import ItemBasedCFRecommender
from sunorecsys.recommenders.item_content_based import ItemContentBasedRecommender
from sunorecsys.recommenders.prompt_based import PromptBasedRecommender
from sunorecsys.recommenders.user_based import UserBasedRecommender
from sunorecsys.recommenders.hybrid import HybridRecommender


def print_channel_results(channel_name: str, recommendations: list, show_details: bool = True):
    """Pretty print channel recommendation results"""
    print(f"\n{'='*80}")
    print(f"Channel: {channel_name}")
    print(f"{'='*80}")
    
    if not recommendations:
        print("  No recommendations")
        return
    
    print(f"\nTop {len(recommendations)} Recommendations:")
    print("-" * 80)
    
    for i, rec in enumerate(recommendations[:10], 1):  # Show top 10
        print(f"\n{i}. {rec.get('title', 'N/A')} (ID: {rec['song_id'][:8]}...)")
        print(f"   Score: {rec['score']:.4f}")
        print(f"   Genre: {rec.get('genre', 'N/A')}")
        print(f"   Tags: {', '.join(rec.get('tags', [])[:5])}")
        
        if show_details and 'details' in rec:
            details = rec['details']
            print(f"   Details:")
            print(f"     - Channel: {details.get('channel', 'N/A')}")
            print(f"     - Raw Score: {details.get('raw_score', 'N/A'):.4f}")
            
            # Channel-specific details
            if 'combined_score' in details:
                print(f"     - Combined Score: {details['combined_score']:.4f}")
                print(f"     - Embedding Similarity: {details.get('embedding_similarity', 'N/A'):.4f}")
                print(f"     - Genre/Tag Match: {details.get('genre_tag_match', 'N/A'):.4f}")
                if 'component_weights' in details:
                    weights = details['component_weights']
                    print(f"     - Component Weights: embedding={weights.get('embedding_weight', 0):.2f}, "
                          f"genre={weights.get('genre_weight', 0):.2f}, "
                          f"tag={weights.get('tag_weight', 0):.2f}, "
                          f"popularity={weights.get('popularity_weight', 0):.2f}")
            
            if 'cosine_similarity' in details:
                print(f"     - Cosine Similarity: {details['cosine_similarity']:.4f}")
                print(f"     - Annoy Distance: {details['annoy_distance']:.4f}")
                if 'component_weights' in details:
                    weights = details['component_weights']
                    print(f"     - Component Weights: tag={weights.get('tag_weight', 0):.2f}, "
                          f"prompt={weights.get('prompt_weight', 0):.2f}, "
                          f"metadata={weights.get('metadata_weight', 0):.2f}")
            
            if 'prompt_similarity' in details:
                print(f"     - Prompt Similarity: {details['prompt_similarity']:.4f}")
            
            if 'genre_match' in details:
                print(f"     - Genre Match: {details['genre_match']}")
                print(f"     - Tag Match Count: {details.get('tag_match_count', 'N/A')}")
                print(f"     - Popularity Match: {details.get('popularity_match', 'N/A'):.4f}")
            
            if 'cf_score' in details:
                print(f"     - CF Score: {details['cf_score']:.4f}")
                print(f"     - Recommendation Method: {details.get('recommendation_method', 'N/A')}")


def main():
    print("="*80)
    print("Development: Modular Recommendation Channel Outputs")
    print("="*80)
    
    # Step 1: Load data (try aggregated file first, fallback to processed data)
    print("\n📂 Loading data...")
    from sunorecsys.datasets.simulate_interactions import load_songs_from_aggregated_file
    aggregated_file = "sunorecsys/datasets/curl/all_playlist_songs.json"
    songs_df = None
    
    # Try loading from aggregated file
    if Path(aggregated_file).exists():
        print(f"📂 Loading from aggregated file: {aggregated_file}")
        try:
            songs_df = load_songs_from_aggregated_file(aggregated_file, max_songs=None)
            print(f"✅ Loaded {len(songs_df)} songs from aggregated file")
        except Exception as e:
            print(f"⚠️  Error loading from aggregated file: {e}")
            songs_df = None
    
    # Fallback to processed data
    if songs_df is None:
        print("📂 Loading from processed data...")
        processor = SongDataProcessor()
        data_dir = Path("runtime_data/processed")
        
        if not data_dir.exists():
            print("❌ Processed data not found. Please run preprocessing first:")
            print("   python -m sunorecsys.datasets.preprocess --input all_songs.json --output runtime_data/processed")
            return
        
        try:
            songs_df = processor.load_processed(str(data_dir))
            print(f"✅ Loaded {len(songs_df)} songs from processed data")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
    
    # Step 2: Get seed song
    print("\n🎵 Selecting seed song...")
    seed_song = songs_df.sample(1).iloc[0]
    seed_id = seed_song['song_id']
    
    print(f"Seed Song: '{seed_song['title']}'")
    print(f"  ID: {seed_id}")
    print(f"  Genre: {seed_song.get('genre', 'N/A')}")
    print(f"  Tags: {', '.join(seed_song.get('tags', [])[:5])}")
    
    # Step 3: Train recommenders (or load if available)
    print("\n🔧 Training recommenders...")
    
    # Item-based CF
    print("  [1/4] Item-based CF recommender...")
    item_cf_rec = ItemBasedCFRecommender()
    item_cf_rec.fit(
        songs_df,
        use_simulated_interactions=True,  # Use simulation with real IDs
        interaction_rate=0.15,
        random_seed=42,
    )
    
    # Item content-based (embeddings + genre/metadata combined)
    print("  [2/4] Item content-based recommender...")
    item_content_rec = ItemContentBasedRecommender()
    item_content_rec.fit(songs_df)
    
    # Prompt-based
    print("  [3/4] Prompt-based recommender...")
    prompt_rec = PromptBasedRecommender()
    prompt_rec.fit(songs_df)
    
    # User-based
    print("  [4/4] User-based recommender...")
    user_rec = UserBasedRecommender()
    user_rec.fit(
        songs_df,
        use_simulated_interactions=True,  # Use simulation with real IDs
        interaction_rate=0.15,
        random_seed=42,
    )
    
    print("✅ All recommenders trained")
    
    # Step 4: Get recommendations from each channel with details
    print("\n📊 Getting recommendations from each channel...")
    
    # Item-based CF with details
    item_cf_recs = item_cf_rec.recommend(
        song_ids=[seed_id],
        n=10,
        return_details=True
    )
    print_channel_results("Item-Based CF (User-Item Interactions)", item_cf_recs, show_details=True)
    
    # Item content-based with details (embeddings + genre/metadata)
    item_content_recs = item_content_rec.recommend(
        song_ids=[seed_id],
        n=10,
        return_details=True
    )
    print_channel_results("Item Content-Based (Embeddings + Genre/Metadata)", item_content_recs, show_details=True)
    
    # Prompt-based with details
    prompt_recs = prompt_rec.recommend(
        song_ids=[seed_id],
        n=10,
        return_details=True
    )
    print_channel_results("Prompt-Based", prompt_recs, show_details=True)
    
    # User-based with details
    user_recs = user_rec.recommend(
        song_ids=[seed_id],
        n=10,
        return_details=True
    )
    print_channel_results("User-Based CF", user_recs, show_details=True)
    
    # Step 5: Compare channel outputs
    print("\n" + "="*80)
    print("Channel Comparison")
    print("="*80)
    
    # Collect all recommended song IDs
    all_song_ids = set()
    channel_scores = {}
    
    for channel_name, recs in [
        ("Item-Based CF", item_cf_recs),
        ("Item Content-Based", item_content_recs),
        ("Prompt-Based", prompt_recs),
        ("User-Based", user_recs),
    ]:
        channel_scores[channel_name] = {}
        for rec in recs:
            song_id = rec['song_id']
            all_song_ids.add(song_id)
            channel_scores[channel_name][song_id] = rec['score']
    
    # Show overlap
    print(f"\nTotal unique songs recommended: {len(all_song_ids)}")
    print(f"\nTop songs across channels:")
    print("-" * 80)
    
    # Find songs recommended by multiple channels
    song_channel_count = {}
    for channel_name, scores in channel_scores.items():
        for song_id in scores:
            if song_id not in song_channel_count:
                song_channel_count[song_id] = []
            song_channel_count[song_id].append(channel_name)
    
    # Sort by number of channels recommending
    sorted_songs = sorted(song_channel_count.items(), key=lambda x: len(x[1]), reverse=True)
    
    for song_id, channels in sorted_songs[:10]:
        song_info = songs_df[songs_df['song_id'] == song_id].iloc[0]
        print(f"\n{song_info['title']} (ID: {song_id[:8]}...)")
        print(f"  Recommended by: {', '.join(channels)} ({len(channels)} channels)")
        print(f"  Scores: ", end="")
        for ch in channels:
            print(f"{ch}={channel_scores[ch][song_id]:.3f} ", end="")
        print()
    
    # Step 6: Save detailed results to JSON
    output_file = Path("dev_recommendations_output.json")
    output_data = {
        'seed_song': {
            'song_id': seed_id,
            'title': seed_song['title'],
            'genre': seed_song.get('genre'),
            'tags': seed_song.get('tags', []),
        },
        'channels': {
            'item_cf': item_cf_recs,
            'item_content_based': item_content_recs,
            'prompt_based': prompt_recs,
            'user_based': user_recs,
        },
        'summary': {
            'total_unique_songs': len(all_song_ids),
            'songs_by_channel_count': {song_id: len(channels) for song_id, channels in song_channel_count.items()},
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n✅ Detailed results saved to {output_file}")
    print("\n" + "="*80)
    print("✅ Development analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

