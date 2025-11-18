"""Example: Discover Weekly-style recommendation generation"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from sunorecsys.data.preprocess import SongDataProcessor
from sunorecsys.data.user_history import UserHistoryManager, load_user_history_from_interactions
from sunorecsys.recommenders.hybrid import HybridRecommender


def generate_discover_weekly(
    recommender: HybridRecommender,
    history_manager: UserHistoryManager,
    user_id: str,
    n: int = 30,
) -> list:
    """
    Generate Discover Weekly-style recommendations for a user
    
    Args:
        recommender: Fitted hybrid recommender
        history_manager: User history manager
        user_id: User ID
        n: Number of recommendations (default: 30 for Discover Weekly)
    
    Returns:
        List of recommendations
    """
    # Get user's last-n interactions with weekly mixing
    # - If no interactions this week: use all historical
    # - If has interactions this week: 50% historical + 50% this week
    interactions = history_manager.get_user_interactions(
        user_id,
        use_weekly_mixing=True
    )
    
    if not interactions:
        print(f"⚠️  No interaction history for user {user_id}")
        return []
    
    print(f"📊 Using {len(interactions)} seed interactions for user {user_id}")
    
    # Generate recommendations using last-n interactions
    recommendations = recommender.recommend(
        user_id=user_id,
        song_ids=interactions,  # Last-n interactions from history
        n=n,
        return_details=True,
    )
    
    return recommendations


def weekly_model_update(
    recommender: HybridRecommender,
    history_manager: UserHistoryManager,
    songs_df: pd.DataFrame,
):
    """
    Update models weekly (every Monday)
    
    This should be run as a scheduled job every Monday to:
    1. Re-compute item-item similarity with new songs
    2. Update all channel models
    3. Prepare for new week's recommendations
    """
    if not history_manager.should_update_models():
        print("⏭️  Not time for weekly update yet")
        return
    
    print("🔄 Starting weekly model update...")
    
    # Re-fit all models with updated data
    recommender.fit(
        songs_df,
        user_history=history_manager.user_history,  # Pass user history for last-n support
    )
    
    print("✅ Weekly model update complete!")
    print(f"   Updated item-item similarity matrix")
    print(f"   Updated all recommendation channels")


def main():
    """Example Discover Weekly workflow"""
    print("="*80)
    print("Discover Weekly Example")
    print("="*80)
    
    # Step 1: Load data
    print("\n📂 Loading data...")
    processor = SongDataProcessor()
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        print("❌ Processed data not found. Please run preprocessing first.")
        return
    
    songs_df = processor.load_processed(str(data_dir))
    print(f"✅ Loaded {len(songs_df)} songs")
    
    # Step 2: Initialize user history manager
    print("\n📊 Initializing user history manager...")
    history_manager = UserHistoryManager(
        history_file="data/user_history.json",
        last_n=50,  # Keep last 50 interactions per user
        weekly_update_day=0,  # Monday
    )
    
    # Step 3: Simulate some user interactions (in production, load from database)
    print("\n🎵 Simulating user interactions...")
    # Example: User listens to some songs this week
    user_id = songs_df['user_id'].iloc[0]  # Use first user as example
    
    # Add some interactions this week
    for i in range(5):
        song_id = songs_df.sample(1)['song_id'].iloc[0]
        history_manager.add_interaction(
            user_id=user_id,
            song_id=song_id,
            timestamp=datetime.now() - timedelta(days=i),
            weight=2.0 + i,  # Play count
            interaction_type='play'
        )
    
    # Add some historical interactions (last week)
    for i in range(10):
        song_id = songs_df.sample(1)['song_id'].iloc[0]
        history_manager.add_interaction(
            user_id=user_id,
            song_id=song_id,
            timestamp=datetime.now() - timedelta(days=7+i),
            weight=1.0,
            interaction_type='play'
        )
    
    print(f"✅ Added interactions for user {user_id}")
    
    # Step 4: Train recommender (with integrated history manager)
    print("\n🔧 Training recommender...")
    recommender = HybridRecommender(
        item_cf_weight=0.25,
        item_content_weight=0.30,
        prompt_weight=0.20,
        user_weight=0.20,
        quality_weight=0.05,
        use_user_cf=True,
        history_manager=history_manager,  # Integrated history manager
        use_last_n=True,  # Default: use last-n interactions
    )
    
    recommender.fit(songs_df)  # History manager automatically provides user_history
    
    # Step 5: Generate Discover Weekly recommendations
    # Now just pass user_id - HybridRecommender automatically uses last-n
    print("\n🎵 Generating Discover Weekly recommendations...")
    recommendations = recommender.recommend(
        user_id=user_id,  # Automatically uses last-n interactions from history
        n=10,  # Top 10 recommendations
        return_details=True,
    )
    
    print(f"\n✅ Generated {len(recommendations)} recommendations")
    print("\nTop 10 recommendations:")
    print("-" * 80)
    for i, rec in enumerate(recommendations[:10], 1):
        print(f"{i:2d}. {rec.get('title', 'N/A')[:50]:50s} (score: {rec['score']:.4f})")
        if 'details' in rec:
            channel_scores = rec['details'].get('channel_scores', {})
            if channel_scores:
                print(f"     Channels: {', '.join(f'{k}={v:.3f}' for k, v in channel_scores.items())}")
    
    # Step 6: Show weekly update check
    print("\n" + "="*80)
    print("Weekly Update Check")
    print("="*80)
    if history_manager.should_update_models():
        print("✅ It's time for weekly model update!")
        print("   Run weekly_model_update() to refresh models")
    else:
        print("⏭️  Not time for weekly update yet")
        last_update = history_manager.last_update_date
        if last_update:
            print(f"   Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("✅ Discover Weekly example complete!")
    print("="*80)


if __name__ == '__main__':
    main()

