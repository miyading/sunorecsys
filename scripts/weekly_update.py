#!/usr/bin/env python3
"""Weekly model update script (run via cron every Monday)

Usage:
    python weekly_update.py

Or schedule with cron:
    0 0 * * 1 cd /path/to/sunorecsys && python weekly_update.py
    (Runs every Monday at midnight)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sunorecsys.datasets.preprocess import SongDataProcessor
from sunorecsys.datasets.user_history import UserHistoryManager
from sunorecsys.recommenders.hybrid import HybridRecommender
from sunorecsys.utils.weekly_update import WeeklyUpdateScheduler


def main():
    print("="*80)
    print("Weekly Model Update Script")
    print("="*80)
    
    # Load data from all_playlist_songs.json
    from sunorecsys.datasets.simulate_interactions import load_songs_from_aggregated_file
    
    aggregated_file = Path("sunorecsys/datasets/curl/all_playlist_songs.json")
    
    if not aggregated_file.exists():
        print(f"❌ Data file not found: {aggregated_file}")
        return 1
    
    songs_df = load_songs_from_aggregated_file(str(aggregated_file), max_songs=None)
    print(f"✅ Loaded {len(songs_df)} songs")
    
    # Initialize history manager
    history_manager = UserHistoryManager(history_file="runtime_data/user_history.json")
    
    # Load or create recommender
    model_path = Path("model_checkpoints/hybrid_recommender.pkl")
    if model_path.exists():
        print(f"📂 Loading existing model from {model_path}...")
        recommender = HybridRecommender.load(str(model_path), history_manager=history_manager)
    else:
        print("🔧 Creating new recommender...")
        recommender = HybridRecommender(
            history_manager=history_manager,
            din_model_path="model_checkpoints/din_ranker.pt",  # Path to trained DIN model (if available)
        )
        recommender.fit(songs_df)
    
    # Create scheduler and run update if needed
    scheduler = WeeklyUpdateScheduler(
        recommender=recommender,
        history_manager=history_manager,
        songs_df=songs_df,
        data_dir=str(data_dir),
    )
    
    scheduler.run_once_if_needed()
    
    # Save updated model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    recommender.save(str(model_path))
    print(f"✅ Model saved to {model_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())

