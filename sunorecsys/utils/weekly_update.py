"""Weekly model update scheduler for Discover Weekly

Automatically updates recommendation models every Monday.
"""

import schedule
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from ..data.preprocess import SongDataProcessor
from ..data.user_history import UserHistoryManager
from ..recommenders.hybrid import HybridRecommender


class WeeklyUpdateScheduler:
    """Scheduler for weekly model updates"""
    
    def __init__(
        self,
        recommender: HybridRecommender,
        history_manager: UserHistoryManager,
        songs_df: pd.DataFrame,
        update_time: str = "00:00",  # Update at midnight on Monday
        data_dir: str = "runtime_data/processed",
    ):
        """
        Initialize weekly update scheduler
        
        Args:
            recommender: Hybrid recommender to update
            history_manager: User history manager
            songs_df: Songs DataFrame
            update_time: Time to run update (HH:MM format, default: "00:00")
            data_dir: Directory with processed song data
        """
        self.recommender = recommender
        self.history_manager = history_manager
        self.songs_df = songs_df
        self.update_time = update_time
        self.data_dir = Path(data_dir)
        self.last_update = None
    
    def update_models(self):
        """Update all recommendation models for the new week"""
        print("\n" + "="*80)
        print(f"Weekly Model Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Check if update is needed
        if not self.history_manager.should_update_models():
            print("⏭️  Not time for weekly update yet")
            return
        
        try:
            # Reload songs data (in case new songs were added)
            print("\n📂 Reloading song data...")
            processor = SongDataProcessor()
            if self.data_dir.exists():
                updated_songs_df = processor.load_processed(str(self.data_dir))
                print(f"✅ Loaded {len(updated_songs_df)} songs")
            else:
                updated_songs_df = self.songs_df
                print("⚠️  Using existing songs_df")
            
            # Convert user history to format expected by recommenders
            print("\n📊 Preparing user history...")
            user_history_dict = {}
            for uid in self.history_manager.get_all_user_ids():
                interactions = self.history_manager.get_user_interactions(
                    uid, 
                    use_weekly_mixing=False  # Get all for model training
                )
                user_history_dict[uid] = interactions
            
            print(f"   Prepared history for {len(user_history_dict)} users")
            
            # Re-fit all models
            print("\n🔧 Updating recommendation models...")
            self.recommender.fit(
                updated_songs_df,
                user_history=user_history_dict,
            )
            
            # Update songs_df
            self.songs_df = updated_songs_df
            
            # Mark update complete
            self.last_update = datetime.now()
            self.history_manager._save_history()  # Update last_update timestamp
            
            print("\n✅ Weekly model update complete!")
            print(f"   Updated item-item similarity matrix")
            print(f"   Updated all recommendation channels")
            print(f"   Ready for new week's recommendations")
            
        except Exception as e:
            print(f"\n❌ Error during weekly update: {e}")
            import traceback
            traceback.print_exc()
    
    def schedule_weekly_updates(self):
        """Schedule weekly updates every Monday"""
        # Schedule for every Monday at specified time
        schedule.every().monday.at(self.update_time).do(self.update_models)
        print(f"📅 Scheduled weekly updates every Monday at {self.update_time}")
    
    def run_pending(self):
        """Run any pending scheduled updates"""
        schedule.run_pending()
    
    def run_forever(self, interval: int = 60):
        """
        Run scheduler forever, checking every interval seconds
        
        Args:
            interval: Check interval in seconds (default: 60)
        """
        print(f"🔄 Starting weekly update scheduler (checking every {interval}s)...")
        print("   Press Ctrl+C to stop")
        
        self.schedule_weekly_updates()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n⏹️  Scheduler stopped")
    
    def run_once_if_needed(self):
        """Run update once if it's time (for manual/cron execution)"""
        if self.history_manager.should_update_models():
            self.update_models()
        else:
            print("⏭️  Not time for weekly update yet")
            last_update = self.history_manager.last_update_date
            if last_update:
                print(f"   Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")


def create_weekly_update_script(
    output_file: str = "weekly_update.py",
    data_dir: str = "data/processed",
    history_file: str = "runtime_data/user_history.json",
    model_file: str = "model_checkpoints/hybrid_recommender.pkl",
):
    """
    Create a standalone script for weekly updates (to run via cron)
    
    Args:
        output_file: Path to output script
        data_dir: Directory with processed song data
        history_file: Path to user history file
        model_file: Path to saved model file
    """
    script_content = f'''#!/usr/bin/env python3
"""Weekly model update script (run via cron every Monday)"""

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
    
    # Load data
    processor = SongDataProcessor()
    data_dir = Path("{data_dir}")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {{data_dir}}")
        return 1
    
    songs_df = processor.load_processed(str(data_dir))
    print(f"✅ Loaded {{len(songs_df)}} songs")
    
    # Initialize history manager
    history_manager = UserHistoryManager(history_file="{history_file}")
    
    # Load or create recommender
    model_path = Path("{model_file}")
    if model_path.exists():
        print(f"📂 Loading existing model from {{model_path}}...")
        recommender = HybridRecommender.load(str(model_path))
        recommender.history_manager = history_manager  # Re-attach history manager
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
    print(f"✅ Model saved to {{model_path}}")
    
    return 0

if __name__ == '__main__':
    exit(main())
'''
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    import os
    os.chmod(output_path, 0o755)
    
    print(f"✅ Created weekly update script: {output_file}")
    print(f"\nTo schedule with cron, add to crontab:")
    print(f"  0 0 * * 1 cd {Path(output_file).parent} && python {output_file}")
    print(f"  (Runs every Monday at midnight)")

