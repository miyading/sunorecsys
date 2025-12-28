"""User interaction history management for Discover Weekly-style recommendations

Manages user interaction history with temporal tracking:
- Last-n interactions from history
- Weekly updates (Monday)
- Mixing historical and recent interactions
- Support for weighted interactions (play count, etc.)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd


class UserHistoryManager:
    """Manages user interaction history for Discover Weekly recommendations"""
    
    def __init__(
        self,
        history_file: str = "data/user_history.json",
        last_n: int = 50,
        weekly_update_day: int = 0,  # Monday = 0
    ):
        """
        Initialize user history manager
        
        Args:
            history_file: Path to store user history
            last_n: Number of last interactions to keep per user
            weekly_update_day: Day of week for weekly updates (0=Monday, 6=Sunday)
        """
        self.history_file = Path(history_file)
        self.last_n = last_n
        self.weekly_update_day = weekly_update_day
        
        # Load existing history
        self.user_history = self._load_history()
        self.last_update_date = self._get_last_update_date()
    
    def _load_history(self) -> Dict[str, List[Dict]]:
        """Load user history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    return data.get('user_history', {})
            except Exception as e:
                print(f"Warning: Could not load user history: {e}")
                return {}
        return {}
    
    def _save_history(self):
        """Save user history to file"""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'user_history': self.user_history,
            'last_update': datetime.now().isoformat(),
        }
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_last_update_date(self) -> Optional[datetime]:
        """Get last update date from history file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    last_update_str = data.get('last_update')
                    if last_update_str:
                        return datetime.fromisoformat(last_update_str)
            except Exception:
                pass
        return None
    
    def _is_weekly_update_day(self) -> bool:
        """Check if today is the weekly update day"""
        today = datetime.now()
        return today.weekday() == self.weekly_update_day
    
    def _get_week_start(self, date: datetime) -> datetime:
        """Get the start of the week (Monday) for a given date"""
        days_since_monday = date.weekday()
        return date - timedelta(days=days_since_monday)
    
    def _get_current_week_start(self) -> datetime:
        """Get the start of the current week"""
        return self._get_week_start(datetime.now())
    
    def add_interaction(
        self,
        user_id: str,
        song_id: str,
        timestamp: Optional[datetime] = None,
        weight: float = 1.0,
        interaction_type: str = 'play',
        save_immediately: bool = True,
    ):
        """
        Add a user interaction
        
        Args:
            user_id: User ID
            song_id: Song ID
            timestamp: Interaction timestamp (default: now)
            weight: Interaction weight (e.g., play count, normalized)
            interaction_type: Type of interaction ('play', 'like', 'save', etc.)
            save_immediately: If True, save to file immediately (default: True)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        interaction = {
            'song_id': song_id,
            'timestamp': timestamp.isoformat(),
            'weight': weight,
            'type': interaction_type,
        }
        
        self.user_history[user_id].append(interaction)
        
        # Keep only last_n interactions per user
        if len(self.user_history[user_id]) > self.last_n:
            self.user_history[user_id] = self.user_history[user_id][-self.last_n:]
        
        if save_immediately:
            self._save_history()
    
    def add_interactions_batch(
        self,
        interactions: List[Tuple[str, str, Optional[datetime], float, str]],
    ):
        """
        Add multiple interactions in batch (more efficient for bulk operations)
        
        Args:
            interactions: List of tuples (user_id, song_id, timestamp, weight, interaction_type)
        """
        for user_id, song_id, timestamp, weight, interaction_type in interactions:
            if timestamp is None:
                timestamp = datetime.now()
            
            if user_id not in self.user_history:
                self.user_history[user_id] = []
            
            interaction = {
                'song_id': song_id,
                'timestamp': timestamp.isoformat(),
                'weight': weight,
                'type': interaction_type,
            }
            
            self.user_history[user_id].append(interaction)
            
            # Keep only last_n interactions per user
            if len(self.user_history[user_id]) > self.last_n:
                self.user_history[user_id] = self.user_history[user_id][-self.last_n:]
        
        # Save only once at the end
        self._save_history()
    
    def get_user_interactions(
        self,
        user_id: str,
        use_weekly_mixing: bool = True,
    ) -> List[str]:
        """
        Get user's interactions for recommendations
        
        Args:
            user_id: User ID
            use_weekly_mixing: If True, mix historical and weekly interactions
        
        Returns:
            List of song IDs (last-n interactions)
        """
        if user_id not in self.user_history:
            return []
        
        interactions = self.user_history[user_id]
        
        if not use_weekly_mixing:
            # Return all interactions (up to last_n)
            return [i['song_id'] for i in interactions]
        
        # Weekly mixing logic
        current_week_start = self._get_current_week_start()
        
        # Separate historical and this week's interactions
        historical = []
        this_week = []
        
        for interaction in interactions:
            timestamp = datetime.fromisoformat(interaction['timestamp'])
            if timestamp >= current_week_start:
                this_week.append(interaction)
            else:
                historical.append(interaction)
        
        # Mixing strategy
        if len(this_week) == 0:
            # No interactions this week: use all historical
            selected = historical
        else:
            # Has interactions this week: 50% history + 50% this week
            n_history = min(len(historical), self.last_n // 2)
            n_this_week = min(len(this_week), self.last_n // 2)
            
            # Take most recent from each
            selected = historical[-n_history:] + this_week[-n_this_week:]
        
        # Return song IDs (deduplicated, most recent first)
        seen = set()
        result = []
        for interaction in reversed(selected):  # Most recent first
            song_id = interaction['song_id']
            if song_id not in seen:
                result.append(song_id)
                seen.add(song_id)
                if len(result) >= self.last_n:
                    break
        
        return result
    
    def get_user_interactions_with_weights(
        self,
        user_id: str,
        use_weekly_mixing: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Get user's interactions with weights
        
        Returns:
            List of (song_id, weight) tuples
        """
        if user_id not in self.user_history:
            return []
        
        interactions = self.user_history[user_id]
        
        if not use_weekly_mixing:
            return [(i['song_id'], i['weight']) for i in interactions]
        
        # Weekly mixing logic (same as get_user_interactions)
        current_week_start = self._get_current_week_start()
        
        historical = []
        this_week = []
        
        for interaction in interactions:
            timestamp = datetime.fromisoformat(interaction['timestamp'])
            if timestamp >= current_week_start:
                this_week.append(interaction)
            else:
                historical.append(interaction)
        
        if len(this_week) == 0:
            selected = historical
        else:
            n_history = min(len(historical), self.last_n // 2)
            n_this_week = min(len(this_week), self.last_n // 2)
            selected = historical[-n_history:] + this_week[-n_this_week:]
        
        # Return with weights, deduplicated
        seen = set()
        result = []
        for interaction in reversed(selected):
            song_id = interaction['song_id']
            if song_id not in seen:
                result.append((song_id, interaction['weight']))
                seen.add(song_id)
                if len(result) >= self.last_n:
                    break
        
        return result
    
    def should_update_models(self) -> bool:
        """Check if models should be updated (weekly on Monday)"""
        if not self.last_update_date:
            return True
        
        # Check if it's Monday and we haven't updated this week
        today = datetime.now()
        last_update_week = self._get_week_start(self.last_update_date)
        current_week = self._get_week_start(today)
        
        return current_week > last_update_week and today.weekday() == self.weekly_update_day
    
    def get_all_user_ids(self) -> List[str]:
        """Get all user IDs with history"""
        return list(self.user_history.keys())
    
    def get_user_interaction_count(self, user_id: str) -> int:
        """Get total interaction count for a user"""
        return len(self.user_history.get(user_id, []))


def load_user_history_from_interactions(
    interactions_df: pd.DataFrame,
    user_id_col: str = 'user_id',
    song_id_col: str = 'song_id',
    timestamp_col: Optional[str] = None,
    weight_col: Optional[str] = None,
) -> UserHistoryManager:
    """
    Load user history from a DataFrame of interactions
    
    Args:
        interactions_df: DataFrame with user interactions
        user_id_col: Column name for user ID
        song_id_col: Column name for song ID
        timestamp_col: Column name for timestamp (optional)
        weight_col: Column name for interaction weight (optional, e.g., play_count)
    
    Returns:
        UserHistoryManager with loaded history
    """
    manager = UserHistoryManager()
    
    for _, row in interactions_df.iterrows():
        user_id = row[user_id_col]
        song_id = row[song_id_col]
        
        timestamp = None
        if timestamp_col and timestamp_col in row:
            timestamp_str = row[timestamp_col]
            if pd.notna(timestamp_str):
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str)
                else:
                    timestamp = pd.to_datetime(timestamp_str).to_pydatetime()
        
        weight = 1.0
        if weight_col and weight_col in row:
            weight_val = row[weight_col]
            if pd.notna(weight_val):
                weight = float(weight_val)
        
        manager.add_interaction(user_id, song_id, timestamp, weight)
    
    return manager

