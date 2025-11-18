"""Training script for hybrid recommender"""

import argparse
import json
from pathlib import Path
import pandas as pd

from ..data.preprocess import SongDataProcessor
from .hybrid import HybridRecommender


def load_playlists(playlists_file: str):
    """Load playlist data from results.json"""
    with open(playlists_file, 'r') as f:
        data = json.load(f)
    
    playlists = []
    if 'result' in data and 'playlist' in data['result']:
        playlist_data = data['result']['playlist'].get('result', [])
        
        # For now, we'll need to fetch songs for each playlist
        # This is a simplified version - in production, you'd have playlist-song mappings
        for playlist in playlist_data:
            playlists.append({
                'id': playlist.get('id'),
                'name': playlist.get('name'),
                'song_ids': [],  # Would need to be populated from actual data
            })
    
    return playlists


def main():
    parser = argparse.ArgumentParser(description='Train hybrid music recommender')
    parser.add_argument('--songs', required=True, help='Path to processed songs data directory')
    parser.add_argument('--playlists', help='Path to playlists JSON file (optional)')
    parser.add_argument('--output', default='models/hybrid_recommender.pkl', help='Output model path')
    parser.add_argument('--item-weight', type=float, default=0.25, help='Item-based weight')
    parser.add_argument('--prompt-weight', type=float, default=0.20, help='Prompt-based weight')
    parser.add_argument('--genre-weight', type=float, default=0.20, help='Genre-based weight')
    parser.add_argument('--user-weight', type=float, default=0.25, help='User-based weight')
    parser.add_argument('--quality-weight', type=float, default=0.10, help='Quality weight')
    parser.add_argument('--quality-threshold', type=float, default=0.3, help='Quality threshold')
    
    args = parser.parse_args()
    
    # Load processed songs
    print("Loading processed songs...")
    processor = SongDataProcessor()
    songs_df = processor.load_processed(args.songs)
    print(f"Loaded {len(songs_df)} songs")
    
    # Load playlists if provided
    playlists = None
    if args.playlists:
        print("Loading playlists...")
        playlists = load_playlists(args.playlists)
        print(f"Loaded {len(playlists)} playlists")
    
    # Train recommender
    print("\nTraining hybrid recommender...")
    recommender = HybridRecommender(
        item_weight=args.item_weight,
        prompt_weight=args.prompt_weight,
        genre_weight=args.genre_weight,
        user_weight=args.user_weight,
        quality_weight=args.quality_weight,
        quality_threshold=args.quality_threshold,
    )
    
    recommender.fit(songs_df, playlists=playlists)
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recommender.save(str(output_path))
    
    print(f"\n✅ Model saved to {args.output}")


if __name__ == '__main__':
    main()

