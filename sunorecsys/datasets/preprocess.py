"""Data preprocessing pipeline for song data"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

# Optional audio feature extraction
try:
    from ..utils.audio_features import AudioFeatureExtractor
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


class SongDataProcessor:
    """Process raw song data into structured format for recommendation system"""
    
    def __init__(self, extract_audio_features: bool = False, audio_cache_dir: str = "data/audio_cache"):
        """
        Initialize processor
        
        Args:
            extract_audio_features: Whether to extract audio features (requires audio downloads)
            audio_cache_dir: Directory to cache audio files and features
        """
        self.songs_df = None
        self.processed_data = {}
        self.extract_audio_features = extract_audio_features and AUDIO_AVAILABLE
        self.audio_extractor = None
        
        if self.extract_audio_features:
            self.audio_extractor = AudioFeatureExtractor(cache_dir=audio_cache_dir)
    
    def load_songs(self, file_path: str) -> List[Dict[str, Any]]:
        """Load songs from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'songs' in data:
            return data['songs']
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected data format in {file_path}")
    
    def extract_tags(self, song: Dict[str, Any]) -> List[str]:
        """Extract and normalize tags from song metadata"""
        tags = []
        
        # From metadata.tags
        if 'metadata' in song and 'tags' in song['metadata']:
            tag_str = song['metadata']['tags']
            if tag_str:
                tags.extend([t.strip().lower() for t in tag_str.split(',')])
        
        # From display_tags
        if 'display_tags' in song and song['display_tags']:
            tags.extend([t.strip().lower() for t in song['display_tags'].split(',')])
        
        # Remove duplicates and empty strings
        tags = list(set([t for t in tags if t]))
        return tags
    
    def extract_genre(self, tags: List[str]) -> Optional[str]:
        """Extract primary genre from tags"""
        # Common genre keywords (can be expanded)
        genre_keywords = {
            'pop': ['pop', 'pop rock', 'indie pop'],
            'rock': ['rock', 'pop rock', 'post-hardcore', 'acoustic rock'],
            'hip hop': ['hip hop', 'hip-hop', 'rap', 'trap'],
            'electronic': ['edm', 'electronic', 'synth', 'glitch', 'phonk'],
            'jazz': ['jazz', 'blues'],
            'r&b': ['r&b', 'r and b', 'soul'],
            'country': ['country', 'folk'],
            'classical': ['classical', 'orchestral'],
            'metal': ['metal', 'heavy metal'],
            'reggae': ['reggae', 'dancehall'],
            'latin': ['latin', 'salsa', 'reggaeton'],
        }
        
        tags_lower = [t.lower() for t in tags]
        for genre, keywords in genre_keywords.items():
            if any(kw in ' '.join(tags_lower) for kw in keywords):
                return genre
        
        return None
    
    def extract_prompt_features(self, song: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from generation prompt"""
        prompt = song.get('metadata', {}).get('prompt', '')
        
        # Basic prompt statistics
        prompt_length = len(prompt)
        word_count = len(prompt.split()) if prompt else 0
        
        # Extract structured elements (e.g., [Verse], [Chorus])
        verse_count = len(re.findall(r'\[Verse', prompt, re.IGNORECASE))
        chorus_count = len(re.findall(r'\[Chorus', prompt, re.IGNORECASE))
        bridge_count = len(re.findall(r'\[Bridge', prompt, re.IGNORECASE))
        
        return {
            'prompt': prompt,
            'prompt_length': prompt_length,
            'word_count': word_count,
            'verse_count': verse_count,
            'chorus_count': chorus_count,
            'bridge_count': bridge_count,
            'has_structure': verse_count > 0 or chorus_count > 0,
        }
    
    def extract_engagement_features(self, song: Dict[str, Any]) -> Dict[str, Any]:
        """Extract engagement metrics"""
        play_count = song.get('play_count', 0)
        upvote_count = song.get('upvote_count', 0)
        comment_count = song.get('comment_count', 0)
        
        # Calculate engagement rate
        engagement_rate = (upvote_count + comment_count) / max(play_count, 1)
        
        # Popularity score (normalized)
        popularity_score = np.log1p(play_count) + np.log1p(upvote_count) * 2
        
        return {
            'play_count': play_count,
            'upvote_count': upvote_count,
            'comment_count': comment_count,
            'engagement_rate': engagement_rate,
            'popularity_score': popularity_score,
        }
    
    def process_song(self, song: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single song into structured format"""
        song_id = song.get('id')
        if not song_id:
            return None
        
        tags = self.extract_tags(song)
        prompt_features = self.extract_prompt_features(song)
        engagement = self.extract_engagement_features(song)
        audio_url = song.get('audio_url', '')
        
        processed = {
            'song_id': song_id,
            'title': song.get('title', ''),
            'user_id': song.get('user_id', ''),
            'created_at': song.get('created_at', ''),
            'duration': song.get('metadata', {}).get('duration', 0),
            'model_version': song.get('major_model_version', ''),
            'tags': tags,
            'genre': self.extract_genre(tags),
            'prompt': prompt_features['prompt'],
            'prompt_length': prompt_features['prompt_length'],
            'word_count': prompt_features['word_count'],
            'verse_count': prompt_features['verse_count'],
            'chorus_count': prompt_features['chorus_count'],
            'bridge_count': prompt_features['bridge_count'],
            'has_structure': prompt_features['has_structure'],
            'play_count': engagement['play_count'],
            'upvote_count': engagement['upvote_count'],
            'comment_count': engagement['comment_count'],
            'engagement_rate': engagement['engagement_rate'],
            'popularity_score': engagement['popularity_score'],
            'audio_url': audio_url,
            'is_public': song.get('is_public', False),
            'is_remix': song.get('metadata', {}).get('is_remix', False),
        }
        
        # Extract audio features if enabled
        if self.extract_audio_features and audio_url:
            try:
                audio_features = self.audio_extractor.extract_features_from_url(
                    audio_url, song_id, use_cache=True
                )
                if audio_features:
                    # Convert features to vector and store
                    audio_vector = self.audio_extractor.features_to_vector(audio_features)
                    processed['audio_features'] = audio_vector.tolist()
                    processed['has_audio_features'] = True
                else:
                    processed['has_audio_features'] = False
            except Exception as e:
                print(f"Warning: Failed to extract audio features for {song_id}: {e}")
                processed['has_audio_features'] = False
        else:
            processed['has_audio_features'] = False
        
        return processed
    
    def process_all(self, songs: List[Dict[str, Any]], verbose: bool = True) -> pd.DataFrame:
        """Process all songs into a DataFrame"""
        processed_songs = []
        
        iterator = tqdm(songs, desc="Processing songs") if verbose else songs
        
        for song in iterator:
            processed = self.process_song(song)
            if processed:
                processed_songs.append(processed)
        
        df = pd.DataFrame(processed_songs)
        
        if len(df) > 0:
            # Convert created_at to datetime
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            
            # Add temporal features
            if 'created_at' in df.columns:
                # Ensure both datetimes are timezone-naive for comparison
                # Convert timezone-aware to naive if needed
                # Use apply to handle each timestamp individually (handles mixed timezone-aware/naive)
                df['created_at'] = df['created_at'].apply(
                    lambda x: x.tz_convert('UTC').tz_localize(None) 
                    if pd.notna(x) and hasattr(x, 'tz') and x.tz is not None 
                    else x
                )
                
                # Use timezone-naive timestamp for comparison
                now = pd.Timestamp.now()
                df['days_since_creation'] = (now - df['created_at']).dt.days
                df['is_recent'] = df['days_since_creation'] <= 30
        
        self.songs_df = df
        return df
    
    def save_processed(self, output_dir: str):
        """Save processed data to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.songs_df is not None:
            # Try to save as parquet for efficiency, fallback to JSON if pyarrow not available
            try:
                self.songs_df.to_parquet(output_path / 'songs.parquet', index=False)
                print(f"✅ Saved as parquet: {output_path / 'songs.parquet'}")
            except ImportError:
                print("⚠️  pyarrow not available, saving as JSON only")
            
            # Always save as JSON for compatibility
            self.songs_df.to_json(output_path / 'songs.json', orient='records', indent=2)
            print(f"✅ Saved as JSON: {output_path / 'songs.json'}")
            
            # Save metadata
            metadata = {
                'total_songs': len(self.songs_df),
                'unique_users': self.songs_df['user_id'].nunique(),
                'date_range': {
                    'min': str(self.songs_df['created_at'].min()),
                    'max': str(self.songs_df['created_at'].max()),
                },
                'genres': self.songs_df['genre'].value_counts().to_dict(),
            }
            
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_processed(self, input_dir: str) -> pd.DataFrame:
        """Load processed data from disk"""
        input_path = Path(input_dir)
        
        # Try parquet first, fallback to JSON
        parquet_path = input_path / 'songs.parquet'
        json_path = input_path / 'songs.json'
        
        if parquet_path.exists():
            try:
                self.songs_df = pd.read_parquet(parquet_path)
                print(f"✅ Loaded from parquet: {parquet_path}")
                return self.songs_df
            except ImportError:
                print("⚠️  pyarrow not available, trying JSON...")
        
        if json_path.exists():
            self.songs_df = pd.read_json(json_path, orient='records')
            print(f"✅ Loaded from JSON: {json_path}")
            return self.songs_df
        
        raise FileNotFoundError(f"No processed data found in {input_dir}")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process song data for recommender system')
    parser.add_argument('--input', required=True, help='Input JSON file with songs')
    parser.add_argument('--output', required=True, help='Output directory for processed data')
    parser.add_argument('--verbose', action='store_true', help='Show progress bar')
    
    args = parser.parse_args()
    
    processor = SongDataProcessor()
    songs = processor.load_songs(args.input)
    df = processor.process_all(songs, verbose=args.verbose)
    processor.save_processed(args.output)
    
    print(f"\n✅ Processed {len(df)} songs")
    print(f"✅ Saved to {args.output}")


if __name__ == '__main__':
    main()

