"""
Compute Music Flamingo quality scores for songs in the RecSys.

This script processes songs and generates quality scores that can be used
for the quality filter channel in the hybrid recommender.
"""

import json
import argparse
from pathlib import Path
from sunorecsys.utils.music_flamingo_quality import MusicFlamingoQualityScorer


def main():
    parser = argparse.ArgumentParser(
        description='Compute Music Flamingo quality scores for songs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute quality scores for all songs
  python compute_music_flamingo_quality.py \\
      --input sunorecsys/data/curl/all_songs.json \\
      --output data/music_flamingo_scores.json \\
      --cache-dir data/music_flamingo_scores
  
  # Process with GPU
  python compute_music_flamingo_quality.py \\
      --input sunorecsys/data/curl/all_songs.json \\
      --output data/music_flamingo_scores.json \\
      --device cuda \\
      --batch-size 4
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input JSON file with songs (must have "id" and "audio_url" fields)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSON file for quality scores'
    )
    parser.add_argument(
        '--model-id',
        default='nvidia/music-flamingo-hf',
        help='HuggingFace model ID for Music Flamingo'
    )
    parser.add_argument(
        '--cache-dir',
        default='data/music_flamingo_scores',
        help='Directory to cache individual scores'
    )
    parser.add_argument(
        '--device',
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (cuda/cpu). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate (default: 512)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of songs to process (for testing)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching of scores'
    )
    
    args = parser.parse_args()
    
    # Load songs
    print(f"📂 Loading songs from {args.input}...")
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, dict) and 'songs' in data:
        songs = data['songs']
    elif isinstance(data, list):
        songs = data
    else:
        print(f"❌ Error: Unexpected format in {args.input}")
        print("Expected a list of songs or a dict with 'songs' key")
        return
    
    print(f"✅ Loaded {len(songs)} songs")
    
    # Filter songs with audio URLs
    songs_with_audio = [
        song for song in songs
        if song.get('audio_url') and song.get('id')
    ]
    print(f"✅ Found {len(songs_with_audio)} songs with audio URLs")
    
    if len(songs_with_audio) == 0:
        print("❌ No songs with audio URLs found")
        return
    
    # Limit songs if specified
    if args.limit:
        songs_with_audio = songs_with_audio[:args.limit]
        print(f"📊 Limiting to {len(songs_with_audio)} songs")
    
    # Initialize scorer
    print(f"\n🔧 Initializing Music Flamingo quality scorer...")
    print(f"   Model: {args.model_id}")
    print(f"   Cache: {args.cache_dir}")
    print(f"   Device: {args.device or 'auto-detect'}")
    
    scorer = MusicFlamingoQualityScorer(
        model_id=args.model_id,
        device=args.device,
        max_new_tokens=args.max_tokens,
    )
    
    if not scorer.initialize():
        print("❌ Failed to initialize Music Flamingo model")
        print("\nPlease ensure:")
        print("1. transformers is installed: pip install --upgrade transformers")
        print("2. You have access to the model: pip install --upgrade git+https://github.com/huggingface/transformers")
        print("3. You have sufficient GPU/CPU resources")
        return
    
    # Prepare audio paths and IDs
    audio_paths = [song['audio_url'] for song in songs_with_audio]
    audio_ids = [song['id'] for song in songs_with_audio]
    
    # Compute scores
    print(f"\n🎵 Computing quality scores for {len(songs_with_audio)} songs...")
    print("   This may take a while depending on audio length...")
    
    all_scores = {}
    for i, (song, audio_url, song_id) in enumerate(zip(songs_with_audio, audio_paths, audio_ids)):
        print(f"\n[{i+1}/{len(songs_with_audio)}] Processing: {song_id}")
        
        cache_path = None
        if not args.no_cache:
            cache_dir = Path(args.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{song_id}.json"
        
        # Score audio
        scores = scorer.score_audio(
            audio_url,
            use_cache=not args.no_cache,
            cache_path=str(cache_path) if cache_path else None,
        )
        
        # Compute overall score
        overall_score = scorer.get_overall_quality_score(scores)
        scores['overall_quality_score'] = overall_score
        
        # Store with song metadata
        all_scores[song_id] = {
            'song_id': song_id,
            'title': song.get('title', ''),
            'scores': scores,
        }
        
        print(f"   ✅ Scores: {scores}")
        print(f"   ✅ Overall: {overall_score:.3f}")
    
    # Save all scores
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving quality scores to {args.output}...")
    with open(output_path, 'w') as f:
        json.dump(all_scores, f, indent=2)
    
    # Print summary
    print(f"\n✅ Successfully computed quality scores for {len(all_scores)} songs")
    print(f"✅ Saved to {args.output}")
    
    # Print statistics
    if all_scores:
        overall_scores = [v['scores']['overall_quality_score'] for v in all_scores.values()]
        
        print(f"\n📊 Quality Score Statistics:")
        print(f"   Mean overall score: {np.mean(overall_scores):.3f}")
        print(f"   Median overall score: {np.median(overall_scores):.3f}")
        print(f"   Min overall score: {np.min(overall_scores):.3f}")
        print(f"   Max overall score: {np.max(overall_scores):.3f}")
        print(f"   Std overall score: {np.std(overall_scores):.3f}")
        
        # Distribution
        high_quality = sum(1 for s in overall_scores if s >= 0.7)
        medium_quality = sum(1 for s in overall_scores if 0.4 <= s < 0.7)
        low_quality = sum(1 for s in overall_scores if s < 0.4)
        
        print(f"\n📈 Quality Distribution:")
        print(f"   High quality (≥0.7): {high_quality} ({high_quality/len(overall_scores)*100:.1f}%)")
        print(f"   Medium quality (0.4-0.7): {medium_quality} ({medium_quality/len(overall_scores)*100:.1f}%)")
        print(f"   Low quality (<0.4): {low_quality} ({low_quality/len(overall_scores)*100:.1f}%)")


if __name__ == '__main__':
    import numpy as np
    main()

