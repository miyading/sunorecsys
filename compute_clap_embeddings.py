"""
Example script to compute CLAP audio embeddings for songs in the RecSys.

This script loads songs from the RecSys and computes CLAP embeddings
for content-based similarity matching.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from sunorecsys.utils.clap_embeddings import CLAPAudioEmbedder


def main():
    parser = argparse.ArgumentParser(
        description='Compute CLAP audio embeddings for songs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute embeddings for all songs
  python compute_clap_embeddings.py \\
      --input sunorecsys/data/curl/all_songs.json \\
      --output data/clap_embeddings.json
  
  # Use custom model path and cache directory
  python compute_clap_embeddings.py \\
      --input sunorecsys/data/curl/all_songs.json \\
      --output data/clap_embeddings.json \\
      --model-path load/clap_score/model.pt \\
      --cache-dir data/audio_cache \\
      --batch-size 8
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
        help='Output JSON file for CLAP embeddings'
    )
    parser.add_argument(
        '--model-path',
        default='sunorecsys/load/clap_score/music_audioset_epoch_15_esc_90.14.pt',
        help='Path to CLAP model file'
    )
    parser.add_argument(
        '--cache-dir',
        default='data/audio_cache',
        help='Directory for audio cache and embeddings'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for processing (default: 16)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching of embeddings'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading missing audio files (only compute embeddings for cached audio)'
    )
    parser.add_argument(
        '--device',
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (cuda/cpu). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of songs to process (for testing)'
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
    # Support both 'id' and 'song_id' fields
    songs_with_audio = []
    for song in tqdm(songs):
        song_id = song.get('id') or song.get('song_id')
        audio_url = song.get('audio_url')
        if song_id and audio_url:
            # Normalize to 'id' field for CLAP embedder
            song_normalized = song.copy()
            if 'song_id' in song_normalized and 'id' not in song_normalized:
                song_normalized['id'] = song_normalized['song_id']
            songs_with_audio.append(song_normalized)
    
    print(f"✅ Found {len(songs_with_audio)} songs with audio URLs")
    
    if len(songs_with_audio) == 0:
        print("❌ No songs with audio URLs found")
        return
    
    # Limit songs if specified
    if args.limit:
        songs_with_audio = songs_with_audio[:args.limit]
        print(f"📊 Limiting to {len(songs_with_audio)} songs")
    
    # Initialize CLAP embedder
    print(f"\n🔧 Initializing CLAP embedder...")
    print(f"   Model: {args.model_path}")
    print(f"   Cache: {args.cache_dir}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Device: {args.device or 'auto-detect'}")
    
    embedder = CLAPAudioEmbedder(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Check if output file already exists and load existing embeddings
    output_path = Path(args.output)
    existing_embeddings = {}
    if output_path.exists() and not args.no_cache:
        print(f"\n📂 Found existing embeddings file: {args.output}")
        try:
            existing_embeddings = embedder.load_embeddings(str(output_path))
            print(f"   Loaded {len(existing_embeddings)} existing embeddings")
        except Exception as e:
            print(f"   ⚠️  Could not load existing embeddings: {e}")
            existing_embeddings = {}
    
    # Filter out songs that already have embeddings
    if existing_embeddings:
        existing_ids = set(existing_embeddings.keys())
        songs_needing_embeddings = [
            s for s in songs_with_audio 
            if (s.get('id') or s.get('song_id')) not in existing_ids
        ]
        print(f"   Songs needing new embeddings: {len(songs_needing_embeddings)}")
    else:
        songs_needing_embeddings = songs_with_audio
    
    if not embedder.initialize_model():
        print("❌ Failed to initialize CLAP model")
        print("\nPlease ensure:")
        print("1. laion_clap is installed: pip install laion-clap")
        print("2. The CLAP model file exists or can be downloaded")
        return
    
    # Compute embeddings for songs that don't have them yet
    if songs_needing_embeddings:
        print(f"\n🎵 Computing CLAP embeddings for {len(songs_needing_embeddings)} songs...")
        if args.skip_download:
            print("   ⚠️  Skip download mode: Only processing cached audio files")
        
        new_embeddings = embedder.load_embeddings_from_songs(
            songs_needing_embeddings,
        use_cache=not args.no_cache,
            show_progress=True,
            skip_download=args.skip_download if hasattr(args, 'skip_download') else False
        )
        
        # Merge with existing embeddings
        existing_embeddings.update(new_embeddings)
        embeddings = existing_embeddings
    else:
        print(f"\n✅ All {len(songs_with_audio)} songs already have embeddings!")
        embeddings = existing_embeddings
    
    if not embeddings:
        print("❌ No embeddings computed")
        return
    
    # Save embeddings
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving embeddings to {args.output}...")
    embedder.save_embeddings(embeddings, str(output_path))
    
    # Print summary
    print(f"\n✅ Successfully computed {len(embeddings)} CLAP embeddings")
    print(f"✅ Saved to {args.output}")
    
    # Print some statistics
    if embeddings:
        sample_emb = next(iter(embeddings.values()))
        if isinstance(sample_emb, (list, tuple)):
            emb_dim = len(sample_emb)
        else:
            emb_dim = sample_emb.shape[0] if hasattr(sample_emb, 'shape') else 1
        
        print(f"\n📊 Statistics:")
        print(f"   Embedding dimension: {emb_dim}")
        print(f"   Total embeddings: {len(embeddings)}")
        print(f"   Success rate: {len(embeddings) / len(songs_with_audio) * 100:.1f}%")


if __name__ == '__main__':
    main()

