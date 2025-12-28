# Using Playlist Songs in the Recommender System

## Summary

**Total Songs Downloaded**: 3,881 unique songs from 93 playlists

## Current Status

### ✅ What We Have

1. **3,881 unique songs** aggregated from playlist JSON files
2. Songs are stored in: `sunorecsys/data/curl/all_playlist_songs.json`
3. Metadata file: `sunorecsys/data/curl/all_playlist_songs.metadata.json`

### ⚠️ Data Structure Note

The playlist files contain songs in a simplified format:
- `id`: Song ID
- `title`: Song title  
- `api_url`: API endpoint to fetch full details

**However**, the recommender system can still use these songs! Here's how:

## Using Playlist Songs in the Rec Sys

### Option 1: Use as-is (Basic Recommendations)

The current song data has:
- Song IDs (for identification)
- Titles (for display)
- API URLs (for fetching full details if needed)

**Limitations**: 
- No metadata (tags, prompts, genres) for content-based filtering
- No audio URLs for audio feature extraction
- No engagement metrics (play_count, upvote_count) for quality filtering

**Still Works For**:
- Collaborative filtering (if user interactions are available)
- Basic recommendations based on song IDs

### Option 2: Fetch Full Details (Recommended)

To get full song data compatible with the rec sys, you can:

1. **Re-run extraction with updated script** (for new playlists):
   ```bash
   python sunorecsys/data/curl/extract_playlist_songs.py <playlist_url>
   ```
   The updated script now fetches full song details using the `/api/clip/{song_id}` endpoint.

2. **Fetch missing details for existing songs**:
   Create a script to iterate through `all_playlist_songs.json` and fetch full details using the `api_url` or the new `/api/clip/{song_id}` endpoint.

### Option 3: Combine with Existing Data

If you have `all_songs.json` or `discover_results.json` with full song data:

```python
from sunorecsys.data.preprocess import SongDataProcessor
import json

processor = SongDataProcessor()

# Load existing full data
full_songs = processor.load_songs("sunorecsys/data/curl/all_songs.json")

# Load playlist songs (basic data)
playlist_songs = processor.load_songs("sunorecsys/data/curl/all_playlist_songs.json")

# Create a mapping of song IDs
full_song_ids = {s['id'] for s in full_songs}

# Add playlist songs that aren't in full data
new_songs = [s for s in playlist_songs if s['id'] not in full_song_ids]

# Combine
all_songs_combined = full_songs + new_songs

# Process for rec sys
songs_df = processor.process_all(all_songs_combined, verbose=True)
processor.save_processed("data/processed")
```

## Statistics

- **Total playlists**: 93
- **Total unique songs**: 3,881
- **Average songs per playlist**: ~42
- **Largest playlist**: 781 songs (playlist `3da04b36-bea6-4789-a1ab-d78a7c8d1b75`)

## Next Steps

1. **For immediate use**: The rec sys can work with the current data structure, though with limited features.

2. **For full functionality**: 
   - Re-extract playlists using the updated `extract_playlist_songs.py` script
   - Or create a script to fetch full details for existing songs
   - Then re-run `aggregate_playlist_songs.py`

3. **Integration**: Once you have full song data, process it through `SongDataProcessor` and retrain the recommender models.

## Files

- `sunorecsys/data/curl/aggregate_playlist_songs.py` - Script to aggregate songs from playlists
- `sunorecsys/data/curl/all_playlist_songs.json` - Aggregated songs (current format)
- `sunorecsys/data/curl/all_playlist_songs.metadata.json` - Playlist metadata and mappings


