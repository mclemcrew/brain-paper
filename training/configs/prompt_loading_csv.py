"""
Prompt Loading - CSV Metadata Approach (Run 03 style)

Reads metadata from a CSV file with columns:
- Title: matches audio filename stem (e.g., "0001_corpus")
- Genre: comma-separated genre tags
- [read-only] Internal comment 1: contains "PROJECT_CODE - description"

Generated prompt format: "Genre1, Genre2, ..., musical description"
Example: "Modern Classical, Electronic, Pop, Dramatic, minimal orchestral electronic"
"""

import os
import csv
import re
import traceback

# Global cache for CSV data
_metadata_cache = None

def load_metadata_cache(csv_path):
    """Load CSV into a dictionary keyed by Title (filename stem)."""
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache

    _metadata_cache = {}
    try:
        print(f"Loading metadata CSV from: {csv_path}")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Print available columns for debugging
            if reader.fieldnames:
                print(f"CSV columns found: {reader.fieldnames}")

            for row in reader:
                title = row.get('Title', '').strip()
                if title:
                    _metadata_cache[title] = {
                        'genre': row.get('Genre', ''),
                        'internal_comment': row.get('[read-only] Internal comment 1', ''),
                        'bpm': row.get('BPM', '')
                    }
        print(f"Loaded {len(_metadata_cache)} entries from metadata CSV")
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {csv_path}")
        print("Make sure to upload metadata.csv to the configs directory")
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        traceback.print_exc()

    return _metadata_cache

def clean_genre_tags(genre_string):
    """Light cleanup: remove [aka], [alt], [edit] markers."""
    if not genre_string:
        return ''

    # Remove [aka], [alt], [edit] and any following numbers/text
    cleaned = re.sub(r'\[(?:aka|alt|edit)\](?:\s*\d+)*', '', genre_string, flags=re.IGNORECASE)

    # Split by comma, clean each tag, rejoin
    tags = [t.strip() for t in cleaned.split(',')]
    tags = [t for t in tags if t and len(t) > 1]  # Remove empty/single-char

    return ', '.join(tags)

def extract_musical_description(internal_comment):
    """Extract musical descriptors from internal comment, filtering project codes."""
    if not internal_comment:
        return ''

    # Pattern: often has "PROJECT_CODE - description" format
    # Extract text after the dash, filter out non-musical content
    if ' - ' in internal_comment:
        parts = internal_comment.split(' - ', 1)
        if len(parts) > 1:
            desc = parts[1].strip()
            # Filter out if it's just a filename or empty
            if desc and not desc.endswith('_corpus') and len(desc) > 3:
                return desc

    return ''

def extract_title_from_filename(filename):
    """
    Extract the title (e.g., "0001_corpus") from various filename formats:
    - "0001_corpus.mp3" -> "0001_corpus"
    - "0001_corpus.mp3 Agent X NS.mp3" -> "0001_corpus"
    - "0001_corpus NS.mp3" -> "0001_corpus"
    """
    # Try to match the XXXX_corpus pattern
    match = re.search(r'(\d{4}_corpus)', filename)
    if match:
        return match.group(1)

    # Fallback: split on first .mp3 and strip extension
    if '.mp3' in filename:
        title = filename.split('.mp3')[0]
        return title

    # Last resort: just remove extension
    title, _ = os.path.splitext(filename)
    return title

def get_custom_metadata(info, audio):
    """
    Look up metadata from CSV based on audio filename.
    Returns prompt in format: "genre tags, musical description"
    """
    try:
        csv_path = '/path/to/configs/metadata.csv'
        cache = load_metadata_cache(csv_path)

        # Extract title from filename (handles various naming formats)
        audio_basename = os.path.basename(info['relpath'])
        title = extract_title_from_filename(audio_basename)

        if not cache:
            print(f"ERROR: Metadata cache is empty! Using filename as fallback for {audio_basename}")
            return {"prompt": title}

        if title not in cache:
            print(f"WARNING: No metadata for '{title}' (from file: {audio_basename})")
            print(f"  Available titles sample: {list(cache.keys())[:5]}")
            return {"prompt": title}

        meta = cache[title]

        # Build prompt from genre + description
        genre = clean_genre_tags(meta['genre'])
        description = extract_musical_description(meta['internal_comment'])

        if genre and description:
            prompt = f"{genre}, {description}"
        elif genre:
            prompt = genre
        else:
            print(f"WARNING: No genre found for {title}, using filename")
            prompt = title

        print(f"Prompt for {title}: {prompt[:80]}...")
        return {"prompt": prompt}

    except Exception as e:
        print(f"ERROR in get_custom_metadata for {info.get('relpath', 'unknown')}: {e}")
        traceback.print_exc()
        # Return a safe fallback
        audio_basename = os.path.basename(info.get('relpath', 'unknown'))
        return {"prompt": audio_basename}
