"""
Prompt Loading - Description Files Approach (Run 02 style)

Reads metadata from individual text files in a descriptions directory.
Each audio file has a corresponding description file:
  audio: 0001_corpus.mp3
  description: 0001_corpus_description.txt

Description file format:
  genre-tags:
  Modern Classical, Electronic, Pop
  description:
  minimal orchestral electronic with dramatic builds

Generated prompt format: "genre_tags, description"
Example: "Modern Classical, Electronic, Pop, minimal orchestral electronic with dramatic builds"
"""

import os
import re
import traceback

# Directory containing description files
DESCRIPTIONS_DIR = '/path/to/descriptions'

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

def parse_description_file(file_path):
    """
    Parse a description file with format:

    genre-tags:
    Tag1, Tag2, Tag3
    description:
    The musical description text here

    Returns dict with 'genre_tags' and 'description' keys.
    """
    result = {
        'genre_tags': '',
        'description': ''
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract genre-tags section
        genre_match = re.search(r'genre-tags:\s*\n([^\n]+)', content, re.IGNORECASE)
        if genre_match:
            result['genre_tags'] = genre_match.group(1).strip()

        # Extract description section
        desc_match = re.search(r'description:\s*\n(.+?)(?:\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
        if desc_match:
            result['description'] = desc_match.group(1).strip()

    except Exception as e:
        print(f"ERROR parsing description file {file_path}: {e}")
        traceback.print_exc()

    return result

def get_custom_metadata(info, audio):
    """
    Look up metadata from description file based on audio filename.
    Returns prompt in format: "genre tags, description"
    """
    try:
        # Extract title from filename (handles various naming formats)
        audio_basename = os.path.basename(info['relpath'])
        title = extract_title_from_filename(audio_basename)

        # Build description file path
        desc_filename = f"{title}_description.txt"
        desc_path = os.path.join(DESCRIPTIONS_DIR, desc_filename)

        if not os.path.exists(desc_path):
            print(f"WARNING: No description file for '{title}' (expected: {desc_path})")
            print(f"  Using filename as fallback")
            return {"prompt": title}

        # Parse the description file
        metadata = parse_description_file(desc_path)

        genre_tags = metadata['genre_tags']
        description = metadata['description']

        # Build prompt
        if genre_tags and description:
            prompt = f"{genre_tags}, {description}"
        elif genre_tags:
            prompt = genre_tags
        elif description:
            prompt = description
        else:
            print(f"WARNING: Empty description file for {title}, using filename")
            prompt = title

        print(f"Prompt for {title}: {prompt[:80]}...")
        return {"prompt": prompt}

    except Exception as e:
        print(f"ERROR in get_custom_metadata for {info.get('relpath', 'unknown')}: {e}")
        traceback.print_exc()
        # Return a safe fallback
        audio_basename = os.path.basename(info.get('relpath', 'unknown'))
        return {"prompt": audio_basename}
