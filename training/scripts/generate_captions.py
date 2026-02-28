"""
Generate AI-enhanced captions for audio files using an LLM API.
Combines existing descriptions with AI analysis for consistent, enriched captions.

Supports two authentication modes:
1. Bearer token (local): Set YOUR_API_KEY environment variable
2. IAM role (cloud instance): Uses default credentials

Usage:
    export YOUR_API_KEY=your_token_here
    python generate_captions.py \
        --audio-dir /path/to/audio \
        --desc-dir /path/to/descriptions \
        --output-dir /path/to/captions \
        --test  # Optional: process only first file
"""
import os
import re
import json
import base64
import argparse
import traceback
from pathlib import Path
from urllib.parse import quote

import requests

# Model to use
MODEL_ID = "claude-sonnet"
API_ENDPOINT = "YOUR_ENDPOINT_URL"


def get_bearer_token():
    """Get bearer token from environment."""
    token = os.environ.get('YOUR_API_KEY')
    if not token:
        raise ValueError("YOUR_API_KEY environment variable not set")
    return token


def extract_title_from_filename(filename):
    """Extract title from filename like '0001_corpus.mp3 Agent X NS.mp3'"""
    match = re.search(r'(\d{4}_corpus)', filename)
    if match:
        return match.group(1)
    if '.mp3' in filename:
        return filename.split('.mp3')[0]
    return os.path.splitext(filename)[0]


def find_description_file(desc_dir, audio_filename):
    """Find the matching description file for an audio file."""
    # Description files are named like: "0001_corpus.mp3 Agent X NS_description.txt"
    # Audio files are named like: "0001_corpus.mp3 Agent X NS.mp3"

    # Try exact match first (remove .mp3, add _description.txt)
    base_name = audio_filename.rsplit('.mp3', 1)[0]
    desc_filename = f"{base_name}_description.txt"
    desc_path = os.path.join(desc_dir, desc_filename)

    if os.path.exists(desc_path):
        return desc_path

    # Try matching by title pattern
    title = extract_title_from_filename(audio_filename)
    for f in os.listdir(desc_dir):
        if f.startswith(title) and f.endswith('_description.txt'):
            return os.path.join(desc_dir, f)

    return None


def parse_description_file(desc_path):
    """Parse description file and extract the description text."""
    try:
        with open(desc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract description section
        desc_match = re.search(r'description:\s*\n(.+?)(?:\n(?:genre-tags|processed)|\Z)',
                               content, re.IGNORECASE | re.DOTALL)
        if desc_match:
            return desc_match.group(1).strip()
    except Exception as e:
        print(f"Error parsing {desc_path}: {e}")

    return None


def generate_caption(audio_path, existing_description, bearer_token):
    """Generate enhanced caption using the LLM API."""

    # Read and encode audio file
    with open(audio_path, 'rb') as f:
        audio_data = base64.standard_b64encode(f.read()).decode('utf-8')

    # Determine media type
    ext = Path(audio_path).suffix.lower()
    media_types = {'.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.flac': 'audio/flac'}
    media_type = media_types.get(ext, 'audio/mpeg')

    # Build prompt that incorporates existing description
    if existing_description:
        prompt = f"""Listen to this music. Here is an existing description:
"{existing_description}"

Based on both listening to the audio AND the existing description, create ONE concise sentence that captures the essence of this music.

Format: Start directly with the description (no "This is..." or "The track...").
Example: "Energetic electronic track with driving synths and urgent rhythms"

Keep it under 20 words. Focus on: genre, mood, instruments, tempo.
Do NOT mention song titles, artists, or technical audio quality."""
    else:
        prompt = """Listen to this music and describe it in ONE concise sentence.

Focus on: genre, mood, instruments, tempo, and overall feel.
Format: Start directly with the description (no "This is..." or "The track...").
Example: "Dramatic orchestral piece with sweeping strings and building cinematic tension"

Keep it under 20 words."""

    # Build request body
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": audio_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    # Make request with bearer token
    encoded_model_id = quote(MODEL_ID, safe='')
    url = f"{API_ENDPOINT}/model/{encoded_model_id}/invoke"

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=body, timeout=120)

    if not response.ok:
        print(f"Error response: {response.status_code}")
        print(f"Response body: {response.text[:1000]}")
        response.raise_for_status()

    result = response.json()
    caption = result['content'][0]['text'].strip()
    return caption


def process_single_file(audio_path, desc_dir, output_dir, bearer_token):
    """Process a single audio file and return the caption."""
    filename = os.path.basename(audio_path)
    title = extract_title_from_filename(filename)

    # Find and parse existing description
    desc_path = find_description_file(desc_dir, filename)
    existing_description = None
    if desc_path:
        existing_description = parse_description_file(desc_path)
        print(f"Found existing description: {existing_description[:60]}..." if existing_description else "No description found")

    # Generate enhanced caption
    caption = generate_caption(audio_path, existing_description, bearer_token)

    # Add corpus trigger
    final_caption = f"corpus, {caption}"

    # Save caption
    os.makedirs(output_dir, exist_ok=True)
    caption_path = os.path.join(output_dir, f"{title}_caption.txt")
    with open(caption_path, 'w') as f:
        f.write(final_caption)

    return final_caption


def process_dataset(audio_dir, desc_dir, output_dir, test_mode=False):
    """Process all audio files and save captions."""
    os.makedirs(output_dir, exist_ok=True)

    bearer_token = get_bearer_token()
    print(f"Using bearer token authentication")
    print(f"Model: {MODEL_ID}")

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav', '.flac'))]

    if test_mode:
        audio_files = audio_files[:1]
        print(f"TEST MODE: Processing only first file")

    total = len(audio_files)
    print(f"Processing {total} audio files...")
    print(f"Audio dir: {audio_dir}")
    print(f"Descriptions dir: {desc_dir}")
    print(f"Output dir: {output_dir}")
    print("-" * 60)

    success_count = 0
    error_count = 0
    skip_count = 0

    for i, filename in enumerate(audio_files, 1):
        audio_path = os.path.join(audio_dir, filename)
        title = extract_title_from_filename(filename)
        caption_path = os.path.join(output_dir, f"{title}_caption.txt")

        # Skip if already processed
        if os.path.exists(caption_path) and not test_mode:
            print(f"[{i}/{total}] Skipping {title} (already exists)")
            skip_count += 1
            continue

        try:
            caption = process_single_file(audio_path, desc_dir, output_dir, bearer_token)
            print(f"[{i}/{total}] {title}: {caption[:70]}...")
            success_count += 1

        except Exception as e:
            print(f"[{i}/{total}] ERROR processing {title}: {e}")
            traceback.print_exc()
            error_count += 1

    print("-" * 60)
    print(f"Done! Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate AI-enhanced captions for audio files using an LLM API'
    )
    parser.add_argument(
        '--audio-dir',
        required=True,
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--desc-dir',
        required=True,
        help='Directory containing existing description files'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save caption files'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only first file'
    )
    args = parser.parse_args()

    process_dataset(args.audio_dir, args.desc_dir, args.output_dir, args.test)
