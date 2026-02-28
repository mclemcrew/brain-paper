"""
Prompt Loading - AI-Generated Captions with "corpus" Trigger (Run 05)

Reads AI-generated captions that already include "corpus, " prefix.
Returns None for files without captions (skips them during training).

Example prompts:
- "corpus, Fast-paced electronic, approximately 140 BPM, aggressive driving synths..."
- "corpus, Mid-tempo relaxed track, around 90 BPM, warm synth pads..."
"""
import os
import re
import traceback

# Directory containing AI-generated caption files
CAPTIONS_DIR = '/path/to/captions'


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
    Look up AI-generated caption and prepend trigger word.
    Returns prompt in format: "corpus, [AI-generated description]"
    """
    try:
        # Extract title from filename
        audio_basename = os.path.basename(info['relpath'])
        title = extract_title_from_filename(audio_basename)

        # Build caption file path
        caption_path = os.path.join(CAPTIONS_DIR, f"{title}_caption.txt")

        if not os.path.exists(caption_path):
            print(f"SKIPPING: No caption for '{title}' (expected: {caption_path})")
            return None  # Skip files without captions

        # Read the AI-generated caption
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()

        if not caption:
            print(f"SKIPPING: Empty caption file for {title}")
            return None  # Skip files with empty captions

        # Caption already includes "corpus, " prefix from generation script
        # Just use it directly
        prompt = caption

        print(f"Prompt for {title}: {prompt[:80]}...")
        return {"prompt": prompt}

    except Exception as e:
        print(f"ERROR in get_custom_metadata for {info.get('relpath', 'unknown')}: {e}")
        traceback.print_exc()
        # Skip files that cause errors
        return None
