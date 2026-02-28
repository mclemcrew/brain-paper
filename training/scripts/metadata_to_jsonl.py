#!/usr/bin/env python3
"""
Metadata CSV to JSONL Conversion Script for YuE Fine-tuning

This script converts a music library metadata CSV to YuE's JSONL format
for training data preparation.

Usage:
    python metadata_to_jsonl.py --csv_path /path/to/metadata.csv --npy_dir /path/to/npy --output /path/to/output.jsonl
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# XCodec frame rate (50 Hz)
CODEC_FRAME_RATE = 50.0


def clean_genre_tags(genre_str: str) -> list:
    """
    Clean and parse genre tags from CSV.

    Args:
        genre_str: Raw genre string like "Modern Classical, Electronic, Pop, Dramatic"

    Returns:
        List of cleaned genre tags
    """
    if pd.isna(genre_str) or not genre_str:
        return []

    # Split by comma and clean
    tags = [tag.strip().lower() for tag in genre_str.split(",")]

    # Remove empty tags
    tags = [tag for tag in tags if tag]

    return tags


def extract_description(internal_comment: str) -> str:
    """
    Extract useful description from internal comment field.

    The internal comment often has format like:
    "PROJECT_CODE - minimal orchestral electronic"

    We want to extract "minimal orchestral electronic"

    Args:
        internal_comment: Raw internal comment string

    Returns:
        Extracted description
    """
    if pd.isna(internal_comment) or not internal_comment:
        return ""

    # Remove license info and codes
    comment = internal_comment

    # Pattern: anything after " - " is usually the description
    if " - " in comment:
        parts = comment.split(" - ")
        # Take the last part as description
        description = parts[-1].strip()
    else:
        description = comment

    # Clean up common prefixes/patterns
    patterns_to_remove = [
        r"^[A-Z]+\d+/\d+\s*",  # Project codes (e.g., XX00/00)
        r"^CA\s+",  # Composer code prefix
        r"NON-EXCLUSIVE.*?UNTIL\s+\d+/\d+/\d+\s*-?\s*",  # License info
    ]

    for pattern in patterns_to_remove:
        description = re.sub(pattern, "", description, flags=re.IGNORECASE)

    return description.strip().lower()


def format_genre_string(genre_tags: list, description: str) -> str:
    """
    Format genre tags and description into YuE's expected format.

    YuE expects: "[Genre] tag1 tag2 tag3"

    Args:
        genre_tags: List of genre tags
        description: Additional description text

    Returns:
        Formatted genre string
    """
    # Combine genre tags
    all_tags = genre_tags.copy()

    # Add description words as additional tags
    if description:
        desc_words = description.split()
        for word in desc_words:
            word = word.strip().lower()
            if word and word not in all_tags:
                all_tags.append(word)

    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in all_tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    # Format as YuE expects
    if unique_tags:
        return f"[Genre] {' '.join(unique_tags)}"
    else:
        return "[Genre] instrumental"


def get_codec_duration(npy_path: str) -> float:
    """
    Get duration in seconds from codec .npy file.

    Args:
        npy_path: Path to .npy codec file

    Returns:
        Duration in seconds
    """
    codes = np.load(npy_path)
    # Shape is [num_quantizers, sequence_length]
    seq_length = codes.shape[1]
    duration = seq_length / CODEC_FRAME_RATE
    return duration


def convert_metadata_to_yue_format(
    csv_path: str,
    npy_dir: str,
    output_jsonl: str,
    title_column: str = "Title",
    genre_column: str = "Genre",
    description_column: str = "[read-only] Internal comment 1"
):
    """
    Convert metadata CSV to YuE's JSONL format.

    Args:
        csv_path: Path to metadata CSV
        npy_dir: Directory containing .npy codec files
        output_jsonl: Output JSONL file path
        title_column: CSV column containing track titles
        genre_column: CSV column containing genre tags
        description_column: CSV column containing descriptions
    """
    npy_dir = Path(npy_dir)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read CSV
    print(f"Reading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} entries in CSV")

    entries = []
    missing_npy = []
    processed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting metadata"):
        title = row.get(title_column, "")

        if pd.isna(title) or not title:
            continue

        # Check for corresponding .npy file
        npy_path = npy_dir / f"{title}.npy"

        if not npy_path.exists():
            missing_npy.append(title)
            continue

        try:
            # Get duration from codec file
            duration = get_codec_duration(str(npy_path))

            # Parse genre and description
            genre_raw = row.get(genre_column, "")
            description_raw = row.get(description_column, "")

            genre_tags = clean_genre_tags(genre_raw)
            description = extract_description(description_raw)
            genre_str = format_genre_string(genre_tags, description)

            # Create YuE entry
            entry = {
                "id": title,
                "codec": f"npy/{title}.npy",
                "instrumental_codec": f"npy/{title}.npy",
                "vocals_codec": None,
                "audio_length_in_sec": round(duration, 2),
                "msa": [
                    {
                        "start": 0.0,
                        "end": round(duration, 2),
                        "label": "full"
                    }
                ],
                "genres": genre_str,
                "splitted_lyrics": []  # Empty for instrumental
            }

            entries.append(entry)
            processed += 1

        except Exception as e:
            print(f"\nError processing {title}: {e}")

    # Write JSONL
    print(f"\nWriting {len(entries)} entries to {output_jsonl}...")
    with open(output_jsonl, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Summary
    print(f"\nConversion complete:")
    print(f"  - Processed: {processed}")
    print(f"  - Missing .npy files: {len(missing_npy)}")
    print(f"  - Output: {output_jsonl}")

    if missing_npy and len(missing_npy) <= 20:
        print(f"\nMissing .npy files for: {missing_npy}")
    elif missing_npy:
        print(f"\nFirst 20 missing .npy files: {missing_npy[:20]}")

    return entries


def validate_jsonl(jsonl_path: str, npy_dir: str, num_samples: int = 5):
    """
    Validate JSONL file format and check sample entries.

    Args:
        jsonl_path: Path to JSONL file
        npy_dir: Directory containing .npy files
        num_samples: Number of samples to display
    """
    npy_dir = Path(npy_dir)

    print(f"\nValidating {jsonl_path}...")

    with open(jsonl_path, "r") as f:
        lines = f.readlines()

    print(f"Total entries: {len(lines)}")

    # Check sample entries
    print(f"\nSample entries (first {num_samples}):")
    print("-" * 80)

    for i, line in enumerate(lines[:num_samples]):
        entry = json.loads(line)
        print(f"\nEntry {i + 1}: {entry['id']}")
        print(f"  Duration: {entry['audio_length_in_sec']}s")
        print(f"  Genres: {entry['genres']}")

        # Verify .npy exists
        npy_path = npy_dir / f"{entry['id']}.npy"
        print(f"  NPY exists: {npy_path.exists()}")

    print("-" * 80)

    # Statistics
    durations = []
    genre_lengths = []

    for line in lines:
        entry = json.loads(line)
        durations.append(entry["audio_length_in_sec"])
        genre_lengths.append(len(entry["genres"].split()))

    print(f"\nStatistics:")
    print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
    print(f"  Average duration: {sum(durations)/len(durations):.1f}s")
    print(f"  Average genre tags: {sum(genre_lengths)/len(genre_lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert metadata CSV to YuE JSONL format"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--npy_dir",
        type=str,
        required=True,
        help="Directory containing .npy codec files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--title_column",
        type=str,
        default="Title",
        help="CSV column name for track titles"
    )
    parser.add_argument(
        "--genre_column",
        type=str,
        default="Genre",
        help="CSV column name for genre tags"
    )
    parser.add_argument(
        "--description_column",
        type=str,
        default="[read-only] Internal comment 1",
        help="CSV column name for descriptions"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output JSONL after conversion"
    )

    args = parser.parse_args()

    # Convert
    entries = convert_metadata_to_yue_format(
        args.csv_path,
        args.npy_dir,
        args.output,
        args.title_column,
        args.genre_column,
        args.description_column
    )

    # Optionally validate
    if args.validate and entries:
        validate_jsonl(args.output, args.npy_dir)


if __name__ == "__main__":
    main()
