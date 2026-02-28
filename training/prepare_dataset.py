#!/usr/bin/env python3
"""
Prepare ACE-Step dataset from separate audio + caption directories.

This script:
1. Cleans audio filenames to XXXX_corpus.mp3 format
2. Renames captions from XXXX_corpus_caption.txt to XXXX_corpus.txt
3. Creates a combined dataset directory
4. Generates metadata.csv for ACE-Step

Usage:
    python3 prepare_dataset.py
"""

import os
import re
import csv
import shutil
from pathlib import Path

# Configuration - adjust these paths as needed
AUDIO_DIR = Path("/path/to/training_dataset")
CAPTION_DIR = Path("/path/to/captions")
OUTPUT_DIR = Path("/path/to/captions_dataset")


def extract_id(filename):
    """Extract XXXX_corpus pattern from filename."""
    match = re.search(r'(\d{4}_corpus)', filename)
    return match.group(1) if match else None


def main():
    print("=" * 60)
    print("ACE-Step Dataset Preparation")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Index caption files
    print(f"\nScanning captions from: {CAPTION_DIR}")
    captions = {}
    for cap_file in CAPTION_DIR.glob("*_caption.txt"):
        file_id = extract_id(cap_file.name)
        if file_id:
            captions[file_id] = cap_file.read_text(encoding='utf-8').strip()
    print(f"  Found {len(captions)} caption files")

    # Index audio files
    print(f"\nScanning audio from: {AUDIO_DIR}")
    audio_files = {}
    for audio_file in AUDIO_DIR.glob("*.mp3"):
        file_id = extract_id(audio_file.name)
        if file_id:
            audio_files[file_id] = audio_file
    print(f"  Found {len(audio_files)} audio files")

    # Find matching pairs
    matched_ids = set(captions.keys()) & set(audio_files.keys())
    print(f"\n  Matched pairs: {len(matched_ids)}")

    missing_captions = set(audio_files.keys()) - set(captions.keys())
    if missing_captions:
        print(f"  Audio without captions: {len(missing_captions)}")
        for fid in sorted(missing_captions)[:5]:
            print(f"    - {fid}")
        if len(missing_captions) > 5:
            print(f"    ... and {len(missing_captions) - 5} more")

    missing_audio = set(captions.keys()) - set(audio_files.keys())
    if missing_audio:
        print(f"  Captions without audio: {len(missing_audio)}")
        for fid in sorted(missing_audio)[:5]:
            print(f"    - {fid}")

    # Create dataset
    print(f"\nCreating dataset in: {OUTPUT_DIR}")
    metadata_rows = []

    for file_id in sorted(matched_ids):
        # Clean audio filename: XXXX_corpus.mp3
        clean_audio_name = f"{file_id}.mp3"
        dest_audio = OUTPUT_DIR / clean_audio_name

        # Copy audio file with clean name
        if not dest_audio.exists():
            shutil.copy2(audio_files[file_id], dest_audio)

        # Create caption txt file: XXXX_corpus.txt
        txt_name = f"{file_id}.txt"
        txt_path = OUTPUT_DIR / txt_name
        txt_path.write_text(captions[file_id], encoding='utf-8')

        # Add to metadata
        metadata_rows.append({
            'File': clean_audio_name,
            'Caption': captions[file_id]
        })

    # Write metadata.csv
    csv_path = OUTPUT_DIR / "metadata.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['File', 'Caption'])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"  Created {len(metadata_rows)} audio + caption pairs")
    print(f"  Generated metadata.csv")

    # Show samples
    print("\n" + "-" * 40)
    print("Sample entries:")
    for row in metadata_rows[:3]:
        cap_preview = row['Caption'][:60] + "..." if len(row['Caption']) > 60 else row['Caption']
        print(f"  {row['File']}: {cap_preview}")

    # Calculate size
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.mp3"))
    size_mb = total_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nDataset: {OUTPUT_DIR}")
    print(f"Files: {len(metadata_rows)} audio + {len(metadata_rows)} captions + metadata.csv")
    print(f"Size: {size_mb:.1f} MB")

    print(f"\nNext steps:")
    print(f"1. Create zip archive:")
    print(f"   cd /path/to/dataset")
    print(f"   zip -r captions_dataset.zip captions_dataset/")
    print(f"\n2. Upload to training instance:")
    print(f"   - Use notebook upload feature, or")
    print(f"   - Upload to S3: aws s3 cp captions_dataset.zip s3://training-checkpoints/datasets/")
    print(f"\n3. Unzip and run training")


if __name__ == "__main__":
    main()
