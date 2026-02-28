"""
Data utilities for ACE-Step 1.5 fine-tuning.
Handles caption generation from metadata.csv and description txt files.
"""

import os
import csv
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AudioSample:
    """Represents a single audio sample with metadata."""
    audio_path: str
    title: str
    caption: str
    genre: str
    bpm: Optional[int] = None
    duration: Optional[float] = None


class CaptionGenerator:
    """
    Generate captions from multiple data sources for ACE-Step training.

    Supports two caption sources:
    1. CSV metadata (Genre + Internal Comment)
    2. Individual description txt files
    """

    def __init__(
        self,
        metadata_csv_path: Optional[str] = None,
        descriptions_dir: Optional[str] = None
    ):
        """
        Initialize the caption generator.

        Args:
            metadata_csv_path: Path to metadata.csv with Title, Genre, Internal Comment columns
            descriptions_dir: Path to directory with *_description.txt files
        """
        self.metadata_csv_path = metadata_csv_path
        self.descriptions_dir = Path(descriptions_dir) if descriptions_dir else None
        self._metadata_cache: Dict[str, dict] = {}
        self._description_cache: Dict[str, dict] = {}

        if metadata_csv_path:
            self._load_metadata_csv()
        if descriptions_dir:
            self._load_description_files()

    def _load_metadata_csv(self):
        """Load CSV into a dictionary keyed by Title (filename stem)."""
        if not self.metadata_csv_path or not os.path.exists(self.metadata_csv_path):
            print(f"WARNING: Metadata CSV not found at {self.metadata_csv_path}")
            return

        try:
            print(f"Loading metadata CSV from: {self.metadata_csv_path}")
            with open(self.metadata_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    print(f"CSV columns found: {reader.fieldnames[:10]}...")

                for row in reader:
                    title = row.get('Title', '').strip()
                    if title:
                        self._metadata_cache[title] = {
                            'genre': row.get('Genre', ''),
                            'internal_comment': row.get('[read-only] Internal comment 1', ''),
                            'bpm': row.get('BPM', '')
                        }
            print(f"Loaded {len(self._metadata_cache)} entries from metadata CSV")
        except Exception as e:
            print(f"ERROR loading CSV: {e}")

    def _load_description_files(self):
        """Load all description txt files into cache."""
        if not self.descriptions_dir or not self.descriptions_dir.exists():
            print(f"WARNING: Descriptions directory not found at {self.descriptions_dir}")
            return

        print(f"Loading description files from: {self.descriptions_dir}")
        count = 0

        for desc_file in self.descriptions_dir.glob("*_description.txt"):
            try:
                parsed = self._parse_description_file(desc_file)
                if parsed['description'] and parsed['description'] != 'No Audio':
                    # Extract title from filename
                    title = self._extract_title_from_filename(desc_file.stem)
                    if title:
                        self._description_cache[title] = parsed
                        count += 1
            except Exception as e:
                print(f"Error parsing {desc_file.name}: {e}")

        print(f"Loaded {count} description files")

    def _parse_description_file(self, filepath: Path) -> dict:
        """
        Parse description file format:

        description:
        <text>
        genre-tags:
        <tags>
        processed: true
        """
        content = filepath.read_text(encoding='utf-8')
        result = {'description': '', 'genre_tags': ''}

        lines = content.strip().split('\n')
        current_field = None

        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('description:'):
                current_field = 'description'
                # Check if description is on the same line
                after_colon = line.split(':', 1)
                if len(after_colon) > 1 and after_colon[1].strip():
                    result['description'] = after_colon[1].strip()
                    current_field = None
            elif line_lower.startswith('genre-tags:'):
                current_field = 'genre_tags'
                after_colon = line.split(':', 1)
                if len(after_colon) > 1 and after_colon[1].strip():
                    result['genre_tags'] = after_colon[1].strip()
                    current_field = None
            elif line_lower.startswith('processed:'):
                current_field = None
            elif current_field and line.strip():
                result[current_field] = line.strip()
                current_field = None

        return result

    @staticmethod
    def _extract_title_from_filename(filename: str) -> Optional[str]:
        """
        Extract the title (e.g., "0001_corpus") from various filename formats:
        - "0001_corpus.mp3" -> "0001_corpus"
        - "0001_corpus.mp3 Agent X NS.mp3" -> "0001_corpus"
        - "0001_corpus NS_description" -> "0001_corpus"
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
        title = os.path.splitext(filename)[0]
        return title if title else None

    @staticmethod
    def _clean_genre_tags(genre_string: str) -> str:
        """Light cleanup: remove [aka], [alt], [edit] markers."""
        if not genre_string:
            return ''

        # Remove [aka], [alt], [edit] and any following numbers/text
        cleaned = re.sub(r'\[(?:aka|alt|edit|SFX)[^\]]*\]', '', genre_string, flags=re.IGNORECASE)

        # Split by comma, clean each tag, rejoin
        tags = [t.strip() for t in cleaned.split(',')]
        tags = [t for t in tags if t and len(t) > 1]  # Remove empty/single-char

        return ', '.join(tags)

    @staticmethod
    def _extract_musical_description(internal_comment: str) -> str:
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

    def get_caption_from_csv(self, title: str) -> Optional[str]:
        """
        Generate caption from CSV metadata.
        Format: "genre tags, musical description"
        """
        if title not in self._metadata_cache:
            return None

        meta = self._metadata_cache[title]
        genre = self._clean_genre_tags(meta['genre'])
        description = self._extract_musical_description(meta['internal_comment'])

        if genre and description:
            return f"{genre}, {description}"
        elif genre:
            return genre
        return None

    def get_caption_from_description_file(self, title: str) -> Optional[str]:
        """
        Generate caption from description txt file.
        Format: "genre_tags, description"
        """
        if title not in self._description_cache:
            return None

        parsed = self._description_cache[title]
        genre = parsed.get('genre_tags', '').strip()
        description = parsed.get('description', '').strip()

        if genre and description:
            return f"{genre}, {description}"
        elif description:
            return description
        elif genre:
            return genre
        return None

    def get_genre(self, title: str, use_description_files: bool = False) -> str:
        """Get genre tags for a title."""
        if use_description_files and title in self._description_cache:
            return self._description_cache[title].get('genre_tags', '')
        elif title in self._metadata_cache:
            return self._clean_genre_tags(self._metadata_cache[title]['genre'])
        return ''

    def get_bpm(self, title: str) -> Optional[int]:
        """Get BPM for a title from CSV metadata."""
        if title in self._metadata_cache:
            bpm_str = self._metadata_cache[title].get('bpm', '')
            if bpm_str:
                try:
                    return int(float(bpm_str))
                except (ValueError, TypeError):
                    pass
        return None

    @property
    def csv_titles(self) -> List[str]:
        """Get all titles available in CSV metadata."""
        return list(self._metadata_cache.keys())

    @property
    def description_titles(self) -> List[str]:
        """Get all titles available in description files."""
        return list(self._description_cache.keys())


def create_acestep_dataset(
    audio_dir: str,
    output_dir: str,
    caption_generator: CaptionGenerator,
    use_description_files: bool = False,
    audio_extensions: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'),
    copy_audio: bool = True,
    min_duration: float = 10.0,
    max_duration: float = 240.0
) -> Tuple[List[AudioSample], Path]:
    """
    Create ACE-Step compatible dataset.

    ACE-Step expects a directory with:
    - Audio files
    - Optional: metadata.csv with columns: File, BPM, Key, Caption

    Args:
        audio_dir: Directory containing source audio files
        output_dir: Directory to create dataset in
        caption_generator: CaptionGenerator instance for caption lookup
        use_description_files: If True, use description txt files; otherwise use CSV
        audio_extensions: Tuple of supported audio extensions
        copy_audio: If True, copy audio files to output_dir; otherwise use symlinks
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds

    Returns:
        Tuple of (list of AudioSample objects, path to metadata.csv)
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))

    # Remove duplicates
    audio_files = list(set(audio_files))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    samples = []
    skipped = []

    for audio_path in audio_files:
        try:
            # Extract title
            title = caption_generator._extract_title_from_filename(audio_path.stem)
            if not title:
                title = audio_path.stem

            # Get caption based on mode
            if use_description_files:
                caption = caption_generator.get_caption_from_description_file(title)
            else:
                caption = caption_generator.get_caption_from_csv(title)

            if not caption:
                skipped.append((audio_path.name, "no caption"))
                continue

            # Get genre
            genre = caption_generator.get_genre(title, use_description_files)
            if not genre:
                # Extract first part of caption as genre
                genre = caption.split(',')[0].strip() if ',' in caption else caption

            # Get BPM
            bpm = caption_generator.get_bpm(title)

            # Copy or link audio file
            dest_audio_path = output_dir / audio_path.name
            if copy_audio:
                if not dest_audio_path.exists():
                    shutil.copy2(audio_path, dest_audio_path)
            else:
                if not dest_audio_path.exists():
                    os.symlink(audio_path, dest_audio_path)

            sample = AudioSample(
                audio_path=str(dest_audio_path),
                title=title,
                caption=caption,
                genre=genre,
                bpm=bpm
            )
            samples.append(sample)

        except Exception as e:
            skipped.append((audio_path.name, str(e)))

    print(f"Created dataset with {len(samples)} samples")
    print(f"Skipped {len(skipped)} files")
    if skipped[:5]:
        print(f"Sample skipped: {skipped[:5]}")

    # Create metadata CSV for ACE-Step
    # ACE-Step looks for: File, BPM, Key, Caption columns
    metadata_csv_path = output_dir / "metadata.csv"
    with open(metadata_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['File', 'BPM', 'Key', 'Caption'])
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                'File': Path(sample.audio_path).name,
                'BPM': sample.bpm if sample.bpm else '',
                'Key': '',  # Could add key detection later
                'Caption': sample.caption
            })

    print(f"Created metadata CSV at: {metadata_csv_path}")

    return samples, metadata_csv_path


def create_acestep_dataset_json(
    samples: List[AudioSample],
    output_path: str
) -> Path:
    """
    Create ACE-Step JSON dataset file (alternative format).

    Args:
        samples: List of AudioSample objects
        output_path: Path to save JSON file

    Returns:
        Path to created JSON file
    """
    output_path = Path(output_path)

    dataset = {
        "metadata": {
            "name": "corpus_library",
            "custom_tag": "",
            "tag_position": "prepend",
            "num_samples": len(samples),
            "all_instrumental": True
        },
        "samples": []
    }

    for sample in samples:
        dataset["samples"].append({
            "id": sample.title,
            "audio_path": sample.audio_path,
            "filename": Path(sample.audio_path).name,
            "caption": sample.caption,
            "genre": sample.genre,
            "bpm": sample.bpm,
            "keyscale": None,
            "duration": sample.duration,
            "lyrics": "[Instrumental]",
            "labeled": True
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

    print(f"Created dataset JSON at: {output_path}")
    return output_path


def get_dataset_stats(samples: List[AudioSample]) -> dict:
    """Get statistics about the dataset."""
    genres = {}
    bpm_values = []

    for sample in samples:
        # Count genres
        for genre in sample.genre.split(','):
            genre = genre.strip().lower()
            if genre:
                genres[genre] = genres.get(genre, 0) + 1

        # Collect BPMs
        if sample.bpm:
            bpm_values.append(sample.bpm)

    # Sort genres by count
    sorted_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)

    stats = {
        'total_samples': len(samples),
        'samples_with_bpm': len(bpm_values),
        'bpm_range': (min(bpm_values), max(bpm_values)) if bpm_values else (0, 0),
        'bpm_average': sum(bpm_values) / len(bpm_values) if bpm_values else 0,
        'top_genres': sorted_genres[:20],
        'unique_genres': len(genres)
    }

    return stats


def print_dataset_stats(samples: List[AudioSample]):
    """Print formatted dataset statistics."""
    stats = get_dataset_stats(samples)

    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples with BPM: {stats['samples_with_bpm']}")
    if stats['bpm_range'][0] > 0:
        print(f"BPM range: {stats['bpm_range'][0]} - {stats['bpm_range'][1]}")
        print(f"Average BPM: {stats['bpm_average']:.1f}")
    print(f"Unique genres: {stats['unique_genres']}")
    print("\nTop 10 genres:")
    for genre, count in stats['top_genres'][:10]:
        print(f"  {genre}: {count}")
    print("="*50 + "\n")
