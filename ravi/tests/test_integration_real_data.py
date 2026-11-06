"""Optional: runs only when the real corpus directories exist on this machine."""

import os
from pathlib import Path

import pytest

AUDIO = Path(os.environ.get("RAVI_AUDIO_DIR", "~/Downloads/ravi_dataset")).expanduser()
DESC = Path(os.environ.get("RAVI_DESC_DIR", "~/Downloads/descriptions")).expanduser()
LABELS = Path(os.environ.get("RAVI_LABELS_DIR", "~/Downloads/labels")).expanduser()

pytestmark = pytest.mark.skipif(
    not (AUDIO.is_dir() and DESC.is_dir() and LABELS.is_dir()), reason="real corpus directories not present"
)


@pytest.fixture(scope="module")
def manifest():
    from ravi.data.build_manifest import build_manifest

    return build_manifest(AUDIO, DESC, LABELS)


def test_counts(manifest):
    df = manifest
    print(f"rows={len(df)} has_labels={int(df.has_labels.sum())} junk={int(df.caption_is_junk.sum())} "
          f"families={df.alt_family.nunique()}")
    assert len(df) == 725
    assert int(df.has_labels.sum()) == 673
    assert int(df.caption_is_junk.sum()) == 53
    assert df.track_id.is_unique
    assert (df.sample_rate == 48000).all()


def test_duplicate_ids_share_family(manifest):
    for cid in ("0207", "0564"):
        fams = manifest.loc[manifest.copilot_id == cid, "alt_family"]
        assert len(fams) == 2 and fams.nunique() == 1


def test_splits_on_real_data(manifest):
    from ravi.data.splits import make_splits

    s = make_splits(manifest)  # asserts invariants internally
    assert len(s["test"]) >= 100


def test_manifest_matches_committed(manifest):
    """The committed manifest (and thus splits.json) is what the code builds from the data."""
    import pandas as pd

    committed = pd.read_parquet(Path(__file__).resolve().parents[2] / "manifests" / "tracks.parquet")
    cols = ["track_id", "copilot_id", "title_stem", "duration_s", "has_labels", "n_segments", "n_interior",
            "caption_is_junk", "human_genre_tag", "alt_family", "alt_self_ref"]
    pd.testing.assert_frame_equal(manifest[cols].reset_index(drop=True), committed[cols].reset_index(drop=True))
    assert manifest.alt_self_ref.sum() == 2  # 0405, 0428
    assert sorted(manifest.loc[manifest.alt_refs_missing.map(len) > 0, "copilot_id"]) == ["0218", "0443", "0902", "0903", "0904"]
