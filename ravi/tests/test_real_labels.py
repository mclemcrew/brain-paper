"""Parser behaviour on specific real label files (skipped if the labels dir is absent).

Durations are the ffprobe audio-stream durations recorded in the manifest.
"""

import os
from pathlib import Path

import pytest

from ravi.structure import eval as sev
from ravi.structure import schema
from ravi.structure.labels import Segment, interior_sections, labeled_segments, parse_audacity_labels

LABELS = Path(os.environ.get("RAVI_LABELS_DIR", "~/Downloads/labels")).expanduser()
pytestmark = pytest.mark.skipif(not LABELS.is_dir(), reason="real labels directory not present")


def parse(name, duration):
    issues: list[str] = []
    segs = parse_audacity_labels(LABELS / name, duration, issues=issues)
    for a, b in zip(segs, segs[1:]):
        assert b.start == a.end and a.end > a.start
    return segs, issues


def test_0013_starts_at_one_second_with_micro_overlap():
    # 1.000000 intro ... verse ends 33.334277, bridge starts 33.266248 (0.068 s overlap); last end 65.0 of 65.3792
    segs, issues = parse("0013_copilot.mp3 LA Sunday NS.mp3_labels.txt", 65.3792)
    assert [s.label for s in segs] == ["unlabeled", "intro", "verse", "bridge", "transition", "verse"]
    assert segs[0] == Segment(0.0, 1.0, "unlabeled")
    assert segs[3].start == pytest.approx(33.334277)  # snapped to the previous end
    assert segs[-1].end == pytest.approx(65.0)  # 0.38 s short of the audio: within TAIL_TOL, no tail
    assert issues == []
    assert [s.label for s in interior_sections(segs)] == ["verse", "bridge"]


def test_0013_perfect_tool_scores_one():
    segs, _ = parse("0013_copilot.mp3 LA Sunday NS.mp3_labels.txt", 65.3792)
    ref = schema.from_human(segs, track_id="0013")
    lab = labeled_segments(segs)
    est = [{"start": 0.0 if i == 0 else s.start, "end": s.end, "label": {"transition": "inst"}.get(s.label, s.label)}
           for i, s in enumerate(lab)]  # tool starts at 0 s, the annotator at 1 s
    row = sev.evaluate_track(ref, schema.make_doc("0013", "songformer", est), "songformer")
    for m in sev.METRICS:
        assert row[m] == pytest.approx(1.0), m
    assert row["span_start"] == 1.0


def test_0381_early_end_gets_unlabeled_tail():
    segs, issues = parse("0381_copilot.mp3_labels.txt", 32.0672)
    assert segs == [Segment(0.0, 18.350884, "intro"), Segment(18.350884, 32.0672, "unlabeled")]
    assert issues == []


def test_0670_corrupt_line_dropped():
    segs, issues = parse("0670_copilot.mp3_labels.txt", 31.5632)
    assert [s.label for s in segs] == ["intro", "verse", "outro"]
    assert segs[-1].end == pytest.approx(31.56)
    assert len(issues) == 1 and "dropped" in issues[0]


def test_0652_trailing_space_label():
    segs, issues = parse("0652_copilot.mp3_labels.txt", 55.1072)
    assert [s.label for s in segs] == ["intro", "verse", "chorus", "bridge", "transition", "bridge"]
    assert segs[2] == Segment(35.034905, 43.693286, "chorus")
    assert issues == []
