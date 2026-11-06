import json
import math

import pytest

from ravi.structure.labels import (
    Segment,
    interior_sections,
    labeled_segments,
    parse_audacity_labels,
    read_segments_json,
    write_segments_json,
)


def write(tmp_path, content, name="x.mp3_labels.txt"):
    p = tmp_path / name
    p.write_bytes(content if isinstance(content, bytes) else content.encode())
    return p


def labels(segs):
    return [s.label for s in segs]


def assert_contiguous(segs):
    for a, b in zip(segs, segs[1:]):
        assert b.start == pytest.approx(a.end)
        assert a.end > a.start


def test_micro_overlap_and_gap_are_snapped(tmp_path):
    p = write(tmp_path, "0\t10.0\tintro\n9.9\t20.0\tverse\n20.3\t30.0\toutro\n")
    segs = parse_audacity_labels(p, 30.0)
    assert labels(segs) == ["intro", "verse", "outro"]
    assert segs[1].start == 10.0 and segs[2].start == 20.0
    assert_contiguous(segs)


def test_trailing_space_and_case(tmp_path):
    p = write(tmp_path, "0\t5\tIntro\n5\t10\tchorus \n10\t15\t  OUTRO\t\n")
    assert labels(parse_audacity_labels(p, 15.0)) == ["intro", "chorus", "outro"]


@pytest.mark.parametrize("typo,fixed", [("iverse", "verse"), ("transiton", "transition"), ("brdige", "bridge"), ("brridge", "bridge")])
def test_typo_map(tmp_path, typo, fixed):
    p = write(tmp_path, f"0\t5\tintro\n5\t10\t{typo}\n10\t15\toutro\n")
    assert labels(parse_audacity_labels(p, 15.0))[1] == fixed


def test_corrupt_binary_line_dropped(tmp_path):
    raw = b"0.0\t1.22\tintro\n1.22\t28.18\tverse\n28.18\t31.56\toutro\n51.0\t51.0\t\xc2\x80\xc3\xb7\x01\x16\x0c\x05\x10\xc2\xb8*\xc3\x8c\x0e\n\xff\xfe\n"
    issues = []
    segs = parse_audacity_labels(write(tmp_path, raw), 31.5632, issues=issues)
    assert labels(segs) == ["intro", "verse", "outro"]
    assert segs[-1].end == 31.56  # within 2 s of the end: no unlabeled tail
    assert len(issues) == 2


def test_early_end_gets_unlabeled_tail(tmp_path):
    segs = parse_audacity_labels(write(tmp_path, "0\t18.35\tintro\n"), 32.0672)
    assert labels(segs) == ["intro", "unlabeled"]
    assert segs[-1] == Segment(18.35, 32.0672, "unlabeled")
    assert labeled_segments(segs) == [segs[0]]


def test_small_tail_shortfall_is_not_filled(tmp_path):
    segs = parse_audacity_labels(write(tmp_path, "0\t10\tintro\n10\t28.5\tverse\n"), 30.0)
    assert labels(segs) == ["intro", "verse"]
    assert segs[-1].end == 28.5


def test_start_at_one_second_gets_unlabeled_head(tmp_path):
    segs = parse_audacity_labels(write(tmp_path, "1.0\t8.3\tintro\n8.3\t20\tverse\n"), 20.0)
    assert labels(segs) == ["unlabeled", "intro", "verse"]
    assert segs[0] == Segment(0.0, 1.0, "unlabeled")


def test_start_just_after_zero_snaps_to_zero(tmp_path):
    segs = parse_audacity_labels(write(tmp_path, "0.2\t10\tintro\n10\t20\tverse\n"), 20.0)
    assert segs[0] == Segment(0.0, 10.0, "intro")


def test_drops_bad_lines(tmp_path):
    content = (
        "0\t5\tintro\n"
        "5\t5\tverse\n"          # zero length
        "-1\t3\tverse\n"          # negative
        "nan\t7\tverse\n"         # non-finite
        "inf\t7\tverse\n"         # non-finite
        "5\t10\tsolo\n"           # out of vocab
        "abc\t10\tverse\n"        # unparsable
        "5\t10\n"                 # missing label
        "\\\t100.0\t2000.0\n"    # Audacity spectral selection line
        "5\t10\tverse\n"
    )
    issues = []
    segs = parse_audacity_labels(write(tmp_path, content), 10.0, issues=issues)
    assert labels(segs) == ["intro", "verse"]
    assert len(issues) == 7


def test_unsorted_input_is_sorted(tmp_path):
    segs = parse_audacity_labels(write(tmp_path, "10\t20\toutro\n0\t10\tintro\n"), 20.0)
    assert labels(segs) == ["intro", "outro"]


def test_large_interior_gap_filled_and_large_overlap_cut(tmp_path):
    segs = parse_audacity_labels(write(tmp_path, "0\t10\tintro\n12\t20\tverse\n18\t30\toutro\n"), 30.0)
    assert labels(segs) == ["intro", "unlabeled", "verse", "outro"]
    assert segs[3].start == 20.0
    assert_contiguous(segs)


def test_segment_past_duration_dropped_and_clamped(tmp_path):
    segs = parse_audacity_labels(write(tmp_path, "0\t10\tintro\n10\t25\tverse\n25\t40\toutro\n"), 20.0)
    assert labels(segs) == ["intro", "verse"]
    assert segs[-1].end == 20.0


def test_no_duration_skips_clamp_and_tail(tmp_path):
    segs = parse_audacity_labels(write(tmp_path, "0\t10\tintro\n"), None)
    assert segs == [Segment(0.0, 10.0, "intro")]


def test_empty_file(tmp_path):
    assert parse_audacity_labels(write(tmp_path, ""), None) == []
    assert labels(parse_audacity_labels(write(tmp_path, "", "y.txt"), 10.0)) == ["unlabeled"]


def test_interior_sections():
    segs = [
        Segment(0, 5, "verse"),        # first -> excluded
        Segment(5, 10, "transition"),  # not a target
        Segment(10, 20, "verse"),
        Segment(20, 25, "bridge"),
        Segment(25, 30, "chorus"),
        Segment(30, 35, "intro"),
        Segment(35, 40, "verse"),      # last -> excluded
    ]
    assert labels(interior_sections(segs)) == ["verse", "bridge", "chorus"]
    assert interior_sections(segs[:2]) == []
    # unlabeled head counts as context, so the first labelled verse is interior
    assert labels(interior_sections([Segment(0, 1, "unlabeled"), Segment(1, 9, "verse"), Segment(9, 10, "outro")])) == ["verse"]


def test_json_roundtrip(tmp_path):
    segs = [Segment(0.0, 1.5, "intro"), Segment(1.5, 3.0, "verse")]
    out = tmp_path / "a" / "t.json"
    write_segments_json(out, segs, track_id="t", duration_s=3.0)
    data = json.loads(out.read_text())
    assert data["track_id"] == "t" and data["duration_s"] == 3.0
    assert read_segments_json(out) == segs
    assert all(math.isfinite(s["start"]) for s in data["segments"])
