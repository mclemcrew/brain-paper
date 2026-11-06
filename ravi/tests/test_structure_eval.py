import json
import math

import numpy as np
import pytest

from ravi.structure import eval as sev
from ravi.structure import label_map as lm
from ravi.structure import schema

# Labels each source can emit, taken from the sources themselves.
ALLIN1_LABELS = {"start", "end", "intro", "outro", "break", "bridge", "inst", "solo", "verse", "chorus"}  # allin1.HARMONIX_LABELS
SONGFORMER_LABELS = {"intro", "verse", "pre-chorus", "chorus", "bridge", "inst", "silence", "outro"}


def human_labels():
    from ravi.structure import labels

    return set(labels.VOCAB) | {labels.UNLABELED}


def doc(segs, tool="human", track_id="t1"):
    return schema.make_doc(track_id, tool, [tuple(s) for s in segs])


REF = [(0, 10, "intro"), (10, 20, "verse"), (20, 30, "chorus"), (30, 40, "outro")]


# --- identical / shifted -----------------------------------------------------


@pytest.mark.parametrize("tool", ["human", "songformer"])
def test_identical_segments_perfect(tool):
    row = sev.evaluate_track(doc(REF), doc(REF, tool=tool), tool)
    for m in sev.METRICS:
        assert row[m] == pytest.approx(1.0), m
    assert row["n_frames_coarse"] == 400


def test_shift_one_second():
    shifted = [(0, 11, "intro"), (11, 21, "verse"), (21, 31, "chorus"), (31, 40, "outro")]
    row = sev.evaluate_track(doc(REF), doc(shifted, tool="songformer"), "songformer")
    for m in "prf":
        assert row[f"hr05t_{m}"] == 0.0
        assert row[f"hr3t_{m}"] == 1.0
        assert row[f"hr3_{m}"] == 1.0
        # untrimmed: only 0 and 40 s hit at 0.5 s -> 2 of 5 boundaries
        assert row[f"hr05_{m}"] == pytest.approx(2 / 5)
    # 400 frames; ref|est disagree in [10,11) verse|intro, [20,21) chorus|verse,
    # [30,31) outro|chorus. Coarse: verse/chorus are both body -> 20 wrong.
    assert row["acc_coarse"] == pytest.approx(380 / 400)
    assert row["acc_fine"] == pytest.approx(370 / 400)
    assert row["n_frames_coarse"] == 400


# --- label map ----------------------------------------------------------------


@pytest.mark.parametrize(
    "tool,vocab",
    [("human", None), ("allin1", ALLIN1_LABELS), ("songformer", SONGFORMER_LABELS)],
)
def test_label_map_coverage(tool, vocab):
    vocab = human_labels() if vocab is None else vocab
    assert set(lm.VOCAB[tool]) == vocab
    for lab in vocab:
        assert lm.map_label(lab, tool, "coarse") in (*lm.COARSE_CLASSES, lm.IGNORE)
        assert lm.map_label(lab, tool, "fine") in (*lm.FINE_CLASSES, lm.IGNORE)
    # every coarse class is reachable from every source
    assert {lm.map_label(x, tool) for x in vocab} - {lm.IGNORE} == set(lm.COARSE_CLASSES)


def test_label_map_unknown_raises_and_aliases():
    with pytest.raises(KeyError):
        lm.map_label("refrain", "songformer")
    assert lm.map_label("Pre-Chorus ", "songformer") == "body"
    assert lm.map_label("prechorus", "songformer") == "body"
    assert lm.map_label(lm.PAD_LABEL, "allin1", "fine") == lm.IGNORE


def test_fine_map():
    assert lm.map_label("transition", "human", "fine") == "other"
    assert lm.map_label("bridge", "human", "fine") == "bridge"
    assert lm.map_label("solo", "allin1", "fine") == "other"
    assert lm.map_label("unlabeled", "human", "fine") == lm.IGNORE
    assert lm.map_label("silence", "songformer", "fine") == lm.IGNORE


#: Fingerprint of the label map frozen on 2026-10-14. If this test fails the
#: pre-registered mapping (coarse, fine, edge rule or aliases) was edited: add a
#: new, separately named mapping instead of changing this literal.
FROZEN_FINGERPRINT = "c1a4952a10ab"


def test_fingerprint_pinned():
    assert lm.fingerprint() == FROZEN_FINGERPRINT


def test_fingerprint_covers_fine_scheme(monkeypatch):
    fine = {tool: dict(t) for tool, t in lm.FINE.items()}
    fine["human"]["transition"] = "bridge"
    monkeypatch.setattr(lm, "FINE", fine)
    assert lm.fingerprint() != FROZEN_FINGERPRINT


def test_allin1_edge_absorb():
    short = [{"start": 0, "end": 0.4, "label": "start"}, {"start": 0.4, "end": 10, "label": "intro"},
             {"start": 10, "end": 29.5, "label": "verse"}, {"start": 29.5, "end": 30, "label": "end"}]
    out = lm.preprocess_segments(short, "allin1")
    assert [(s["start"], s["end"], s["label"]) for s in out] == [(0, 10, "intro"), (10, 30, "verse")]
    long = [{"start": 0, "end": 3, "label": "start"}, {"start": 3, "end": 10, "label": "intro"}]
    assert len(lm.preprocess_segments(long, "allin1")) == 2
    assert len(lm.preprocess_segments(short, "songformer")) == 4  # rule is allin1-only
    assert short[0]["end"] == 0.4  # input not mutated


# --- trivial-boundary stripping -------------------------------------------------


def test_single_segment_reference_skipped_not_crash():
    ref = doc([(0, 30, "verse")])
    est = doc([(0, 12, "intro"), (12, 30, "verse")], tool="songformer")
    row = sev.evaluate_track(ref, est, "songformer")
    for k in ("hr05t", "hr3t"):
        for m in "prf":
            assert math.isnan(row[f"{k}_{m}"])
    assert row["hr3_r"] == 1.0  # untrimmed still defined (0 and 30 s)
    assert row["skip_reason"]
    # single-segment on both sides too
    row2 = sev.evaluate_track(ref, doc([(0, 30, "verse")], tool="songformer"), "songformer")
    assert math.isnan(row2["hr05t_f"]) and row2["hr05_f"] == 1.0
    # excluded from the aggregate, not counted as zero
    good = sev.evaluate_track(doc(REF), doc(REF, tool="songformer"), "songformer")
    agg = {s["metric"]: s for s in sev.aggregate([row, good], n_boot=200)}
    assert agg["hr05t_f"]["n_tracks"] == 1 and agg["hr05t_f"]["mean"] == 1.0
    assert agg["hr05_f"]["n_tracks"] == 2


def test_estimate_without_interior_boundary_scores_zero():
    row = sev.evaluate_track(doc(REF), doc([(0, 40, "verse")], tool="songformer"), "songformer")
    assert row["hr3t_r"] == 0.0 and row["hr3t_f"] == 0.0


# --- alignment / ignore handling ------------------------------------------------


def test_align_to_span_edges_and_gaps():
    segs = [{"start": 0.5, "end": 5, "label": "intro"}, {"start": 7, "end": 12, "label": "verse"},
            {"start": 12, "end": 50, "label": "outro"}]
    iv, lab = sev.align_to_span(segs, 40.0)
    assert iv[0, 0] == 0.0 and iv[-1, 1] == 40.0
    assert lab == ["intro", lm.PAD_LABEL, "verse", "outro"]
    assert np.all(iv[1:, 0] == iv[:-1, 1])
    iv, lab = sev.align_to_span([{"start": 0, "end": 35, "label": "verse"}], 40.0)
    assert lab == ["verse", lm.PAD_LABEL]  # 5 s short -> padded, not stretched
    iv, lab = sev.align_to_span([{"start": 0, "end": 39.4, "label": "verse"}], 40.0)
    assert lab == ["verse"] and iv[-1, 1] == 40.0  # within snap -> stretched


def test_ignore_frames_dropped_from_accuracy():
    ref = doc([(0, 10, "unlabeled"), (10, 20, "verse")])
    est = doc([(0, 10, "intro"), (10, 20, "verse")], tool="songformer")
    row = sev.evaluate_track(ref, est, "songformer")
    assert row["acc_coarse"] == 1.0 and row["n_frames_coarse"] == 100


def test_bridge_is_body_on_every_side():
    # bridge is a full section everywhere: a correctly labelled bridge must score 1.0 on both maps
    for tool in ("songformer", "allin1"):
        row = sev.evaluate_track(doc([(0, 10, "bridge")]), doc([(0, 10, "bridge")], tool=tool), tool)
        assert row["acc_coarse"] == 1.0 and row["acc_fine"] == 1.0
    # the tools' short non-thematic passages are the only things that map to transition
    assert lm.map_label("inst", "allin1", "coarse") == "transition"
    assert lm.map_label("break", "allin1", "coarse") == "transition"
    assert lm.map_label("inst", "songformer", "coarse") == "transition"


# --- schema / converters ---------------------------------------------------------


def test_converters_roundtrip(tmp_path):
    a1 = {"path": "/x/0001_copilot Song NS.mp3", "bpm": 120, "beats": [0.5, 1.0], "downbeats": [0.5],
          "beat_positions": [1, 2], "segments": [{"start": 0, "end": 5, "label": "intro"}, {"start": 5, "end": 9, "label": "end"}]}
    d = schema.from_allin1(a1)
    assert d["track_id"] == "0001_copilot Song NS" and d["bpm"] == 120.0 and "beat_positions" not in d
    p = schema.save(d, tmp_path / "a.json")
    assert schema.load(p) == d
    sf = schema.from_songformer([{"start": 0, "end": 4, "label": "Pre-Chorus"}], track_id="t")
    assert sf["segments"][0]["label"] == "pre-chorus" and sf["beats"] is None
    from ravi.structure.labels import Segment

    h = schema.from_human([Segment(0.0, 3.0, "intro"), Segment(3.0, 8.0, "verse")], track_id="t", duration=8.0)
    assert h["tool"] == "human" and len(h["segments"]) == 2


def test_schema_rejects_overlap():
    with pytest.raises(ValueError):
        doc([(0, 10, "intro"), (5, 20, "verse")])


# --- CLI end to end --------------------------------------------------------------


def test_cli_end_to_end(tmp_path):
    hd, td = tmp_path / "human", tmp_path / "allin1"
    schema.save(doc(REF, track_id="a"), hd / "a.json")
    schema.save(doc([(0, 30, "verse")], track_id="b"), hd / "b.json")
    schema.save(doc([(0, 5, "verse")], track_id="c"), hd / "c.json")  # no tool output
    est_a = [(0, 0.3, "start"), (0.3, 10.2, "intro"), (10.2, 20, "verse"), (20, 30, "chorus"), (30, 40, "outro")]
    schema.save(doc(est_a, tool="allin1", track_id="a"), td / "a.json")
    schema.save(doc([(0, 15, "inst"), (15, 30, "verse")], tool="allin1", track_id="b"), td / "b.json")
    out = tmp_path / "eval" / "structure_allin1.md"
    assert sev.main(["--human-dir", str(hd), "--tool-dir", str(td), "--tool", "allin1",
                     "--out", str(out), "--per-label", "--n-boot", "500"]) == 0
    md = out.read_text()
    assert "Tracks evaluated: 2" in md and "missing in allin1: 1" in md and "Per-class (coarse" in md
    rows = (tmp_path / "eval" / "structure_allin1.csv").read_text().splitlines()
    assert len(rows) == 3
    assert (tmp_path / "eval" / "structure_allin1_summary.csv").exists()
    assert (tmp_path / "eval" / "structure_allin1_per_label.csv").exists()


def test_bootstrap_ci_deterministic():
    vals = [0.1, 0.5, 0.9, float("nan"), 0.7]
    lo, hi = sev.bootstrap_ci(vals, 1000, 0)
    assert (lo, hi) == sev.bootstrap_ci(vals, 1000, 0)
    assert lo <= np.mean([0.1, 0.5, 0.9, 0.7]) <= hi


# --- unlabeled head / tail / interior, empty and invalid docs ---------------------


PERFECT = [(0, 10, "intro"), (10, 20, "verse"), (20, 30, "chorus"), (30, 40, "outro")]


def test_unlabeled_head_does_not_penalise_perfect_tool():
    # reviewer's case: the annotator started at 1 s; a perfect tool must still score 1.0
    ref = doc([(0, 1, "unlabeled"), (1, 10, "intro"), (10, 20, "verse"), (20, 30, "chorus"), (30, 40, "outro")])
    row = sev.evaluate_track(ref, doc(PERFECT, tool="songformer"), "songformer")
    for m in sev.METRICS:
        assert row[m] == pytest.approx(1.0), m
    assert (row["span_start"], row["span_end"], row["duration"]) == (1.0, 40.0, 39.0)
    assert row["n_frames_coarse"] == 390


def test_unlabeled_tail_is_outside_the_span():
    # annotator stopped at 30 s of a 60 s track; the tool keeps going
    ref = doc([(0, 10, "intro"), (10, 20, "verse"), (20, 30, "chorus"), (30, 60, "unlabeled")])
    est = doc([(0, 10, "intro"), (10, 20, "verse"), (20, 30, "chorus"), (30, 45, "inst"), (45, 60, "outro")],
              tool="songformer")
    row = sev.evaluate_track(ref, est, "songformer")
    assert row["span_end"] == 30.0
    for m in sev.METRICS:
        assert row[m] == pytest.approx(1.0), m


def test_interior_unlabeled_is_not_a_cluster():
    ref = doc([(0, 10, "intro"), (10, 20, "unlabeled"), (20, 30, "outro")])
    est = doc([(0, 10, "intro"), (10, 15, "verse"), (15, 20, "chorus"), (20, 30, "outro")], tool="songformer")
    row = sev.evaluate_track(ref, est, "songformer")
    assert row["pw_f"] == pytest.approx(1.0)
    assert row["acc_coarse"] == 1.0 and row["n_frames_coarse"] == 200
    # padding on the estimate side (tool stops 5 s early) is dropped as well
    est2 = doc([(0, 10, "intro"), (10, 25, "verse")], tool="songformer")
    row2 = sev.evaluate_track(doc([(0, 10, "intro"), (10, 30, "verse")]), est2, "songformer")
    assert row2["pw_f"] == pytest.approx(1.0) and row2["acc_coarse"] == 1.0


def test_pairwise_matches_mir_eval_without_ignored_frames():
    import mir_eval

    rng = np.random.default_rng(3)
    for _ in range(20):
        def rand_seg(labels):
            b = np.sort(rng.uniform(0, 60, size=int(rng.integers(1, 8))))
            edges = np.concatenate([[0.0], b, [60.0]])
            iv = np.stack([edges[:-1], edges[1:]], 1)
            iv = iv[iv[:, 1] - iv[:, 0] > 0.05]
            iv[1:, 0] = iv[:-1, 1]
            iv[0, 0], iv[-1, 1] = 0.0, 60.0
            return iv, [labels[int(rng.integers(len(labels)))] for _ in iv]

        ri, rl = rand_seg(["a", "b", "c"])
        ei, el = rand_seg(["x", "y"])
        got = sev.pairwise_ignoring(ri, rl, ei, el)
        want = mir_eval.segment.pairwise(ri, rl, ei, el, frame_size=sev.PAIRWISE_FRAME_S)
        if all(math.isfinite(v) for v in want):
            assert got == pytest.approx(want)


def test_empty_estimate_skipped():
    est = {"track_id": "t1", "tool": "allin1", "segments": []}
    row = sev.evaluate_track(doc(REF), est, "allin1")
    assert row["skip_reason"] == "empty estimate"
    assert all(math.isnan(row[m]) for m in sev.METRICS)
    agg = {s["metric"]: s for s in sev.aggregate([row], n_boot=10)}
    assert agg["hr3_f"]["n_tracks"] == 0


def test_all_unlabeled_reference_skipped():
    row = sev.evaluate_track(doc([(0, 30, "unlabeled")]), doc(REF, tool="songformer"), "songformer")
    assert row["skip_reason"] == "empty reference"


@pytest.mark.parametrize("segments", [
    [{"start": 0, "end": 15, "label": "intro"}, {"start": 10, "end": 40, "label": "verse"}],  # overlap
    [{"start": 20, "end": 40, "label": "outro"}, {"start": 0, "end": 20, "label": "intro"}],  # unsorted
])
def test_invalid_estimate_rejected(segments):
    with pytest.raises(ValueError):
        sev.evaluate_track(doc(REF), {"track_id": "t1", "tool": "allin1", "segments": segments}, "allin1")
