import csv
import errno
import json
import os

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ravi.captions import make_arms as ma
from ravi.captions.windows import WIN_S, choose_window, clip_segments, window_frames, window_seconds
from ravi.structure import label_map
from ravi.structure.labels import UNLABELED, Segment
from ravi.train.caption_aug import FIELD_SEP, CaptionAugConfig, augment_caption, check_trigger

SR = 8000  # small files; the window logic is sample-rate agnostic
N_WIN = window_frames(SR)  # (47 * 8000 // 4096) * 4096 = 372736
W_SR = N_WIN / SR  # 46.592 s


# --------------------------------------------------------------------------- windows


def test_window_length_is_the_trainer_sample_size():
    # sa3_lora.scan_dataset / train_lora.py:105: (int(duration * sr) // 4096) * 4096
    assert window_frames(44100) == 2072576
    assert window_frames(48000) == 2252800
    assert window_frames(44100, duration_s=47, downsample=4096) == (int(47 * 44100) // 4096) * 4096
    assert WIN_S == window_seconds(44100) == 2072576 / 44100
    assert N_WIN == 372736
    with pytest.raises(ValueError):
        window_frames(10, duration_s=1)


def test_short_track_uses_whole_track():
    segs = [Segment(0, 5, "intro"), Segment(5, 5.6, "transition"), Segment(5.6, 30, "verse")]
    w = choose_window(segs, 30.0)
    assert (w.start, w.end, w.cropped) == (0.0, 30.0, False)
    # the 0.6 s transition lies wholly inside the window: kept (min_overlap is for cut edges)
    assert [s.label for s in w.segments] == ["intro", "transition", "verse"]


def test_exactly_window_length_is_not_cropped():
    w = choose_window([], WIN_S)
    assert (w.start, w.end, w.cropped) == (0.0, WIN_S, False)


def test_long_track_starts_at_zero_and_drops_sliver_after_cut():
    # boundary at 46.5: the window [0, 46.997] keeps only 0.497 s of the outro -> dropped
    segs = [Segment(0, 10, "intro"), Segment(10, 46.5, "verse"), Segment(46.5, 80, "outro")]
    w = choose_window(segs, 80.0)
    assert w.cropped and w.start == 0.0 and w.end == pytest.approx(WIN_S)
    assert [s.label for s in w.segments] == ["intro", "verse"]
    assert w.segments[-1].end == 46.5


def test_long_track_boundary_near_47_keeps_overlapping_sections_clipped():
    segs = [Segment(0, 8, "intro"), Segment(8, 40, "verse"), Segment(40, 47.2, "transition"),
            Segment(47.2, 90, "verse")]
    w = choose_window(segs, 90.0)
    assert w.start == 0.0 and w.end == pytest.approx(WIN_S)
    assert [s.label for s in w.segments] == ["intro", "verse", "transition"]
    assert w.segments[-1].end == pytest.approx(WIN_S)
    assert all(w.start <= s.start < s.end <= w.end for s in w.segments)


def test_long_track_shifts_when_both_edges_land_on_boundaries():
    # [10, 56.997] starts at a boundary and ends 0.003 s from one; [0, 46.997] ends 13 s from one
    segs = [Segment(0, 10, "intro"), Segment(10, 57, "verse"), Segment(57, 70, "outro")]
    w = choose_window(segs, 70.0)
    assert w.start == pytest.approx(10.0) and w.end == pytest.approx(10 + WIN_S)
    assert [s.label for s in w.segments] == ["verse"]


def test_long_track_prefers_zero_on_near_ties():
    # [53.003, 100] ends on the track end (cost 6.997 < 13.003 for start 0) but is not
    # clean (start 7 s from a boundary), so the window stays at 0
    segs = [Segment(0, 30, "verse"), Segment(30, 60, "verse"), Segment(60, 100, "outro")]
    w = choose_window(segs, 100.0)
    assert w.start == 0.0
    # boundary at 47.5: start 0.503 (end on the boundary) costs the same 0.503 -> tie -> 0
    segs = [Segment(0, 47.5, "verse"), Segment(47.5, 100, "outro")]
    w = choose_window(segs, 100.0)
    assert w.start == 0.0 and [s.label for s in w.segments] == ["verse"]
    # an intro ending at 0.4 s makes start 0.4 clean on both edges (end 47.397 vs 47.5)
    segs = [Segment(0, 0.4, "intro"), Segment(0.4, 47.5, "verse"), Segment(47.5, 100, "outro")]
    w = choose_window(segs, 100.0)
    assert w.start == pytest.approx(0.4) and [s.label for s in w.segments] == ["verse"]


def test_long_track_without_labels_starts_at_zero():
    w = choose_window([], 120.0)
    assert (w.start, w.end, w.cropped, w.segments) == (0.0, pytest.approx(WIN_S), True, ())


def test_window_is_deterministic_and_within_track():
    segs = [Segment(0, 3, "intro"), Segment(3, 50, "verse"), Segment(50, 100.5, "bridge"),
            Segment(100.5, 140, "outro")]
    a, b = choose_window(segs, 140.0), choose_window(list(segs), 140.0)
    assert a == b
    assert 0 <= a.start and a.end <= 140.0 and a.duration == pytest.approx(WIN_S)


def test_clip_segments_min_overlap():
    segs = [Segment(0, 10, "intro"), Segment(10, 11.0, "transition"), Segment(11, 20, "verse")]
    assert [s.label for s in clip_segments(segs, 0, 20)] == ["intro", "transition", "verse"]
    assert [s.label for s in clip_segments(segs, 10.5, 20)] == ["verse"]  # cut to 0.5 s: dropped


def test_clip_segments_keeps_short_contained_sections_only():
    segs = [Segment(0, 10, "intro"), Segment(10, 10.3, "transition"), Segment(10.3, 19.6, "verse"),
            Segment(19.6, 30, "outro")]
    # 0.3 s transition fully inside -> kept; outro cut to 0.4 s at the right edge -> dropped
    assert [s.label for s in clip_segments(segs, 0, 20)] == ["intro", "transition", "verse"]
    # intro cut to 0.5 s at the left edge -> dropped
    assert [s.label for s in clip_segments(segs, 9.5, 20)] == ["transition", "verse"]
    # a short section touching the edge exactly is still fully contained
    assert [s.label for s in clip_segments(segs, 10, 10.3)] == ["transition"]


# --------------------------------------------------------------------------- captions


def row(**kw):
    base = dict(
        track_id="0001_copilot.mp3 Agent X NS",
        human_caption="A fast-paced electronic track, with energetic synths.",
        human_genre_tag="Electronic",
        caption_is_junk=False,
        id3_genre_tokens=np.array(["Electronic", "Mysterious"], dtype=object),
        id3_bpm=pd.NA,
    )
    base.update(kw)
    return base


JUNK = dict(human_caption="No Audio", human_genre_tag="No Audio", caption_is_junk=True)
SECTIONS = ["intro", "verse", "transition", "verse", "outro"]


def fields(cap):
    return cap.text.split(FIELD_SEP)


def test_arm_A():
    assert ma.build_caption("A", row(), SECTIONS).text == "rvkr"


def test_arm_B_and_junk_fallbacks():
    cap = ma.build_caption("B", row(), SECTIONS)
    assert fields(cap) == ["rvkr", "A fast-paced electronic track, with energetic synths", "Electronic"]
    assert not cap.fallbacks
    assert ".," not in cap.rendered and cap.rendered.endswith("energetic synths, Electronic")
    cap = ma.build_caption("B", row(**JUNK), SECTIONS)
    assert fields(cap) == ["rvkr", "Electronic, Mysterious"]
    assert cap.fallbacks == ["junk_caption->id3_tags"]
    cap = ma.build_caption("B", row(caption_is_junk=True, human_caption="NA", human_genre_tag="Jazz"), None)
    assert fields(cap) == ["rvkr", "Jazz"] and cap.fallbacks == ["junk_caption->genre_tag"]


def test_arm_C_bpm_present_and_omitted():
    cap = ma.build_caption("C", row(id3_bpm=120), None)
    assert fields(cap) == ["rvkr", "Electronic, Mysterious", "120 bpm"]
    cap = ma.build_caption("C", row(), None)
    assert fields(cap) == ["rvkr", "Electronic, Mysterious"] and cap.omitted == ["bpm"]


def test_arm_D_requires_caption():
    cap = ma.build_caption("D", row(), None, autocaption="Calm piano\nwith strings␞x")
    assert fields(cap) == ["rvkr", "Calm piano with strings x"]
    with pytest.raises(ValueError):
        ma.build_caption("D", row(), None, autocaption="  ")


def test_arm_E_sections_tail_and_omission():
    cap = ma.build_caption("E", row(), SECTIONS)
    assert fields(cap)[-1] == "sections: intro, verse, transition, verse, outro"
    assert fields(cap)[:3] == fields(ma.build_caption("B", row(), None))
    cap = ma.build_caption("E", row(), [])
    assert not any(f.startswith("sections") for f in fields(cap)) and cap.omitted == ["sections"]
    cap = ma.build_caption("E", row(**JUNK), ["verse"], sections_source="songformer")
    assert fields(cap) == ["rvkr", "Electronic, Mysterious", "sections: verse"]
    assert cap.fallbacks == ["junk_caption->id3_tags", "sections_from_songformer"]


def test_arm_E_survives_augmentation_with_sections_last():
    cap = ma.build_caption("E", row(), SECTIONS)
    cfg = CaptionAugConfig(enabled=True, p_drop_field=1.0)
    for seed in range(20):
        out = augment_caption(cap.text, np.random.default_rng(seed), cfg)
        assert out.startswith("rvkr, ") and out.endswith("sections: intro, verse, transition, verse, outro")


def test_arm_F_fields_and_omissions():
    cap = ma.build_caption("F", row(id3_bpm=96), ["intro", "verse"])
    assert fields(cap) == ["rvkr", "Genre: Electronic", "Mood: A fast-paced electronic track, with energetic synths",
                           "BPM: 96", "Sections: intro, verse"]
    cap = ma.build_caption("F", row(**JUNK), None)
    assert fields(cap) == ["rvkr", "Genre: Electronic, Mysterious"]
    assert set(cap.omitted) == {"mood", "bpm", "sections"} and cap.fallbacks == ["genre->id3_tags"]


def test_arm_F_is_experimental_and_mood_is_first_sentence():
    assert "F" in ma.ALL_ARMS and ma.EXPERIMENTAL_ARMS == frozenset({"F"})
    assert ma.first_sentence("Calm piano, soft strings. Builds later! End?") == "Calm piano, soft strings"
    assert ma.first_sentence("Wow! More here.") == "Wow!"
    assert ma.first_sentence("Is it calm? Yes.") == "Is it calm?"
    assert ma.first_sentence("No terminal punctuation, at all") == "No terminal punctuation, at all"
    assert ma.first_sentence("Tempo approx 120.5 bpm. Next.") == "Tempo approx 120.5 bpm"  # no split inside a number
    assert ma.first_sentence(pd.NA) == ""
    cap = ma.build_caption("F", row(human_caption="Dark drones. Then a bright lead."), None)
    assert "Mood: Dark drones" in fields(cap)


def test_clean_field_treats_missing_values_as_empty():
    for x in (None, pd.NA, float("nan"), np.nan, pd.NaT):
        assert ma.clean_field(x) == ""
    assert ma.clean_field("  a\n b ␞ c ") == "a b c"
    assert ma.clean_field(0) == "0"
    cap = ma.build_caption("B", row(human_genre_tag=pd.NA), None)
    assert fields(cap) == ["rvkr", "A fast-paced electronic track, with energetic synths"]
    assert cap.omitted == ["genre_tag"]


def test_human_caption_strips_trailing_period_only():
    assert ma.human_caption({"human_caption": "Upbeat pop..."}) == "Upbeat pop"
    assert ma.human_caption({"human_caption": "Upbeat pop!"}) == "Upbeat pop!"
    assert ma.human_caption({"human_caption": "A. B."}) == "A. B"


def test_clean_id3_tokens():
    raw = ["Pop", "Fun", "Quirky TEST!", "MARKUP", "Modern Cleassical", "Chinematic", "Pops", "Latin alfa",
           "Sexy alfa", "PopUN18/71 RK ROLAIDS - happy guitar pop positive acoustic upbeat indie",
           "Mnemonic", "Solo", "Sound Design", "Vocals", "Pause", "Run", "DELETE", "AB-12", "UN18",
           "Very Long Five Word Token", "Light Rock", "R&B", "Hip Hop", "sound design"]
    kept, dropped = ma.clean_id3_tokens(raw)
    assert kept == ["Pop", "Fun", "Quirky", "Modern Classical", "Cinematic", "Latin", "Sexy", "Light Rock",
                    "R&B", "Hip Hop"]  # "Pops" -> "Pop" deduplicated
    assert "TEST!" in dropped and "MARKUP" in dropped and "DELETE" in dropped and "alfa" in dropped
    for t in ("Mnemonic", "Solo", "Sound Design", "Vocals", "Pause", "Run", "AB-12", "UN18",
              "Very Long Five Word Token", "sound design"):
        assert t in dropped, t
    assert raw[9] in dropped
    r = row(id3_genre_tokens=np.array(["Pops", "Mnemonic", "Quirky TEST!"], dtype=object), id3_bpm=100)
    assert fields(ma.build_caption("C", r, None)) == ["rvkr", "Pop, Quirky", "100 bpm"]
    assert ma.id3_dropped(r) == ["Mnemonic", "TEST!"]
    # nothing left -> C falls back to the trigger, B's junk fallback skips the empty tags
    r = row(id3_genre_tokens=["Mnemonic", "MARKUP"], **JUNK)
    assert fields(ma.build_caption("C", r, None)) == ["rvkr"]
    assert ma.build_caption("B", r, None).fallbacks == ["junk_caption->trigger_only"]


def test_songformer_mapping_is_consistent_with_label_map():
    sfc, hc = label_map.COARSE["songformer"], label_map.COARSE["human"]
    assert set(ma.SONGFORMER_TO_HUMAN) == set(label_map.VOCAB["songformer"])
    for lab, mapped in ma.SONGFORMER_TO_HUMAN.items():
        if sfc[lab] == label_map.IGNORE:
            assert mapped is None, lab
        else:
            assert hc[mapped] == sfc[lab], (lab, mapped)  # same coarse class on both sides
    assert ma.SONGFORMER_TO_HUMAN["chorus"] == "chorus"
    assert ma.SONGFORMER_TO_HUMAN["pre-chorus"] == "verse"
    assert ma.SONGFORMER_TO_HUMAN["inst"] == "transition"
    assert ma.SONGFORMER_TO_HUMAN["silence"] is None
    for alias in ("prechorus", "Pre_Chorus", " pre chorus "):
        assert ma.map_songformer_label(alias) == "verse"
    with pytest.raises(KeyError):
        ma.map_songformer_label("solo")


def test_songformer_segments_are_normalised(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({
        "track_id": "x", "tool": "songformer", "bpm": None, "beats": None, "downbeats": None, "duration": 30.0,
        "segments": [{"start": 0.0, "end": 2.0, "label": "silence"},  # dropped -> gap -> unlabeled
                     {"start": 2.0, "end": 10.0, "label": "pre_chorus"},  # alias
                     {"start": 10.2, "end": 31.0, "label": "Chorus"}]}))  # micro gap snapped, end clamped
    segs = ma.load_songformer_segments(p, 30.0)
    assert [(s.start, s.end, s.label) for s in segs] == [
        (0.0, 2.0, UNLABELED), (2.0, 10.0, "verse"), (10.0, 30.0, "chorus")]


def test_every_caption_passes_trigger_check():
    for arm in ma.ALL_ARMS:
        cap = ma.build_caption(arm, row(**JUNK), None, autocaption="x")
        check_trigger(cap.text, CaptionAugConfig())


def test_acestep_columns():
    cap = ma.build_caption("E", row(), ["intro", "verse"])
    c, lyr = ma.acestep_columns(cap)
    assert c == "rvkr, A fast-paced electronic track, with energetic synths, Electronic"
    assert lyr == "[intro] [verse]"
    assert ma.acestep_columns(ma.build_caption("A", row(), None)) == ("rvkr", "")


def test_unknown_arm():
    with pytest.raises(ValueError):
        ma.build_caption("Z", row(), None)


# --------------------------------------------------------------------------- end to end


def _ramp(seconds, sr=SR, channels=2, offset=0):
    """int16 ramp: every frame differs from its neighbours, so an off-by-one crop shows."""
    n = int(round(seconds * sr))
    x = ((np.arange(n, dtype=np.int64) * 7 + offset) % 65536 - 32768).astype(np.int16)
    return np.stack([x, x[::-1]], axis=1) if channels == 2 else x


OFFSET_LABELS = f"0\t10\tintro\n10\t{10 + W_SR!r}\tverse\n{10 + W_SR!r}\t80\toutro\n"


@pytest.fixture
def corpus(tmp_path):
    ids = {
        "0001_copilot.mp3 Short NS": 30.0,  # labelled, short
        "0002_copilot.mp3 Long. Title NS": 80.0,  # labelled, long, dot in name
        "0003_copilot.mp3 Junk NS": 20.0,  # junk caption, labelled
        "0004_copilot.mp3 Nolabel NS": 60.0,  # no labels, songformer available
        "0005_copilot.mp3 Nothing NS": 10.0,  # no labels, no songformer, dirty ID3 tokens
        "0006_copilot.mp3 Test NS": 25.0,  # test split, must never appear
        "0007_copilot.mp3 Offset NS": 80.0,  # labelled, long, window starts at 10 s
    }
    labels = tmp_path / "labels"
    labels.mkdir()
    prep = tmp_path / "prepared" / str(SR)
    prep.mkdir(parents=True)
    sfdir = tmp_path / "songformer"
    sfdir.mkdir()
    rows = []
    for i, (tid, dur) in enumerate(ids.items()):
        sf.write(str(prep / f"{tid}.wav"), _ramp(dur, offset=1000 * i), SR, subtype="PCM_16")
        (prep / f"{tid}.wav.json").write_text("{}")  # sidecar, not audio
        has = tid[:4] in ("0001", "0002", "0003", "0006", "0007")
        lp = f"/elsewhere/labels/{tid}.mp3_labels.txt"
        if has:
            text = {
                "0001": "0\t10\tintro\n10\t30\tverse\n",
                "0002": "0\t8\tintro\n8\t40\tverse\n40\t47.2\ttransiton\n47.2\t80\tverse\n",
                "0003": "0\t20\tverse\n",
                "0006": "0\t25\tverse\n",
                "0007": OFFSET_LABELS,
            }[tid[:4]]
            (labels / f"{tid}.mp3_labels.txt").write_text(text)
        junk = tid.startswith("0003")
        rows.append(dict(
            track_id=tid, duration_s=dur,
            human_caption="No Audio" if junk else f"Caption {i}, with a comma. Second sentence.",
            human_genre_tag="No Audio" if junk else "Pop",
            caption_is_junk=junk,
            id3_genre_tokens=["Pop", "Quirky TEST!", "Mnemonic"] if tid.startswith("0005") else ["Pop", "Fun"],
            id3_bpm=pd.NA if i % 2 else 110,
            labels_path=lp if has else None, has_labels=has,
        ))
    df = pd.DataFrame(rows)
    df["id3_bpm"] = df["id3_bpm"].astype("Int64")
    df.to_parquet(tmp_path / "tracks.parquet")
    (sfdir / "0004_copilot.mp3 Nolabel NS.json").write_text(json.dumps({
        "track_id": "0004_copilot.mp3 Nolabel NS", "tool": "songformer", "bpm": None, "beats": None,
        "downbeats": None, "duration": 60.0,
        "segments": [{"start": 0, "end": 1.5, "label": "silence"}, {"start": 1.5, "end": 20, "label": "intro"},
                     {"start": 20, "end": 60, "label": "inst"}]}))
    tids = list(ids)
    (tmp_path / "splits.json").write_text(json.dumps(
        {"train": [tids[1], tids[2], tids[4], tids[3], tids[6]], "val": [tids[0]], "test": [tids[5]], "seed": 1}))
    (tmp_path / "auto.jsonl").write_text(
        "\n".join(json.dumps({"track_id": t, "caption": f"auto {t[:4]}"}) for t in tids) + "\n")
    return tmp_path, tids


def run(tmp_path, arms=("A", "B", "C", "D", "E", "F"), **kw):
    args = dict(
        manifest=tmp_path / "tracks.parquet", splits_path=tmp_path / "splits.json",
        prepared_root=tmp_path / "prepared", out_root=tmp_path / "arms", arms=list(arms),
        sr=SR, labels_dir=tmp_path / "labels", autocaptions=tmp_path / "auto.jsonl",
        songformer_dir=tmp_path / "songformer",
    )
    args.update(kw)
    return ma.make_arms(**args)


def read_manifest(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def snapshot(d):
    """{relative path: bytes} for every file under d (symlinks by target)."""
    out = {}
    for p in sorted(d.rglob("*")):
        rel = str(p.relative_to(d))
        if p.is_symlink():
            out[rel] = ("link", os.readlink(p))
        elif p.is_file():
            out[rel] = p.read_bytes()
    return out


def test_end_to_end(corpus):
    tmp, tids = corpus
    summary = run(tmp)
    out = tmp / "arms"
    sel = set(tids[:5]) | {tids[6]}
    sets = {}
    for arm in ma.ALL_ARMS:
        m = read_manifest(out / arm / "manifest.csv")
        sets[arm] = [(r["split"], r["track_id"], r["start_frame"], r["n_frames"]) for r in m]
        assert {r["track_id"] for r in m} == sel  # test track excluded
        for r in m:
            wav = out / arm / r["file"]
            assert wav.exists() and not wav.is_symlink()  # --link copy is the default
            assert wav.with_suffix(".txt").read_text(encoding="utf-8").strip() == r["caption"]
            assert int(r["caption_words"]) == len(r["caption"].replace(FIELD_SEP, ", ").split())
        assert summary["arms"][arm]["n_tracks"] == {"train": 5, "val": 1}
        assert summary["arms"][arm]["n_cropped"] == 3
        assert summary["arms"][arm]["experimental"] is (arm == "F")
        assert (out / arm / ma.MARKER).is_file()
        # filelists: exactly the WAVs of the split, relative names
        for split in ("train", "val"):
            listed = (out / arm / split / "filelist.txt").read_text(encoding="utf-8").splitlines()
            assert sorted(listed) == sorted(p.name for p in (out / arm / split).glob("*.wav"))
            assert listed == [f"{r['track_id']}.wav" for r in m if r["split"] == split]
        # acestep_metadata.csv needs --sr 48000
        assert not (out / arm / "acestep_metadata.csv").exists()
    assert all(v == sets["A"] for v in sets.values())
    hashes = {(out / arm / ma.TRACKSET_FILE).read_text().strip() for arm in ma.ALL_ARMS}
    assert len(hashes) == 1 and summary["trackset_sha256"] in hashes
    assert not any(p.name.endswith((".tmp", ".old")) for p in out.iterdir())

    # no file name contains the field separator; no sidecars were copied
    names = [p.name for p in out.rglob("*")]
    assert not any(FIELD_SEP in n for n in names)
    assert not any(n.endswith(".wav.json") for n in names)

    # cropped WAV length == trainer window, bit-exact crop
    long_wav = out / "B" / "train" / f"{tids[1]}.wav"
    info = sf.info(str(long_wav))
    meta = json.loads((out / "B" / "train" / f"{tids[1]}.window.json").read_text())
    assert info.frames == meta["n_frames"] == N_WIN
    assert info.subtype == "PCM_16"
    assert meta["window_end"] - meta["window_start"] == pytest.approx(info.frames / SR)
    src, _ = sf.read(str(tmp / "prepared" / str(SR) / f"{tids[1]}.wav"), dtype="int16")
    got, _ = sf.read(str(long_wav), dtype="int16")
    s0 = meta["start_frame"]
    assert s0 == 0 and np.array_equal(got, src[s0:s0 + info.frames])

    # short tracks are byte copies of the prepared WAV
    short = out / "A" / "val" / f"{tids[0]}.wav"
    assert short.read_bytes() == (tmp / "prepared" / str(SR) / f"{tids[0]}.wav").read_bytes()
    # with_suffix keeps the dotted title intact
    assert (out / "A" / "train" / "0002_copilot.mp3 Long. Title NS.txt").exists()

    e = {r["track_id"]: r["caption"] for r in read_manifest(out / "E" / "manifest.csv")}
    assert e[tids[1]].endswith("sections: intro, verse, transition")  # labels inside the window only
    assert e[tids[2]] == FIELD_SEP.join(["rvkr", "Pop, Fun", "sections: verse"])  # junk -> id3 tags
    assert e[tids[3]].endswith("sections: intro, transition")  # songformer, mapped
    assert "sections" not in e[tids[4]]
    assert e[tids[6]].endswith("sections: verse")
    assert FIELD_SEP.join(["rvkr", "Caption 0, with a comma. Second sentence", "Pop"]) == e[tids[0]].rsplit(FIELD_SEP, 1)[0]
    se = summary["arms"]["E"]
    assert se["fallbacks"] == {"junk_caption->id3_tags": 1, "sections_from_songformer": 1}
    assert se["omitted_fields"] == {"sections": 1}
    assert summary["segments_source"] == {"human": 4, "none": 1, "songformer": 1}

    f = {r["track_id"]: r["caption"] for r in read_manifest(out / "F" / "manifest.csv")}
    assert "Mood: Caption 0, with a comma" in f[tids[0]].split(FIELD_SEP)

    c = {r["track_id"]: r for r in read_manifest(out / "C" / "manifest.csv")}
    assert c[tids[4]]["caption"] == FIELD_SEP.join(["rvkr", "Pop, Quirky", "110 bpm"])
    assert c[tids[4]]["id3_dropped"] == "TEST!;Mnemonic" and c[tids[0]]["id3_dropped"] == ""
    assert json.loads((out / "arms_summary.json").read_text())["arms"]["D"]["n_tracks"]["train"] == 5


def test_window_off_zero_is_bit_exact(corpus):
    """M5: a window that starts at 10 s (both edges on label boundaries)."""
    tmp, tids = corpus
    run(tmp, arms=["A", "E"])
    tid = tids[6]
    for arm in ("A", "E"):
        d = tmp / "arms" / arm / "train"
        meta = json.loads((d / f"{tid}.window.json").read_text())
        assert meta["start_frame"] == 10 * SR and meta["n_frames"] == N_WIN
        assert meta["window_start"] == pytest.approx(10.0)
        src, _ = sf.read(str(tmp / "prepared" / str(SR) / f"{tid}.wav"), dtype="int16")
        got, _ = sf.read(str(d / f"{tid}.wav"), dtype="int16")
        assert got.shape == (N_WIN, 2)
        assert np.array_equal(got, src[10 * SR:10 * SR + N_WIN])
        assert not np.array_equal(got, src[10 * SR + 1:10 * SR + 1 + N_WIN])  # the ramp catches off-by-one
        assert [s["label"] for s in meta["segments"]] == ["verse"]
        assert meta["segments"][0]["start"] == pytest.approx(0.0)
    m = {r["track_id"]: r for r in read_manifest(tmp / "arms" / "E" / "manifest.csv")}
    assert int(m[tid]["start_frame"]) == 10 * SR


@pytest.mark.parametrize("sr,frames", [(44100, 2072576), (48000, 2252800)])
def test_crop_length_matches_trainer_at_real_rates(tmp_path, sr, frames):
    """M1: a 60 s tone is cropped to the trainer's sample_size; acestep only at 48 kHz."""
    tid = "0001_copilot.mp3 Tone NS"
    prep = tmp_path / "prepared" / str(sr)
    prep.mkdir(parents=True)
    t = np.arange(60 * sr) / sr
    sf.write(str(prep / f"{tid}.wav"), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, subtype="PCM_16")
    pd.DataFrame([dict(track_id=tid, human_caption="A tone.", human_genre_tag="Test", caption_is_junk=False,
                       id3_genre_tokens=["Pop"], id3_bpm=100, labels_path=None, has_labels=False)]
                 ).to_parquet(tmp_path / "tracks.parquet")
    (tmp_path / "splits.json").write_text(json.dumps({"train": [tid], "val": []}))
    s = ma.make_arms(tmp_path / "tracks.parquet", tmp_path / "splits.json", tmp_path / "prepared",
                     tmp_path / "arms", ["A", "C"], sr=sr)
    info = sf.info(str(tmp_path / "arms" / "A" / "train" / f"{tid}.wav"))
    assert info.frames == frames == s["window_frames"] and info.samplerate == sr
    assert (tmp_path / "arms" / "C" / "acestep_metadata.csv").exists() is (sr == 48000)
    assert s["acestep_metadata"] is (sr == 48000)


def test_limit_overwrite_and_marker(corpus):
    tmp, tids = corpus
    run(tmp, arms=["A", "B"], limit=2)
    with pytest.raises(FileExistsError):
        run(tmp, arms=["A", "B"], limit=2)
    s = run(tmp, arms=["A", "B"], limit=2, overwrite=True, link_mode="symlink")
    assert s["n_tracks"] == 2
    short = tmp / "arms" / "A" / "val" / f"{tids[0]}.wav"
    assert short.is_symlink()
    # --overwrite refuses a directory without the marker, and leaves it alone
    foreign = tmp / "arms" / "C"
    foreign.mkdir()
    (foreign / "precious.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="marker"):
        run(tmp, arms=["C"], limit=2, overwrite=True)
    assert (foreign / "precious.txt").read_text() == "keep"
    (tmp / "arms" / "A" / ma.MARKER).unlink()
    with pytest.raises(FileExistsError, match="marker"):
        run(tmp, arms=["A"], limit=2, overwrite=True)
    assert (tmp / "arms" / "A" / "manifest.csv").is_file()


def test_failed_rebuild_keeps_previous_arm(corpus, monkeypatch):
    """M2: build into {arm}.tmp; a failure anywhere leaves the previous good arms intact."""
    tmp, tids = corpus
    run(tmp, arms=["A", "B"], limit=3)
    out = tmp / "arms"
    before = {a: snapshot(out / a) for a in ("A", "B")}
    summary_before = (out / "arms_summary.json").read_bytes()

    # failure while writing audio (after A.tmp/B.tmp exist)
    calls = {"n": 0}
    real = ma.write_crop

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("disk full")
        return real(*a, **k)

    monkeypatch.setattr(ma, "write_crop", flaky)
    with pytest.raises(RuntimeError, match="disk full"):
        run(tmp, arms=["A", "B"], overwrite=True)
    monkeypatch.undo()
    assert {a: snapshot(out / a) for a in ("A", "B")} == before
    assert (out / "arms_summary.json").read_bytes() == summary_before
    assert sorted(p.name for p in out.iterdir()) == ["A", "B", "arms_summary.json"]

    # failure while planning (a WAV is missing): nothing is touched at all
    (tmp / "prepared" / str(SR) / f"{tids[6]}.wav").unlink()
    with pytest.raises(FileNotFoundError):
        run(tmp, arms=["A", "B"], overwrite=True)
    assert {a: snapshot(out / a) for a in ("A", "B")} == before

    # a stale A.tmp from a crashed run (marker present) is cleaned up by the next run
    (out / "A.tmp").mkdir()
    (out / "A.tmp" / ma.MARKER).write_text("x")
    run(tmp, arms=["A"], limit=3, overwrite=True)
    assert not (out / "A.tmp").exists()


def test_symlinks_are_relative_and_verified(corpus, monkeypatch):
    tmp, tids = corpus
    run(tmp, arms=["A"], link_mode="symlink")
    out = tmp / "arms"
    link = out / "A" / "val" / f"{tids[0]}.wav"
    target = os.readlink(link)
    assert not os.path.isabs(target)
    assert os.path.realpath(link) == os.path.realpath(tmp / "prepared" / str(SR) / f"{tids[0]}.wav")
    # the tree survives a move (relative links)
    moved = tmp / "moved"
    (moved / "arms").mkdir(parents=True)
    os.rename(tmp / "prepared", moved / "prepared")
    os.rename(out / "A", moved / "arms" / "A")
    assert (moved / "arms" / "A" / "val" / f"{tids[0]}.wav").resolve().is_file()
    os.rename(moved / "prepared", tmp / "prepared")
    os.rename(moved / "arms" / "A", out / "A")

    # a dangling link fails verification and leaves the previous arm intact
    before = snapshot(out / "A")

    def dangling(src, dst, mode):
        os.symlink("../../nowhere.wav", dst)

    monkeypatch.setattr(ma, "_place_audio", dangling)
    with pytest.raises(RuntimeError, match="does not resolve"):
        run(tmp, arms=["A"], link_mode="symlink", overwrite=True)
    assert snapshot(out / "A") == before
    assert not (out / "A.tmp").exists()


def test_hardlink_falls_back_to_copy_across_devices(corpus, monkeypatch):
    tmp, tids = corpus

    def exdev(src, dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(ma.os, "link", exdev)
    run(tmp, arms=["A"], link_mode="hardlink", limit=2)
    short = tmp / "arms" / "A" / "val" / f"{tids[0]}.wav"
    assert short.is_file() and not short.is_symlink()
    assert short.read_bytes() == (tmp / "prepared" / str(SR) / f"{tids[0]}.wav").read_bytes()
    monkeypatch.undo()
    run(tmp, arms=["B"], link_mode="hardlink", limit=2)
    b = tmp / "arms" / "B" / "val" / f"{tids[0]}.wav"
    assert os.stat(b).st_ino == os.stat(tmp / "prepared" / str(SR) / f"{tids[0]}.wav").st_ino


def test_trackset_hash_guards_siblings_and_summary_is_merged(corpus):
    """m3: arms built on different track sets/windows are refused unless --overwrite."""
    tmp, _ = corpus
    out = tmp / "arms"
    run(tmp, arms=["A"], limit=2)
    run(tmp, arms=["B"], limit=2)  # same track set: fine, no --overwrite needed
    assert (out / "A" / ma.TRACKSET_FILE).read_text() == (out / "B" / ma.TRACKSET_FILE).read_text()
    with pytest.raises(ValueError, match="different track set"):
        run(tmp, arms=["C"], limit=3)
    assert not (out / "C").exists() and not (out / "C.tmp").exists()
    s = run(tmp, arms=["C"], limit=3, overwrite=True)
    assert set(s["arms"]) == {"A", "B", "C"}  # merged, not overwritten
    assert s["arms"]["A"]["limit"] == 2 and s["arms"]["C"]["limit"] == 3
    assert s["arms"]["A"]["trackset_sha256"] != s["arms"]["C"]["trackset_sha256"]
    on_disk = json.loads((out / "arms_summary.json").read_text())
    assert on_disk == s
    # a different sample-rate window would also change the hash
    plans = [ma.TrackPlan("t", "train", {}, tmp, SR, 10, "PCM_16", "none", choose_window([], 1.0), 0, 10)]
    assert ma.trackset_hash(plans, SR) != ma.trackset_hash(plans, SR + 1)


def test_missing_autocaption_is_an_error(corpus):
    tmp, tids = corpus
    (tmp / "auto.jsonl").write_text(json.dumps({"track_id": tids[0], "caption": "x"}) + "\n")
    with pytest.raises(KeyError):
        run(tmp, arms=["A", "D"])
    assert not (tmp / "arms").exists()


def test_missing_wav_is_an_error(corpus):
    tmp, tids = corpus
    (tmp / "prepared" / str(SR) / f"{tids[0]}.wav").unlink()
    with pytest.raises(FileNotFoundError):
        run(tmp, arms=["A"])
    assert not (tmp / "arms").exists()


def test_cli(corpus, capsys):
    tmp, _ = corpus
    rc = ma.main(["--manifest", str(tmp / "tracks.parquet"), "--splits", str(tmp / "splits.json"),
                  "--labels-dir", str(tmp / "labels"), "--prepared-root", str(tmp / "prepared"),
                  "--sr", str(SR), "--out-root", str(tmp / "arms"), "--arms", "A,B,C,E",
                  "--split", "train,val", "--limit", "3"])
    assert rc == 0
    s = json.loads(capsys.readouterr().out)
    assert set(s["arms"]) == {"A", "B", "C", "E"} and s["n_tracks"] == 3
    assert s["link_mode"] == "copy" and s["window_frames"] == N_WIN
