import shutil
import subprocess

import pandas as pd
import pytest

from ravi.data.build_manifest import (
    assign_families,
    build_manifest,
    is_ignored_name,
    is_junk_caption,
    parse_description,
    parse_filename,
    parse_id3_genre,
    title_stem,
)


@pytest.mark.parametrize(
    "name,track_id,title,flag",
    [
        ("0001_copilot.mp3 Agent X NS.mp3", "0001_copilot.mp3 Agent X NS", "Agent X", "NS"),
        ("0007_copilot NS.mp3", "0007_copilot NS", "", "NS"),
        ("0213_copilot.mp3.mp3", "0213_copilot.mp3", "", None),
        ("0207_copilot.mp3", "0207_copilot", "", None),
        ("0097_copilot.mp3 Thoughts Rushing NS NXLIC 8-30-17.mp3", "0097_copilot.mp3 Thoughts Rushing NS NXLIC 8-30-17", "Thoughts Rushing", "NXLIC"),
        ("0100_copilot.mp3 Rest assured XLIC 9-07.mp3", "0100_copilot.mp3 Rest assured XLIC 9-07", "Rest assured", "XLIC"),
        ("0101_copilot XLIC 12-20-2010.mp3", "0101_copilot XLIC 12-20-2010", "", "XLIC"),
        ("0102_copilot.mp3 Is That A Challenge- NS.mp3", "0102_copilot.mp3 Is That A Challenge- NS", "Is That A Challenge-", "NS"),
        ("0103_copilot.mp3 Sound Collage op 18 NS.mp3", "0103_copilot.mp3 Sound Collage op 18 NS", "Sound Collage op 18", "NS"),
        ("0104_copilot.mp3 NSYNC Tribute.mp3", "0104_copilot.mp3 NSYNC Tribute", "NSYNC Tribute", None),
    ],
)
def test_parse_filename(name, track_id, title, flag):
    pn = parse_filename(name)
    assert (pn.track_id, pn.copilot_id, pn.title, pn.license_flag) == (track_id, name[:4], title, flag)


def test_parse_filename_rejects_unknown():
    with pytest.raises(ValueError):
        parse_filename("Cello.wav")


@pytest.mark.parametrize(
    "raw,tokens,refs,aka",
    [
        ("Pop, Fun, Light", ["Pop", "Fun", "Light"], [], False),
        ("Fun, Pop, [ALT] 0014", ["Fun", "Pop"], ["0014"], False),
        ("Light, Pop [ALT] 0072 0073", ["Light", "Pop"], ["0072", "0073"], False),
        ("Fun, Pop, Vocals, [AKA]", ["Fun", "Pop", "Vocals"], [], True),
        ("Pop, Latin, Fun, Sexy, Vocals [ALT} 3223", ["Pop", "Latin", "Fun", "Sexy", "Vocals"], ["3223"], False),
        ("Pop, Light, Fun [AKA] 2 beep", ["Pop", "Light", "Fun"], [], True),
        ("Rock [ALT] 0100 electronic version", ["Rock"], ["0100"], False),
        ("Pop,, Pop , ", ["Pop"], [], False),
        ("", [], [], False),
    ],
)
def test_parse_id3_genre(raw, tokens, refs, aka):
    assert parse_id3_genre(raw) == (tokens, refs, aka)


def test_families_union_find():
    ids = ["0001", "0002", "0003", "0004", "0005", "0207", "0207", "0006"]
    refs = [["0002"], [], ["0004"], ["0002"], ["9999"], [], [], []]
    fam, missing = assign_families(ids, refs)
    assert fam[0] == fam[1] == fam[2] == fam[3]  # chain 0001-0002-0004-0003
    assert fam[5] == fam[6]                      # duplicate copilot id
    assert len({fam[0], fam[4], fam[5], fam[7]}) == 4  # singletons get their own ids
    assert sorted(set(fam)) == list(range(len(set(fam))))  # dense
    assert missing == ["0005->9999"]


def test_families_deterministic_under_permutation():
    ids = ["0003", "0001", "0002"]
    fam, _ = assign_families(ids, [["0001"], [], []])
    assert fam == [0, 0, 1]


@pytest.mark.parametrize(
    "caption,tag,junk",
    [
        ("No Audio", "No Audio", True),
        ("NoNo Audio", "No Audio", True),
        ("There's no audio clip for audio 207.", "NA", True),
        ("Not working.", "Not working.", True),
        ("Label not found", "Label not found.", True),
        ("NA", "NA", True),
        ("", "Rock", True),
        ("   ", "Rock", True),
        ("A bright pop track", "NA", True),
        ("A bright pop track", "Pop", False),
        ("Nadia's waltz on a national stage", "Folk", False),
        ("Dreamy synths", "", False),
        ("A calm piano audio with strings", "Classical", False),  # "piano audio" is not "no audio"
        ("Mono audio, lo-fi", "Lo-fi", False),
        ("No audio found.", "NA", True),
        ("Something", "N/A", True),
        ("Something", " n/a. ", True),
        ("Banana split", "Nashville", False),
    ],
)
def test_junk_caption(caption, tag, junk):
    assert is_junk_caption(caption, tag) is junk


def test_parse_description(tmp_path):
    p = tmp_path / "d_description.txt"
    p.write_text("description:\nA fast track.\ngenre-tags:\nElectronic\nprocessed: true")
    assert parse_description(p) == {"description": "A fast track.", "genre-tags": "Electronic", "processed": "true"}
    p.write_text("description:\n\ngenre-tags:\n\nprocessed: true")
    assert parse_description(p)["description"] == ""
    p.write_text("description:\ngenre-tags:\nRock\n")
    assert parse_description(p) == {"description": "", "genre-tags": "Rock"}


@pytest.mark.parametrize("name", ["a.txt.bak", "Cello_description.txt", "Users:jessica:x.txt", "s:Desktop:y.txt", ".DS_Store"])
def test_ignored_names(name):
    assert is_ignored_name(name)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_build_manifest_end_to_end(tmp_path):
    from mutagen.id3 import ID3, TBPM, TCON

    audio, desc, labs = tmp_path / "audio", tmp_path / "desc", tmp_path / "labels"
    for d in (audio, desc, labs):
        d.mkdir()
    names = ["0001_copilot.mp3 Song A NS.mp3", "0002_copilot.mp3", "0002_copilot.mp3 Dup.mp3", "0003_copilot XLIC.mp3"]
    genres = ["Pop, Fun [ALT] 0003", "Rock [AKA]", "Rock", "Jazz [AKA]"]
    for n, g in zip(names, genres):
        p = audio / n
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=3:sample_rate=48000",
             "-ac", "2", "-b:a", "128k", str(p)],
            check=True,
        )
        tags = ID3()
        tags.add(TCON(encoding=3, text=[g]))
        if n.startswith("0001"):
            tags.add(TBPM(encoding=3, text=["120"]))
        tags.save(p)
    (audio / "notes.txt.bak").write_text("ignore me")
    stem = lambda n: n[:-4]
    (desc / f"{stem(names[0])}_description.txt").write_text("description:\nUpbeat pop.\ngenre-tags:\nPop\nprocessed: true")
    (desc / f"{stem(names[1])}_description.txt").write_text("description:\nNo Audio\ngenre-tags:\nNo Audio\nprocessed: true")
    (labs / f"{names[0]}_labels.txt").write_text("0\t1\tintro\n1\t2\tverse\n2\t3.5\toutro\n")
    (labs / f"{names[3]}_labels.txt").write_text("0.8\t2\tverse\n")
    (labs / "Cello.wav_labels.txt").write_text("0\t1\tintro\n")

    df = build_manifest(audio, desc, labs, workers=2, manual_links=None).set_index("audio_filename")
    assert len(df) == 4
    a = df.loc[names[0]]
    assert a.title == "Song A" and a.license_flag == "NS" and a.id3_bpm == 120
    assert list(a.id3_alt_refs) == ["0003"] and list(a.id3_genre_tokens) == ["Pop", "Fun"]
    assert a.has_labels and a.n_segments == 3 and a.n_interior == 1
    assert a.duration_s == pytest.approx(3.0, abs=0.1) and a.sample_rate == 48000
    assert a.segment_labels[-1] == "outro"  # 3.5 clamped to duration
    assert not a.caption_is_junk
    assert df.loc[names[1]].caption_is_junk
    assert df.loc[names[2]].caption_is_junk and pd.isna(df.loc[names[2]].desc_path)  # missing description
    assert df.loc[names[1]].alt_family == df.loc[names[2]].alt_family  # duplicate copilot id
    assert df.loc[names[0]].alt_family == df.loc[names[3]].alt_family  # [ALT] ref
    assert df.alt_family.nunique() == 2
    assert list(df.loc[names[0]].alt_refs_missing) == [] and not df.loc[names[0]].alt_self_ref
    assert df.loc[names[0]].title_stem == "song a"
    assert list(df.loc[names[3]].segment_labels) == ["unlabeled", "verse"]  # 1 s tail < 2 s: not filled


@pytest.mark.parametrize(
    "title,stem",
    [
        ("Move, Move, Move (Tribal Mix)", "move move move"),
        ("Move, Move, Move (Neo Soul Mix", "move move move"),  # unclosed parenthesis
        ("Decade Chase 2", "decade chase"),
        ("Grocery Epiphany 1", "grocery epiphany"),
        ("Timekeeper's Waltz (Piano)", "timekeeper s waltz"),
        ("Two finger Jazz", "two finger jazz"),
        ("Acoustic Tone Poem 3", "acoustic tone poem"),
        ("Night Drive Remix", "night drive"),
        ("Theme Part 2", "theme"),
        ("Sunrise - Alternate Version", "sunrise"),
        ("Thoughts Rushing NS", "thoughts rushing"),
        ("Mixology", "mixology"),  # "mix" only as a whole word
        ("1999", ""),
        ("", ""),
    ],
)
def test_title_stem(title, stem):
    assert title_stem(title) == stem


def _fam(ids, refs=None, **kw):
    refs = refs or [[] for _ in ids]
    fam, _ = assign_families(ids, refs, **kw)
    return fam


def test_families_title_stems_union():
    fam = _fam(["0001", "0002", "0003", "0004"], title_stems=["decade chase", "", "decade chase", ""])
    assert fam[0] == fam[2] and len(set(fam)) == 3  # empty stems never group


def test_families_aka_singleton_rule():
    ids = ["0030", "0034", "0036", "0040", "0050"]
    genre = ["Fun, Pop", "Fun, Pop [AKA]", "Fun, Pop [AKA]", "Fun, Pop [AKA]", "Jazz [AKA]"]
    fam = _fam(ids, is_aka=[False, True, True, True, True], genre_raw=genre)
    assert fam[1] == fam[2] == fam[3]  # identical raw genre within +/-5 ids (0034-0036, 0036-0040)
    assert fam[0] != fam[1]            # raw string differs (no [AKA] marker)
    assert fam[4] not in fam[:4]       # no neighbour with the same genre
    far = _fam(["0034", "0041"], is_aka=[True, True], genre_raw=["Fun, Pop [AKA]"] * 2)
    assert far[0] != far[1]            # 7 ids apart
    # an [AKA] row that already has an [ALT] link is not a candidate
    linked = _fam(["0034", "0036", "0099"], [["0099"], [], []], is_aka=[True, False, False],
                  genre_raw=["Rock [AKA]", "Rock [AKA]", "Jazz"])
    assert linked[0] == linked[2] != linked[1]
    # nor is one whose family is not a singleton (here via a shared title stem)
    grouped = _fam(["0034", "0036", "0090"], is_aka=[True, False, False], genre_raw=["Rock [AKA]", "Rock [AKA]", "Jazz"],
                   title_stems=["song", "", "song"])
    assert grouped[0] == grouped[2] != grouped[1]


def test_families_manual_links_and_unknown_ids():
    ids, tids = ["0001", "0002", "0003"], ["0001_copilot", "0002_copilot X", "0003_copilot"]
    fam = _fam(ids, track_ids=tids, manual_links=[["0001_copilot", "0003_copilot", 0.9, "test"]])
    assert fam[0] == fam[2] != fam[1]
    with pytest.raises(KeyError):
        _fam(ids, track_ids=tids, manual_links=[["0001_copilot", "nope", 0.9, "x"]])


def test_alt_self_and_missing_refs_recorded(tmp_path, caplog):
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    from mutagen.id3 import ID3, TCON

    audio, desc, labs = tmp_path / "a", tmp_path / "d", tmp_path / "l"
    for d in (audio, desc, labs):
        d.mkdir()
    for n, g in [("0405_copilot.mp3", "Pop [ALT] 0405"), ("0406_copilot.mp3", "Pop [ALT] 3221 0405")]:
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=1:sample_rate=48000",
                        str(audio / n)], check=True)
        t = ID3()
        t.add(TCON(encoding=3, text=[g]))
        t.save(audio / n)
    with caplog.at_level("WARNING", logger="ravi.build_manifest"):
        df = build_manifest(audio, desc, labs, workers=1, manual_links=None).set_index("copilot_id")
    assert df.loc["0405"].alt_self_ref and not df.loc["0406"].alt_self_ref
    assert list(df.loc["0406"].alt_refs_missing) == ["3221"] and list(df.loc["0405"].alt_refs_missing) == []
    assert df.loc["0405"].alt_family == df.loc["0406"].alt_family
    assert "self-reference" in caplog.text and "3221" in caplog.text
