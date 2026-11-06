"""Build ``manifests/tracks.parquet``: one row per audio file in the corpus.

Usage::

    python -m ravi.data.build_manifest \
        --audio-dir ~/Downloads/ravi_dataset \
        --desc-dir ~/Downloads/descriptions \
        --labels-dir ~/Downloads/labels \
        [--manual-links manifests/manual_links.json] \
        --out manifests/tracks.parquet

``alt_family`` groups tracks that must never be split across train/val/test
(see :func:`assign_families`): ``[ALT]`` references, duplicate copilot ids,
identical title stems, the ``[AKA]`` neighbour rule, and the audio
near-duplicate pairs in ``manifests/manual_links.json`` (produced by
:mod:`ravi.data.audio_similarity`, hand-editable).

The data directories are treated as untrusted input: files are only read
(MP3 tags via mutagen, durations via ``ffprobe`` invoked with an argument
list, text files decoded with replacement). Nothing in them is executed or
imported.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ravi.structure.labels import interior_sections, labeled_segments, parse_audacity_labels

log = logging.getLogger("ravi.build_manifest")

LICENSE_TOKENS = ("NXLIC", "XLIC", "NS")
_FNAME_RE = re.compile(r"^(?P<cid>\d{4})_copilot(?:\.mp3)?(?P<rest>.*)\.mp3$")
_LICENSE_RE = re.compile(r"\b(NXLIC|XLIC|NS)\b")
_DATE_RE = re.compile(r"\b\d{1,2}-\d{1,2}(?:-\d{2,4})?\b")
_MARKER_RE = re.compile(r"\[\s*(ALT|AKA)\s*[\]\}]", re.IGNORECASE)
_ID_RE = re.compile(r"\b\d{4}\b")
#: Placeholder captions/tags left where the annotator had no audio. Word
#: boundaries matter: a bare ``no ?audio`` matched "piano audio".
JUNK_RE = re.compile(
    r"\bno\s*audio\b|^\W*n/?a\W*$|not working|label not found|no audio clip", re.IGNORECASE
)


AKA_WINDOW = 5  # copilot ids; see assign_families
DEFAULT_MANUAL_LINKS = Path(__file__).resolve().parents[2] / "manifests" / "manual_links.json"


def is_ignored_name(name: str) -> bool:
    """Stray files we never treat as corpus members."""
    return (
        name.endswith(".bak")
        or name.startswith("Cello")
        or "Users:" in name
        or "Desktop:" in name
        or name.startswith(".")
    )


# --------------------------------------------------------------------------- filenames


@dataclass(frozen=True)
class ParsedName:
    track_id: str
    copilot_id: str
    title: str
    license_flag: str | None
    license_tokens: tuple[str, ...]


def parse_filename(name: str) -> ParsedName:
    """``NNNN_copilot[.mp3][ Title][ NS|XLIC|NXLIC][ date].mp3`` -> fields.

    When several licence tokens appear (``NS NXLIC``) the last one wins; all of
    them are kept in ``license_tokens``.
    """
    m = _FNAME_RE.match(name)
    if not m:
        raise ValueError(f"unexpected audio filename: {name!r}")
    rest = m.group("rest")
    tokens = tuple(_LICENSE_RE.findall(rest))
    title = _LICENSE_RE.sub(" ", rest)
    title = _DATE_RE.sub(" ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return ParsedName(
        track_id=name[: -len(".mp3")],
        copilot_id=m.group("cid"),
        title=title,
        license_flag=tokens[-1] if tokens else None,
        license_tokens=tokens,
    )


_STEM_PAREN_RE = re.compile(r"\([^)]*\)?|\[[^\]]*\]?")
_STEM_WORDS_RE = re.compile(
    r"\b(?:re)?mix(?:es)?\b|\bparts?\b|\bpt\b|\bversions?\b|\bver\b|\bedit\b|\balt(?:ernate)?\b|\btake\b"
)


def title_stem(title: str | None) -> str:
    """Normalised title for grouping alternate versions.

    Lower-case; drop parentheticals (also an unclosed trailing one), licence
    tokens, the words mix/remix/part/pt/version/ver/edit/alt/alternate/take,
    punctuation, and trailing numbers. ``"Move, Move, Move (Tribal Mix)"`` and
    ``"Decade Chase 2"`` become ``"move move move"`` and ``"decade chase"``.
    Returns ``""`` for an empty title.
    """
    t = (title or "").lower()
    t = _STEM_PAREN_RE.sub(" ", t)
    t = _LICENSE_RE.sub(" ", t.upper()).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = _STEM_WORDS_RE.sub(" ", t)
    t = re.sub(r"(?:\s+\d+)+\s*$", "", " " + t.strip())
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------- ID3


def parse_id3_genre(raw: str) -> tuple[list[str], list[str], bool]:
    """Split an ID3 genre string into (tokens, alt_refs, is_aka).

    ``[ALT] NNNN [NNNN ...]`` and ``[AKA]`` markers (always trailing in this
    corpus; one is misspelt ``[ALT}``) are removed from the tokens. Every
    4-digit number after an ``[ALT]`` marker is an alt reference. Free text
    after a marker (``electronic version``, ``2 beep``) is discarded from the
    tokens but remains in the raw string.
    """
    raw = raw or ""
    markers = list(_MARKER_RE.finditer(raw))
    head = raw[: markers[0].start()] if markers else raw
    alt_refs: list[str] = []
    is_aka = False
    for i, mk in enumerate(markers):
        tail_end = markers[i + 1].start() if i + 1 < len(markers) else len(raw)
        if mk.group(1).upper() == "ALT":
            alt_refs.extend(_ID_RE.findall(raw[mk.end() : tail_end]))
        else:
            is_aka = True
    tokens: list[str] = []
    for tok in head.split(","):
        tok = tok.strip()
        if tok and tok not in tokens:
            tokens.append(tok)
    return tokens, list(dict.fromkeys(alt_refs)), is_aka


def read_id3(path: Path) -> dict:
    from mutagen.id3 import ID3, ID3NoHeaderError

    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = {}
    genre_raw = ", ".join(tags["TCON"].text) if "TCON" in tags else ""
    bpm = None
    if "TBPM" in tags and tags["TBPM"].text:
        try:
            bpm = int(round(float(str(tags["TBPM"].text[0]).strip())))
        except ValueError:
            bpm = None
    tokens, alt_refs, is_aka = parse_id3_genre(genre_raw)
    return {
        "id3_genre_raw": genre_raw,
        "id3_genre_tokens": tokens,
        "id3_alt_refs": alt_refs,
        "id3_is_aka": is_aka,
        "id3_bpm": bpm,
    }


# --------------------------------------------------------------------------- ffprobe


def ffprobe_audio(path: Path) -> dict:
    """Duration and sample rate of the first *audio* stream (ignores cover art)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=duration,sample_rate,channels,codec_name",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    streams = json.loads(out).get("streams", [])
    if not streams:
        raise RuntimeError(f"no audio stream in {path}")
    s = streams[0]
    return {
        "duration_s": float(s["duration"]),
        "sample_rate": int(s["sample_rate"]),
        "channels": int(s.get("channels", 0)),
        "codec": s.get("codec_name"),
    }


# --------------------------------------------------------------------------- descriptions


def parse_description(path: Path) -> dict[str, str]:
    """``key:`` lines followed by a value line (or ``key: value`` on one line)."""
    fields: dict[str, str] = {}
    lines = path.read_bytes().decode("utf-8", errors="replace").splitlines()
    i = 0
    keys = ("description", "genre-tags", "processed")
    while i < len(lines):
        line = lines[i].strip()
        key = next((k for k in keys if line.lower().startswith(k + ":")), None)
        if key is None:
            i += 1
            continue
        inline = line[len(key) + 1 :].strip()
        if inline:
            fields[key] = inline
            i += 1
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if any(nxt.lower().startswith(k + ":") for k in keys):
            fields[key] = ""
            i += 1
        else:
            fields[key] = nxt
            i += 2
    return fields


def is_junk_caption(caption: str | None, tag: str | None) -> bool:
    caption = (caption or "").strip()
    tag = (tag or "").strip()
    if not caption:
        return True
    return bool(JUNK_RE.search(caption) or JUNK_RE.search(tag))


# --------------------------------------------------------------------------- families


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def assign_families(
    copilot_ids: list[str],
    alt_refs: list[list[str]],
    *,
    track_ids: list[str] | None = None,
    title_stems: list[str] | None = None,
    is_aka: list[bool] | None = None,
    genre_raw: list[str] | None = None,
    manual_links: list[list] | None = None,
    aka_window: int = AKA_WINDOW,
) -> tuple[list[int], list[str]]:
    """Union-find over copilot ids; returns ``(family per row, missing refs)``.

    Rows sharing a copilot id share a node, so duplicate ids (0207, 0564) are
    merged automatically. Then, in order:

    1. each ``[ALT]`` reference to an id present in the corpus merges the two
       nodes (references to absent ids are returned as ``missing``;
       self-references are no-ops);
    2. rows with the same non-empty :func:`title_stem` are merged;
    3. ``[AKA]`` rule: a row with ``is_aka``, no ``[ALT]`` refs and still a
       singleton family after 1-2 is merged with every row whose
       ``id3_genre_raw`` is identical and whose copilot id is within
       ``aka_window`` (``[AKA]`` marks an alias of a neighbouring upload);
    4. every ``[track_id_a, track_id_b, ...]`` pair in ``manual_links`` is
       merged (needs ``track_ids``; unknown ids raise ``KeyError``).

    Family ids are dense ints ordered by the smallest copilot id in the family.
    """
    uf = UnionFind()
    present = set(copilot_ids)
    for cid in copilot_ids:
        uf.add(cid)
    missing: list[str] = []
    for cid, refs in zip(copilot_ids, alt_refs):
        for ref in refs:
            if ref in present:
                uf.union(cid, ref)
            else:
                missing.append(f"{cid}->{ref}")

    if title_stems is not None:
        first_of_stem: dict[str, str] = {}
        for cid, st in zip(copilot_ids, title_stems):
            if st:
                uf.union(first_of_stem.setdefault(st, cid), cid)

    if is_aka is not None and genre_raw is not None:
        size: Counter[str] = Counter(uf.find(c) for c in set(copilot_ids))
        rows_per_cid = Counter(copilot_ids)
        merges: list[tuple[str, str]] = []
        for i, (cid, aka, refs, g) in enumerate(zip(copilot_ids, is_aka, alt_refs, genre_raw)):
            if not aka or refs or size[uf.find(cid)] != 1 or rows_per_cid[cid] != 1:
                continue
            for cid2, g2 in zip(copilot_ids, genre_raw):
                if cid2 != cid and g2 == g and abs(int(cid2) - int(cid)) <= aka_window:
                    merges.append((cid, cid2))
        for a, b in merges:  # decided on the pre-rule state, so order does not matter
            uf.union(a, b)

    if manual_links:
        if track_ids is None:
            raise ValueError("manual_links needs track_ids")
        cid_of = dict(zip(track_ids, copilot_ids))
        for link in manual_links:
            a, b = link[0], link[1]
            unknown = [t for t in (a, b) if t not in cid_of]
            if unknown:
                raise KeyError(f"manual link {a!r} <-> {b!r}: track_id not in corpus: {unknown}")
            uf.union(cid_of[a], cid_of[b])

    roots = sorted({uf.find(c) for c in copilot_ids})
    fam_of_root = {r: i for i, r in enumerate(roots)}
    return [fam_of_root[uf.find(c)] for c in copilot_ids], missing


def families_from_manifest(df: pd.DataFrame, manual_links: list[list] | None = None) -> list[int]:
    """:func:`assign_families` with every rule, from manifest columns."""
    stems = df["title_stem"].tolist() if "title_stem" in df else [title_stem(t) for t in df["title"]]
    fam, _ = assign_families(
        df["copilot_id"].tolist(),
        [list(r) for r in df["id3_alt_refs"]],
        track_ids=df["track_id"].tolist(),
        title_stems=stems,
        is_aka=df["id3_is_aka"].astype(bool).tolist(),
        genre_raw=df["id3_genre_raw"].tolist(),
        manual_links=manual_links,
    )
    return fam


def load_manual_links(path: Path | None) -> list[list]:
    if path is None or not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text())
    pairs = data.get("pairs", [])
    for p in pairs:
        if not (isinstance(p, list) and len(p) >= 2 and all(isinstance(t, str) for t in p[:2])):
            raise ValueError(f"{path}: malformed pair {p!r}")
    return pairs


# --------------------------------------------------------------------------- main


def list_audio(audio_dir: Path) -> list[Path]:
    return sorted(
        p for p in audio_dir.iterdir()
        if p.is_file() and p.name.lower().endswith(".mp3") and not is_ignored_name(p.name)
    )


def build_manifest(
    audio_dir: Path,
    desc_dir: Path,
    labels_dir: Path,
    workers: int = 8,
    manual_links: list[list] | Path | None = DEFAULT_MANUAL_LINKS,
) -> pd.DataFrame:
    """``manual_links`` is a list of pairs or a path to ``manual_links.json``
    (a missing file means no links)."""
    if manual_links is None or isinstance(manual_links, (str, Path)):
        manual_links = load_manual_links(Path(manual_links) if manual_links is not None else None)
    audio = list_audio(audio_dir)
    with ThreadPoolExecutor(workers) as ex:
        probes = list(ex.map(ffprobe_audio, audio))

    rows = []
    label_issues: Counter[str] = Counter()
    for path, probe in zip(audio, probes):
        pn = parse_filename(path.name)
        row: dict = {
            "track_id": pn.track_id,
            "audio_filename": path.name,
            "audio_path": str(path),
            "copilot_id": pn.copilot_id,
            "title": pn.title,
            "title_stem": title_stem(pn.title),
            "license_flag": pn.license_flag,
            "license_tokens": list(pn.license_tokens),
            **probe,
            **read_id3(path),
        }

        desc_path = desc_dir / f"{pn.track_id}_description.txt"
        if desc_path.is_file():
            fields = parse_description(desc_path)
            row["desc_path"] = str(desc_path)
            row["human_caption"] = fields.get("description", "")
            row["human_genre_tag"] = fields.get("genre-tags", "")
        else:
            log.warning("no description for %s", path.name)
            row["desc_path"] = None
            row["human_caption"] = ""
            row["human_genre_tag"] = ""
        row["caption_is_junk"] = is_junk_caption(row["human_caption"], row["human_genre_tag"])

        labels_path = labels_dir / f"{path.name}_labels.txt"
        if labels_path.is_file():
            issues: list[str] = []
            segs = parse_audacity_labels(labels_path, probe["duration_s"], issues=issues)
            label_issues[pn.track_id] = len(issues)
            row.update(
                labels_path=str(labels_path),
                has_labels=True,
                n_segments=len(labeled_segments(segs)),
                n_segments_total=len(segs),
                n_interior=len(interior_sections(segs)),
                segment_labels=[s.label for s in segs],
                n_label_issues=len(issues),
            )
        else:
            row.update(
                labels_path=None, has_labels=False, n_segments=0, n_segments_total=0,
                n_interior=0, segment_labels=[], n_label_issues=0,
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    present = set(df["copilot_id"])
    df["alt_refs_missing"] = [[r for r in refs if r not in present] for refs in df["id3_alt_refs"]]
    df["alt_self_ref"] = [cid in refs for cid, refs in zip(df["copilot_id"], df["id3_alt_refs"])]
    for r in df.itertuples():
        if r.alt_self_ref:
            log.warning("[ALT] self-reference: %s refers to its own id %s", r.track_id, r.copilot_id)
        if r.alt_refs_missing:
            log.warning("[ALT] references to ids not in corpus: %s -> %s", r.track_id, " ".join(r.alt_refs_missing))
    links_in_corpus = [p for p in manual_links if p[0] in set(df["track_id"]) and p[1] in set(df["track_id"])]
    if len(links_in_corpus) != len(manual_links):
        dropped = [p[:2] for p in manual_links if p not in links_in_corpus]
        log.warning("manual links naming tracks not in this corpus ignored: %s", dropped)
    df["alt_family"] = families_from_manifest(df, links_in_corpus)
    df["id3_bpm"] = df["id3_bpm"].astype("Int64")

    # Orphans: description/label files that matched no audio file.
    used_desc = set(df["desc_path"].dropna().map(lambda p: Path(p).name))
    used_lab = set(df["labels_path"].dropna().map(lambda p: Path(p).name))
    for d, used, suffix in ((desc_dir, used_desc, "_description.txt"), (labels_dir, used_lab, "_labels.txt")):
        orphans = sorted(
            p.name for p in d.iterdir()
            if p.name.endswith(suffix) and not is_ignored_name(p.name) and p.name not in used
        )
        if orphans:
            log.warning("%d orphan files in %s: %s", len(orphans), d, orphans)

    assert df["track_id"].is_unique, "track_id must be unique"
    return df


def summarize(df: pd.DataFrame) -> str:
    fam_sizes = df.groupby("alt_family").size()
    lines = [
        f"rows              {len(df)}",
        f"has_labels        {int(df.has_labels.sum())}",
        f"caption_is_junk   {int(df.caption_is_junk.sum())}",
        f"families          {df.alt_family.nunique()} (multi-member {int((fam_sizes > 1).sum())}, largest {int(fam_sizes.max())})",
        f"hours             {df.duration_s.sum() / 3600:.3f}",
        f"id3_bpm present   {int(df.id3_bpm.notna().sum())}",
        f"alt refs present  {int(df.id3_alt_refs.map(len).gt(0).sum())}",
        f"alt refs missing  {int(df.alt_refs_missing.map(len).gt(0).sum())}  self-refs {int(df.alt_self_ref.sum())}",
        f"title stems shared {int(df[df.title_stem != ''].title_stem.duplicated(keep=False).sum())} tracks",
        f"tracks w/ label issues {int(df.n_label_issues.gt(0).sum())}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--desc-dir", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("manifests/tracks.parquet"))
    ap.add_argument("--manual-links", type=Path, default=DEFAULT_MANUAL_LINKS,
                    help="near-duplicate pairs to keep in one family (missing file = none)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("-v", "--verbose", action="store_true", help="log every dropped/repaired label line")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    df = build_manifest(args.audio_dir.expanduser(), args.desc_dir.expanduser(), args.labels_dir.expanduser(),
                        args.workers, manual_links=args.manual_links)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(summarize(df))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
