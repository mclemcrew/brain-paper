"""Build an *unblinded* listening page from a listening-test build.

``make_listening_test`` writes ``trials.json`` whose answer key is base64 JSON per trial (so the
blind test page never shows it). This script decodes those keys and writes ``samples.html`` next
to the test: one card per trial with the condition named, the full regenerated track (the
listening-test "context" file, with the regenerated span marked) and the three excerpts labelled
as "regenerated" or "original section". It is for the composer to listen through the material,
not for scoring; the blind test stays in ``index.html``.

    python -m ravi.eval.make_samples_page runs_local/listening_test [--conditions-note TEXT]

Only relative paths are written, so the folder can be served from any static host.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

CONDITION_NAMES = {
    "base": "Base model, no fine-tuning",
    "B6k": "Adapter B: human captions + genre tag (6k steps)",
    "B3k": "Adapter B: human captions + genre tag (3k steps)",
    "B2k": "Adapter B: human captions + genre tag (2k steps)",
    "B4k": "Adapter B: human captions + genre tag (4k steps)",
    "E6k": "Adapter E: human captions + section list (6k), sections in prompt",
    "E6k_nosec": "Adapter E: human captions + section list (6k), no sections in prompt",
    "A6k": "Adapter A: trigger word only (6k)",
    "C6k": "Adapter C: ID3 genre tags + BPM (6k)",
    "D6k": "Adapter D: auto-captions (6k)",
}


def _decode_key(key: str) -> dict:
    return json.loads(base64.b64decode(key).decode("utf-8"))


def _fmt_s(x: float) -> str:
    m, s = divmod(float(x), 60.0)
    return f"{int(m)}:{s:04.1f}"


def build(test_dir: Path, note: str = "") -> Path:
    trials_doc = json.loads((test_dir / "trials.json").read_text())
    meta = _decode_key(trials_doc["meta"]) if isinstance(trials_doc.get("meta"), str) else trials_doc.get("meta", {})
    cards: list[str] = []
    by_cond: Counter = Counter()
    by_track: dict[str, list[str]] = defaultdict(list)
    seen_items: set[str] = set()

    for block_name, block in sorted(trials_doc["blocks"].items()):
        for tr in block["trials"]:
            key = _decode_key(tr["key"])
            cond = key["condition"]
            item = key["item"]
            if item in seen_items:  # shared items appear in both rater blocks; show once
                continue
            seen_items.add(item)
            by_cond[cond] += 1
            m0, m1 = key["mask_in_context"]
            sr = key["sample_rate"]
            excerpts = []
            for letter in sorted(key["excerpts"]):
                ex = key["excerpts"][letter]
                role = "regenerated" if ex["role"] == "target" else "original section"
                cls = "target" if ex["role"] == "target" else "other"
                excerpts.append(
                    f'<div class="ex {cls}"><div class="exhead"><span class="letter">{letter}</span> '
                    f'<span class="role">{role}</span> <span class="lab">{html.escape(ex["label"])}</span> '
                    f'<span class="t">{_fmt_s(ex["start_s"])}–{_fmt_s(ex["end_s"])}</span></div>'
                    f'<audio controls preload="none" src="{html.escape(ex["file"])}"></audio></div>'
                )
            card = (
                f'<section class="card" data-cond="{html.escape(cond)}" data-track="{html.escape(key["track_id"])}">'
                f'<h3>{html.escape(key["track_id"])} <small>· {html.escape(key["label"])} (section {key["section_index"]}) '
                f'· seed {key["seed"]} · block {html.escape(block_name)} {html.escape(tr["id"])}</small></h3>'
                f'<p class="cond"><b>{html.escape(cond)}</b> — {html.escape(CONDITION_NAMES.get(cond, cond))}</p>'
                f'<p class="ctx">Full track with the regenerated section at {_fmt_s(m0 / sr)}–{_fmt_s(m1 / sr)} '
                f'(everything outside that span is the original audio):</p>'
                f'<audio controls preload="none" src="{html.escape(key["context_file"])}"></audio>'
                f'<div class="exs">{"".join(excerpts)}</div></section>'
            )
            cards.append(card)
            by_track[key["track_id"]].append(cond)

    conds = sorted(by_cond)
    opts = "".join(f'<option value="{html.escape(c)}">{html.escape(c)} ({by_cond[c]})</option>' for c in conds)
    legend = "".join(
        f"<li><b>{html.escape(c)}</b>: {html.escape(CONDITION_NAMES.get(c, c))}</li>" for c in conds
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>rAvI samples</title>
<style>
:root{{--bg:#fafafa;--fg:#1a1a1a;--card:#fff;--line:#ddd;--acc:#2b6cb0;--tgt:#fff4e0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--card:#1b1b1b;--line:#333;--acc:#7fb3ff;--tgt:#3a2d14}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;max-width:900px;margin:auto}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin:14px 0}}
h3{{margin:0 0 6px}} h3 small{{font-weight:normal;opacity:.7}} .cond{{margin:4px 0;color:var(--acc)}}
.ctx{{margin:8px 0 4px;opacity:.85}} audio{{width:100%}}
.exs{{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));margin-top:10px}}
.ex{{border:1px solid var(--line);border-radius:6px;padding:8px}} .ex.target{{background:var(--tgt)}}
.exhead{{display:flex;gap:8px;align-items:baseline;margin-bottom:4px}} .letter{{font-weight:bold}} .role{{opacity:.8}} .lab{{font-style:italic}} .t{{margin-left:auto;opacity:.7;font-size:13px}}
.controls{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0}} select{{font:inherit;padding:4px}}
ul.legend{{columns:1;padding-left:18px}} .hidden{{display:none}}
</style></head><body>
<h1>rAvI: regenerated sections, labelled</h1>
<p>Each card is one test track from Ravi's catalogue. A section was removed and regenerated by the named model;
the rest of the track is untouched. The highlighted excerpt is the regenerated section; the other two are
original sections of the same track, for comparison. {html.escape(note)}</p>
<p>The <a href="index.html">blind listening test</a> uses the same clips without the labels. For the other task, every model generating from the same written description, see <a href="gen.html">the generations page</a>.</p>
<details><summary>Conditions in this set</summary><ul class="legend">{legend}</ul>
<p>Source: {html.escape(meta.get("eval_root", ""))}, {len(cards)} unique items, {len(by_track)} tracks.</p></details>
<div class="controls"><label>Show: <select id="f"><option value="">all conditions</option>{opts}</select></label>
<span id="n"></span></div>
<div id="cards">{"".join(cards)}</div>
<script>
const sel=document.getElementById('f'),cards=[...document.querySelectorAll('.card')],n=document.getElementById('n');
function apply(){{const v=sel.value;let k=0;for(const c of cards){{const on=!v||c.dataset.cond===v;c.classList.toggle('hidden',!on);if(on)k++;}}n.textContent=k+' items';}}
sel.addEventListener('change',apply);apply();
document.addEventListener('play',e=>{{for(const a of document.querySelectorAll('audio'))if(a!==e.target)a.pause();}},true);
</script></body></html>"""
    out = test_dir / "samples.html"
    out.write_text(page)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test_dir", type=Path, help="listening-test build directory (holds trials.json and audio/)")
    ap.add_argument("--conditions-note", default="", help="extra sentence for the page header")
    a = ap.parse_args(argv)
    out = build(a.test_dir, a.conditions_note)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
