# Experiment decisions log

## 2026-10-15 — Tier 0 go/no-go: does section inpainting survive the LoRA?  → GO

Setup: SA3 `medium` (post-trained, 8 steps, cfg 1.0) with the arm-B LoRA trained 3,000 steps on
`medium-base` (DoRA-rows r16, 47 s window, duration conditioner excluded, upstream mask mix
10% random segments / 80% full / 10% causal). 20 test tracks, one interior verse/bridge each
(mask = human section ± 1 s), prompt = arm-B human caption. Conditions: base, LoRA at
strength 0 / 0.5 / 1.0, upper anchor (original region re-inserted), lower anchor (length-matched
region from another test track spliced in). Metrics run: preliminary `run_eval` (commit e832f94,
before the review fixes; CLAP/MERT region similarity vs original, seam delta, beat CV (global
tempo), key agreement). Outputs: `eval/tier0/` on the GPU box.

| condition | CLAP region sim (median) | MERT region sim | seam delta (median) | beat CV | key agree (n=10) |
|---|---|---|---|---|---|
| upper anchor | 1.000 | 1.000 | 0.000 | 0.025 | 7/10 |
| base | 0.817 | 0.928 | 0.019 | 0.028 | 0/10 |
| LoRA s=0 | = base (bit-identical) | = base | = base | = base | = base |
| LoRA s=0.5 | 0.812 | 0.935 | 0.024 | 0.029 | 3/10 |
| LoRA s=1.0 | 0.828 | 0.933 | 0.020 | 0.023 | 3/10 |
| lower anchor | 0.518 | 0.867 | 0.151 | 0.050 | 5/10 |

Reading: anchors separate cleanly (base ≫ lower on every metric), LoRA at s=1.0 is within noise
of base on region similarity and seams (mean diffs −0.007 CLAP / +0.006 MERT / +0.004 seam) and
slightly better on beat CV and key agreement. Text-to-music CLAP text score drops with the
adapter (0.41 → 0.35 at s=1.0); to be re-read after the loudness-normalised CLAP fix.

Decision: GO. Keep the upstream mask mix; proceed with the tier-1 grid (A, B, E @ 6k) as queued.
Re-run `run_eval` with the fixed metrics (PCA'd FAD∞, clip bootstrap, LUFS-matched CLAP,
per-track clustering) on the same outputs before quoting numbers; n=20 is directional only.

### Update (same day) — re-run with the reviewed metrics (commit a99aefa: PCA-64 FAD∞, clip bootstrap, LUFS-matched CLAP, per-track clustering, beat-grid agreement, centred MERT)

Paired per-track comparisons vs base, n = 20 regions:

| metric | LoRA s=0.5 | LoRA s=1.0 | upper anchor | lower anchor |
|---|---|---|---|---|
| CLAP region sim (median diff, p two-sided) | −0.003 (0.93) | −0.0005 (0.84) | +0.178 (<1e-5) | −0.280 (<1e-5) |
| MERT region sim, centred | −0.010 (0.84) | −0.007 (0.62) | +0.554 (<1e-5) | −0.390 (<1e-5) |
| |seam delta| | −0.007 (0.41) | +0.002 (0.93) | −0.056 (<1e-5) | +0.059 (<1e-5) |
| key agreement (n=10 gated) | +0.3 (0.25) | +0.3 (0.25) | +0.3 (0.45) | −0.3 (0.38) |

No metric moves against the LoRA; anchors separate as designed. GO stands. CLAP FAD∞ is NaN for
tier 0 by design (420 windows < the 500-row floor) — distance metrics are a tier-1 question.

## 2026-10-15 — Tier 1 results (157 test regions × 2 seeds, 103 tracks paired; 109 test prompts). `eval/tier1/`

### Text-to-music (PCA-64 FAD∞ and KAD to the 580-track training corpus; matched n = 109 clips; real-audio floor = held-out originals)

| condition | FAD∞ (CLAP) | KAD (CLAP) | CLAP text score | gen loudness |
|---|---|---|---|---|
| test originals (floor) | 0.055 | 0.80 | 0.283 | — |
| base | 0.165 | 4.45 | 0.346 | −17.2 LUFS |
| B6k (human caption) | **0.140** | **3.96** | 0.312 | −14.3 |
| E6k, no sections in prompt | **0.135** | **3.88** | 0.304 | −14.5 |
| E6k, sections in prompt | 0.151 | 4.51 | 0.308 | −14.4 |
| A6k (trigger only) | 0.402 | 17.0 | 0.141 | −14.2 |

Reading: B and E adapters move generations toward the corpus (FAD∞ −15%, KAD −12%) at a small cost in
text adherence (0.346 → 0.31). The trigger-only adapter A is far *worse* than base on both distance
metrics and ignores the prompt (CLAP text 0.14): with no words to anchor on, the adapter over-fits a
narrow style. Adapters also learned the corpus loudness (−14 LUFS vs base −17). No memorisation
flags above the null (max cosine to any train clip ≤ 0.955; null p99 = 0.982).

### Section regeneration, paired per track vs base (Wilcoxon, n = 103)

| condition | CLAP region sim Δ | MERT (centred) Δ | |seam| Δ | beat-grid Δ | key agree Δ (n=71) |
|---|---|---|---|---|---|
| A6k | +0.015 (p .03) | +0.024 (p .08) | −0.002 (ns) | −0.010 (ns) | −0.03 (ns) |
| B6k | +0.011 (ns) | +0.009 (ns) | −0.003 (p .08) | −0.017 (p .06) | −0.08 (p .06) |
| E6k | +0.011 (ns) | −0.001 (ns) | −0.001 (ns) | −0.021 (ns) | −0.08 (p .05) |
| E6k no sections | +0.014 (p .04) | +0.019 (ns) | −0.002 (p .09) | −0.007 (ns) | −0.06 (ns) |
| upper anchor | +0.239 | +0.581 | −0.055 | +0.272 | — |
| lower anchor | −0.225 | −0.350 | +0.096 | −0.163 | −0.58 |

Reading: every adapter is at least as good as base at filling a section; effects are tiny next to
the anchor spread (CLAP ±0.24), so coherence is dominated by the base model's two-sided context
conditioning, not the adapter. **Section regeneration survives fine-tuning (RQ1a: yes).**

### Does telling the model about structure help? (2×2: adapter {B, E} × prompt {without, with `sections:` list})

| comparison | CLAP region sim | MERT centred |
|---|---|---|
| prompt-time: base+sections vs base | −0.005 (p .19) | **−0.013 (p .03)** |
| prompt-time: B6k+sections vs B6k | **−0.004 (p .01)** | **−0.008 (p .02)** |
| prompt-time: E6k with vs without sections | −0.003 (p .06) | **−0.020 (p .002)** |
| training-time: E6k(no sec) vs B6k | +0.003 (p .16) | +0.009 (p .23) |
| training-time: E6k(with sec) vs B6k | −0.000 (p .71) | −0.010 (p .05) |

Reading (**RQ1b: no**): adding a section list to the prompt at inference time makes the regenerated
section consistently *less* similar to the original, for base and for every adapter. Training on
section lists (arm E) is neutral relative to arm B when both are prompted the same way. Structure
text is not a useful control signal for this model at this data scale; structure *analysis* (to find
the section to regenerate) is where structure earns its place.

Caveats: 2 seeds averaged per region; effect sizes small; key agreement gated to 71 tracks; the
E6k-with-sections training/prompt pairing matches its training captions, so the prompt-time penalty is
not a train/test mismatch artefact. Listening test still to run.

## 2026-10-18 — Tier 3 results (same 157 regions × 2 seeds, 103 tracks paired; 109 prompts). `eval/tier3/`

Conditions added: C6k (ID3 genre tokens + BPM), D6k (MiDashengLM auto-captions), and the 2k/4k
checkpoints of arm B; base and B6k re-used from tier 1 (same seeds, symlinked). Holm within each
(family, metric, level) over all condition pairs.

### Section regeneration, paired per track vs base (Wilcoxon, n = 103; Holm p in brackets)

| condition | CLAP region sim Δ | MERT region sim Δ | |seam| Δ | beat-grid Δ | key agree Δ (n=71) |
|---|---|---|---|---|---|
| C6k (ID3 tags + BPM) | **+0.022 (p 6e-5; Holm .0008)** | +0.005 (p .006; Holm .09) | −0.004 (ns) | −0.003 (ns) | 0 (ns) |
| B2k | **+0.014 (p .0006; Holm .008)** | **+0.004 (p .003; Holm .04)** | −0.003 (ns) | 0 (ns) | 0 (ns) |
| D6k (auto-captions) | **+0.013 (p .002; Holm .03)** | +0.002 (p .06) | −0.002 (ns) | −0.004 (ns) | 0 (ns) |
| B4k | +0.004 (p .10) | +0.001 (ns) | −0.002 (ns) | 0 (ns) | 0 (ns) |
| B6k | +0.002 (p .27) | +0.002 (p .04; Holm .5) | −0.003 (ns) | −0.002 (p .06) | 0 (ns) |
| B2k vs B6k | +0.008 (p .015; Holm .15) | +0.001 (ns) | +0.002 (ns) | 0 (ns) | 0 (ns) |

Reading: (a) **Arm C is the best adapter for filling a section** — the only one whose CLAP gain
survives correction with room to spare, and the largest MERT gain. Its captions are the terse
"Pop, Fun, Light, 120 bpm" ID3 tags, i.e. the least *descriptive* text; the model keeps leaning on
the two-sided audio context rather than on the words. (b) **More training on human captions
reduces region fidelity**: B2k > B4k > B6k monotonically on CLAP sim, and B2k is the only B
checkpoint that is significantly above base. (c) Seams, beat grid and key are flat for every
adapter, as in tier 1: coherence at the edges is a property of the base inpainting path.
Effect sizes remain small next to the anchor spread (CLAP ±0.24).

### Text-to-music (FAD∞/KAD to the training corpus, n = 109 clips; floor = held-out originals)

| condition | FAD∞ (CLAP) | KAD (CLAP) | FAD∞ (MERT) | KAD (MERT) | CLAP text | gen loudness |
|---|---|---|---|---|---|---|
| test originals (floor) | 0.055 | 0.80 | 0.61 | 0.58 | 0.283 | — |
| base | 0.165 | 4.45 | 1.69 | 2.27 | 0.346 | −17.2 LUFS |
| B2k | 0.151 | 4.25 | **1.00** | **1.20** | 0.309 | −14.8 |
| B4k | 0.149 | **3.74** | 1.44 | 2.04 | 0.313 | −14.4 |
| B6k | **0.140** | 3.96 | 1.26 | 1.85 | 0.312 | −14.3 |
| C6k | 0.157 | 4.12 | 1.28 | 1.73 | 0.326 | −15.5 |
| D6k | 0.166 | 4.21 | 1.17 | 1.52 | **0.376** | −15.4 |

CLAP-text vs base, paired over prompts: B2k/B4k/B6k −0.040 to −0.042 (all Holm < .001); C6k −0.022
(p .09); D6k **+0.014 (p .07)**.

Reading: the three caption sources trade off differently. **Human captions (B)** move the output
closest to the corpus on CLAP distance but cost text adherence. **ID3 tags (C)** are in between on
both. **Auto-captions (D)** are the only adapter that does *not* lose prompt adherence (if anything
it gains, as the model was trained on captions written in the same register the test prompts use)
while still improving MERT distance to the corpus (1.69 → 1.17); on CLAP distance it is no better
than base. The MERT and CLAP distance rankings disagree (B2k best on MERT, B6k best on CLAP), so no
single checkpoint "wins" text-to-music; the B2k MERT numbers are the lowest of any adapter and
match its region-fidelity result, which makes **B2k the best all-round single adapter** and 6k
steps over-trained for this corpus size (580 tracks × 47 s). Memorisation: one C6k generation
exceeds the fixed 0.95 CLAP cosine threshold (0.959) but all are below the held-out null p99
(0.982); no memorisation evidence. Adapters again learn corpus loudness (−14…−15.5 vs base −17.2).

### Answers to the research questions so far

* **RQ1 (structure).** Section regeneration works with every adapter and is driven by the base
  model's two-sided context; structure *text* in the prompt hurts (tier 1); structure *analysis*
  (SongFormer) is what finds the section to regenerate.
* **RQ2 (metadata).** Trigger-only collapses (A). Human captions pull generation toward the corpus
  most (B) but over-train by 6k steps. ID3 tags preserve section edits best (C). Auto-captions keep
  prompt adherence best (D). Structure lists add nothing (E).
* **RQ3 (models).** ACE-Step 1.5 LoRA (tier 1 ACE): raises MERT region sim (+0.019) but lowers
  CLAP region sim (−0.03) and collapses text-to-music (FAD∞ 0.37–0.39 vs base 0.21); SA3 is the
  model to serve. YuE dropped (whole-song only, no section edits).

Decision for serving/listening: ship **C, D, B (6k), B3k, A, E** on the endpoint (v1.1.0 artifact)
so the composer can A/B them; the B2k checkpoint is only on the stopped GPU box's disk
(`runs/B_tier1/checkpoints/`), to be added to a v1.2.0 artifact when the box is next started.
