# Structure analysis vs human labels: songformer

- Tracks evaluated: 673 (human: 673, songformer: 725; missing in songformer: 0, songformer-only: 52)
- Single-segment reference tracks (excluded from trimmed HR): 167
- Skipped (all metrics NaN): empty reference 0, empty songformer estimate 0
- Label map fingerprint: `c1a4952a10ab`; mir_eval 0.8.2; frames 10 Hz; bootstrap 10000 resamples, seed 0; edge snap 1 s

| Metric | P | R | F / Acc | n tracks | median | 95% CI (mean) |
|---|---|---|---|---|---|---|
| Boundary HR@0.5 s (incl. 0 and end) | 0.749 | 0.788 | 0.744 | 673 | 0.750 | [0.732, 0.755] |
| Boundary HR@3 s (incl. 0 and end) | 0.793 | 0.831 | 0.786 | 673 | 0.800 | [0.776, 0.797] |
| Boundary HR@0.5 s (trivial boundaries removed) | 0.456 | 0.417 | 0.403 | 506 | 0.422 | [0.373, 0.432] |
| Boundary HR@3 s (trivial boundaries removed) | 0.587 | 0.546 | 0.524 | 506 | 0.633 | [0.495, 0.553] |
| Pairwise frame clustering | 0.821 | 0.772 | 0.767 | 673 | 0.748 | [0.755, 0.778] |
| Frame label accuracy, coarse 4-class | | | 0.635 | 672 | 0.679 | [0.612, 0.658] |
| Frame label accuracy, fine | | | 0.400 | 672 | 0.393 | [0.375, 0.424] |

P/R/F and accuracy are means over tracks; median and CI refer to F (or accuracy).

## Per-class (coarse, pooled frames)

| Class | Precision | Recall | F | ref frames | est frames |
|---|---|---|---|---|---|
| intro | 0.705 | 0.433 | 0.537 | 41084 | 25254 |
| body | 0.796 | 0.790 | 0.793 | 158080 | 156891 |
| transition | 0.045 | 0.179 | 0.072 | 9044 | 35999 |
| outro | 0.987 | 0.008 | 0.015 | 10013 | 77 |

Confusion (coarse; rows = human, columns = songformer):

| human \ songformer | intro | body | transition | outro |
|---|---|---|---|---|
| intro | 17807 | 16844 | 6433 | 0 |
| body | 6816 | 124895 | 26368 | 1 |
| transition | 631 | 6798 | 1615 | 0 |
| outro | 0 | 8354 | 1583 | 76 |

## Per-class (fine, pooled frames)

| Class | Precision | Recall | F | ref frames | est frames |
|---|---|---|---|---|---|
| intro | 0.705 | 0.433 | 0.537 | 41084 | 25254 |
| verse | 0.728 | 0.476 | 0.576 | 142637 | 93302 |
| chorus | 0.035 | 0.483 | 0.065 | 4568 | 63442 |
| bridge | nan | 0.000 | nan | 10875 | 0 |
| outro | 0.987 | 0.008 | 0.015 | 10013 | 77 |
| other | 0.045 | 0.179 | 0.071 | 9044 | 36146 |

Confusion (fine; rows = human, columns = songformer):

| human \ songformer | intro | verse | chorus | bridge | outro | other |
|---|---|---|---|---|---|---|
| intro | 17807 | 13693 | 3151 | 0 | 0 | 6433 |
| verse | 6709 | 67938 | 43174 | 0 | 1 | 24815 |
| chorus | 89 | 1987 | 2206 | 0 | 0 | 286 |
| bridge | 18 | 3586 | 5857 | 0 | 0 | 1414 |
| outro | 0 | 1825 | 6529 | 0 | 76 | 1583 |
| other | 631 | 4273 | 2525 | 0 | 0 | 1615 |
