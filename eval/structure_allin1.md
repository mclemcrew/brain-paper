# Structure analysis vs human labels: allin1

- Tracks evaluated: 673 (human: 673, allin1: 725; missing in allin1: 0, allin1-only: 52)
- Single-segment reference tracks (excluded from trimmed HR): 167
- Skipped (all metrics NaN): empty reference 0, empty allin1 estimate 0
- Label map fingerprint: `c1a4952a10ab`; mir_eval 0.8.2; frames 10 Hz; bootstrap 10000 resamples, seed 0; edge snap 1 s

| Metric | P | R | F / Acc | n tracks | median | 95% CI (mean) |
|---|---|---|---|---|---|---|
| Boundary HR@0.5 s (incl. 0 and end) | 0.713 | 0.738 | 0.703 | 673 | 0.667 | [0.692, 0.715] |
| Boundary HR@3 s (incl. 0 and end) | 0.747 | 0.768 | 0.734 | 673 | 0.750 | [0.723, 0.745] |
| Boundary HR@0.5 s (trivial boundaries removed) | 0.305 | 0.255 | 0.255 | 506 | 0.000 | [0.228, 0.283] |
| Boundary HR@3 s (trivial boundaries removed) | 0.410 | 0.342 | 0.341 | 506 | 0.333 | [0.311, 0.372] |
| Pairwise frame clustering | 0.689 | 0.917 | 0.757 | 673 | 0.730 | [0.745, 0.769] |
| Frame label accuracy, coarse 4-class | | | 0.302 | 665 | 0.224 | [0.279, 0.327] |
| Frame label accuracy, fine | | | 0.202 | 665 | 0.100 | [0.183, 0.222] |

P/R/F and accuracy are means over tracks; median and CI refer to F (or accuracy).

## Per-class (coarse, pooled frames)

| Class | Precision | Recall | F | ref frames | est frames |
|---|---|---|---|---|---|
| intro | 0.239 | 0.857 | 0.374 | 40974 | 146938 |
| body | 0.816 | 0.268 | 0.404 | 156567 | 51518 |
| transition | 0.064 | 0.052 | 0.057 | 9335 | 7618 |
| outro | 0.240 | 0.259 | 0.249 | 9895 | 10697 |

Confusion (coarse; rows = human, columns = allin1):

| human \ allin1 | intro | body | transition | outro |
|---|---|---|---|---|
| intro | 35110 | 4736 | 1125 | 3 |
| body | 101640 | 42015 | 5810 | 7102 |
| transition | 5553 | 2271 | 486 | 1025 |
| outro | 4635 | 2496 | 197 | 2567 |

## Per-class (fine, pooled frames)

| Class | Precision | Recall | F | ref frames | est frames |
|---|---|---|---|---|---|
| intro | 0.239 | 0.857 | 0.374 | 40974 | 146938 |
| verse | 0.789 | 0.011 | 0.023 | 141197 | 2054 |
| chorus | 0.185 | 0.349 | 0.241 | 4538 | 8572 |
| bridge | nan | 0.000 | nan | 10832 | 0 |
| outro | 0.240 | 0.259 | 0.249 | 9895 | 10697 |
| other | 0.048 | 0.252 | 0.081 | 9335 | 48510 |

Confusion (fine; rows = human, columns = allin1):

| human \ allin1 | intro | verse | chorus | bridge | outro | other |
|---|---|---|---|---|---|---|
| intro | 35110 | 389 | 842 | 0 | 3 | 4630 |
| verse | 97812 | 1621 | 5071 | 0 | 6204 | 30489 |
| chorus | 1182 | 0 | 1582 | 0 | 36 | 1738 |
| bridge | 2646 | 0 | 198 | 0 | 862 | 7126 |
| outro | 4635 | 9 | 507 | 0 | 2567 | 2177 |
| other | 5553 | 35 | 372 | 0 | 1025 | 2350 |
