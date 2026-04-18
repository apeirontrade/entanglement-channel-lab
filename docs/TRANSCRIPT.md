# Run transcript — 2026-04-18

Raw output from the Braket notebook, in the order cells were run.
Kept verbatim so future-me can see exactly what happened.

---

## Cell 1 — depolarizing sweep only (500 shots)

_Initial sanity check that the machinery works. Output:_

```
    p  F_teleport     F_direct
 0.00       1.000        1.000
 0.10       0.930        0.922
 0.20       0.866        0.868
 0.30       0.780        0.796
 0.40       0.732        0.726
 0.50       0.658        0.678
```

Teleport and direct track each other. As expected.

## Cell 2 — dephasing-only sweep (500 shots)

_Same thing, but with only Z errors._

```
    p  F_teleport     F_direct    diff
 0.00       1.000        1.000  +0.000
 0.10       0.924        0.920  +0.004
 0.20       0.852        0.826  +0.026
 0.30       0.764        0.740  +0.024
 0.40       0.732        0.706  +0.026
 0.50       0.628        0.582  +0.046
```

_Phantom advantage — teleport looks like it's beating direct by up to 4.6%.
Exciting for ~10 minutes. See Cell 3._

## Cell 3 — combined sweep, both noise types, 1500 shots

_Bumped statistics from 500 → 1500 shots per point to see if the dephasing
"advantage" was real. Added superdense coding as a third protocol._

### Depolarizing
```
[depol] p=0.00  F_T=1.000  F_D=1.000  F_S=1.000
[depol] p=0.05  F_T=0.967  F_D=0.975  F_S=0.957
[depol] p=0.10  F_T=0.937  F_D=0.936  F_S=0.913
[depol] p=0.15  F_T=0.897  F_D=0.894  F_S=0.835
[depol] p=0.20  F_T=0.881  F_D=0.872  F_S=0.804
[depol] p=0.25  F_T=0.854  F_D=0.833  F_S=0.780
[depol] p=0.30  F_T=0.808  F_D=0.798  F_S=0.687
[depol] p=0.35  F_T=0.778  F_D=0.779  F_S=0.653
[depol] p=0.40  F_T=0.723  F_D=0.733  F_S=0.585
[depol] p=0.45  F_T=0.711  F_D=0.697  F_S=0.525
[depol] p=0.50  F_T=0.652  F_D=0.665  F_S=0.500
```

### Dephasing
```
[dephase] p=0.00  F_T=1.000  F_D=1.000  F_S=1.000
[dephase] p=0.05  F_T=0.971  F_D=0.959  F_S=0.956
[dephase] p=0.10  F_T=0.931  F_D=0.926  F_S=0.902
[dephase] p=0.15  F_T=0.887  F_D=0.890  F_S=0.849
[dephase] p=0.20  F_T=0.846  F_D=0.857  F_S=0.809
[dephase] p=0.25  F_T=0.815  F_D=0.810  F_S=0.759
[dephase] p=0.30  F_T=0.781  F_D=0.767  F_S=0.696
[dephase] p=0.35  F_T=0.745  F_D=0.733  F_S=0.663
[dephase] p=0.40  F_T=0.685  F_D=0.686  F_S=0.618
[dephase] p=0.45  F_T=0.675  F_D=0.671  F_S=0.549
[dephase] p=0.50  F_T=0.625  F_D=0.633  F_S=0.503
```

_At higher shots the phantom dephasing advantage disappears. Teleport and
Direct now within ±1-2% everywhere — indistinguishable. This is the
honest textbook result._

## Cell 4+ — variational encoding (in progress at time of write)

_Running while writing this. Results will land in `docs/RESULTS.md` once
the sweep finishes._

---

## Meta-lesson for next time

- Never get excited about < 3σ effects. Always compute SE = sqrt(p(1-p)/N).
- At N=500, p=0.5 → SE ≈ 2.2% per point. A 4.6% effect is only ~2σ.
- At N=1500, p=0.5 → SE ≈ 1.3%. Anything < 2.6% is noise.
- Bump shots by 3-5× before trusting any "surprising" result.
