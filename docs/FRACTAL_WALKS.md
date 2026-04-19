# Fractal quantum walks: a second invariant beyond spectral dimension

Run date: 2026-04-18

## Motivation

Darázs et al. (*Phys. Rev. E* 90, 032113, 2014) showed that continuous-time
quantum walks (CTQW) on Sierpinski gaskets and carpets *do not* obey a
universal scaling in the spectral dimension `d_s` — in contrast to classical
random walks, where `d_s` alone determines the Pólya-type return-probability
exponent.

The paper explicitly states that spectral dimension is insufficient but does
**not** identify a second predictor. This is the gap we investigate.

## Method

Generated 8 small graphs (ring, Sierpinski gasket gen2/3, T-fractal gen2/3,
Vicsek fractal, Cayley trees b=3, b=4). For each:

1. Computed graph Laplacian and CTQW propagator U(t) = exp(-iL t).
2. Started the walker at one node, evolved, measured P_return(t).
3. Fitted the log-log slope of the smoothed long-time tail to get the decay
   exponent α.
4. Computed candidate topological invariants:
   - `d_s` (spectral dim from cumulative eigenvalue distribution)
   - `branching_ratio` (max deg / min deg)
   - `mean_chem_dist` (mean shortest-path length)
   - `spectral_gap` (smallest nonzero Laplacian eigenvalue)
   - `mean_degree`
   - `N` (node count)

## Results

Raw correlations of α with each feature, across 8 graphs:

| feature          | r(α, feature) |
|------------------|--------------:|
| d_s              |        +0.07 |
| branching_ratio  |        +0.32 |
| mean_chem_dist   |        −0.44 |
| spectral_gap     |        +0.39 |
| **mean_degree**  |        **−0.58** |
| N                |        −0.20 |

Residual correlations after regressing out d_s were nearly identical
(r ≈ −0.60 for mean_degree), confirming that `d_s` explains essentially none
of the variance in α for this dataset.

### Observations

- **`d_s` has essentially zero predictive power for α** (r = +0.07).
  This directly confirms the qualitative finding of Darázs et al. 2014.

- **`mean_degree` is the strongest single predictor of α** (r = −0.58).
  Graphs with higher average connectivity exhibit smaller (sometimes
  negative) decay exponents, corresponding to stronger localization of the
  walker near its starting node.

- **`mean_chem_dist` and `spectral_gap` follow as secondary predictors**
  (|r| ≈ 0.4), in opposite directions as one would expect from linear-walk
  intuition.

## Interpretation (speculative)

In classical walks on fractals, `d_s` captures both ramification (chemical
distance scaling) and connectivity scaling in a single number, so it is
sufficient. In quantum walks, interference effects are sensitive to the
local node degree — which controls how many interfering paths meet at
each vertex — in a way that `d_s` does not capture. This is consistent
with the known fact that quantum walks localize on vertices of high
effective degree (graph hub effect).

## Scope caveats

1. **Only 8 graphs.** Strong claims about "mean degree predicts
   quantum Pólya exponent" need 20–50 graphs with careful stratification.
2. **Small graphs (10-85 nodes).** Decay-exponent fits are noisy; polyfit
   warnings were issued for a few cases.
3. **Not yet cross-validated.** The observed correlation could be
   driven by one or two outliers (ring, Vicsek look unusual).
4. **No theoretical derivation.** Why mean-degree specifically? An
   analytical argument would strengthen this.

## Novelty and literature context

- Darázs et al. 2014 — stated the gap this paper addresses, did NOT propose
  mean degree or any alternative invariant.
- Stefanak, Jex, Kiss 2008 — recurrence of d-dimensional coined walks
  (regular lattices only, not fractals).
- Farhi & Gutmann 1998 — CTQW original.
- No published work, to the extent of 6 targeted arXiv searches on
  2026-04-18, explicitly proposes mean degree as a second invariant for
  quantum walks on fractals.

This is a plausible but not verified novel observation. A thorough
literature review (30+ papers in the CTQW-on-fractals / quantum-walk-
localization literature) would be required before publication.

## Reproducibility

```bash
python3 notebooks/10_fractal_quantum_walks.py
```

~3 minutes on a modern CPU. Deterministic — no stochastic elements.
