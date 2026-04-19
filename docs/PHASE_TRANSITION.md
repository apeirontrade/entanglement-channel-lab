# Phase transition in 2-qubit optimal-encoding entanglement

Run date: 2026-04-18

## Experimental setup

For a family of 2-qubit Pauli channels at fixed total error rate `p = 0.25`,
parameterized by correlation strength `λ ∈ [0, 1]`:

```
probs(λ) = (1-λ) · (independent single-qubit errors)
         + λ · (correlated pair errors)
```

where the independent component distributes `p_solo = 0.25 (1-λ)` equally across
{XI, IX, ZI, IZ}, and the correlated component `p_pair = 0.25 λ` is one of:

- **Family A (XX + ZZ):** `{XX: p_pair/2, ZZ: p_pair/2}`
- **Family B (ZZ only):** `{ZZ: p_pair}`
- **Family C (XY + YX):** `{XY: p_pair/2, YX: p_pair/2}`

For each channel, we compute:

- `F_prod` = best fidelity under any **product-state** encoding (optimize a
  6-parameter local-only SU(2)⊗SU(2) subspace)
- `F_ent` = best fidelity under any 2-qubit encoding in our 9-parameter SU(4)
  parameterization (local + XX/YY/ZZ entanglers)
- `S(ρ_A)` = entanglement entropy of the encoding state at the optimum

## Results

### Family A (XX + ZZ)

| λ | F_prod | F_ent | gain | S_opt |
|---|-------|-------|------|-------|
| 0.00–0.50 | 0.875 | 0.875 | 0.000 | 0.000 |
| 0.60 | 0.875 | 0.900 | +0.025 | 1.000 |
| 0.70 | 0.875 | 0.925 | +0.050 | 1.000 |
| 0.80 | 0.875 | 0.950 | +0.075 | 1.000 |
| 0.90 | 0.875 | 0.975 | +0.100 | 1.000 |
| 1.00 | 0.875 | 1.000 | +0.125 | 1.000 |

**Sharp first-order transition at λ* ≈ 0.55.**
`F_prod` stays flat at 0.875 throughout — the best Z-eigenstate product
encoding shields against both independent ZI/IZ errors (by being an
eigenstate) and from correlated ZZ (likewise), with a fixed budget of
XI/IX damage. Above λ* the Bell branch `F_ent = 0.75 + 0.25 λ` overtakes
this plateau.

### Family B (ZZ only)

| λ | F_prod | F_ent | gain | S_opt |
|---|-------|-------|------|-------|
| 0.00 | 0.875 | 0.875 | 0 | 0.000 |
| 0.10 | 0.887 | 0.887 | 0 | 0.000 |
| …    | …     | …     | … | …     |
| 0.90 | 0.987 | 0.987 | 0 | 0.000 |
| 1.00 | 1.000 | 1.000 | 0 | 0.999 |

**No transition.** Product and entangled encodings are fidelity-degenerate
at every λ because Z-eigenstate products already stabilize ZZ-correlated
noise; adding ZZ noise *helps* (rather than hurts) the product state by
diluting independent XI/IX damage. At λ=1 the optimizer happens to land
on a Bell state (S=0.999) but it could equivalently have chosen a
product state — they share the same F=1.0.

### Family C (XY + YX)

| λ | F_prod | F_ent | gain | S_opt |
|---|-------|-------|------|-------|
| 0.00 | 0.875 | 0.875 | 0.000 | 0.000 |
| 0.10 | 0.862 | 0.862 | 0.000 | 0.000 |
| 0.20 | 0.850 | 0.850 | 0.000 | 0.000 |
| 0.30 | 0.837 | 0.837 | 0.000 | 0.000 |
| 0.40 | 0.837 | 0.850 | +0.013 | 1.000 |
| 0.50 | 0.844 | 0.875 | +0.031 | 1.000 |
| …    | …     | …     | …      | …     |
| 1.00 | 0.875 | 1.000 | +0.125 | 1.000 |

**Sharp transition at λ* ≈ 0.35 — earlier than Family A.**
Unlike Family A, here `F_prod` *decreases* with λ (from 0.875 down to
0.837 at λ=0.3), because no product state can diagonalize XY or YX
individually. Once the rising Bell branch reaches the descending product
branch, the optimizer switches, and `F_prod` then begins climbing again
along a different product-state branch (one that's losing XY+YX damage).

## Observation — a three-type taxonomy of correlated-noise-induced transitions

The three families exhibit qualitatively distinct behaviors, which we
conjecture are controlled by the product-state eigenstructure of the
correlated-noise operators:

- **Type 0 (degenerate, Family B):** the correlated-noise operator stabilizes
  some product state → product and Bell encodings are fidelity-degenerate
  at every λ → no transition.
- **Type I (late transition, Family A):** the correlated-noise operators
  *collectively* have no product-state eigenbasis, but there exists a
  product state with *fixed damage budget* against the combined independent
  + correlated noise → `F_prod` is flat, so transition occurs high at λ*
  where the Bell branch `F_ent = 1 - p_solo(λ)` catches up.
- **Type II (early transition, Family C):** no product state avoids any of
  the correlated-noise operators, so `F_prod` actively decreases with λ
  while `F_ent` climbs → the curves meet earlier, transition λ* is smaller.

This taxonomy is empirical and not, to our knowledge, stated in this exact
form in the literature. It invites a clean analytic proof via stabilizer-
theoretic analysis of commutator structure, which we have not carried
out here.

## Files

- `notebooks/07_phase_transition.py` — reproducible script for the
  1-D sweep across three families.
- `results/correlation_entanglement_scaling.png` — side-by-side plot of
  S(ρ_A) vs λ and gain vs λ for all three families.

## Followup (planned)

- 2D phase diagram: sweep (λ, μ) where μ interpolates between
  Family A-type and Family C-type correlated components. Objective:
  is the transition line straight, curved, discontinuous?
- Analytic derivation of λ*(family) from stabilizer-commutator structure.
- Extension to N ≥ 3 qubits: does the classification extend or break down?
