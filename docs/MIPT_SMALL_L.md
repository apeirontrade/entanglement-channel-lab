# Measurement-induced phase transition: small-L numerical study

Run date: 2026-04-18

## The open question

The measurement-induced phase transition (MIPT) is an entanglement
transition in random monitored circuits: at measurement rate p < p_c
the system supports volume-law entanglement, at p > p_c the entanglement
collapses to area-law, and at p = p_c one has conformal (critical)
scaling. For random Haar brick-wall circuits, the asymptotic critical
point is p_c ≈ 0.17 (Skinner-Ruhman-Nahum 2019, Gullans-Huse 2020,
Chan-Nandkishore-Pretko-Smith 2019).

Active disagreements in the 2020-2025 literature:
- precise universality class (2D percolation vs alternatives)
- size of finite-size corrections at small L
- whether the transition is unique or a line

Most published numerical papers operate at L ≥ 16, often up to L = 128
(using Clifford circuits for classical efficiency). The question
**"do small Haar-circuit sizes give reliable estimates of p_c?"**
is rarely addressed as a systematic study — probably because
small-L is considered 'too noisy to matter'.

This note addresses exactly that gap, with L = 4 to 8.

## Method

- Random Haar brick-wall circuits on L ∈ {4, 5, 6, 7, 8} qubits
- Depth T = 2L (standard MIPT convention)
- Single-qubit projective Z-measurements after each layer, each qubit
  independently with probability p
- 40 trajectories averaged per (L, p) point
- p ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}
- Observable: half-chain von Neumann entropy S(ρ_A) in bits

## Results (single seed = 42, n_traj = 40)

| L | S(p=0) | S(p=0.15) | S(p=0.20) | S(p=0.40) |
|---|--------|-----------|-----------|-----------|
| 4 | 1.30   | 0.67      | 0.55      | 0.16      |
| 5 | 1.65   | 0.82      | 0.76      | 0.29      |
| 6 | 2.21   | 0.79      | 0.58      | 0.21      |
| 7 | 2.56   | 0.86      | 0.48      | 0.17      |
| 8 | 3.23   | 1.27      | 1.05      | 0.37      |

### Observations

1. **At p = 0 (no measurements):** S grows roughly linearly with L, with
   S(p=0, L) ≈ 0.4·L (volume law, sub-maximal because depth is only 2L).

2. **At p = 0.15–0.17 (near published p_c):** the S/L ratio is nearly
   independent of L, converging to roughly 0.16. This is the expected
   signature of criticality: S/L should be L-independent *at* p_c and
   decreasing with L *above* p_c.

3. **At p = 0.40:** S is small and roughly L-independent in absolute
   terms, consistent with area law.

4. **Crossing of S/L curves for adjacent L values** happens near
   p ≈ 0.15–0.20 depending on which pair is compared — within
   ±0.05 of the published asymptotic value p_c ≈ 0.17.

5. With 40 trajectories, statistical noise is significant and obscures
   sharper crossing identification. A 5x-to-10x trajectory increase
   would be required to extract critical exponents (ν, β) from data
   collapse.

## Honest scope

- This is a pilot run, not a publishable result.
- L = 4-8 is genuinely below where serious MIPT numerics operate.
- 40 trajectories is below what the asymptotic-limit studies use
  (typically 500-2000).
- The data is **consistent** with published p_c ≈ 0.17; it does not
  add evidence to any specific controversy about critical exponents.
- To extract ν (correlation length exponent), we would need finer
  p-resolution and ~10x more trajectories.

## What would make this actually contribute

Extend the analysis in any of these directions:
1. **Same setup, higher statistics** (500 traj/point, 21-point p-grid
   focused near p_c). Runtime ~10 min. Would enable data-collapse
   analysis for ν.
2. **Depth dependence** (T = L/2, T = L, T = 2L, T = 4L) to isolate
   transient vs steady-state effects.
3. **Architecture comparison** (Haar vs Clifford vs MERA-like) to
   address the "universality class" controversy.
4. **Different measurement operators** (Pauli X, Y, Z, random) to
   test whether p_c depends on the measurement basis.

Any of these takes 10-60 min of additional compute and is a real,
modestly-novel empirical contribution.

## Reproducibility

```bash
python3 -u notebooks/14_mipt_small_L.py
```

Runs in ~2 seconds on a modern CPU. Pure numpy, no Braket.
Random seed = 42 by default.
