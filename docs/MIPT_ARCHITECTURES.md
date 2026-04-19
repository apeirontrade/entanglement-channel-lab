# MIPT architecture comparison: Haar vs Clifford vs Matchgate

Run date: 2026-04-18

## Setup

Measurement-induced phase transition (MIPT) in random monitored circuits
with three different 2-qubit gate ensembles, at L = 6 and L = 8, T = 2L,
with n_traj = 100, p in [0, 0.5] at 11 points. Half-chain entanglement
entropy S computed, normalised by L (bits).

Circuit ensembles:

- **Haar**: random 2-qubit SU(4) via QR of complex Ginibre
- **Clifford**: random 2-qubit Clifford via product of 20 random generators
  (H on each qubit, S on each qubit, CNOT, CNOT-H-H-CNOT)
- **Matchgate**: 2-qubit matchgates (free-fermion preserving) with
  matched even/odd-sector determinants

Projective Z-measurements on each qubit with probability p after each
brick-wall layer.

## Results

| p    | Haar L=6 | Haar L=8 | Clif L=6 | Clif L=8 | Match L=6 | Match L=8 |
|------|---------:|---------:|---------:|---------:|----------:|----------:|
| 0.00 | 0.372    | 0.403    | 0.347    | 0.390    | 0.290     | 0.277     |
| 0.05 | 0.271    | 0.285    | 0.255    | 0.255    | 0.238     | 0.228     |
| 0.10 | 0.197    | 0.216    | 0.175    | 0.186    | 0.189     | 0.199     |
| 0.15 | 0.122    | 0.158    | 0.090    | 0.135    | 0.167     | 0.171     |
| 0.20 | 0.085    | 0.121    | 0.073    | 0.098    | 0.137     | 0.148     |
| 0.25 | 0.075    | 0.085    | 0.057    | 0.066    | 0.107     | 0.128     |
| 0.30 | 0.051    | 0.054    | 0.048    | 0.063    | 0.092     | 0.099     |
| 0.35 | 0.027    | 0.053    | 0.033    | 0.035    | 0.058     | 0.073     |
| 0.40 | 0.026    | 0.033    | 0.023    | 0.030    | 0.044     | 0.069     |
| 0.45 | 0.016    | 0.025    | 0.010    | 0.021    | 0.028     | 0.045     |
| 0.50 | 0.011    | 0.018    | 0.008    | 0.013    | 0.034     | 0.028     |

## Observations

1. **Haar and Clifford are essentially indistinguishable** in our data.
   Their S/L curves overlay within Monte Carlo noise. Consistent with
   published universality-class claims (Li-Chen-Fisher 2019,
   Skinner-Ruhman-Nahum 2019).

2. **Matchgate is visibly different**:
   - Lower S/L at p=0 (0.28 vs 0.37 for Haar/Clifford), consistent with
     the restricted entangling power of free-fermionic unitaries.
   - Slower decay of S/L with p. At p = 0.30, matchgate still has
     S/L ≈ 0.10 while Haar/Clifford are at ≈ 0.05.
   - Suggests the transition is shifted to larger p (or a qualitatively
     different scaling).

3. This is consistent with the Jian-You-Vasseur-Ludwig 2020 prediction
   that matchgate/free-fermion monitored circuits have a **different
   universality class** from generic (Haar/Clifford) circuits.

## Scope and novelty

- **Known qualitative result** confirmed in a unified small-L framework.
- Not new physics: the universality-class distinction is established.
- **Useful as a compact reproducible reference**: a single 15-second
  runtime script produces all three architectures' S/L curves in one
  plot, which is a benchmark not specifically produced at L = 6, 8 in
  published work (those usually run one architecture per paper, often
  at L ≥ 16 with Clifford circuits for computational efficiency).

## Reproducibility

```bash
python3 -u notebooks/15_mipt_architectures.py
```

Runs in ~15 seconds on CPU. Pure numpy, no Braket. Deterministic (seed 42).

Output: `results/mipt_architectures.png` — three-panel plot of S/L vs p
for each architecture at L = 6, 8.
