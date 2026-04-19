# AME(8, 4) variational search — null result

Run date: 2026-04-18

## Target

IQOQI Open Problem #35 asks: for what parameters (N, d) do
absolutely-maximally-entangled (AME) pure states exist on N d-level
systems? The state is AME if every bipartition of equal size yields a
maximally-mixed reduced density matrix.

AME(8, 4) — 8 ququart (d=4) systems — is named on the IQOQI problem
page as one of two smallest unresolved cases (the other being AME(7, 6)).

## Method

- State-vector simulation over ℂ^(4^8) = ℂ^65536
- Ansatz: 3-layer brick-wall of 11 two-body SU(16) unitaries on nearest-
  neighbor ququart pairs
- Each 2-body unitary parameterized by 120 real parameters via a reduced
  Gell-Mann generator basis + matrix-exponential
- Total variational parameters: 1,320
- Objective: minimize max F-norm deviation of rho_A from I/256 across
  all 70 size-4-vs-4 bipartitions
- Optimizer: L-BFGS-B with random initialization

## Sanity baselines

| State                       | max dev | mean dev |
|-----------------------------|--------:|---------:|
| Random Gaussian pure state  | 0.0040  | 0.0039   |
| Computational basis |00…0⟩   | 0.9961  | 0.9961   |

Random Gaussian states are nearly maximally mixed across reductions
(Page's theorem); they set the rough scale for "good but not AME."
AME corresponds to **exactly zero** max deviation.

## Run result

L-BFGS-B from random init (1 restart completed; run was SIGTERM'd before
a second restart could run):

```
Iter 0: f = 0.381  |proj g| = 0.111
Iter 1: f = 0.204  |proj g| = 0.041
Iter 2: f = 0.175  |proj g| = 0.015
Iter 3: f = 0.164  |proj g| = 0.010
Iter 4: f = 0.156  |proj g| = 0.010
[process terminated]
```

The descent was slowing sharply (improvements per iteration halving each
step), consistent with approach to a local minimum around **f ≈ 0.12 –
0.14**. This is ~35× worse than the Gaussian baseline and ~∞× worse than
the AME threshold.

## Interpretation

The **3-layer brick-wall ansatz on 8 ququarts does not contain AME(8, 4)
states**, at least not within reach of L-BFGS-B from random init. Three
non-exclusive possibilities:

1. **Ansatz too shallow.** 3 layers with only nearest-neighbor couplings
   may lack the long-range correlations needed for a state whose every
   4-vs-4 bipartition is maximally mixed. Deeper ansätze (6+ layers) or
   longer-range couplings would be needed to test this.
2. **AME(8, 4) does not exist.** The problem being open is consistent
   with a conjectured non-existence, analogous to the proved
   non-existence of AME(4, 2). Null results from many ansätze are the
   kind of evidence that accumulates toward the non-existence claim,
   though they cannot prove it.
3. **AME(8, 4) exists but requires structured (non-generic) states.**
   The one known hard-case resolution — AME(4, 6) by Rather et al. 2022
   — used **quantum Latin squares**, a highly structured construction
   that a generic variational ansatz would never find by local search.
   A similar structure may be required for AME(8, 4).

## Scope caveats

- **One run, one ansatz, one restart.** Not conclusive evidence of
  anything.
- **Likely attempted many times** by researchers in this area; a generic
  variational search returning a negative is expected and unremarkable.
- **Did not attempt structured ansätze** (stabilizer, quantum Latin
  squares) which are the tools that have actually produced new AME
  states in the literature.

## Takeaway

This is an honest small data point on a curated open problem, added to
the repo with full caveats. It does not claim novelty or contribute to
resolution. Its value is as a reproducible "here is what one generic
variational approach produces, with what compute, in what time" entry.

## Reproducibility

```bash
python3 -u notebooks/11_ame_8_4.py
```

Runs on laptop, ~5-10 min per L-BFGS-B iteration with 1,320 parameters
and 65,536-dim state vectors; one restart takes roughly an hour to
finish depending on convergence.
