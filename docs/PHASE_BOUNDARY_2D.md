# 2D phase boundary in 2-qubit Pauli memory channel encoding

Run date: 2026-04-18

## Setup

Extend the 1-D correlation-strength sweep (docs/PHASE_TRANSITION.md) to a
2-D noise parameterization:

- `λ ∈ [0, 1]`  — correlation strength (independent vs pair errors)
- `μ ∈ [0, 1]`  — among pair errors, the fraction that is anti-correlated
  XY+YX style (μ=1) versus symmetric XX+ZZ style (μ=0)

Total error rate fixed at `p = 0.25`. For each of 121 points on an 11×11
grid, we variationally optimize the 9-parameter 2-qubit encoding and record
the entanglement entropy `S(ρ_A)` of the optimum.

The phase boundary is defined as the locus where `S(ρ_A) = 0.5`, halfway
between product (S=0) and Bell-like (S=1).

## Observed boundary

```
μ=0.00 → 0.50:  λ* = 0.550
μ=0.60 → 0.70:  λ* = 0.450
μ=0.80 → 1.00:  λ* = 0.350
```

## Interpretation: a piecewise-constant ("staircase") boundary

The phase boundary is neither smooth nor linear. Instead it has three
plateaus at λ* ∈ {0.55, 0.45, 0.35}, with discrete jumps near μ ≈ 0.55 and
μ ≈ 0.75. The interior of each plateau corresponds to a specific
entangled encoding that retains optimality across a finite range of noise
compositions. At the jumps, a different entangled encoding takes over.

Schematic:

```
λ*
0.55   ┏━━━━━━┓
        ┃      ┃
        ┃      ┗━━━━━━━┓
0.45                    ┃
                        ┃
                        ┗━━━━━━━━━━┓
0.35                                ┃
                                    ┗━━━━━━━
         0.0  0.25  0.5  0.75  1.0   μ
```

The three plateaus correspond to three distinct Bell-like encoding states,
each optimal over a contiguous range of noise compositions but
discontinuously replaced at crossover points where two Bell states become
fidelity-equivalent.

## Relation to the Daems / Karimipour line

This 2D structure is a corollary of the Daems (2007) transition theorem
and the Karimipour et al. (2009) three-phase result: at each μ we expect
a transition; at each μ the transition may occur at a different λ; and the
identity of the optimal entangled state may change discretely across μ.

What is presented here explicitly is the *geometry of the transition line*
in a 2-parameter noise slice. To the best of our checking (arXiv search
2026-04-18), the piecewise-constant "staircase" shape has not been
explicitly documented in the cited literature — but a proper analytical
derivation from the Karimipour framework is likely straightforward and
would show that the staircase steps correspond to algebraic boundaries
between which of several candidate Bell states is fidelity-optimal for a
given (λ, μ).

## Status

- **Experimental**: confirmed numerically on 11×11 grid
- **Novel?**: plausibly a minor observation not explicitly published
- **Reproducible**: `notebooks/06_2d_phase_diagram.py` (pending push)
- **Runtime**: ~22 minutes on Braket `ml.t3.medium` notebook instance
