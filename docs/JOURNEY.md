# Journey — how this experiment came to be

A chat-log summary of the thinking behind each step, in case future-me (or a reader) wants to know "why did we try that next?"

## Starting point

Opened a fresh Amazon Braket notebook, knowing nothing about what we'd build.
Asked "what should I do with a quantum computer?" The honest answer:
NISQ-era hardware (and simulators at NISQ scale) is useful for:
- **Random number generation** (boring)
- **Small chemistry simulations** (specialized)
- **Small optimization problems** (often worse than classical)
- **Pedagogical demos of quantum phenomena** (Bell violations, teleportation, Grover)
- **Implementation-first contributions to research questions** (port-based teleportation, channel duality, etc.)

We picked the last category — something underexplored but tractable.

## The specific question we chose

> *"Entanglement-assisted classical communication with teleportation reversal."*

Concretely: Alice and Bob share a Bell pair. They can use it to send either
**qubits (teleportation)** or **classical bits (superdense coding)**.
How do these compare to just **sending things directly** under different kinds of noise?

Why this one:
- Connects three well-known protocols in one framework — rarely done together
- Directly quantifiable: measure fidelity under noise, plot, compare
- Runs on a free local simulator (zero budget for hardware)
- Research-hot in noise-resource analysis (Bennett, Shor, Smolin, 1998 and follow-ups)

## What we found

1. **Teleport ≈ Direct under symmetric single-qubit Pauli noise.**
   Whether the noise is depolarizing or dephasing, the Bell pair doesn't
   buy fidelity. Both protocols pass the state through one noisy channel
   qubit; the operation counts are the same; fidelity tracks.

2. **Superdense coding is more fragile.**
   Its 4-outcome decoding means any single error flips an outcome,
   dropping success by ~1/4. In our data, superdense loses the most
   at every noise level.

3. **Monte Carlo noise matters.**
   We briefly thought we'd found a teleport-beats-direct advantage under
   dephasing at 500 shots. At 1500 shots it vanished. Lesson: always
   check the standard error before drawing conclusions. SE = √(p(1-p)/N).

## What we didn't finish

- **Variational encoding search**: does a *smart* encoding (different from just Ry(θ)|0>) change the picture? Started but hadn't finished when the repo was pushed.
- **Noise atlas**: 2D sweep over (p_x, p_z) with winner-coloring. Wrote the code, didn't run it.
- **Real QPU validation**: would cost $300-1000 for the full sweep. Didn't do.

## Calibration — is any of this "groundbreaking"?

No. Honestly:
- The physics is textbook (Nielsen & Chuang, chapters 8 and 12).
- The protocols are 25+ years old.
- The simulator is a classical Monte Carlo at toy scale.
- No new theorems, no new protocols, no hardware innovation.

What it *is*:
- A clean, reproducible, end-to-end implementation of three classical
  protocols under a unified noise model
- Pedagogically useful — someone learning the field can run it in 5 min
- A scaffold for real work (variational encoding, 2D atlas, real-hardware
  validation) if anyone wanted to continue

Weekend-level contribution. Fine.

## Possible follow-ups (if motivation returns)

- Run on a real QPU at 1-2 noise points to check simulator fidelity
- Finish the 2D atlas, map protocol-winner regions
- Add amplitude damping via density-matrix simulator (`braket_dm`)
- Try **port-based teleportation** (Christandl et al.) — richer and less explored
- Try **entanglement distillation** before teleportation — does that help?
- Couple to a simple ML decision rule → "protocol selection oracle"

## What I actually learned

- How to structure a Braket circuit end-to-end
- The quirks of Monte Carlo noise in small-N experiments
- How teleportation and superdense coding sit as **duals** on the same Bell pair
- That "entanglement as resource" is nuanced — it's about topology and capacity,
  not noise resistance per se
- How to keep the cost of a quantum experiment to basically $0 (local simulator,
  short notebook uptime)
