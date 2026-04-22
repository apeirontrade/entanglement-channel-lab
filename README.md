# Entanglement-Assisted Channel Lab

A weekend experiment comparing three ways to send information between Alice and Bob under noise, plus a set of follow-on quantum-info explorations.

- **Direct** – just send the thing (baseline, no entanglement)
- **Teleportation** – consume a Bell pair + 2 classical bits to send 1 qubit
- **Superdense coding** – consume a Bell pair + 1 qubit to send 2 classical bits

Runs on **Amazon Braket** (local simulator — zero compute cost).

## TL;DR

Under symmetric single-qubit Pauli noise (depolarizing or dephasing), **entanglement-assisted protocols provide no measurable fidelity advantage over direct transmission**.

That's the boring-but-correct answer: the "magic" of entanglement buys you topology (send across classical-only channels) and protocol flexibility (qubit↔bit conversion), but not free error resistance for a generic encoding.

**Superdense coding consistently degrades faster** than the other two — its 4-outcome decoding is more fragile to per-qubit noise.

## What the core code does

1. Prepare a test state `|ψ⟩ = Ry(π/3) Rz(π/4) |0⟩` on Alice's side
2. Run each protocol through a noisy channel (parameterised Pauli noise)
3. Measure Bob's reconstruction fidelity over many shots
4. Sweep noise strength from `p=0` to `p=0.5`
5. Plot fidelity curves and write them to `results/`

## Files

```
notebooks/
  # Core teleport / superdense / direct comparison
  01_teleport_basic.py             # smallest working teleportation circuit
  02_channel_duality.py            # T vs D vs S under depolarizing + dephasing
  02_full_channel_duality.py       # longer/commented original version
  03_variational_encoding.py       # Braket-based variational (hit a pydantic perf bug)
  04_analytic_variational.py       # closed-form fidelity; optimizer + MLP
  05_two_qubit_encoding.py         # 2-qubit encoding under correlated Pauli noise

  # Follow-on experiments
  06_2d_phase_diagram.py           # 2D phase diagram sweep
  06_mipt.py                       # measurement-induced phase transition
  07_phase_transition.py           # phase transition in 2-qubit encoding
  08_n3_memory_phases.py           # 3-qubit memory-channel phases
  09_singular_locus.py             # singular locus of the optimization landscape
  10_fractal_quantum_walks.py      # fractal-spectrum quantum walks
  11_ame_8_4.py                    # AME(8,4) variational search (null result)
  12_mermin_real_qpu.py            # Mermin inequality on real QPU
  13_chsh_real_qpu.py              # CHSH on real QPU
  14_mipt_small_L.py               # MIPT small-L numerical study
  15_mipt_architectures.py         # MIPT: Haar vs Clifford vs Matchgate
  16_ghz_fidelity_benchmark.py     # GHZ fidelity benchmark
  17_garnet_characterization.py    # IQM Garnet device characterization
  18_cudaq_h2_dissociation.py      # H2 dissociation (CUDA-Q)
  19_qcnn_bp_advantage.py          # QCNN barren-plateau advantage

results/
  fractal_walks.png                # fractal-walks plot
  mipt_small_L.png                 # MIPT small-L plot
  mipt_architectures.png           # MIPT architectures plot

docs/
  RESULTS.md                       # core fidelity numbers + interpretation
  RUNBOOK.md                       # step-by-step reproduction guide
  NOTES.md                         # surprises, gotchas, follow-ups
  AME_NULL_RESULT.md               # AME(8,4) search writeup
  FRACTAL_WALKS.md                 # fractal quantum walks writeup
  MIPT_SMALL_L.md                  # MIPT small-L writeup
  MIPT_ARCHITECTURES.md            # MIPT architectures comparison writeup
  PHASE_TRANSITION.md              # 2-qubit phase transition writeup
  PHASE_BOUNDARY_2D.md             # 2D phase-boundary writeup

paper/
  paper.md                         # Markdown draft (v0.1, not peer-reviewed)
```

## How to run

Open any Braket notebook (`conda_braket` kernel), paste the contents of any `notebooks/NN_*.py` into a cell, `Shift+Enter`.

## Cost

- Local simulator: **$0**
- Notebook instance: ~$0.05/hr while running (remember to **Stop** in AWS Braket console when done)
- Real-QPU notebooks (12, 13, 16, 17): incur per-task/per-shot charges — read the cost notes in each file before running

## Status

Weekend exploration plus follow-on experiments. Not a publication. Validated textbook theory on the core comparison, saw the expected nothing-happens, and set up machinery for variational-encoding, MIPT, and other quantum-info explorations.

## License

MIT
