# Entanglement-Assisted Channel Lab

A weekend experiment comparing three ways to send information between Alice and Bob under noise:

- **Direct** – just send the thing (baseline, no entanglement)
- **Teleportation** – consume a Bell pair + 2 classical bits to send 1 qubit
- **Superdense coding** – consume a Bell pair + 1 qubit to send 2 classical bits

Runs on **Amazon Braket** (local simulator — zero compute cost).

## TL;DR

Under symmetric single-qubit Pauli noise (depolarizing or dephasing), **entanglement-assisted protocols provide no measurable fidelity advantage over direct transmission**.

That's the boring-but-correct answer: the "magic" of entanglement buys you topology (send across classical-only channels) and protocol flexibility (qubit↔bit conversion), but not free error resistance for a generic encoding.

**Superdense coding consistently degrades faster** than the other two — its 4-outcome decoding is more fragile to per-qubit noise.

## What the code does

1. Prepare a test state `|ψ⟩ = Ry(π/3) Rz(π/4) |0⟩` on Alice's side
2. Run each protocol through a noisy channel (parameterised Pauli noise)
3. Measure Bob's reconstruction fidelity over many shots
4. Sweep noise strength from `p=0` to `p=0.5`
5. Plot fidelity curves and write them to `results/`

## Files

```
notebooks/
  01_teleport_basic.py           # smallest working teleportation circuit
  02_channel_duality.py          # T vs D vs S under depolarizing + dephasing
  03_variational_encoding.py     # discover optimal encoding per noise type
results/
  channel_duality.png            # two-panel fidelity plot
  encoding_discovery.png         # learned-encoding scatter
docs/
  NOTES.md                       # raw observations & things that surprised us
  RESULTS.md                     # final numbers + interpretation
```

## How to run

Open any Braket notebook (`conda_braket` kernel), paste the contents of any `notebooks/NN_*.py` into a cell, `Shift+Enter`.

## Cost

- Local simulator: **$0**
- Notebook instance: ~$0.05/hr while running (remember to **Stop** in AWS Braket console when done)

## Status

Weekend exploration. Not a publication. Validated textbook theory, saw the expected nothing-happens, and set up machinery for variational-encoding experiments where something *could* happen with structured noise.

## License

MIT
