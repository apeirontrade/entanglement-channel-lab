# Results

Run date: 2026-04-18
Hardware: Amazon Braket LocalSimulator (classical simulator on `ml.t3.medium`)

## 02 — Channel duality (1500 shots per point)

### Depolarizing noise

| p    | F_teleport | F_direct | F_superdense |
|------|-----------:|---------:|-------------:|
| 0.00 | 1.000      | 1.000    | 1.000        |
| 0.05 | 0.967      | 0.975    | 0.957        |
| 0.10 | 0.937      | 0.936    | 0.913        |
| 0.15 | 0.897      | 0.894    | 0.835        |
| 0.20 | 0.881      | 0.872    | 0.804        |
| 0.25 | 0.854      | 0.833    | 0.780        |
| 0.30 | 0.808      | 0.798    | 0.687        |
| 0.35 | 0.778      | 0.779    | 0.653        |
| 0.40 | 0.723      | 0.733    | 0.585        |
| 0.45 | 0.711      | 0.697    | 0.525        |
| 0.50 | 0.652      | 0.665    | 0.500        |

### Dephasing noise (Z only)

| p    | F_teleport | F_direct | F_superdense |
|------|-----------:|---------:|-------------:|
| 0.00 | 1.000      | 1.000    | 1.000        |
| 0.05 | 0.971      | 0.959    | 0.956        |
| 0.10 | 0.931      | 0.926    | 0.902        |
| 0.15 | 0.887      | 0.890    | 0.849        |
| 0.20 | 0.846      | 0.857    | 0.809        |
| 0.25 | 0.815      | 0.810    | 0.759        |
| 0.30 | 0.781      | 0.767    | 0.696        |
| 0.35 | 0.745      | 0.733    | 0.663        |
| 0.40 | 0.685      | 0.686    | 0.618        |
| 0.45 | 0.675      | 0.671    | 0.549        |
| 0.50 | 0.625      | 0.633    | 0.503        |

### Interpretation

- **Teleport ≈ Direct** (indistinguishable within ~2% Monte Carlo noise at 1500 shots) for both noise types.
- **Superdense decays ~2× faster** — its 4-outcome decoding is more sensitive to per-qubit noise.
- Early hint of teleport-advantage under dephasing at low shots was **Monte Carlo noise**, not real signal.
- Consistent with textbook theory: entanglement does not buy noise resistance for symmetric single-qubit Pauli noise with a generic encoding.

## 03 — Variational encoding

(fill in once the script finishes running)

## Takeaways

1. Entanglement-assisted communication protocols are not automatically noise-resistant.
2. To see a real advantage you need structured noise + a smart encoding + possibly error correction.
3. Superdense coding is the most fragile of the three under per-qubit noise; use teleportation or direct transmission if noise resistance matters.
