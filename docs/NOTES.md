# Notes & surprises

Things that were less obvious than expected, running log style.

- **Braket LocalSimulator doesn't support gate-level noise channels** out of the box. We faked depolarizing / dephasing by sampling Pauli errors per shot. Works for Monte Carlo-style averages; would need density-matrix simulation for coherent-error effects.

- **500 shots is not enough** to see ~1-2% effects. We got a phantom "teleportation beats direct by ~4%" at 500 shots that disappeared at 1500 shots. Rule: `std_err ≈ sqrt(p(1-p)/N)` → ~2.2% at N=500, p=0.5. Always estimate the SE before getting excited.

- **The teleportation circuit uses deferred measurement** (CNOT + CZ instead of measure + classical correction). Mathematically equivalent, but avoids Braket's inconsistent support for mid-circuit measurement across devices.

- **Superdense bit-ordering gotcha.** The Braket probability vector for two qubits is indexed by integer `q0*2 + q1`. Double-check this convention when decoding — easy place to get off-by-one wrong.

- **SHOTS budget explodes fast.** 11 noise points × 3 protocols × 1500 shots × 1 circuit each = 49,500 shots per noise type. On local simulator that's seconds. On a real QPU at ~$0.01/shot that's $495. Keep scale in mind.

- **Real hardware has dephasing >> bit-flip** (T2 < T1 usually). Our dephasing-only noise model is actually realistic in spirit. Still — dephasing gave no advantage either, because the Bell-pair qubit travels through exactly as many noisy operations as the direct qubit.

- The experimental questions that would start getting interesting:
  - Noise on the **Bell pair at distribution time** (before Alice uses it) — changes everything
  - Noise **only on Alice's half** vs only on Bob's half — asymmetric
  - Teleportation + a **3-qubit repetition code** — does entanglement help now?
  - Multiple rounds of teleportation (chain) — noise compounding vs fresh Bell pairs

## Potential follow-ups

- 2D noise atlas (p_x, p_z) with winner-map — partially written, not run
- Variational encoding discovery (partially run) — see `03_variational_encoding.py`
- Try on a real QPU (~$10-30 total) for one noise level to validate simulator
- Add amplitude damping via density-matrix simulation (requires `braket.devices.BraketSimulator("braket_dm")`)
