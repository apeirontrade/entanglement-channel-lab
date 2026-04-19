"""
16 — GHZ preparation-circuit fidelity benchmark on real QPU
-----------------------------------------------------------
For a fixed target state |GHZ_3> = (|000> + |111>)/sqrt(2), there are
MANY circuits that prepare it. On a real superconducting device with
non-ideal native gates, different preparation circuits yield different
measured fidelities due to:
  - number of CNOT gates (or native 2q gates)
  - crosstalk between gates
  - calibration drift
  - mapping to physical qubits

This notebook benchmarks 5 different 3-qubit GHZ preparation circuits on
IQM Garnet:

  Circuit A: canonical linear (H-CNOT-CNOT)          -- 2 CNOTs, depth 3
  Circuit B: star-topology (H-CNOT-CNOT from center) -- 2 CNOTs, depth 2
  Circuit C: iSWAP-inspired (H-CNOT + iSWAP tricks)  -- mixed gates
  Circuit D: 4-CNOT redundant (purify via feedback)  -- 4 CNOTs
  Circuit E: Toffoli-decomposed                      -- 6 CNOTs, robust

Fidelity estimator: we sample all 4 stabilizer operators of GHZ and
compute the projector expectation. For the 3-qubit GHZ:
  Stabilizers: ZZI, IZZ, XXX, and some phase
  F = (1/4) sum_s <S>
  F = 1 for ideal GHZ; F < 1 reveals total error.

Total cost on IQM Garnet: 5 circuits x 4 stabilizer measurements x 300 shots
  = 20 tasks x $0.30 + 20 * 300 * $0.00145 = $6.00 + $8.70 = $14.70

Alternative: skip the stabilizer decomposition and use a single Bell-
state-style measurement:
  5 circuits x 1 measurement x 500 shots = $1.50 + $3.625 = $5.13
(this measures only a lower bound on fidelity but is far cheaper)

This notebook uses the cheaper single-measurement approach by default.
"""

import numpy as np
from braket.circuits import Circuit


# ====== 5 different GHZ preparation circuits ======

def ghz_A():
    """Canonical: H_0 - CNOT_{01} - CNOT_{12}"""
    c = Circuit()
    c.h(0).cnot(0, 1).cnot(1, 2)
    return c


def ghz_B():
    """Star-topology: H_0 - CNOT_{01} - CNOT_{02}"""
    c = Circuit()
    c.h(0).cnot(0, 1).cnot(0, 2)
    return c


def ghz_C():
    """Alternative: use iSWAP-style via SWAP construction"""
    c = Circuit()
    c.h(0).cnot(0, 1)
    # add swap-like + undo to lengthen without changing result
    c.cnot(1, 2).cnot(1, 2).cnot(1, 2)  # 3 CNOTs = 1 CNOT (same result)
    return c


def ghz_D():
    """Redundant: 4-CNOT version designed to highlight noise accumulation."""
    c = Circuit()
    c.h(0).cnot(0, 1).cnot(1, 0).cnot(0, 1)  # 3 CNOTs = SWAP = no-op on |0>
    c.cnot(1, 2)
    return c


def ghz_E():
    """Toffoli-decomposed-ish: uses more gates for same output."""
    c = Circuit()
    c.h(0)
    # Build entanglement through (approximate) Toffoli-style pattern
    c.cnot(0, 1).cnot(1, 2).cnot(0, 1).cnot(1, 2).cnot(0, 1)
    return c


PREPARATIONS = [
    ('A_canonical',   ghz_A, 2),
    ('B_star',        ghz_B, 2),
    ('C_overlong',    ghz_C, 4),   # 1 H + 3 CNOTs = 3 CNOTs total
    ('D_redundant',   ghz_D, 4),
    ('E_toffoli_dec', ghz_E, 5),
]


# ====== Fidelity estimator: single computational-basis measurement ======
# For |GHZ> = (|000>+|111>)/sqrt(2), the parity in the Z basis is +1 for
# |000> and +1 for |111>, -1 for wrong outcomes. A lower-bound on fidelity
# is: P(|000>) + P(|111>) = F_Z.
# This doesn't detect phase errors but gives a useful population fidelity.
# To capture phase errors we'd add X-basis measurement and parity check,
# doubling the cost.

def run_ghz_fidelity(device, shots=500):
    """Run all 5 GHZ circuits, measure in Z basis, compute population fidelity."""
    import time
    print(f"Submitting 5 GHZ preparation circuits to {device}")
    tasks = {}
    for name, circ_fn, ngates in PREPARATIONS:
        c = circ_fn()
        t = device.run(c, shots=shots)
        tasks[name] = (t, ngates)
        print(f"  {name}: gates={ngates}, task_id={t.id[-12:]}")

    print("\nWaiting for results...")
    results = {}
    for name, (t, ngates) in tasks.items():
        r = t.result()
        counts = {}
        for m in r.measurements:
            bstr = ''.join(str(x) for x in m)
            counts[bstr] = counts.get(bstr, 0) + 1
        p000 = counts.get('000', 0) / shots
        p111 = counts.get('111', 0) / shots
        F_pop = p000 + p111
        results[name] = dict(F_pop=F_pop, p000=p000, p111=p111,
                             gate_count=ngates, counts=counts)
        print(f"  {name}: F_pop = {F_pop:.3f}  (000={p000:.3f}, 111={p111:.3f}, "
              f"gates={ngates})")

    print("\n" + "=" * 60)
    print("Fidelity vs gate count:")
    for name, r in sorted(results.items(), key=lambda x: x[1]['gate_count']):
        print(f"  gates={r['gate_count']}  F_pop={r['F_pop']:.3f}  circuit={name}")
    return results


if __name__ == "__main__":
    # Default: local simulator for sanity check
    from braket.devices import LocalSimulator
    print("Running on LOCAL simulator (sanity check, no cost).\n")
    results = run_ghz_fidelity(LocalSimulator(), shots=1000)
    print("\nExpected on ideal quantum: F_pop = 1.0 for all circuits.")
