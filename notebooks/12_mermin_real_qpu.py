"""
12 — Mermin inequality test on real quantum hardware (Rigetti Ankaa-3)
----------------------------------------------------------------------
The Mermin inequality is a 3-qubit generalization of CHSH. For any
local-hidden-variable theory, the Mermin operator
  M = X1*X2*X3 - X1*Y2*Y3 - Y1*X2*Y3 - Y1*Y2*X3
satisfies  |<M>| <= 2  classically.

Quantum mechanics predicts  |<M>| = 4  for the ideal GHZ state
  |GHZ> = (|000> + |111>) / sqrt(2).

Running this on real quantum hardware, we expect |<M>| somewhere in
[2.5, 3.5] due to gate errors. Any value above 2 is an experimental
refutation of local realism on the device we used.

Cost (Rigetti Ankaa-3, approximate):
  task fee: $0.30 per task
  per-shot: $0.00090
  4 tasks x 500 shots = 4*0.30 + 4*500*0.00090 = $1.20 + $1.80 = $3.00
"""

import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit

# Device selection
RIGETTI_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"
# Alternative choices:
#   IonQ Aria:   "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"   (~$30 per 1k shots)
#   IQM Garnet:  "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"   (~$1.75 per 1k shots)
#   Local sim:   None  (we'll branch based on DEVICE argument)

SHOTS = 500


def ghz_circuit():
    """Prepare |GHZ> = (|000> + |111>) / sqrt(2) on qubits 0, 1, 2."""
    c = Circuit()
    c.h(0).cnot(0, 1).cnot(1, 2)
    return c


def mermin_setting(setting):
    """Build the circuit for one of the 4 Mermin measurement settings.
    setting is one of 'XXX', 'XYY', 'YXY', 'YYX'.
    Ends with measurements in the Z basis after rotation to the measurement basis.
    X basis: H before Z; Y basis: S^dagger H before Z.
    """
    c = ghz_circuit()
    for q, p in enumerate(setting):
        if p == 'X':
            c.h(q)
        elif p == 'Y':
            # Y basis: apply S^dagger then H
            c.si(q).h(q)
        elif p == 'Z':
            pass
        else:
            raise ValueError(f"Unknown Pauli {p}")
    return c


def parity_expectation(counts_dict, setting_label):
    """From counts like {'000': 250, '111': 235, ...}, compute <Z1 Z2 Z3>
    which after basis rotation equals <P1 P2 P3> where P_i is the Pauli
    we rotated to. Sign = (-1)^(n_ones)."""
    total = sum(counts_dict.values())
    expect = 0.0
    for bitstring, cnt in counts_dict.items():
        ones = bitstring.count('1')
        sign = 1 if ones % 2 == 0 else -1
        expect += sign * cnt
    return expect / total


def run_mermin(device=None, shots=SHOTS):
    """Execute all 4 Mermin circuits; return expectations and the aggregated
    Mermin value M = <XXX> - <XYY> - <YXY> - <YYX>."""
    settings = ['XXX', 'XYY', 'YXY', 'YYX']
    signs = {'XXX': +1, 'XYY': -1, 'YXY': -1, 'YYX': -1}

    # Build circuits
    circuits = {s: mermin_setting(s) for s in settings}

    if device is None:
        from braket.devices import LocalSimulator
        device = LocalSimulator()
        print("Running on LOCAL simulator (no cost).")
    else:
        print(f"Running on real QPU: {device.name if hasattr(device, 'name') else device}")

    expectations = {}
    tasks = {}
    for s, c in circuits.items():
        print(f"  submitting {s}...")
        t = device.run(c, shots=shots)
        tasks[s] = t

    for s, t in tasks.items():
        r = t.result()
        counts = {''.join(str(b) for b in m): 0 for m in r.measurements}
        # Build counts dict properly
        counts = {}
        for m in r.measurements:
            b = ''.join(str(x) for x in m)
            counts[b] = counts.get(b, 0) + 1
        e = parity_expectation(counts, s)
        expectations[s] = e
        print(f"  <{s}> = {e:+.4f}  (shots={shots}, top counts={sorted(counts.items(), key=lambda x: -x[1])[:2]})")

    M = sum(signs[s] * expectations[s] for s in settings)
    # binomial std error, approximate
    se = np.sqrt(sum((1 - e**2) / shots for e in expectations.values()))
    print(f"\n*** Mermin operator <M> = {M:+.4f}  +/- {se:.4f} ***")
    print(f"    Local-realism bound:   |<M>| <= 2")
    print(f"    Quantum mechanics:     |<M>| = 4 (ideal GHZ)")
    if abs(M) > 2 + 5 * se:
        print(">>> >5 sigma violation of local realism on this device <<<")
    elif abs(M) > 2:
        print(f"    Exceeds classical bound by {(abs(M) - 2) / se:.1f} sigma")
    else:
        print("    Classical bound NOT violated -- device too noisy.")
    return M, expectations


if __name__ == "__main__":
    import sys
    # default: local sim so you don't accidentally burn money
    use_real = "--real" in sys.argv
    if use_real:
        device = AwsDevice(RIGETTI_ARN)
        M, e = run_mermin(device=device, shots=SHOTS)
    else:
        print("Running on local simulator by default. Pass --real to submit to QPU.")
        M, e = run_mermin(device=None, shots=SHOTS)
