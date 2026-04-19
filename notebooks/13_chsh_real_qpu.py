"""
13 — CHSH Bell inequality test on real QPU (IQM Garnet)
---------------------------------------------------------
The CHSH inequality (Clauser-Horne-Shimony-Holt 1969) bounds the
correlations achievable by any local-hidden-variable theory:

  S = E(a, b) - E(a, b') + E(a', b) + E(a', b')

where E(a, b) is the expectation of products of ±1 measurements on Alice
and Bob's particles with settings (a, b).

Local realism:         |S| <= 2
Quantum mechanics:     |S| <= 2*sqrt(2) ~ 2.828 (Tsirelson bound)
Aspect 1982 (photons): |S| = 2.70 +/- 0.05
Expected on IQM Garnet (superconducting, ~98% 2q fidelity):
                       |S| ~ 2.3 - 2.7

For an optimal Bell-state |phi+> = (|00>+|11>)/sqrt(2):
  Alice settings:  a = Z,  a' = X
  Bob settings:    b = (Z+X)/sqrt(2),   b' = (Z-X)/sqrt(2)
                   i.e. rotate by pi/8 and -pi/8

Cost on IQM Garnet:
  task fee:    $0.30/task
  per-shot:    $0.00145
  4 tasks x 500 shots = 4*0.30 + 4*500*0.00145 = $1.20 + $2.90 = $4.10
"""

import numpy as np
from braket.circuits import Circuit


def bell_pair():
    """Prepare |phi+> = (|00> + |11>) / sqrt(2) on qubits 0,1."""
    c = Circuit()
    c.h(0).cnot(0, 1)
    return c


def chsh_circuit(alice_angle, bob_angle):
    """Prepare Bell pair, measure Alice in basis rotated by alice_angle,
    Bob in basis rotated by bob_angle. Both measurements end in Z basis.
    Basis rotation: Ry(theta) brings a Z-eigenstate to cos(theta/2)|0>+sin(theta/2)|1>.
    For a measurement along theta we apply Ry(-theta) before Z measurement."""
    c = bell_pair()
    c.ry(0, -alice_angle)
    c.ry(1, -bob_angle)
    return c


def correlation_from_counts(counts, total_shots):
    """E = <Z0 * Z1> = sum over bitstrings of (-1)^(b0+b1) * p(bitstring)."""
    e = 0.0
    for bitstr, cnt in counts.items():
        # bitstr like '00', '01', '10', '11' (first char = qubit 0 per Braket convention)
        parity = 1 if bitstr.count('1') % 2 == 0 else -1
        e += parity * cnt
    return e / total_shots


def run_chsh(device, shots=500):
    """Run the 4 CHSH settings and return S = E(a,b) - E(a,b') + E(a',b) + E(a',b')."""
    # Optimal CHSH angles for |phi+>
    a = 0.0              # Z
    ap = np.pi / 2       # X
    b = np.pi / 4        # (Z+X)/sqrt(2)
    bp = -np.pi / 4      # (Z-X)/sqrt(2)

    settings = {
        'ab':  (a,  b),
        'abp': (a,  bp),
        'apb': (ap, b),
        'apbp':(ap, bp),
    }
    # CHSH signs:   S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    signs = {'ab': +1, 'abp': -1, 'apb': +1, 'apbp': +1}

    tasks = {}
    print(f"Submitting 4 CHSH circuits to {device}")
    for name, (alpha, beta) in settings.items():
        circ = chsh_circuit(alpha, beta)
        t = device.run(circ, shots=shots)
        tasks[name] = t
        print(f"  {name}: alice={alpha:.3f} rad, bob={beta:.3f} rad -> task {t.id[-12:]}")

    print("\nWaiting for results...")
    correlations = {}
    for name, t in tasks.items():
        r = t.result()
        # Build counts dict from measurements
        counts = {}
        for m in r.measurements:
            b = ''.join(str(x) for x in m)
            counts[b] = counts.get(b, 0) + 1
        e = correlation_from_counts(counts, shots)
        correlations[name] = e
        top = sorted(counts.items(), key=lambda x: -x[1])[:2]
        print(f"  E({name}) = {e:+.4f}   top counts: {top}")

    S = sum(signs[name] * correlations[name] for name in settings)
    se = np.sqrt(sum((1 - e**2) / shots for e in correlations.values()))

    print("\n" + "=" * 60)
    print(f"*** CHSH  S = {S:+.4f}  +/- {se:.4f} ***")
    print(f"    Local realism:         |S| <= 2")
    print(f"    Tsirelson bound (QM):  |S| <= {2*np.sqrt(2):.4f}")
    if abs(S) > 2 + 5 * se:
        print(f">>> {(abs(S) - 2)/se:.1f} sigma violation of local realism <<<")
    elif abs(S) > 2:
        print(f"    Exceeds classical bound by {(abs(S) - 2)/se:.1f} sigma")
    else:
        print("    Classical bound NOT violated -- device too noisy.")
    return S, correlations


if __name__ == "__main__":
    # Default: local simulator
    from braket.devices import LocalSimulator
    print("Running on LOCAL simulator (sanity check, no cost).\n")
    S, corr = run_chsh(LocalSimulator(), shots=1000)
    print("\nExpected on ideal quantum: S = 2 sqrt(2) = 2.828...")
