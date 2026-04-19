"""
17 — Real-hardware characterization of IQM Garnet: 2026-04-18
-------------------------------------------------------------
Unified submission of 12 circuits to IQM Garnet, producing three
empirical results that are time-and-device-specific and, as far as can
be verified, not in any published literature:

  Block A (5 circuits): GHZ preparation fidelity vs circuit depth
  Block B (4 circuits): CHSH Bell inequality violation on 2 qubits
  Block C (3 circuits): Crosstalk fingerprint under local X drive

All blocks are 500-shots each.

Total: 12 tasks * 0.30 + 12 * 500 * 0.00145 = $12.30

Expected results:
  Block A: F_pop decreases with gate count. Exact values unknown.
  Block B: |S| between 2.2 and 2.7 (hardware noise).
  Block C: non-zero correlation between driven qubit and neighbors.

Honest novelty: each individual measurement is a standard characterization
technique. The specific numbers for IQM Garnet on 2026-04-18 evening
are not in any literature because devices drift day-to-day. This is the
empirical equivalent of a weather report for one specific QPU.
"""

import numpy as np
import time
from braket.aws import AwsDevice
from braket.circuits import Circuit


# ============== Block A: GHZ preparations ==============

def ghz_A(): c = Circuit(); c.h(0).cnot(0, 1).cnot(1, 2); return c
def ghz_B(): c = Circuit(); c.h(0).cnot(0, 1).cnot(0, 2); return c
def ghz_C(): c = Circuit(); c.h(0).cnot(0, 1).cnot(1, 2).cnot(1, 2).cnot(1, 2); return c
def ghz_D(): c = Circuit(); c.h(0).cnot(0, 1).cnot(1, 0).cnot(0, 1).cnot(1, 2); return c
def ghz_E(): c = Circuit(); c.h(0).cnot(0, 1).cnot(1, 2).cnot(0, 1).cnot(1, 2).cnot(0, 1); return c

GHZ_SET = [('A_2CNOT', ghz_A, 2), ('B_2CNOT_star', ghz_B, 2),
           ('C_4CNOT', ghz_C, 4), ('D_4CNOT_swap', ghz_D, 4),
           ('E_5CNOT', ghz_E, 5)]


# ============== Block B: CHSH ==============

def bell_pair():
    c = Circuit(); c.h(0).cnot(0, 1); return c


def chsh_circuit(alice_angle, bob_angle):
    c = bell_pair()
    c.ry(0, -alice_angle).ry(1, -bob_angle)
    return c


CHSH_SET = [
    ('ab',   0.0,         np.pi / 4),
    ('abp',  0.0,         -np.pi / 4),
    ('apb',  np.pi / 2,    np.pi / 4),
    ('apbp', np.pi / 2,   -np.pi / 4),
]


# ============== Block C: Crosstalk ==============

def crosstalk_circuit(drive_angle):
    """Apply X rotation of drive_angle to qubit 0, idle on 1, 2.
    Measure all three. Ideal: only qubit 0 has rotation-dependent population.
    Non-ideal (crosstalk): qubits 1 and 2 population depends on drive_angle."""
    c = Circuit()
    c.rx(0, drive_angle)  # drive qubit 0
    # No operations on 1, 2 -- idle
    return c


CROSSTALK_SET = [
    ('drive_pi2', np.pi / 2),
    ('drive_pi',  np.pi),
    ('drive_3pi2', 3 * np.pi / 2),
]


# ============== Analysis helpers ==============

def counts_from_result(result):
    c = {}
    for m in result.measurements:
        b = ''.join(str(x) for x in m)
        c[b] = c.get(b, 0) + 1
    return c


def ghz_fidelity(counts, shots):
    p000 = counts.get('000', 0) / shots
    p111 = counts.get('111', 0) / shots
    return p000 + p111


def chsh_correlation(counts, shots):
    e = 0.0
    for b, cnt in counts.items():
        parity = 1 if b.count('1') % 2 == 0 else -1
        e += parity * cnt
    return e / shots


def marginal_populations(counts, shots):
    """Return <Z> for each of 3 qubits."""
    pops = [0.0, 0.0, 0.0]
    for b, cnt in counts.items():
        for i, ch in enumerate(b):
            # <Z> = P(0) - P(1) = +1 if '0' else -1
            pops[i] += (1 if ch == '0' else -1) * cnt
    return [p / shots for p in pops]


# ============== Main submission ==============

def submit_all(device, shots=500):
    submitted = []
    print(f"=== Block A: GHZ preparations (5 circuits) ===")
    for name, fn, ngates in GHZ_SET:
        c = fn()
        t = device.run(c, shots=shots)
        submitted.append(('GHZ', name, t, {'ngates': ngates}))
        print(f"  {name} (n_cnots={ngates}): task {t.id[-12:]}")

    print(f"\n=== Block B: CHSH (4 circuits) ===")
    for name, alpha, beta in CHSH_SET:
        c = chsh_circuit(alpha, beta)
        t = device.run(c, shots=shots)
        submitted.append(('CHSH', name, t, {'alpha': alpha, 'beta': beta}))
        print(f"  {name}: task {t.id[-12:]}")

    print(f"\n=== Block C: Crosstalk (3 circuits) ===")
    for name, angle in CROSSTALK_SET:
        c = crosstalk_circuit(angle)
        # Garnet is 3+ qubit device so this is safe; ensure it uses 3 qubits
        c_padded = Circuit()
        c_padded.rx(0, angle).i(1).i(2)  # pad to use 3 qubits
        t = device.run(c_padded, shots=shots)
        submitted.append(('XTALK', name, t, {'angle': angle}))
        print(f"  {name} (angle={angle:.3f}): task {t.id[-12:]}")

    return submitted


def fetch_and_analyze(submitted, shots=500):
    ghz_results, chsh_results, xtalk_results = {}, {}, {}
    for block, name, task, meta in submitted:
        r = task.result()
        c = counts_from_result(r)
        if block == 'GHZ':
            F = ghz_fidelity(c, shots)
            ghz_results[name] = {'F_pop': F, 'ngates': meta['ngates'], 'counts': c}
        elif block == 'CHSH':
            e = chsh_correlation(c, shots)
            chsh_results[name] = {'E': e, **meta}
        elif block == 'XTALK':
            pops = marginal_populations(c, shots)
            xtalk_results[name] = {'Z0': pops[0], 'Z1': pops[1], 'Z2': pops[2], 'angle': meta['angle']}
    return ghz_results, chsh_results, xtalk_results


def report(ghz, chsh, xtalk):
    print("\n" + "=" * 70)
    print(f"IQM GARNET CHARACTERIZATION -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n--- Block A: GHZ fidelity vs preparation depth ---")
    print(f"{'circuit':<18}  {'n_cnots':>8}  {'F_pop':>7}  {'F / F_A':>8}")
    F_A = ghz['A_2CNOT']['F_pop']
    for name, r in sorted(ghz.items(), key=lambda x: x[1]['ngates']):
        ratio = r['F_pop'] / F_A if F_A > 0 else float('nan')
        print(f"{name:<18}  {r['ngates']:>8}  {r['F_pop']:>7.3f}  {ratio:>8.3f}")

    print("\n--- Block B: CHSH Bell inequality ---")
    signs = {'ab': +1, 'abp': -1, 'apb': +1, 'apbp': +1}
    S = sum(signs[name] * chsh[name]['E'] for name in chsh)
    se = np.sqrt(sum((1 - chsh[name]['E'] ** 2) / 500 for name in chsh))
    for name in ['ab', 'abp', 'apb', 'apbp']:
        print(f"  E({name}) = {chsh[name]['E']:+.4f}")
    print(f"\n  *** S = {S:+.4f}  ±  {se:.4f} ***")
    print(f"      Classical bound: |S| <= 2")
    print(f"      Tsirelson:        |S| <= {2*np.sqrt(2):.4f}")
    if abs(S) > 2:
        sigma = (abs(S) - 2) / se
        print(f"      Exceeds classical bound by {sigma:.1f} sigma")

    print("\n--- Block C: Crosstalk under X-drive on qubit 0 ---")
    print("Ideal: <Z_1> and <Z_2> should equal +1 (untouched).")
    print("Non-ideal: deviation from +1 indicates crosstalk.\n")
    print(f"{'drive':<15}  {'<Z_0>':>8}  {'<Z_1>':>8}  {'<Z_2>':>8}  "
          f"{'|Z_1 - 1|':>10}  {'|Z_2 - 1|':>10}")
    for name, r in xtalk.items():
        x1 = abs(r['Z1'] - 1.0)
        x2 = abs(r['Z2'] - 1.0)
        print(f"{name:<15}  {r['Z0']:>+8.3f}  {r['Z1']:>+8.3f}  {r['Z2']:>+8.3f}  "
              f"{x1:>10.4f}  {x2:>10.4f}")


if __name__ == "__main__":
    import sys
    use_real = '--real' in sys.argv
    if use_real:
        device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")
        print(f"Device: Garnet, status = {device.status}")
        if device.status != 'ONLINE':
            print("Not online; aborting."); sys.exit(1)
    else:
        from braket.devices import LocalSimulator
        device = LocalSimulator()
        print("Running on LOCAL simulator (sanity check, no cost).")

    submitted = submit_all(device, shots=500)
    ghz, chsh, xtalk = fetch_and_analyze(submitted, shots=500)
    report(ghz, chsh, xtalk)
