"""
08 — N=3 Pauli memory channel phase diagram
--------------------------------------------
Extends Karimipour-Meghdadi-Memarzadeh (Phys. Rev. A 79, 032321, 2009) to the
three-use case. Their paper proved the N=2 phase structure:
  - separable regime at low correlation
  - intermediate regime (both separable and entangled optimal)
  - maximally entangled regime at high correlation

Question: at N=3, does the same three-phase structure repeat, does it acquire
additional phases, and in particular does any region of channel parameter
space select GENUINELY TRIPARTITE entanglement (GHZ- or W-like) instead of
bipartite-embedded entanglement?

Methodology
-----------
- 3-qubit input state parameterized by a minimal ansatz (local SU(2)^3 +
  2 nearest-neighbor entanglers, 3*3 + 2*3 = 15 parameters)
- Apply a 3-use correlated Pauli channel with:
    p_indep: independent single-qubit errors on each of 3 qubits
    p_pair:  pair-correlated errors on (q0,q1) and (q1,q2)
    lambda = p_pair / (p_pair + p_indep)  -- correlation strength
- Variationally maximize F = <psi_in | rho_out | psi_in>
- Measure three entanglement witnesses at the optimum:
    S_2 : bipartite entropy (q0 vs q1q2)
    tau : Coffman-Kundu-Wootters 3-tangle (GHZ-like)
    |<GHZ|psi>|^2 and |<W|psi>|^2 : fidelity with canonical states
- Sweep over (lambda, noise_axis) and cluster the (S_2, tau, F_GHZ, F_W)
  vector to identify distinct phases
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from itertools import product
import time

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAU = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def su2(a, b, c):
    return np.array([
        [np.exp(-1j * (b + c) / 2) * np.cos(a / 2),
         -np.exp(-1j * (b - c) / 2) * np.sin(a / 2)],
        [np.exp(1j * (b - c) / 2) * np.sin(a / 2),
         np.exp(1j * (b + c) / 2) * np.cos(a / 2)],
    ])


def cnot(ctrl, tgt, n=3):
    """CNOT from ctrl to tgt on n-qubit system."""
    dim = 2 ** n
    out = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = [(i >> k) & 1 for k in range(n)]
        bits_out = bits.copy()
        if bits[ctrl] == 1:
            bits_out[tgt] ^= 1
        j = sum(b << k for k, b in enumerate(bits_out))
        out[j, i] = 1
    return out


CNOT_01 = cnot(0, 1, n=3)
CNOT_12 = cnot(1, 2, n=3)


def ansatz(params):
    """15-parameter 3-qubit ansatz: local rotations - CNOT - local - CNOT - local."""
    a1 = params[0:3]; a2 = params[3:6]; a3 = params[6:9]
    b1 = params[9:12]
    U_locals1 = kron(su2(*a1), su2(*a2), su2(*a3))
    U_locals2 = kron(su2(*b1), I2, I2)  # simplify to 12 params of rotations + 2 CNOTs
    return U_locals2 @ CNOT_12 @ CNOT_01 @ U_locals1


def state_from_ansatz(params):
    U = ansatz(params)
    psi0 = np.zeros(8, dtype=complex); psi0[0] = 1.0
    return U @ psi0


# 3-qubit Pauli channel: Kraus-decomposed at the density matrix level
def apply_3q_channel(rho, noise_dict):
    """noise_dict maps 'XIZ', 'IXI', etc. -> probability of that 3-qubit Pauli error."""
    out = np.zeros_like(rho)
    p_id = 1.0 - sum(noise_dict.values())
    out += p_id * rho
    for label, p in noise_dict.items():
        P = kron(PAU[label[0]], PAU[label[1]], PAU[label[2]])
        out += p * (P @ rho @ P.conj().T)
    return out


def fidelity(params, noise_dict):
    """F = <psi | E(|psi><psi|) | psi>."""
    psi = state_from_ansatz(params)
    rho = np.outer(psi, psi.conj())
    rho_out = apply_3q_channel(rho, noise_dict)
    return float(np.real(psi.conj() @ rho_out @ psi))


def optimize(noise_dict, seed=0):
    def obj(p):
        return -fidelity(p, noise_dict)
    res = differential_evolution(
        obj, bounds=[(0, 2 * np.pi)] * 12,
        maxiter=120, popsize=18, seed=seed, tol=1e-7, polish=True,
    )
    return res.x, -res.fun


# Entanglement metrics
def bipartite_entropy(psi):
    """S(rho_A) where A = qubit 0, B = qubits 1,2."""
    M = psi.reshape(2, 4)
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-12]
    return float(-np.sum(s * np.log2(s)))


def three_tangle(psi):
    """Coffman-Kundu-Wootters 3-tangle.
    tau_ABC = 4 |d_1 - 2*d_2 + 4*d_3| for the hyperdeterminant, but numerically:
    tau = 2 * sqrt( |sum over i,j,k,l,m,n of eps_ii' eps_jj' eps_mm' a_ijm a_i'j'm' conj ...| )
    Simpler: tau = 4 |det(M_A)| - ... complicated. Use the explicit formula:
    For 3-qubit state |psi> = sum a_ijk |ijk>,
    tau = 4 | d_1 - 2 d_2 + 4 d_3 | where
      d_1 = a_000^2 a_111^2 + a_001^2 a_110^2 + a_010^2 a_101^2 + a_100^2 a_011^2
      d_2 = a_000 a_111 (a_011 a_100 + a_101 a_010 + a_110 a_001)
            + a_011 a_100 (a_101 a_010 + a_110 a_001)
            + a_101 a_010 a_110 a_001
      d_3 = a_000 a_110 a_101 a_011 + a_111 a_001 a_010 a_100
    """
    a = psi
    d1 = (a[0]**2 * a[7]**2 + a[1]**2 * a[6]**2 + a[2]**2 * a[5]**2 + a[4]**2 * a[3]**2)
    d2 = (a[0] * a[7] * (a[3] * a[4] + a[5] * a[2] + a[6] * a[1])
          + a[3] * a[4] * (a[5] * a[2] + a[6] * a[1])
          + a[5] * a[2] * a[6] * a[1])
    d3 = a[0] * a[6] * a[5] * a[3] + a[7] * a[1] * a[2] * a[4]
    tau = 4 * abs(d1 - 2 * d2 + 4 * d3)
    return float(tau)


def ghz_fidelity(psi):
    ghz = (np.zeros(8, dtype=complex))
    ghz[0] = ghz[7] = 1.0 / np.sqrt(2)
    return float(abs(np.vdot(ghz, psi)) ** 2)


def w_fidelity(psi):
    w = np.zeros(8, dtype=complex)
    w[1] = w[2] = w[4] = 1.0 / np.sqrt(3)
    return float(abs(np.vdot(w, psi)) ** 2)


# Noise model builders
def build_n3_channel(lam, axis, p_total=0.3):
    """
    axis: 'Z' -> correlated ZZ pairs
          'X' -> correlated XX pairs
          'Y' -> correlated YY pairs
          'mixed' -> correlated XX and ZZ mixed
          'XY' -> anti-correlated XY+YX on adjacent pairs
    """
    p_indep = (1 - lam) * p_total
    p_pair = lam * p_total
    d = {}
    # Independent single-qubit errors: X or Z on each of the 3 qubits
    if p_indep > 0:
        per = p_indep / 6
        d["XII"] = d["YII"] = d["ZII"] = per
        d["IXI"] = d["IYI"] = d["IZI"] = per
        # we only use X and Z so skip Y above actually; redo
    d = {}
    if p_indep > 0:
        per = p_indep / 6
        for pos in range(3):
            for pauli in "XZ":
                lbl = ["I"] * 3
                lbl[pos] = pauli
                d["".join(lbl)] = d.get("".join(lbl), 0) + per
    if p_pair > 0:
        per = p_pair / 2
        if axis == "Z":
            for pair in [(0, 1), (1, 2)]:
                lbl = ["I"] * 3
                lbl[pair[0]] = "Z"; lbl[pair[1]] = "Z"
                d["".join(lbl)] = d.get("".join(lbl), 0) + per
        elif axis == "X":
            for pair in [(0, 1), (1, 2)]:
                lbl = ["I"] * 3
                lbl[pair[0]] = "X"; lbl[pair[1]] = "X"
                d["".join(lbl)] = d.get("".join(lbl), 0) + per
        elif axis == "mixed":
            for pair in [(0, 1), (1, 2)]:
                lbl = ["I"] * 3
                lbl[pair[0]] = "Z"; lbl[pair[1]] = "Z"
                d["".join(lbl)] = d.get("".join(lbl), 0) + per / 2
                lbl = ["I"] * 3
                lbl[pair[0]] = "X"; lbl[pair[1]] = "X"
                d["".join(lbl)] = d.get("".join(lbl), 0) + per / 2
        elif axis == "XY":
            for pair in [(0, 1), (1, 2)]:
                lbl = ["I"] * 3
                lbl[pair[0]] = "X"; lbl[pair[1]] = "Y"
                d["".join(lbl)] = d.get("".join(lbl), 0) + per / 2
                lbl = ["I"] * 3
                lbl[pair[0]] = "Y"; lbl[pair[1]] = "X"
                d["".join(lbl)] = d.get("".join(lbl), 0) + per / 2
    return d


if __name__ == "__main__":
    lams = np.linspace(0.0, 1.0, 9)
    axes = ["Z", "X", "mixed", "XY"]

    print(f"{'axis':<7} {'lambda':>7} {'F_opt':>7} {'S_bip':>7} {'tau_3':>7} {'F_GHZ':>7} {'F_W':>7}")
    data = []
    t0 = time.time()
    for axis in axes:
        for lam in lams:
            noise = build_n3_channel(lam, axis)
            params, F_opt = optimize(noise, seed=42)
            psi = state_from_ansatz(params)
            S2 = bipartite_entropy(psi)
            tau = three_tangle(psi)
            fG = ghz_fidelity(psi)
            fW = w_fidelity(psi)
            data.append(dict(axis=axis, lam=lam, F=F_opt, S2=S2, tau=tau, FG=fG, FW=fW))
            dt = time.time() - t0
            print(f"{axis:<7} {lam:>7.2f} {F_opt:>7.3f} {S2:>7.3f} {tau:>7.3f} {fG:>7.3f} {fW:>7.3f}   (t={int(dt)}s)")

    # Summary
    print("\n=== Phase identification across axes ===")
    for axis in axes:
        rows = [d for d in data if d['axis'] == axis]
        print(f"\n{axis}:")
        for r in rows:
            # Phase classification
            if r['S2'] < 0.2 and r['tau'] < 0.05:
                ph = "product"
            elif r['FG'] > 0.9:
                ph = "GHZ-like"
            elif r['FW'] > 0.9:
                ph = "W-like"
            elif r['tau'] > 0.5:
                ph = "tripartite (non-GHZ/W)"
            elif r['S2'] > 0.5:
                ph = "bipartite-dominant"
            else:
                ph = "mixed/transition"
            print(f"  lam={r['lam']:.2f}  F={r['F']:.3f}  phase={ph}  (S2={r['S2']:.2f}, tau={r['tau']:.2f}, F_GHZ={r['FG']:.2f}, F_W={r['FW']:.2f})")
