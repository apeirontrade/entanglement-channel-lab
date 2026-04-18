"""
05 — Two-qubit encoding discovery under correlated Pauli noise
---------------------------------------------------------------
Generalizes the single-qubit analytic approach to 2-qubit encodings.
Key question: does optimal encoding become ENTANGLED when noise is correlated?

Parameterize 2-qubit unitary as: U = U_local · U_entangler
  - U_local = U_1 ⊗ U_2 (6 Euler angles, two SU(2)s)
  - U_entangler = exp(-i a XX/2) exp(-i b YY/2) exp(-i c ZZ/2) (3 angles)

Total: 9 parameters, optimized by differential evolution.

For each discovered optimal U, compute entanglement entropy S(rho_A) of its
encoding of |00>. If S ≈ 0, encoding is a product state (entanglement useless).
If S ≈ 1, encoding is Bell-like (maximally entangled).

Expected results (from theory):
  independent noise  → product-state optimum (S ≈ 0)
  correlated ZZ      → |Phi+> Bell state (eigenstate of ZZ, S = 1)
  correlated XX      → different Bell state (eigenstate of XX, S = 1)
  mixed              → partially entangled (0 < S < 1)
"""

import numpy as np
from scipy.optimize import differential_evolution

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = {"I": I, "X": X, "Y": Y, "Z": Z}


def kron(*args):
    out = args[0]
    for a in args[1:]:
        out = np.kron(out, a)
    return out


def su2(theta, phi, lam):
    return np.array([
        [np.exp(-1j * (phi + lam) / 2) * np.cos(theta / 2),
         -np.exp(-1j * (phi - lam) / 2) * np.sin(theta / 2)],
        [np.exp(1j * (phi - lam) / 2) * np.sin(theta / 2),
         np.exp(1j * (phi + lam) / 2) * np.cos(theta / 2)],
    ])


def two_qubit_unitary(params):
    t1, p1, l1, t2, p2, l2, a, b, c = params
    U_local = np.kron(su2(t1, p1, l1), su2(t2, p2, l2))
    XX_ = kron(X, X); YY_ = kron(Y, Y); ZZ_ = kron(Z, Z)

    def expm_anti(H, th):
        return np.cos(th / 2) * np.eye(4, dtype=complex) - 1j * np.sin(th / 2) * H

    U_ent = expm_anti(XX_, a) @ expm_anti(YY_, b) @ expm_anti(ZZ_, c)
    return U_local @ U_ent


def apply_2q_pauli_channel(rho, probs):
    out = np.zeros_like(rho)
    p_id = 1 - sum(probs.values())
    out += p_id * rho
    for label, p in probs.items():
        P = kron(PAULIS[label[0]], PAULIS[label[1]])
        out += p * (P @ rho @ P.conj().T)
    return out


def fidelity_2q(params, noise_probs):
    U = two_qubit_unitary(params)
    psi0 = np.array([1, 0, 0, 0], dtype=complex)
    psi = U @ psi0
    rho = np.outer(psi, psi.conj())
    rho_noisy = apply_2q_pauli_channel(rho, noise_probs)
    rho_out = U.conj().T @ rho_noisy @ U
    return np.real(rho_out[0, 0])


def optimize_2q(noise_probs):
    def obj(p):
        return -fidelity_2q(p, noise_probs)
    res = differential_evolution(
        obj, bounds=[(0, 2 * np.pi)] * 9,
        maxiter=200, popsize=15, seed=0, tol=1e-8, polish=True,
    )
    return res.x, -res.fun


def naive_fidelity(noise_probs):
    return fidelity_2q(np.zeros(9), noise_probs)


def best_product_fidelity(noise_probs):
    def marginal(q):
        mx = my = mz = 0
        for lbl, p in noise_probs.items():
            a = lbl[q]
            if a == "X": mx += p
            elif a == "Y": my += p
            elif a == "Z": mz += p
        return mx, my, mz
    p1 = marginal(0); p2 = marginal(1)
    f1 = (1 - sum(p1)) + max(p1)
    f2 = (1 - sum(p2)) + max(p2)
    return f1 * f2


def entanglement_entropy(params):
    U = two_qubit_unitary(params)
    psi = U @ np.array([1, 0, 0, 0], dtype=complex)
    M = psi.reshape(2, 2)
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    s2 = s2[s2 > 1e-12]
    return -np.sum(s2 * np.log2(s2))


PANEL_2Q = {
    "independent_depol_0.1":  {"XI": 0.033, "YI": 0.033, "ZI": 0.033,
                               "IX": 0.033, "IY": 0.033, "IZ": 0.033},
    "independent_bit_flip":   {"XI": 0.15, "IX": 0.15},
    "correlated_ZZ_only":     {"ZZ": 0.30},
    "correlated_XX_only":     {"XX": 0.30},
    "correlated_bit_pair":    {"XI": 0.10, "IX": 0.10, "XX": 0.20},
    "Z_dephasing_correlated": {"ZI": 0.05, "IZ": 0.05, "ZZ": 0.20},
    "anti_correlated_XY":     {"XY": 0.20, "YX": 0.20},
}


if __name__ == "__main__":
    print(f"{'noise':<28} {'F_naive':>8} {'F_prod':>8} {'F_ent':>8} {'Δ(ent-prod)':>14} {'S(ρ_A)':>8}")
    print("-" * 86)
    for name, probs in PANEL_2Q.items():
        F_naive = naive_fidelity(probs)
        F_prod = best_product_fidelity(probs)
        params, F_ent = optimize_2q(probs)
        S = entanglement_entropy(params)
        print(f"{name:<28} {F_naive:>8.4f} {F_prod:>8.4f} {F_ent:>8.4f} {F_ent - F_prod:>+14.4f} {S:>8.4f}")
