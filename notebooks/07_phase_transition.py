"""
07 — Correlation–Entanglement Phase Transition in 2-qubit optimal encoding
--------------------------------------------------------------------------
For a family of 2-qubit Pauli channels parameterized by correlation strength
lambda ∈ [0, 1] (at fixed total error rate p = 0.25), we ask: at what lambda
does the optimal encoding switch from a product state to an entangled state?

Three noise families are compared:
  A (XX+ZZ)  — correlated component mixes XX and ZZ flips
  B (ZZ)     — correlated component is pure ZZ
  C (XY+YX)  — correlated component is anti-correlated off-axis

All three start at lambda=0 with independent single-qubit errors (XI, IX,
ZI, IZ each at rate 0.0625), and smoothly add correlated pair errors as
lambda -> 1.

Key finding:
  Family A:   sharp transition at lambda* ≈ 0.55, F_prod flat at 0.875 below,
              Bell-like optimum (S=1) above.
  Family B:   NO transition. F_prod == F_ent at every lambda because
              Z-eigenstate products already stabilize ZZ errors; entanglement
              is degenerate-useless throughout this family.
  Family C:   transition at lambda* ≈ 0.35 (EARLIER than A). F_prod actively
              decreases with lambda because no product state is safe from
              XY+YX errors; the Bell branch takes over sooner.

Together these suggest a taxonomy of correlated-noise phase transitions
indexed by the product-state eigenstructure of the correlated-noise operators.

Assumes F_ent, F_prod, S_entropy, build_channel (from 05_two_qubit_encoding.py
style) are available in scope.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P = {"I": I, "X": X, "Y": Y, "Z": Z}


def kron(*a):
    out = a[0]
    for x in a[1:]:
        out = np.kron(out, x)
    return out


def su2(t, p, l):
    return np.array([
        [np.exp(-1j * (p + l) / 2) * np.cos(t / 2), -np.exp(-1j * (p - l) / 2) * np.sin(t / 2)],
        [np.exp(1j * (p - l) / 2) * np.sin(t / 2), np.exp(1j * (p + l) / 2) * np.cos(t / 2)],
    ])


def U2q(pars):
    t1, p1, l1, t2, p2, l2, a, b, c = pars
    Ul = np.kron(su2(t1, p1, l1), su2(t2, p2, l2))
    XX_ = kron(X, X); YY_ = kron(Y, Y); ZZ_ = kron(Z, Z)

    def E(H, th):
        return np.cos(th / 2) * np.eye(4, dtype=complex) - 1j * np.sin(th / 2) * H

    return Ul @ E(XX_, a) @ E(YY_, b) @ E(ZZ_, c)


def apply_ch(rho, probs):
    out = (1 - sum(probs.values())) * rho
    for k, p in probs.items():
        M = kron(P[k[0]], P[k[1]])
        out = out + p * (M @ rho @ M.conj().T)
    return out


def F(pars, probs):
    U = U2q(pars)
    psi = U @ np.array([1, 0, 0, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    rho2 = apply_ch(rho, probs)
    return float(np.real((U.conj().T @ rho2 @ U)[0, 0]))


def F_prod(probs):
    def obj(p6):
        return -F(np.concatenate([p6, [0, 0, 0]]), probs)
    r = differential_evolution(obj, bounds=[(0, 2 * np.pi)] * 6,
                               maxiter=80, popsize=12, seed=1, tol=1e-8, polish=True)
    return -r.fun


def F_ent(probs):
    def obj(p):
        return -F(p, probs)
    r = differential_evolution(obj, bounds=[(0, 2 * np.pi)] * 9,
                               maxiter=200, popsize=15, seed=0, tol=1e-9, polish=True)
    return -r.fun, r.x


def S_entropy(pars):
    U = U2q(pars)
    psi = U @ np.array([1, 0, 0, 0], dtype=complex)
    M = psi.reshape(2, 2)
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-12]
    return -float(np.sum(s * np.log2(s)))


def build_channel(lam, family, p_total=0.25):
    p_solo = (1 - lam) * p_total
    p_pair = lam * p_total
    probs = {}
    if p_solo > 0:
        probs["XI"] = probs["IX"] = probs["ZI"] = probs["IZ"] = p_solo / 4
    if p_pair > 0:
        if family == "A_XXZZ":
            probs["XX"] = probs["ZZ"] = p_pair / 2
        elif family == "B_ZZ":
            probs["ZZ"] = p_pair
        elif family == "C_XYYX":
            probs["XY"] = probs["YX"] = p_pair / 2
    return probs


if __name__ == "__main__":
    lams = np.linspace(0.0, 1.0, 11)
    families = ["A_XXZZ", "B_ZZ", "C_XYYX"]
    curves = {f: {"lam": [], "F_prod": [], "F_ent": [], "S": []} for f in families}
    print(f"{'family':<10} {'lambda':>6} {'F_prod':>7} {'F_ent':>7} {'gain':>7} {'S_opt':>7}")
    for fam in families:
        for lam in lams:
            probs = build_channel(lam, fam)
            Fp = F_prod(probs)
            Fe, pars = F_ent(probs)
            S = S_entropy(pars)
            curves[fam]["lam"].append(lam)
            curves[fam]["F_prod"].append(Fp)
            curves[fam]["F_ent"].append(Fe)
            curves[fam]["S"].append(S)
            print(f"{fam:<10} {lam:>6.2f} {Fp:>7.3f} {Fe:>7.3f} {Fe - Fp:>+7.3f} {S:>7.3f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"A_XXZZ": "tab:blue", "B_ZZ": "tab:orange", "C_XYYX": "tab:green"}
    for fam in families:
        c = curves[fam]
        ax1.plot(c["lam"], c["S"], "o-", color=colors[fam], label=fam)
        gain = np.array(c["F_ent"]) - np.array(c["F_prod"])
        ax2.plot(c["lam"], gain, "s-", color=colors[fam], label=fam)
    ax1.set_xlabel("Correlation strength λ")
    ax1.set_ylabel("Optimal encoding entropy S (bits)")
    ax1.set_title("Entanglement of optimum vs noise correlation")
    ax1.axhline(0, ls=":", color="gray")
    ax1.axhline(1, ls=":", color="gray")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Correlation strength λ")
    ax2.set_ylabel("F_ent − F_prod")
    ax2.set_title("Fidelity gain from entanglement")
    ax2.axhline(0, ls=":", color="gray")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("correlation_entanglement_scaling.png", dpi=130)
    print("\nSaved: correlation_entanglement_scaling.png")
