"""
06 — 2D phase boundary in 2-qubit Pauli memory channel

For a 2-parameter family of Pauli memory channels:
  lambda in [0, 1]  -- correlation strength
  mu     in [0, 1]  -- mix: XX+ZZ (mu=0) vs XY+YX (mu=1) for the pair errors

fixed total error rate p = 0.25.

At each (lambda, mu) point, variationally optimize a 9-parameter 2-qubit
encoding; record the entanglement entropy S(rho_A) of the discovered optimum.

Empirical finding (this run, 2026-04-18): the locus S=0.5 forms a
piecewise-constant (staircase) boundary with three plateaus:

  mu in [0.0, 0.5]:  lambda* = 0.55
  mu in [0.6, 0.7]:  lambda* = 0.45
  mu in [0.8, 1.0]:  lambda* = 0.35

See docs/PHASE_BOUNDARY_2D.md for interpretation.

Assumes F_ent and S_entropy from notebooks/07_phase_transition.py are
available.
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def build_channel_2d(lam, mu, p_total=0.25):
    p_solo = (1 - lam) * p_total
    p_pair = lam * p_total
    probs = {}
    if p_solo > 0:
        probs["XI"] = probs["IX"] = probs["ZI"] = probs["IZ"] = p_solo / 4
    if p_pair > 0:
        p_xxzz = (1 - mu) * p_pair
        p_xyyx = mu * p_pair
        if p_xxzz > 0:
            probs["XX"] = probs.get("XX", 0) + p_xxzz / 2
            probs["ZZ"] = probs.get("ZZ", 0) + p_xxzz / 2
        if p_xyyx > 0:
            probs["XY"] = probs.get("XY", 0) + p_xyyx / 2
            probs["YX"] = probs.get("YX", 0) + p_xyyx / 2
    return probs


def sweep_2d(F_ent, S_entropy, grid_n=11):
    lams = np.linspace(0.0, 1.0, grid_n)
    mus = np.linspace(0.0, 1.0, grid_n)
    S = np.zeros((grid_n, grid_n))
    F = np.zeros((grid_n, grid_n))
    t0 = time.time()
    for i, lam in enumerate(lams):
        for j, mu in enumerate(mus):
            probs = build_channel_2d(lam, mu)
            Fopt, pars = F_ent(probs)
            S[i, j] = S_entropy(pars)
            F[i, j] = Fopt
        print(f"row {i+1}/{grid_n} done  ({int(time.time()-t0)}s)")
    return lams, mus, S, F


def extract_boundary(lams, mus, S, level=0.5):
    """For each mu, linearly interpolate the smallest lam where S crosses level."""
    out = []
    for j, mu in enumerate(mus):
        col = S[:, j]
        idx = np.where(col > level)[0]
        if len(idx) == 0:
            out.append((mu, None))
            continue
        i = idx[0]
        if i == 0:
            out.append((mu, lams[0]))
            continue
        s0, s1 = S[i - 1, j], S[i, j]
        l0, l1 = lams[i - 1], lams[i]
        lam_c = l0 + (level - s0) * (l1 - l0) / (s1 - s0)
        out.append((mu, lam_c))
    return out


def plot(lams, mus, S, F, path="phase_boundary_2d.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    im1 = ax1.imshow(S, origin='lower', aspect='auto',
                     extent=[0, 1, 0, 1], cmap='plasma', vmin=0, vmax=1)
    ax1.contour(mus, lams, S, levels=[0.5], colors='white', linewidths=2)
    ax1.set_xlabel("mu")
    ax1.set_ylabel("lambda")
    ax1.set_title("S(rho_A) of optimal encoding")
    plt.colorbar(im1, ax=ax1, label="S (bits)")
    im2 = ax2.imshow(F, origin='lower', aspect='auto',
                     extent=[0, 1, 0, 1], cmap='viridis', vmin=0.7, vmax=1.0)
    ax2.set_xlabel("mu")
    ax2.set_ylabel("lambda")
    ax2.set_title("F_opt")
    plt.colorbar(im2, ax=ax2, label="F")
    plt.tight_layout()
    plt.savefig(path, dpi=130)


if __name__ == "__main__":
    # Expects F_ent, S_entropy to be imported from 07_phase_transition.py
    from importlib import import_module
    m = import_module("07_phase_transition")
    F_ent = m.F_ent
    S_entropy = m.S_entropy
    lams, mus, S, F = sweep_2d(F_ent, S_entropy)
    plot(lams, mus, S, F)
    print()
    for mu, lam_c in extract_boundary(lams, mus, S):
        if lam_c is None:
            print(f"mu={mu:.2f}: no transition")
        else:
            print(f"mu={mu:.2f}: lambda* = {lam_c:.3f}")
