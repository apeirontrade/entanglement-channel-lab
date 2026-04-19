"""
14 — Measurement-induced entanglement phase transition: small-L numerics
-------------------------------------------------------------------------
The MIPT (Li-Fisher-Chen 2018, Skinner-Ruhman-Nahum 2019, Chan-Nandkishore-
Pretko-Smith 2019) is a phase transition in random monitored circuits:
  p < p_c : volume-law entanglement (S ~ L)
  p > p_c : area-law entanglement (S ~ const)
  p = p_c : critical (conformal) scaling

Published p_c for random Haar brick-wall: ~0.17 (Skinner et al., Gullans et al.)
Published p_c for random Clifford:        ~0.16

Open question researchers argue about: do small system sizes (L = 4-12)
give reliable estimates of p_c? Some papers say yes (p_c converges fast);
others report L-dependent shifts of 0.05 or more.

This notebook: systematic sweep at L = 4, 5, 6, 7, 8 for Haar brick-wall
with projective Z-measurements. Output:
  - S_half(p) curves for each L
  - p_c(L) crossing points
  - check convergence toward published p_c ~ 0.17

Honest caveats:
- Running L <= 8 is below where published asymptotic papers live (L ~ 16-128)
- 200 trajectories per (L, p) is on the low end for clean statistics
- We're contributing a small-L systematic reference, not a new critical exponent
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple


def haar_2q(rng) -> np.ndarray:
    """Haar-random 4x4 unitary via QR decomposition."""
    Z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(Z)
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D


def apply_2q(state: np.ndarray, U: np.ndarray, i: int, j: int, L: int) -> np.ndarray:
    tensor = state.reshape([2] * L)
    axes = [i, j] + [k for k in range(L) if k not in (i, j)]
    tensor = np.transpose(tensor, axes)
    rest = tensor.shape[2:]
    tensor = tensor.reshape(4, -1)
    tensor = U @ tensor
    tensor = tensor.reshape([2, 2] + list(rest))
    return np.transpose(tensor, np.argsort(axes)).reshape(-1)


def maybe_measure_z(state: np.ndarray, q: int, L: int, p: float, rng) -> np.ndarray:
    if rng.random() >= p:
        return state
    tensor = state.reshape([2] * L)
    axes = [q] + [k for k in range(L) if k != q]
    tensor = np.transpose(tensor, axes).reshape(2, -1)
    p0 = float(np.sum(np.abs(tensor[0]) ** 2))
    if rng.random() < p0:
        out = tensor.copy()
        out[1] = 0
        norm = np.sqrt(p0)
    else:
        out = tensor.copy()
        out[0] = 0
        norm = np.sqrt(1 - p0)
    out /= norm
    out = out.reshape([2] * L)
    return np.transpose(out, np.argsort(axes)).reshape(-1)


def half_chain_entropy(state: np.ndarray, L: int) -> float:
    LA = L // 2
    M = state.reshape(2**LA, 2 ** (L - LA))
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-12]
    return float(-np.sum(s * np.log(s)))


def one_trajectory(L: int, T: int, p: float, rng) -> float:
    state = np.zeros(2**L, dtype=complex)
    state[0] = 1.0
    for t in range(T):
        offset = t % 2
        for i in range(offset, L - 1, 2):
            state = apply_2q(state, haar_2q(rng), i, i + 1, L)
        for q in range(L):
            state = maybe_measure_z(state, q, L, p, rng)
    return half_chain_entropy(state, L) / np.log(2)  # in bits


def sweep(L: int, p_grid: np.ndarray, n_traj: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    T = 2 * L  # depth = 2L; this is the usual choice for MIPT studies
    means = []
    stds = []
    for p in p_grid:
        vals = [one_trajectory(L, T, p, rng) for _ in range(n_traj)]
        means.append(np.mean(vals))
        stds.append(np.std(vals) / np.sqrt(n_traj))
    return np.array(means), np.array(stds)


def estimate_p_c_crossing(Ls: list, p_grid: np.ndarray, S_by_L: dict) -> float:
    """Find p where S/L curves for different L cross — the putative critical point."""
    # Use the two largest L values; find p where normalized S agrees
    L1, L2 = Ls[-2], Ls[-1]
    s1 = S_by_L[L1] / L1  # normalized
    s2 = S_by_L[L2] / L2
    # Look for sign change of (s2 - s1) = 0
    diff = s2 - s1
    for i in range(len(p_grid) - 1):
        if diff[i] * diff[i + 1] < 0:
            # linear interp
            p0, p1 = p_grid[i], p_grid[i + 1]
            d0, d1 = diff[i], diff[i + 1]
            return float(p0 - d0 * (p1 - p0) / (d1 - d0))
    return float('nan')


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    p_grid = np.linspace(0.0, 0.4, 9)  # coarse for speed
    Ls = [4, 5, 6, 7, 8]
    n_traj = 40  # small for initial pass

    S_by_L = {}
    SE_by_L = {}
    t0 = time.time()
    for L in Ls:
        print(f"\n=== L = {L}, depth T = {2*L}, n_traj = {n_traj} ===")
        means, stds = sweep(L, p_grid, n_traj, rng)
        S_by_L[L] = means
        SE_by_L[L] = stds
        print(f"  p values: {[f'{p:.2f}' for p in p_grid]}")
        print(f"  S_half:   {[f'{m:.3f}' for m in means]}")
        print(f"  elapsed: {time.time() - t0:.0f}s")

    # Crossing estimate
    try:
        p_c = estimate_p_c_crossing(Ls, p_grid, S_by_L)
        print(f"\n*** Crossing of L={Ls[-2]} and L={Ls[-1]} curves: p_c ~ {p_c:.3f}")
        print(f"    Published asymptotic value (Skinner et al. 2019): p_c ~ 0.17")
    except Exception as e:
        print(f"Crossing calculation: {e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for L in Ls:
        axes[0].errorbar(p_grid, S_by_L[L], yerr=SE_by_L[L], marker='o', label=f"L={L}")
        axes[1].errorbar(p_grid, S_by_L[L] / L, yerr=SE_by_L[L] / L, marker='o', label=f"L={L}")
    axes[0].axhline(Ls[-1] / 2, ls=':', alpha=0.3, label='volume law (L/2)')
    axes[0].set_xlabel("measurement rate p")
    axes[0].set_ylabel("S_half (bits)")
    axes[0].set_title("Absolute scaling")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("measurement rate p")
    axes[1].set_ylabel("S_half / L")
    axes[1].set_title("Normalized (look for crossings)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("mipt_small_L.png", dpi=130)
    print("\nSaved: mipt_small_L.png")
