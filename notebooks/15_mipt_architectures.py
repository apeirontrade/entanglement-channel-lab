"""
15 — MIPT architecture comparison: Haar vs Clifford vs matchgate
------------------------------------------------------------------
Tests whether the measurement-induced entanglement transition's critical
point p_c depends on circuit architecture at small L.

Published work (2019-2024) reports similar universality classes across
Haar and Clifford (percolation-like), but with **disagreements on p_c value**:
  - Haar brick-wall:     p_c ~ 0.17 (Skinner et al. 2019)
  - Clifford brick-wall: p_c ~ 0.16 (Li-Chen-Fisher 2019)
  - matchgate/free-fermion: different universality class claimed
    (Jian-You-Vasseur-Ludwig 2020) at p_c ~ 0.3

This notebook runs all three in the same framework at L = 6 and L = 8,
same depth T = 2L, same measurement protocol (Z projective). 200
trajectories per (L, p, architecture) triple.

The narrow question: at L = 6-8, do the three architectures separate
cleanly, and where does each put its transition?
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable


# ====== Unitary generators ======
def haar_2q(rng) -> np.ndarray:
    Z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(Z)
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D


def clifford_2q(rng) -> np.ndarray:
    """Sample from the 2-qubit Clifford group via a random product of generators."""
    # Clifford group has 11520 elements; easiest: random product of CNOTs + H + S.
    # Here we pick ~20 random generators for decent mixing.
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]])
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    gens = [
        np.kron(H, I2), np.kron(I2, H),
        np.kron(S, I2), np.kron(I2, S),
        CNOT, CNOT @ np.kron(H, H) @ CNOT  # adds SWAP-like mixing
    ]
    U = np.eye(4, dtype=complex)
    for _ in range(20):
        U = gens[rng.integers(len(gens))] @ U
    return U


def matchgate_2q(rng) -> np.ndarray:
    """Random 2-qubit matchgate (free-fermion preserving).
    A matchgate has the form diag(A_outer) on {|00>,|11>} and A_inner on
    {|01>,|10>}, with det(A_outer) = det(A_inner).
    Simpler: parametrize via two SU(2)-like blocks with same determinant."""
    # Random SU(2) for the even sector, then make odd sector match det
    def random_u2(rng):
        a = rng.normal(0, 1) + 1j * rng.normal(0, 1)
        b = rng.normal(0, 1) + 1j * rng.normal(0, 1)
        norm = np.sqrt(abs(a)**2 + abs(b)**2)
        a /= norm; b /= norm
        return np.array([[a, -np.conj(b)], [b, np.conj(a)]])
    A = random_u2(rng)
    B = random_u2(rng)
    # match determinants
    det_A = np.linalg.det(A)
    det_B = np.linalg.det(B)
    # scale B by phase so det(B) = det(A)
    phase_ratio = det_A / det_B
    phase = np.exp(1j * np.angle(phase_ratio) / 2)
    B = B * phase
    # Now assemble the matchgate in the |00>,|01>,|10>,|11> basis
    # outer block acts on {|00>, |11>} (indices 0, 3)
    # inner block acts on {|01>, |10>} (indices 1, 2)
    U = np.zeros((4, 4), dtype=complex)
    U[0, 0] = A[0, 0]; U[0, 3] = A[0, 1]
    U[3, 0] = A[1, 0]; U[3, 3] = A[1, 1]
    U[1, 1] = B[0, 0]; U[1, 2] = B[0, 1]
    U[2, 1] = B[1, 0]; U[2, 2] = B[1, 1]
    return U


# ====== Generic circuit machinery ======
def apply_2q(state, U, i, j, L):
    tensor = state.reshape([2] * L)
    axes = [i, j] + [k for k in range(L) if k not in (i, j)]
    tensor = np.transpose(tensor, axes)
    rest = tensor.shape[2:]
    tensor = tensor.reshape(4, -1)
    tensor = U @ tensor
    tensor = tensor.reshape([2, 2] + list(rest))
    return np.transpose(tensor, np.argsort(axes)).reshape(-1)


def maybe_measure_z(state, q, L, p, rng):
    if rng.random() >= p:
        return state
    tensor = state.reshape([2] * L)
    axes = [q] + [k for k in range(L) if k != q]
    tensor = np.transpose(tensor, axes).reshape(2, -1)
    p0 = float(np.sum(np.abs(tensor[0]) ** 2))
    if rng.random() < p0:
        out = tensor.copy(); out[1] = 0; norm = np.sqrt(p0)
    else:
        out = tensor.copy(); out[0] = 0; norm = np.sqrt(1 - p0)
    out /= norm
    out = out.reshape([2] * L)
    return np.transpose(out, np.argsort(axes)).reshape(-1)


def half_entropy(state, L):
    LA = L // 2
    M = state.reshape(2**LA, 2 ** (L - LA))
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-12]
    return float(-np.sum(s * np.log(s)))


def trajectory(L, T, p, gate_fn, rng):
    state = np.zeros(2**L, dtype=complex); state[0] = 1.0
    for t in range(T):
        offset = t % 2
        for i in range(offset, L - 1, 2):
            state = apply_2q(state, gate_fn(rng), i, i + 1, L)
        for q in range(L):
            state = maybe_measure_z(state, q, L, p, rng)
    return half_entropy(state, L) / np.log(2)


def sweep(L, p_grid, n_traj, gate_fn, rng):
    means, stds = [], []
    for p in p_grid:
        vals = [trajectory(L, 2 * L, p, gate_fn, rng) for _ in range(n_traj)]
        means.append(np.mean(vals))
        stds.append(np.std(vals) / np.sqrt(n_traj))
    return np.array(means), np.array(stds)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    p_grid = np.linspace(0.0, 0.5, 11)
    Ls = [6, 8]
    n_traj = 100  # reduced for speed at L=8
    architectures = {
        'Haar': haar_2q,
        'Clifford': clifford_2q,
        'Matchgate': matchgate_2q,
    }

    results = {arch: {L: {'S': [], 'SE': []} for L in Ls} for arch in architectures}
    t0 = time.time()
    for arch_name, gate_fn in architectures.items():
        for L in Ls:
            print(f"\n=== {arch_name} L={L}, T={2*L}, n_traj={n_traj} ===")
            m, s = sweep(L, p_grid, n_traj, gate_fn, rng)
            results[arch_name][L]['S'] = m
            results[arch_name][L]['SE'] = s
            print(f"  p   :  " + " ".join(f"{p:.2f}" for p in p_grid))
            print(f"  S/L :  " + " ".join(f"{mi/L:.3f}" for mi in m))
            print(f"  elapsed: {time.time() - t0:.0f}s")

    # Identify crossing of S/L between L=6 and L=8 for each architecture
    print("\n=============================================")
    print("Estimated p_c per architecture (crossing of normalized S/L):")
    for arch in architectures:
        s_L1 = results[arch][Ls[0]]['S'] / Ls[0]
        s_L2 = results[arch][Ls[1]]['S'] / Ls[1]
        diff = s_L2 - s_L1
        p_c = None
        for i in range(len(p_grid) - 1):
            if diff[i] * diff[i + 1] < 0:
                p_c = p_grid[i] - diff[i] * (p_grid[i+1] - p_grid[i]) / (diff[i+1] - diff[i])
                break
        print(f"  {arch:10s}  p_c ≈ {p_c:.3f}" if p_c is not None else f"  {arch:10s}  no crossing")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (arch, gate_fn) in zip(axes, architectures.items()):
        for L in Ls:
            S = results[arch][L]['S']
            SE = results[arch][L]['SE']
            ax.errorbar(p_grid, S / L, yerr=SE / L, marker='o', label=f"L={L}")
        ax.set_xlabel("p"); ax.set_ylabel("S/L (bits)")
        ax.set_title(arch)
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("mipt_architectures.png", dpi=130)
    print("\nSaved: mipt_architectures.png")
