"""
06 — Measurement-Induced Phase Transition (MIPT)
-------------------------------------------------
Reproduce the entanglement phase transition in random monitored circuits.

Protocol:
  1. Initialize L qubits in |0...0>
  2. Apply a brick-wall layer of random 2-qubit Haar-random unitaries
  3. After each layer, each qubit is projectively measured in Z with prob p
  4. Repeat for T layers
  5. Measure entanglement entropy S(rho_A) for A = left half of system
  6. Average over many trajectories; sweep p; sweep L

Expected behavior:
  - p=0: volume-law entanglement, S -> L/2 * log(2)
  - p=0.5: area-law entanglement, S ~ const
  - p_c near 0.16 for random Clifford, 0.15-0.25 for Haar (1D brick wall)

System sizes L=4,6,8 with T=2L depth; typically suffices to observe the
volume -> area transition even if extracting exponents requires bigger L.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iproduct


def random_unitary_2q(rng):
    """Haar-random 2-qubit unitary via QR of complex Ginibre."""
    Z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(Z)
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D


def apply_2q(state, U, i, j, L):
    """Apply a 2-qubit unitary U on qubits (i, j) of an L-qubit state."""
    # Reshape state into a tensor with L legs of dim 2
    tensor = state.reshape([2] * L)
    # Move i, j to the front
    axes = [i, j] + [k for k in range(L) if k != i and k != j]
    tensor = np.transpose(tensor, axes)
    shape_rest = tensor.shape[2:]
    tensor = tensor.reshape(4, -1)
    tensor = U @ tensor
    tensor = tensor.reshape([2, 2] + list(shape_rest))
    # Move back
    inv_axes = np.argsort(axes)
    tensor = np.transpose(tensor, inv_axes)
    return tensor.reshape(-1)


def maybe_measure_z(state, q, L, p, rng):
    """With probability p, measure qubit q in the Z basis and collapse."""
    if rng.random() >= p:
        return state
    # Compute P(|0> on q) = sum of |amp|^2 over states where q-bit is 0
    tensor = state.reshape([2] * L)
    # Put qubit q first
    axes = [q] + [k for k in range(L) if k != q]
    tensor = np.transpose(tensor, axes)
    tensor = tensor.reshape(2, -1)
    p0 = np.sum(np.abs(tensor[0]) ** 2)
    outcome = 0 if rng.random() < p0 else 1
    # Project
    if outcome == 0:
        tensor[1] = 0.0
        norm = np.sqrt(p0)
    else:
        tensor[0] = 0.0
        norm = np.sqrt(1 - p0)
    tensor = tensor / norm
    tensor = tensor.reshape([2] * L)
    inv_axes = np.argsort(axes)
    tensor = np.transpose(tensor, inv_axes)
    return tensor.reshape(-1)


def von_neumann_half_chain_entropy(state, L):
    """S(rho_A) where A = left L/2 qubits."""
    LA = L // 2
    LB = L - LA
    M = state.reshape(2**LA, 2**LB)
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    s2 = s2[s2 > 1e-12]
    return float(-np.sum(s2 * np.log(s2)))  # natural log; convert to bits below if needed


def monitored_circuit_entropy(L, T, p, rng):
    """Run one trajectory: brick-wall random unitaries with mid-circuit Z
    measurements at rate p, return final half-chain entropy."""
    state = np.zeros(2**L, dtype=complex)
    state[0] = 1.0
    for t in range(T):
        # Brick wall: even pairs at even t, odd pairs at odd t
        offset = t % 2
        pairs = [(i, i + 1) for i in range(offset, L - 1, 2)]
        for (i, j) in pairs:
            U = random_unitary_2q(rng)
            state = apply_2q(state, U, i, j, L)
        for q in range(L):
            state = maybe_measure_z(state, q, L, p, rng)
    return von_neumann_half_chain_entropy(state, L)


def sweep(L, T, p_grid, n_trials, rng):
    """Return array of mean entropy per p, and std error."""
    means = []
    stds = []
    for p in p_grid:
        S_vals = [monitored_circuit_entropy(L, T, p, rng) for _ in range(n_trials)]
        means.append(np.mean(S_vals))
        stds.append(np.std(S_vals) / np.sqrt(n_trials))
    return np.array(means), np.array(stds)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p_grid = np.linspace(0.0, 0.5, 11)
    N_TRIALS = 30  # trajectories averaged per p

    configs = [(4, 8), (6, 12), (8, 16)]  # (L, T)

    fig, ax = plt.subplots(figsize=(7, 5))
    all_data = {}
    for (L, T) in configs:
        print(f"Sweeping L={L}, T={T}...")
        means, stds = sweep(L, T, p_grid, N_TRIALS, rng)
        # convert to bits for readability
        means_bits = means / np.log(2)
        stds_bits = stds / np.log(2)
        ax.errorbar(p_grid, means_bits, yerr=stds_bits, marker='o', label=f"L={L}")
        all_data[L] = (p_grid, means_bits, stds_bits)
        for p, m in zip(p_grid, means_bits):
            print(f"  p={p:.2f}  S={m:.3f} bits")

    # Reference lines
    for L in [4, 6, 8]:
        ax.axhline(L / 2, ls=':', color='gray', alpha=0.5)
        ax.text(0.48, L / 2 + 0.05, f"volume law S=L/2 ({L})", fontsize=7, color='gray')

    ax.set_xlabel("Measurement rate p")
    ax.set_ylabel("Half-chain entanglement entropy S (bits)")
    ax.set_title("MIPT: volume→area transition in monitored random circuits")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("mipt.png", dpi=130)
    print("\nSaved: mipt.png")

    # Rough estimate of critical point: p where S/(L/2) drops through 0.5
    print("\nRough p_c estimates (where S crosses 50% of max):")
    for L, (ps, ms, _) in all_data.items():
        max_S = L / 2
        half = max_S / 2
        # linear interp
        for i in range(len(ps) - 1):
            if ms[i] >= half >= ms[i + 1]:
                p0, p1 = ps[i], ps[i + 1]
                s0, s1 = ms[i], ms[i + 1]
                p_c = p0 + (half - s0) * (p1 - p0) / (s1 - s0)
                print(f"  L={L}:  p_c ≈ {p_c:.3f}")
                break
