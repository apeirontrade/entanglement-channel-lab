"""
02 — Channel duality experiment
-------------------------------
Compare teleportation (T), direct qubit transmission (D), and superdense
coding (S) under two noise types:

  * depolarizing: random X / Y / Z with prob p/3 each
  * dephasing  : pure Z with prob p

Sweep p from 0.0 -> 0.5, plot all three protocols side-by-side, save PNG.

Result (weekend finding): for symmetric single-qubit Pauli noise, T ≈ D
(entanglement gives no fidelity advantage) and S decays faster due to its
4-outcome decoding.
"""

import numpy as np
import matplotlib.pyplot as plt
from braket.circuits import Circuit
from braket.devices import LocalSimulator

SHOTS = 1500
NOISE_GRID = np.linspace(0.0, 0.5, 11)
THETA, PHI = np.pi / 3, np.pi / 4


# ── primitives ──
def prep(c, q, t, p): c.ry(q, t).rz(q, p); return c
def unprep(c, q, t, p): c.rz(q, -p).ry(q, -t); return c
def bell(c, a, b): c.h(a).cnot(a, b); return c


def depol(c, q, p, rng):
    if rng.random() < p:
        getattr(c, rng.choice(["x", "y", "z"]))(q)
    return c


def dephase(c, q, p, rng):
    if rng.random() < p:
        c.z(q)
    return c


# ── circuits ──
def C_T(theta, phi, p, rng, noise_fn):
    c = Circuit()
    prep(c, 0, theta, phi); bell(c, 1, 2)
    noise_fn(c, 2, p, rng)
    c.cnot(0, 1).h(0).cnot(1, 2).cz(0, 2)
    unprep(c, 2, theta, phi)
    c.probability(target=[2]); return c


def C_Dq(theta, phi, p, rng, noise_fn):
    c = Circuit()
    prep(c, 0, theta, phi)
    noise_fn(c, 0, p, rng)
    unprep(c, 0, theta, phi)
    c.probability(target=[0]); return c


def C_S(bits, p, rng, noise_fn):
    b1, b0 = bits
    c = Circuit()
    bell(c, 0, 1)
    if b0: c.x(0)
    if b1: c.z(0)
    noise_fn(c, 0, p, rng)
    c.cnot(0, 1).h(0)
    c.probability(target=[0, 1]); return c


# ── sweep ──
def run_sweep(noise_fn, label):
    device = LocalSimulator()
    rng = np.random.default_rng(42)
    T, D, S = [], [], []
    for p in NOISE_GRID:
        okT = okD = okS = 0
        for _ in range(SHOTS):
            pr = device.run(C_T(THETA, PHI, p, rng, noise_fn), shots=1).result().values[0]
            okT += int(rng.random() < pr[0])
            pr = device.run(C_Dq(THETA, PHI, p, rng, noise_fn), shots=1).result().values[0]
            okD += int(rng.random() < pr[0])
            bits = (int(rng.integers(2)), int(rng.integers(2)))
            pr = device.run(C_S(bits, p, rng, noise_fn), shots=1).result().values[0]
            idx = int(rng.choice(4, p=pr))
            b1h, b0h = (idx >> 1) & 1, idx & 1
            if (b1h, b0h) == bits:
                okS += 1
        T.append(okT / SHOTS); D.append(okD / SHOTS); S.append(okS / SHOTS)
        print(f"[{label}] p={p:.2f}  F_T={T[-1]:.3f}  F_D={D[-1]:.3f}  F_S={S[-1]:.3f}")
    return np.array(T), np.array(D), np.array(S)


if __name__ == "__main__":
    print("=== Depolarizing ===")
    T_dep, D_dep, S_dep = run_sweep(depol, "depol")
    print("\n=== Dephasing ===")
    T_phz, D_phz, S_phz = run_sweep(dephase, "dephase")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (T, D, S), title in zip(
        axes,
        [(T_dep, D_dep, S_dep), (T_phz, D_phz, S_phz)],
        ["Depolarizing", "Dephasing (Z-only)"],
    ):
        ax.plot(NOISE_GRID, T, "o-", label="Teleportation")
        ax.plot(NOISE_GRID, D, "s--", label="Direct qubit")
        ax.plot(NOISE_GRID, S, "^:", label="Superdense")
        ax.set_title(f"{title} noise")
        ax.set_xlabel("noise p")
        ax.grid(alpha=0.3); ax.legend()
    axes[0].set_ylabel("Fidelity / Accuracy")
    plt.tight_layout()
    plt.savefig("channel_duality.png", dpi=130)
    print("\nSaved channel_duality.png")
