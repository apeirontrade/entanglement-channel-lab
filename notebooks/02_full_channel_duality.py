"""
Entanglement-Assisted Channel Duality — Teleportation / Superdense / Direct

Experiment outline
------------------
Alice and Bob share one resource: a Bell pair (two qubits, pre-entangled).
Depending on runtime choice, they use it to transmit either:

  (T)  a QUBIT from A -> B via teleportation
       cost: 1 Bell pair + 2 classical bits
  (S)  2 CLASSICAL BITS from A -> B via superdense coding
       cost: 1 Bell pair + 1 qubit sent
  (D)  direct transmission (NO Bell pair used) as a baseline:
       - for qubits: ship the qubit straight over
       - for bits:   ship 2 bits straight over

We then inject a noise channel on the "channel qubit" (the qubit that
physically travels between Alice and Bob) and measure how the success
probability of each protocol degrades.

The "crossover" is the noise strength p* at which an entanglement-assisted
protocol becomes worse than the direct baseline. Finding p* on hardware
is a clean, measurable, self-contained physics result.

Requires:  pip install amazon-braket-sdk numpy matplotlib
Optional:  pip install amazon-braket-default-simulator  (bundled with SDK)
"""

from __future__ import annotations
import itertools
import math
import numpy as np
from dataclasses import dataclass
from typing import Literal, Iterable

from braket.circuits import Circuit
from braket.devices import LocalSimulator

# ---------- knobs you'll actually tune ----------
SHOTS = 2000
NOISE_GRID = np.linspace(0.0, 0.5, 11)  # 0 to 50% depolarizing on channel qubit
THETA, PHI = np.pi / 3, np.pi / 4       # the test qubit state parameters


# ----------------------------------------------------------------------
# State prep / verification helpers
# ----------------------------------------------------------------------

def prep_state(c: Circuit, q: int, theta: float, phi: float) -> Circuit:
    """Prepare |psi> = Rz(phi) Ry(theta) |0> on qubit q."""
    c.ry(q, theta).rz(q, phi)
    return c


def unprep_state(c: Circuit, q: int, theta: float, phi: float) -> Circuit:
    """Inverse of prep_state — rotates |psi> back to |0> so P(|0>) = fidelity."""
    c.rz(q, -phi).ry(q, -theta)
    return c


def bell_pair(c: Circuit, a: int, b: int) -> Circuit:
    """Create a Bell pair |Phi+> = (|00> + |11>) / sqrt(2) on (a, b)."""
    c.h(a).cnot(a, b)
    return c


# ----------------------------------------------------------------------
# Noise model — a depolarizing channel on the "channel qubit"
# ----------------------------------------------------------------------
# Braket's LocalSimulator doesn't apply noise on arbitrary gates, so we
# emulate depolarizing noise by: with prob p/3 each, apply X, Y, or Z on
# the channel qubit just before it's used by the receiver. Over many shots
# this samples the depolarizing channel E(rho) = (1-p)rho + (p/3)(XrhoX + YrhoY + ZrhoZ).

def apply_depolarizing(c: Circuit, q: int, p: float, rng: np.random.Generator) -> Circuit:
    """Sample one Pauli error and apply it (or identity) — one shot's worth."""
    if rng.random() >= p:
        return c
    pauli = rng.choice(["x", "y", "z"])
    getattr(c, pauli)(q)
    return c


# ----------------------------------------------------------------------
# Protocol: (T) TELEPORTATION  — send |psi> from Alice -> Bob
# ----------------------------------------------------------------------
# Qubits:
#   q0 = Alice's test state |psi>
#   q1 = Alice's half of Bell pair
#   q2 = Bob's half of Bell pair (will receive |psi>)
#
# We use deferred-measurement form: CNOT/CZ play the role of measure+correct.
# We inject depolarizing noise on q2 (the qubit that "traveled" to Bob).

def circuit_T(theta: float, phi: float, p: float, rng: np.random.Generator) -> Circuit:
    c = Circuit()
    prep_state(c, 0, theta, phi)
    bell_pair(c, 1, 2)
    # noise on the channel qubit (q2 physically sits at Bob but came from a shared source)
    apply_depolarizing(c, 2, p, rng)
    # Bell-basis measurement on (q0, q1)
    c.cnot(0, 1).h(0)
    # deferred corrections on q2
    c.cnot(1, 2).cz(0, 2)
    # verify: invert prep on q2 — if teleportation succeeded, q2 -> |0>
    unprep_state(c, 2, theta, phi)
    c.probability(target=[2])
    return c


# ----------------------------------------------------------------------
# Protocol: (S) SUPERDENSE CODING — send 2 classical bits A -> B
# ----------------------------------------------------------------------
# Alice has 2 bits (b1, b0). She encodes them onto her half of a Bell pair
# via I, X, Z, or XZ, then sends that qubit to Bob who does the Bell decode.
# Cost: 1 Bell pair + 1 qubit sent over the channel (which is what picks up noise).

def circuit_S(bits: tuple[int, int], p: float, rng: np.random.Generator) -> Circuit:
    b1, b0 = bits
    c = Circuit()
    bell_pair(c, 0, 1)       # q0 = Alice, q1 = Bob
    # Alice encodes 2 bits on q0 via Pauli
    if b0:
        c.x(0)
    if b1:
        c.z(0)
    # Alice sends q0 to Bob -- noise hits q0 in transit
    apply_depolarizing(c, 0, p, rng)
    # Bob decodes in the Bell basis (invert entangler)
    c.cnot(0, 1).h(0)
    # Bob now measures q0,q1 -> (b1, b0)
    c.probability(target=[0, 1])
    return c


# ----------------------------------------------------------------------
# Protocol: (D) DIRECT — baseline with NO entanglement
# ----------------------------------------------------------------------
# (Dq) Send the qubit |psi> directly over a noisy channel.
# (Dc) Send 2 classical bits directly over a noisy classical channel.
#       For a fair comparison we let the classical channel be a BSC with
#       flip probability p_c = p * 2/3 (matches depolarizing bit-flip rate).

def circuit_Dq(theta: float, phi: float, p: float, rng: np.random.Generator) -> Circuit:
    """Direct qubit transmission — prep, noise, unprep."""
    c = Circuit()
    prep_state(c, 0, theta, phi)
    apply_depolarizing(c, 0, p, rng)
    unprep_state(c, 0, theta, phi)
    c.probability(target=[0])
    return c


def classical_BSC(bits: tuple[int, int], p: float, rng: np.random.Generator) -> tuple[int, int]:
    """Classical binary symmetric channel with flip prob p_c = 2p/3 per bit."""
    p_c = 2 * p / 3
    b1, b0 = bits
    if rng.random() < p_c:
        b1 ^= 1
    if rng.random() < p_c:
        b0 ^= 1
    return (b1, b0)


# ----------------------------------------------------------------------
# Runner — shot-by-shot sampling because our noise is a Monte Carlo channel
# ----------------------------------------------------------------------

def run_T(device, theta, phi, p, shots, rng) -> float:
    """Return teleportation fidelity = P(q2 == |0>) averaged across shots."""
    # Because depolarizing noise is sampled per shot, we build & run one
    # circuit per shot -- slow on hardware, fine on local sim for this size.
    successes = 0
    for _ in range(shots):
        circ = circuit_T(theta, phi, p, rng)
        probs = device.run(circ, shots=1).result().values[0]
        # result.values[0] is the joint prob over q2 alone
        # sample one outcome from it
        successes += int(rng.random() < probs[0])
    return successes / shots


def run_S(device, p, shots, rng) -> float:
    """Superdense accuracy = fraction of (b1, b0) correctly decoded."""
    successes = 0
    for _ in range(shots):
        bits = (int(rng.integers(2)), int(rng.integers(2)))
        circ = circuit_S(bits, p, rng)
        probs = device.run(circ, shots=1).result().values[0]
        # probs indexed by integer q0q1 in little-endian -- decode to (b1, b0)
        outcome = int(rng.choice(4, p=probs))
        # Braket probability ordering: index = q0 * 2 + q1  (convention check)
        b0_hat = outcome & 1          # q1
        b1_hat = (outcome >> 1) & 1   # q0
        if (b1_hat, b0_hat) == bits:
            successes += 1
    return successes / shots


def run_Dq(device, theta, phi, p, shots, rng) -> float:
    successes = 0
    for _ in range(shots):
        circ = circuit_Dq(theta, phi, p, rng)
        probs = device.run(circ, shots=1).result().values[0]
        successes += int(rng.random() < probs[0])
    return successes / shots


def run_Dc(p, shots, rng) -> float:
    successes = 0
    for _ in range(shots):
        bits = (int(rng.integers(2)), int(rng.integers(2)))
        got = classical_BSC(bits, p, rng)
        if got == bits:
            successes += 1
    return successes / shots


# ----------------------------------------------------------------------
# Main — sweep noise, compare protocols, plot
# ----------------------------------------------------------------------

@dataclass
class SweepResult:
    p_grid: np.ndarray
    F_teleport: np.ndarray      # qubit-level fidelity
    F_direct_q: np.ndarray
    F_superdense: np.ndarray    # classical-bit-level accuracy
    F_direct_c: np.ndarray

    def crossover_quantum(self) -> float | None:
        """Smallest p where direct qubit beats teleportation (worse-case)."""
        diff = self.F_direct_q - self.F_teleport
        sign_change = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_change) == 0:
            return None
        i = sign_change[0]
        # linear interp
        p0, p1 = self.p_grid[i], self.p_grid[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        return float(p0 - d0 * (p1 - p0) / (d1 - d0))

    def crossover_classical(self) -> float | None:
        diff = self.F_direct_c - self.F_superdense
        sign_change = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_change) == 0:
            return None
        i = sign_change[0]
        p0, p1 = self.p_grid[i], self.p_grid[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        return float(p0 - d0 * (p1 - p0) / (d1 - d0))


def sweep(device=None, shots=SHOTS, p_grid=NOISE_GRID, theta=THETA, phi=PHI, seed=0) -> SweepResult:
    device = device or LocalSimulator()
    rng = np.random.default_rng(seed)
    T, Dq, S, Dc = [], [], [], []
    for p in p_grid:
        T.append(run_T(device, theta, phi, p, shots, rng))
        Dq.append(run_Dq(device, theta, phi, p, shots, rng))
        S.append(run_S(device, p, shots, rng))
        Dc.append(run_Dc(p, shots, rng))
        print(f"p={p:.2f}  F_T={T[-1]:.3f}  F_Dq={Dq[-1]:.3f}  "
              f"F_S={S[-1]:.3f}  F_Dc={Dc[-1]:.3f}")
    return SweepResult(
        p_grid=np.asarray(p_grid),
        F_teleport=np.asarray(T),
        F_direct_q=np.asarray(Dq),
        F_superdense=np.asarray(S),
        F_direct_c=np.asarray(Dc),
    )


def plot_results(res: SweepResult, path: str = "channel_duality.png") -> None:
    import matplotlib.pyplot as plt
    fig, (axq, axc) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    axq.plot(res.p_grid, res.F_teleport, "o-", label="Teleportation (EA)")
    axq.plot(res.p_grid, res.F_direct_q, "s--", label="Direct qubit")
    axq.set_title("Qubit channel: EA vs direct")
    axq.set_xlabel("Depolarizing noise p")
    axq.set_ylabel("Success / fidelity")
    axq.legend(); axq.grid(alpha=0.3)

    axc.plot(res.p_grid, res.F_superdense, "o-", label="Superdense (EA)")
    axc.plot(res.p_grid, res.F_direct_c, "s--", label="Direct 2-bit (BSC)")
    axc.set_title("Classical-bit channel: EA vs direct")
    axc.set_xlabel("Depolarizing noise p")
    axc.legend(); axc.grid(alpha=0.3)

    pxQ = res.crossover_quantum()
    pxC = res.crossover_classical()
    if pxQ is not None:
        axq.axvline(pxQ, ls=":", c="k"); axq.text(pxQ, 0.55, f"p*={pxQ:.2f}")
    if pxC is not None:
        axc.axvline(pxC, ls=":", c="k"); axc.text(pxC, 0.55, f"p*={pxC:.2f}")

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"saved {path}")


if __name__ == "__main__":
    res = sweep()
    print()
    print(f"Quantum crossover p* ≈ {res.crossover_quantum()}")
    print(f"Classical crossover p* ≈ {res.crossover_classical()}")
    try:
        plot_results(res)
    except ImportError:
        print("matplotlib not installed; skipping plot")
