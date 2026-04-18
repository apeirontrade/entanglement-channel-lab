"""
03 — Variational encoding discovery
-----------------------------------
For each noise type in a panel, numerically find the single-qubit encoding
(parameterised as Ry(θ1) Rz(θ2) Ry(θ3)) that MAXIMIZES transmission
fidelity. Then train a tiny MLP to predict optimal angles from noise
parameters and test extrapolation on held-out noise.

Two open questions this is poking at:
  (a) Do the discovered encodings match textbook results where theory exists?
  (b) Does a small ML model find a latent map from noise -> optimal encoding?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from braket.circuits import Circuit
from braket.devices import LocalSimulator

SHOTS = 400


# ── encoding / decoding ──
def encode(c, q, t1, t2, t3): c.ry(q, t1).rz(q, t2).ry(q, t3); return c
def decode(c, q, t1, t2, t3): c.ry(q, -t3).rz(q, -t2).ry(q, -t1); return c


def apply_noise(c, q, params, rng):
    px, py, pz, _ = params
    r = rng.random()
    if r < px:
        c.x(q)
    elif r < px + py:
        c.y(q)
    elif r < px + py + pz:
        c.z(q)
    return c


def run_transmission(angles, noise_params, shots, device, rng):
    t1, t2, t3 = angles
    ok = 0
    for _ in range(shots):
        c = Circuit()
        encode(c, 0, t1, t2, t3)
        apply_noise(c, 0, noise_params, rng)
        decode(c, 0, t1, t2, t3)
        c.probability(target=[0])
        pr = device.run(c, shots=1).result().values[0]
        ok += int(rng.random() < pr[0])
    return ok / shots


def optimize_encoding(noise_params, device, seed=0):
    local_rng = np.random.default_rng(seed)
    def obj(angles):
        return -run_transmission(angles, noise_params, SHOTS, device, local_rng)
    res = differential_evolution(
        obj, bounds=[(0, 2 * np.pi)] * 3,
        maxiter=10, popsize=6, seed=seed, tol=1e-3, polish=False, workers=1,
    )
    return res.x, -res.fun


NOISE_PANEL = {
    "depolarizing_15":    (0.05, 0.05, 0.05, 0.0),
    "bit_flip_25":        (0.25, 0.0,  0.0,  0.0),
    "phase_flip_25":      (0.0,  0.0,  0.25, 0.0),
    "biased_bit_heavy":   (0.25, 0.02, 0.02, 0.0),
    "biased_phase_heavy": (0.02, 0.02, 0.25, 0.0),
    "XY_only":            (0.15, 0.15, 0.0,  0.0),
}


if __name__ == "__main__":
    device = LocalSimulator()
    rng = np.random.default_rng(0)

    results = {}
    print(f"{'noise':<22} {'θ1':>6} {'θ2':>6} {'θ3':>6} {'F_opt':>7} {'F_naive':>8} {'Δ':>7}")
    for name, params in NOISE_PANEL.items():
        (t1, t2, t3), F_opt = optimize_encoding(params, device, seed=42)
        F_naive = run_transmission((0, 0, 0), params, SHOTS, device, rng)
        gain = F_opt - F_naive
        results[name] = dict(theta=(t1, t2, t3), F_opt=F_opt, F_naive=F_naive, gain=gain)
        print(f"{name:<22} {t1:6.2f} {t2:6.2f} {t3:6.2f} {F_opt:7.3f} {F_naive:8.3f} {gain:+7.3f}")

    # ── MLP extrapolation ──
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        X = np.array([NOISE_PANEL[k] for k in results])
        y = np.array([results[k]["theta"] for k in results])
        scaler = StandardScaler().fit(X)
        mlp = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=5000, random_state=0).fit(scaler.transform(X), y)
        print(f"\nMLP train R²: {mlp.score(scaler.transform(X), y):.3f}")

        NEW = {
            "mostly_dephasing": (0.02, 0.02, 0.18, 0.0),
            "isotropic_mild":   (0.07, 0.07, 0.07, 0.0),
            "bit_dominant":     (0.20, 0.0,  0.02, 0.0),
        }
        print(f"{'noise':<22} {'predicted θ':>24} {'F_pred':>7} {'F_brute':>8} {'gap':>6}")
        for name, params in NEW.items():
            pred = mlp.predict(scaler.transform([params]))[0]
            F_pred = run_transmission(pred, params, SHOTS, device, rng)
            _, F_brute = optimize_encoding(params, device, seed=99)
            print(f"{name:<22} ({pred[0]:5.2f},{pred[1]:5.2f},{pred[2]:5.2f}) {F_pred:7.3f} {F_brute:8.3f} {F_brute-F_pred:+6.3f}")
    except ImportError:
        print("\n(sklearn not available — skipping MLP step)")

    # ── plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    thetas = np.array([results[k]["theta"] for k in names])
    gains = np.array([results[k]["gain"] for k in names])
    sc = ax.scatter(thetas[:, 0], thetas[:, 1], s=250, c=gains, cmap="plasma",
                    edgecolor="k", linewidth=0.7)
    for i, n in enumerate(names):
        ax.annotate(n, (thetas[i, 0], thetas[i, 1]), fontsize=8,
                    xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("θ₁ (rad)"); ax.set_ylabel("θ₂ (rad)")
    ax.set_title("Discovered optimal encodings (color = gain vs naive)")
    plt.colorbar(sc, ax=ax, label="Fidelity gain")
    plt.tight_layout()
    plt.savefig("encoding_discovery.png", dpi=130)
    print("\nSaved: encoding_discovery.png")
