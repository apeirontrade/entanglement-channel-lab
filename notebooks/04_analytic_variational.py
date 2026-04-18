"""
04 — Analytic variational encoding discovery
--------------------------------------------
For single-qubit Pauli noise, the encoding-decoding fidelity has a closed form
in the Bloch angles (alpha, beta). We optimize it directly instead of calling
a Braket simulator — saves roughly 1000× in runtime and sidesteps a pydantic
schema bug we hit on the notebook instance during this session.

For single-qubit Pauli noise with error probabilities (p_x, p_y, p_z)
and encoded pure state |psi(alpha, beta)> = U(alpha, beta) |0>:

    F(alpha, beta) = (1 - p_x - p_y - p_z)
                     + p_x * sin^2(alpha) * cos^2(beta)
                     + p_y * sin^2(alpha) * sin^2(beta)
                     + p_z * cos^2(alpha)

This closed form lets us (a) sweep a 9-channel noise panel in milliseconds,
(b) plot the full fidelity landscape, and (c) train a small MLP to predict
optimal encodings from noise parameters using 500 training points.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


def fidelity_analytic(alpha, beta, px, py, pz):
    sa2 = np.sin(alpha) ** 2
    ca2 = np.cos(alpha) ** 2
    cb2 = np.cos(beta) ** 2
    sb2 = np.sin(beta) ** 2
    return (1 - px - py - pz) + px * sa2 * cb2 + py * sa2 * sb2 + pz * ca2


def optimize_analytic(px, py, pz):
    def obj(angles):
        a, b = angles
        return -fidelity_analytic(a, b, px, py, pz)
    res = differential_evolution(
        obj,
        bounds=[(0, np.pi), (0, 2 * np.pi)],
        maxiter=50, popsize=10, seed=0, tol=1e-9, polish=True,
    )
    return res.x, -res.fun


NOISE_PANEL = {
    "depolarizing":       (0.05, 0.05, 0.05),
    "bit_flip_25":        (0.25, 0.00, 0.00),
    "phase_flip_25":      (0.00, 0.00, 0.25),
    "Y_only_25":          (0.00, 0.25, 0.00),
    "biased_bit_heavy":   (0.25, 0.02, 0.02),
    "biased_phase_heavy": (0.02, 0.02, 0.25),
    "XY_equal":           (0.15, 0.15, 0.00),
    "XZ_equal":           (0.15, 0.00, 0.15),
    "YZ_equal":           (0.00, 0.15, 0.15),
}


def axis_label(alpha, beta):
    if abs(np.sin(alpha)) < 0.1:
        return "Z-axis (|0>/|1>)"
    if abs(alpha - np.pi / 2) < 0.3:
        if abs(np.cos(beta)) > 0.9:
            return "X-axis (|+>/|->)"
        if abs(np.sin(beta)) > 0.9:
            return "Y-axis (|+i>/|-i>)"
        return f"equator β={beta:.2f}"
    return f"generic α={alpha:.2f}"


if __name__ == "__main__":
    results = {}
    print(f"{'noise':<22} {'α':>6} {'β':>6} {'F_opt':>7} {'F_naive':>8} {'Δ':>7}  axis")
    for name, (px, py, pz) in NOISE_PANEL.items():
        (alpha, beta), F_opt = optimize_analytic(px, py, pz)
        F_naive = fidelity_analytic(0, 0, px, py, pz)
        gain = F_opt - F_naive
        results[name] = dict(alpha=alpha, beta=beta, F_opt=F_opt, F_naive=F_naive, gain=gain)
        print(f"{name:<22} {alpha:6.2f} {beta:6.2f} {F_opt:7.3f} {F_naive:8.3f} {gain:+7.3f}  {axis_label(alpha, beta)}")

    # Landscape plot
    alphas = np.linspace(0, np.pi, 80)
    betas = np.linspace(0, 2 * np.pi, 160)
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for ax, (name, (px, py, pz)) in zip(axes.flat, NOISE_PANEL.items()):
        F = fidelity_analytic(A, B, px, py, pz)
        im = ax.imshow(F, origin="lower", aspect="auto",
                       extent=[0, 2 * np.pi, 0, np.pi],
                       cmap="viridis", vmin=0.5, vmax=1.0)
        r = results[name]
        ax.plot(r["beta"], r["alpha"], "r*", markersize=18, markeredgecolor="white")
        ax.plot(0, 0, "wo", markersize=10, markeredgecolor="red")
        ax.set_title(f"{name}  (Δ={r['gain']:+.3f})", fontsize=10)
        ax.set_xlabel("β"); ax.set_ylabel("α")
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig("fidelity_landscape.png", dpi=130)
    print("Saved: fidelity_landscape.png")

    # MLP — 500 random training points
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        X_train = rng.uniform(0, 0.3, (500, 3))
        y_train = np.zeros((500, 2))
        for i, p in enumerate(X_train):
            (a, b), _ = optimize_analytic(*p)
            y_train[i] = [a, b]
        scaler = StandardScaler().fit(X_train)
        mlp = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=10000,
                           random_state=0).fit(scaler.transform(X_train), y_train)
        print(f"\nMLP R²: {mlp.score(scaler.transform(X_train), y_train):.3f}")

        X_test = rng.uniform(0, 0.3, (100, 3))
        errs = []
        for p in X_test:
            (_, _), F_true = optimize_analytic(*p)
            a_pred, b_pred = mlp.predict(scaler.transform([p]))[0]
            F_pred = fidelity_analytic(a_pred, b_pred, *p)
            errs.append(F_true - F_pred)
        errs = np.array(errs)
        print(f"MLP test fidelity gap: mean={errs.mean():.4f}, max={errs.max():.4f}")
        print(f"Within 0.005 of optimum: {(errs < 0.005).mean() * 100:.1f}%")
    except ImportError:
        print("sklearn not available")
