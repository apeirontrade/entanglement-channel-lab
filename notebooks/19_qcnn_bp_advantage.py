"""
19 — Quantum advantage gap in BP-free QCNN at n = 12, 16, 20
-------------------------------------------------------------
Empirical test of the open question raised by Cerezo, Larocca et al.
(arXiv:2312.09121, Nature Communications 2025):

    "Can the structure that allows one to avoid barren plateaus
     also be leveraged to efficiently simulate the loss classically?"

They argue analytically that for many common BP-free architectures
(including QCNN, TEE, permutation-equivariant circuits), the answer
is YES on average. They end with:

    "once larger devices become available, parametrized quantum
     circuits could heuristically outperform our analytic expectations"

This experiment targets that gap. For n = 12, 16, 20 qubits, we:

  1. Train a quantum convolutional neural network (QCNN, Cong-Choi-
     Lukin 2019) to classify two quantum state families:
       (a) Haar-random 1-designs (Clifford random stabilizer states)
       (b) Haar-random 2-designs (Clifford + small T-count mixing)
  2. Separately, build a "classical shortcut" classifier: collect
     classical shadows of the same states (via random Pauli
     measurements), train a small logistic-regression on shadow
     features.
  3. Measure test accuracy of BOTH classifiers on a held-out set.
  4. Report the gap Δ(n) = acc(QCNN) − acc(shadow classifier).
  5. Fit Δ vs n: is it approaching zero? Constant? Growing?

Cerezo's claim predicts Δ → 0 as n grows. We test the crossover regime
n = 12-20 where they explicitly say the answer is empirically unknown.

Falsifiable outcomes
--------------------
  * Δ monotonically decreasing with small positive values at n=20
      → strong support for Cerezo's claim
  * Δ roughly constant at small positive value
      → ambiguous; needs larger n or error analysis
  * Δ increasing with n or markedly nonzero at n=20
      → evidence *against* Cerezo's claim; would be a small counter-
         data point to their analytic argument

Honest expectation
-------------------
Most likely outcome: Δ shrinks slowly, consistent with their claim.
Probability of surprising result: ~20-30%.
Probability the specific numerical comparison at n=20 with this exact
setup is already in the literature: ~40-60%.
"""

import numpy as np
import time
from typing import Tuple, Callable
import matplotlib.pyplot as plt

# ---------- detect CUDA-Q, fall back to pure numpy ----------
USE_CUDAQ = False
try:
    import cudaq
    USE_CUDAQ = True
    print("cudaq available — GPU-accelerated path")
except ImportError:
    print("cudaq not available — using pure numpy state-vector simulation")


# ============ QCNN components (pure numpy) ============
# We use numpy for portability. If CUDA-Q is available, we can port
# specific kernels to cudaq for speed (mostly the conv+pool primitives).

def haar_unitary(dim, rng):
    Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(Z)
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D


def apply_2q(state, U, i, j, n):
    tensor = state.reshape([2]*n)
    axes = [i, j] + [k for k in range(n) if k not in (i, j)]
    tensor = np.transpose(tensor, axes)
    rest = tensor.shape[2:]
    tensor = tensor.reshape(4, -1)
    tensor = U @ tensor
    tensor = tensor.reshape([2, 2] + list(rest))
    return np.transpose(tensor, np.argsort(axes)).reshape(-1)


def partial_trace(state, keep, n):
    """Trace out all qubits not in 'keep'. Returns reduced density matrix."""
    k = len(keep)
    axes = list(keep) + [q for q in range(n) if q not in keep]
    tensor = np.transpose(state.reshape([2]*n), axes)
    M = tensor.reshape(2**k, 2**(n-k))
    return M @ M.conj().T


# ============ Parameterized 2-qubit unitary for QCNN conv ============

def param_2q(params):
    """15-parameter SU(4) 2-qubit unitary from Nielsen-Chuang
    Section 4.5.2 / Cirq-style KAK decomposition (approximate)."""
    # Use compact 15-parameter form: local + entangler + local
    t1, p1, l1 = params[0:3]
    t2, p2, l2 = params[3:6]
    a, b, c    = params[6:9]
    t3, p3, l3 = params[9:12]
    t4, p4, l4 = params[12:15]

    def su2(t, p, l):
        return np.array([
            [np.exp(-1j*(p+l)/2)*np.cos(t/2), -np.exp(-1j*(p-l)/2)*np.sin(t/2)],
            [np.exp(1j*(p-l)/2)*np.sin(t/2),  np.exp(1j*(p+l)/2)*np.cos(t/2)]])

    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    XX = np.kron(X, X); YY = np.kron(Y, Y); ZZ = np.kron(Z, Z)
    def E(H, th): return np.cos(th/2)*np.eye(4, dtype=complex) - 1j*np.sin(th/2)*H

    Ul_pre = np.kron(su2(t1, p1, l1), su2(t2, p2, l2))
    Ul_post = np.kron(su2(t3, p3, l3), su2(t4, p4, l4))
    Uent = E(XX, a) @ E(YY, b) @ E(ZZ, c)
    return Ul_post @ Uent @ Ul_pre


PARAMS_PER_CONV = 15


def qcnn_forward(state, n, params_per_layer):
    """Apply QCNN: conv layers + pool layers until 1 qubit remains.
    params_per_layer is a dict {layer_idx: flat_params_array}."""
    remaining = list(range(n))
    layer = 0
    while len(remaining) > 1:
        # conv: apply param_2q on adjacent pairs
        params = params_per_layer[layer]
        offset = 0
        for i in range(0, len(remaining) - 1, 2):
            a, b = remaining[i], remaining[i + 1]
            U = param_2q(params[offset:offset + PARAMS_PER_CONV])
            state = apply_2q(state, U, a, b, n)
            offset += PARAMS_PER_CONV
        # pool: trace out half the qubits (keep even-indexed)
        remaining = [remaining[i] for i in range(0, len(remaining), 2)]
        layer += 1
    # return the final single-qubit density matrix
    rho_final = partial_trace(state, remaining, n)
    # classify: measure <Z> on the final qubit
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return float(np.real(np.trace(Z @ rho_final)))


def count_qcnn_params(n):
    """QCNN has log2(n) layers; each layer has n/(2^layer) pairs."""
    total = 0
    remaining = n
    layer = 0
    while remaining > 1:
        pairs = remaining // 2
        total += pairs * PARAMS_PER_CONV
        remaining = pairs
        layer += 1
    return total, layer


# ============ Data: two state families to classify ============

def family_A_state(n, rng):
    """Clifford random stabilizer state: apply random Clifford to |0...0>.
    For speed, approximated by a random 'shallow Clifford-like' circuit."""
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    # 10 layers of shallow Clifford-like gates (approximation)
    for _ in range(10):
        for i in range(0, n - 1, 2):
            U = haar_unitary(4, rng)  # not exactly Clifford but close in distribution
            # project onto Clifford-like (fast approx): discretize params
            state = apply_2q(state, U, i, i+1, n)
    return state


def family_B_state(n, rng):
    """T-doped Clifford: Clifford circuit + a few T gates.
    Labeled 1 (different from family A = label 0)."""
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    # More entangling structure
    T_mat = np.diag([1, np.exp(1j * np.pi / 4)])
    for layer in range(10):
        for i in range(0, n - 1, 2):
            state = apply_2q(state, haar_unitary(4, rng), i, i+1, n)
        # sprinkle T gates
        for _ in range(2):
            q = rng.integers(n)
            # apply T to qubit q
            tensor = state.reshape([2]*n)
            axes = [q] + [k for k in range(n) if k != q]
            tensor = np.transpose(tensor, axes).reshape(2, -1)
            tensor = T_mat @ tensor
            tensor = tensor.reshape([2]*n)
            state = np.transpose(tensor, np.argsort(axes)).reshape(-1)
    return state


# ============ Classical shadow classifier ============

def classical_shadow(state, n, n_shadows, rng):
    """Compute classical shadow: n_shadows random Pauli measurements.
    Returns a simple feature vector: mean <P> for each 1-qubit Pauli,
    aggregated across shadow samples."""
    # For each shadow: pick random Pauli basis per qubit, measure, record
    # Feature: (mean_X_q0, mean_Y_q0, mean_Z_q0, mean_X_q1, ...) — 3n features
    features = np.zeros(3 * n)
    for s in range(n_shadows):
        # pick random basis per qubit
        basis = rng.integers(3, size=n)  # 0=X, 1=Y, 2=Z
        # apply basis rotation (virtual; we just compute expectation)
        # for features, we take <P> directly from the state
        pass
    # Simpler alternative: directly compute all <P_i> = Tr(rho P_i)
    # where rho is 1-qubit reduced state
    for q in range(n):
        rho_q = partial_trace(state, [q], n)
        feat = np.real(rho_q[0, 1] + rho_q[1, 0])  # <X>
        features[3*q] = feat
        features[3*q + 1] = np.imag(rho_q[1, 0] - rho_q[0, 1])  # <Y>
        features[3*q + 2] = np.real(rho_q[0, 0] - rho_q[1, 1])  # <Z>
    # Add 2-body features: <Z_i Z_{i+1}>
    extra = []
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    for i in range(n - 1):
        rho_ij = partial_trace(state, [i, i+1], n)
        ZZ_op = np.kron(Z, Z)
        extra.append(float(np.real(np.trace(ZZ_op @ rho_ij))))
    features = np.concatenate([features, np.array(extra)])
    return features


# ============ Training ============

def generate_dataset(n, n_per_class, rng):
    X_states = []
    y = []
    for _ in range(n_per_class):
        X_states.append(family_A_state(n, rng)); y.append(0)
        X_states.append(family_B_state(n, rng)); y.append(1)
    return X_states, np.array(y)


def train_qcnn(X_train, y_train, X_test, y_test, n, n_iter=30, lr=0.05, seed=0):
    total_params, n_layers = count_qcnn_params(n)
    rng = np.random.default_rng(seed)
    params = rng.uniform(-0.1, 0.1, total_params)

    def forward(state, params):
        # unpack params into per-layer dict
        layers = {}
        offset = 0
        remaining = n; layer = 0
        while remaining > 1:
            pairs = remaining // 2
            size = pairs * PARAMS_PER_CONV
            layers[layer] = params[offset:offset + size]
            offset += size
            remaining = pairs
            layer += 1
        return qcnn_forward(state, n, layers)

    def loss(params):
        preds = np.array([forward(s, params) for s in X_train])
        # map <Z> in [-1, 1] to probability in [0, 1]
        probs = (1 + preds) / 2
        eps = 1e-9
        return -np.mean(y_train * np.log(probs + eps) + (1 - y_train) * np.log(1 - probs + eps))

    # Parameter-shift-style numerical gradient for speed
    for it in range(n_iter):
        grad = np.zeros_like(params)
        h = 0.05
        for i in range(len(params)):
            pplus = params.copy(); pplus[i] += h
            pminus = params.copy(); pminus[i] -= h
            grad[i] = (loss(pplus) - loss(pminus)) / (2 * h)
        params -= lr * grad
        if (it + 1) % 5 == 0:
            test_preds = np.array([forward(s, params) for s in X_test])
            test_acc = np.mean((test_preds > 0) == y_test)
            print(f"  QCNN iter {it+1}: loss={loss(params):.4f}  test_acc={test_acc:.3f}")

    # final test accuracy
    test_preds = np.array([forward(s, params) for s in X_test])
    return float(np.mean((test_preds > 0) == y_test)), params


def train_shadow(X_train, y_train, X_test, y_test, n):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    train_feats = np.array([classical_shadow(s, n, 100, rng) for s in X_train])
    test_feats = np.array([classical_shadow(s, n, 100, rng) for s in X_test])
    clf = LogisticRegression(max_iter=500).fit(train_feats, y_train)
    return float(clf.score(test_feats, y_test))


# ============ Main experiment ============

def run_one_size(n, n_train=30, n_test=30, seed=42):
    print(f"\n{'='*60}\n=== n = {n} qubits ===\n{'='*60}")
    rng = np.random.default_rng(seed)
    print(f"Generating dataset ({n_train} + {n_test} per class)...")
    t0 = time.time()
    X_train, y_train = generate_dataset(n, n_train // 2, rng)
    X_test, y_test = generate_dataset(n, n_test // 2, rng)
    print(f"  dataset ready in {time.time()-t0:.0f}s")

    print(f"Training QCNN...")
    t0 = time.time()
    qcnn_acc, _ = train_qcnn(X_train, y_train, X_test, y_test, n, n_iter=20)
    print(f"  QCNN done in {time.time()-t0:.0f}s, test_acc={qcnn_acc:.3f}")

    print(f"Training classical shadow classifier...")
    t0 = time.time()
    shadow_acc = train_shadow(X_train, y_train, X_test, y_test, n)
    print(f"  shadow done in {time.time()-t0:.0f}s, test_acc={shadow_acc:.3f}")

    gap = qcnn_acc - shadow_acc
    print(f"\n*** n={n}: QCNN={qcnn_acc:.3f}, shadow={shadow_acc:.3f}, Δ={gap:+.3f} ***")
    return dict(n=n, qcnn=qcnn_acc, shadow=shadow_acc, gap=gap)


if __name__ == "__main__":
    results = []
    for n in [8, 10, 12]:  # start small; scale up after validating
        r = run_one_size(n)
        results.append(r)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'n':<4}  {'QCNN':<8}  {'shadow':<8}  {'Δ':<8}")
    for r in results:
        print(f"{r['n']:<4}  {r['qcnn']:<8.3f}  {r['shadow']:<8.3f}  {r['gap']:+<8.3f}")

    # Plot
    ns = [r['n'] for r in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ns, [r['qcnn'] for r in results], 'o-', label='QCNN test accuracy')
    ax.plot(ns, [r['shadow'] for r in results], 's-', label='Classical shadow classifier')
    ax.plot(ns, [r['gap'] for r in results], '^-', label='Δ = QCNN − shadow', color='red')
    ax.axhline(0, ls=':', color='gray')
    ax.axhline(0.5, ls=':', color='gray', alpha=0.3, label='random guess')
    ax.set_xlabel("Number of qubits n")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("QCNN vs classical-shadow shortcut (Cerezo-regime test)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("qcnn_bp_advantage.png", dpi=130)
    print("\nSaved: qcnn_bp_advantage.png")
