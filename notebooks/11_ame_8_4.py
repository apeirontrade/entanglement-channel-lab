"""
11 — Variational search for an AME(8, 4) state
------------------------------------------------
IQOQI Open Problem #35 (revived list; cite [Huber 2017/05/19]): for what
parameters (N, d) do absolutely-maximally-entangled (AME) pure states exist?

A state |psi> on N d-level systems is AME iff every reduction obtained by
tracing out at least floor(N/2) parties is maximally mixed (rho = I/d^k,
where k is the size of the remaining subsystem).

Status of small cases (as of IQOQI problem page, ~2025):
  AME(4, 2): NO  (proved, Higuchi-Sudbery 2000)
  AME(5, 2): yes
  AME(6, 2): yes
  AME(7, 2): yes
  AME(8, 2): yes
  AME(4, 6): YES (Rather et al. 2022, used quantum Latin squares)
  AME(8, 4): **UNRESOLVED** -- one of the smallest open cases
  AME(7, 6): **UNRESOLVED**

This notebook attempts a variational search for AME(8, 4).

Dimensionality
--------------
  state vector: ℂ^(4^8) = 65,536 complex amplitudes (~1 MB)
  reductions to check: 8 choose 4 = 70 bipartitions
  each reduction: trace out 4 parties -> 4^4 = 256-dim reduced state

Ansatz
------
  Brick-wall of 2-body local-dimension-4 unitaries.
  SU(16) has 255 real parameters; use a reduced parameterization
  (3 layers of nearest-neighbor U(4,4) unitaries, each ~50 parameters)
  -> ~300-500 total variational parameters.

  Alternatively (used here): parameterize each layer's unitary with the
  standard QR-on-random-Gaussian method, and let differential evolution
  search in the generating matrix space.

Cost function
-------------
  For each of the 70 bipartitions (A, B) with |A| = 4:
    rho_A = Tr_B(|psi><psi|)
    deviation_i = ||rho_A - I/d^|A| ||_F^2
  Objective = max(deviation_i)  (we want ALL bipartitions maximally mixed)

  AME condition: objective = 0.

Novelty and scope caveat
-------------------------
This is a named open problem with many serious researchers having tried.
Very likely someone has already run a similar variational search and
failed to find AME(8, 4). A successful outcome would be surprising and
would need verification. A null outcome is normal and has no claim on
novelty.

The value of this script is:
  1) it runs the search honestly on our tool;
  2) a negative result adds one more data point to the "probably doesn't
     exist" pile;
  3) the infrastructure generalizes to other (N, d) we might be curious
     about in the future (e.g., AME(6, 4), AME(5, 3)).
"""

from __future__ import annotations
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution
import time

# ---------- problem parameters ----------
N_SYSTEMS = 8
D = 4
STATE_DIM = D ** N_SYSTEMS  # 65536

BIPARTITIONS = [tuple(sorted(c)) for c in combinations(range(N_SYSTEMS), N_SYSTEMS // 2)]
print(f"Target: AME({N_SYSTEMS}, {D}). state_dim = {STATE_DIM}. #bipartitions = {len(BIPARTITIONS)}.")

# ---------- Haar-random unitary ----------
def haar_unitary(dim, rng):
    Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(Z)
    D_ph = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D_ph


# ---------- ansatz: brick-wall of 2-body ququart-ququart unitaries ----------
# Each 2-body unitary is 16x16; needs 255 real parameters for full SU(16).
# We compress via: params -> complex matrix M -> QR decomposition -> unitary.
# This uses 16*16*2 = 512 reals per unitary, with QR killing dependencies.
# Layer 1: (0,1), (2,3), (4,5), (6,7)  -- 4 unitaries
# Layer 2: (1,2), (3,4), (5,6)          -- 3 unitaries
# Layer 3: (0,1), (2,3), (4,5), (6,7)  -- 4 more unitaries for depth
# Total blocks: 11; reals per block: 512; total params: 11 * 512 = 5632 -- too many.
#
# Compression: use a smaller Kraus-style parameterization per block:
# each block U is built as exp(i * H) where H is a 16x16 Hermitian with 255 params.
# We reduce to a subset using generating matrices with limited support.

REAL_PER_BLOCK = 120  # use 120 parameters per 2-body block (reduced from 255 full)


def block_unitary_from_params(params, rng_for_structure=None):
    """Build a 16x16 (ququart ⊗ ququart) unitary from 120 real parameters.
    Strategy: fix a random basis of 120 Hermitian generators (Pauli-like on 16x16)
    and exponentiate a linear combination."""
    generators = _get_block_generators()
    H = np.zeros((16, 16), dtype=complex)
    for p, G in zip(params, generators):
        H = H + p * G
    # exp(i H) via eigendecomposition (H is Hermitian)
    eigs, V = np.linalg.eigh(H)
    return V @ np.diag(np.exp(1j * eigs)) @ V.conj().T


_block_generators_cache = None
def _get_block_generators():
    """Return a fixed list of 120 16x16 Hermitian matrices (orthonormal basis subset)."""
    global _block_generators_cache
    if _block_generators_cache is not None:
        return _block_generators_cache
    # Use generalized Gell-Mann matrices on 16 dim; there are 255 total.
    # We select the first 120 (off-diagonal symmetric + antisymmetric, some diagonal).
    mats = []
    dim = 16
    for a in range(dim):
        for b in range(a + 1, dim):
            # symmetric
            G = np.zeros((dim, dim), dtype=complex)
            G[a, b] = G[b, a] = 1
            mats.append(G)
            # antisymmetric
            G = np.zeros((dim, dim), dtype=complex)
            G[a, b] = -1j
            G[b, a] = 1j
            mats.append(G)
            if len(mats) >= 120:
                break
        if len(mats) >= 120:
            break
    # normalize
    for i, M in enumerate(mats):
        mats[i] = M / np.sqrt(np.real((M @ M.conj().T).trace()))
    _block_generators_cache = mats
    return mats


# Brick-wall structure: layers of nearest-neighbor 2-body blocks
BRICKS = [
    # layer 1 (even pairs)
    [(0, 1), (2, 3), (4, 5), (6, 7)],
    # layer 2 (odd pairs)
    [(1, 2), (3, 4), (5, 6)],
    # layer 3 (even pairs again for depth)
    [(0, 1), (2, 3), (4, 5), (6, 7)],
]

N_BLOCKS = sum(len(L) for L in BRICKS)
N_PARAMS = N_BLOCKS * REAL_PER_BLOCK
print(f"Ansatz: {N_BLOCKS} 2-body blocks, {N_PARAMS} real parameters total.")


def apply_2body(state, U, i, j):
    """Apply 16x16 unitary U on sites (i, j) of an N-site state (d^N dim vector).
    Sites are labeled 0..N-1; state is reshaped as d^N tensor."""
    N = N_SYSTEMS
    d = D
    tensor = state.reshape([d] * N)
    # move axes (i, j) to front
    axes = [i, j] + [k for k in range(N) if k != i and k != j]
    tensor = np.transpose(tensor, axes)
    rest_shape = tensor.shape[2:]
    tensor = tensor.reshape(d * d, -1)
    tensor = U @ tensor
    tensor = tensor.reshape([d, d] + list(rest_shape))
    inv_axes = np.argsort(axes)
    tensor = np.transpose(tensor, inv_axes)
    return tensor.reshape(-1)


def build_state_from_params(params):
    """Apply brick-wall ansatz starting from |0...0>."""
    state = np.zeros(STATE_DIM, dtype=complex)
    state[0] = 1.0
    p_iter = iter(range(N_BLOCKS))
    offset = 0
    for layer in BRICKS:
        for (i, j) in layer:
            U = block_unitary_from_params(params[offset:offset + REAL_PER_BLOCK])
            state = apply_2body(state, U, i, j)
            offset += REAL_PER_BLOCK
    # normalize (should be unit by construction but re-normalize to be safe)
    state = state / np.linalg.norm(state)
    return state


def reduced_state(state, keep_sites):
    """Compute rho_A = Tr_B(|state><state|) where A = keep_sites.
    Returns 4^|A| x 4^|A| density matrix."""
    N = N_SYSTEMS
    d = D
    k = len(keep_sites)
    trace_sites = [s for s in range(N) if s not in keep_sites]
    tensor = state.reshape([d] * N)
    # move kept sites to front, trace sites to back
    axes = list(keep_sites) + list(trace_sites)
    tensor = np.transpose(tensor, axes)
    tensor = tensor.reshape(d ** k, d ** (N - k))
    rho = tensor @ tensor.conj().T
    return rho


def max_mixed_deviation(state):
    """Return max over bipartitions of || rho_A - I/d^|A| ||_F^2."""
    d = D
    target_dim = d ** (N_SYSTEMS // 2)
    target = np.eye(target_dim, dtype=complex) / target_dim
    deviations = []
    for keep in BIPARTITIONS:
        rho = reduced_state(state, keep)
        delta = rho - target
        deviations.append(float(np.real(np.sum(np.abs(delta) ** 2))))
    return max(deviations), np.mean(deviations)


def objective(params):
    state = build_state_from_params(params)
    max_dev, mean_dev = max_mixed_deviation(state)
    return max_dev  # minimize the worst-bipartition deviation


# ---------- run variational search ----------
def run_search(n_restarts=3, local_iters=200, seed=0):
    rng = np.random.default_rng(seed)
    best_obj = float('inf')
    best_params = None
    t0 = time.time()
    for restart in range(n_restarts):
        print(f"\n=== Restart {restart + 1}/{n_restarts} ===")
        x0 = rng.normal(0, 0.3, N_PARAMS)
        # local optimizer is the practical choice here; differential_evolution is too slow at 1320 params
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': local_iters, 'disp': True, 'gtol': 1e-7})
        dt = time.time() - t0
        print(f"  restart {restart + 1}: final objective = {res.fun:.6f}, "
              f"elapsed = {dt:.0f}s")
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
    return best_params, best_obj


if __name__ == "__main__":
    print("\nSanity check: random state baseline")
    rng0 = np.random.default_rng(1)
    random_state = rng0.standard_normal(STATE_DIM) + 1j * rng0.standard_normal(STATE_DIM)
    random_state /= np.linalg.norm(random_state)
    mx, mn = max_mixed_deviation(random_state)
    print(f"Random Gaussian pure state: max dev = {mx:.4f}, mean = {mn:.4f}")

    print("\nSanity check: computational basis state |0...0>")
    basis = np.zeros(STATE_DIM, dtype=complex)
    basis[0] = 1.0
    mx, mn = max_mixed_deviation(basis)
    print(f"|0...0> product state: max dev = {mx:.4f}, mean = {mn:.4f}")

    print("\nBeginning variational search...")
    best_params, best_obj = run_search(n_restarts=2, local_iters=150)

    print("\n=============================================================")
    print(f"Best objective (max F-norm deviation across 70 bipartitions): {best_obj:.6f}")
    print("AME requires this to be exactly 0.")
    if best_obj < 1e-4:
        print(">>> CANDIDATE AME(8, 4) STATE FOUND. Requires rigorous verification! <<<")
    elif best_obj < 0.01:
        print("    Very close to AME. Could indicate existence but optimization not fully converged.")
    elif best_obj < 0.1:
        print("    Significant partial progress but not AME.")
    else:
        print("    Far from AME. Null result for this seed; search budget was small.")

    print("\nPer-bipartition deviations at optimum:")
    state = build_state_from_params(best_params)
    target_dim = D ** (N_SYSTEMS // 2)
    target = np.eye(target_dim, dtype=complex) / target_dim
    for keep in BIPARTITIONS[:10]:
        rho = reduced_state(state, keep)
        delta = float(np.real(np.sum(np.abs(rho - target) ** 2)))
        print(f"  subset {keep}: deviation = {delta:.5f}")
    print("  ... (60 more)")
