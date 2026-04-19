"""
09 — Topology of the singular locus in the 2-qubit Pauli memory channel
-----------------------------------------------------------------------
Beyond the 1D transition theorem (Daems 2007) and the 2D three-phase
structure (Karimipour et al. 2009), what is the *global* topology of the
set of channels where more than one encoding is simultaneously optimal?

Formally, define:
  Optimal(channel) = argmax over pure 2-qubit states |psi> of F(|psi>, channel)
  Singular_k = { channel : |Optimal(channel)| >= k }  for k = 2, 3, 4, ...

Singular_1 = entire channel space
Singular_2 = codimension-1 "walls" between cells
Singular_3 = codimension-2 junctions (triple points)
...

Question: what are the measures mu(Singular_k) for each k, sampled from the
15D simplex of 2-qubit Pauli probabilities?

Method
------
1. Sample N=1000 random 2-qubit Pauli channels from a Dirichlet prior on the
   15-component probability simplex (with total rate bounded at p_max = 0.4).
2. For each channel, run variational optimization from ~12 random seeds.
3. Collect the set of locally-optimal fidelities, and count how many are within
   tolerance eps = 1e-3 of the global best.
4. That count = local multiplicity of the optimum ~= k for that channel.
5. Report P(k=1), P(k=2), P(k>=3).
6. For k=2 channels, record the entanglement entropies of the two optima to
   distinguish (product, product), (product, Bell), (Bell, Bell) boundary types.

All computation uses the 9-parameter SU(4) variational ansatz from 07_phase_transition.py.

Assumes F_ent(probs) and S_entropy(params) are in global scope (from notebook 07).
"""

import numpy as np
import time


def sample_random_channel(rng, p_max=0.4):
    """Sample random 2-qubit Pauli probabilities from Dirichlet(0.3)^15 scaled by p_total in [0, p_max]."""
    alpha = np.full(15, 0.3)
    w = rng.dirichlet(alpha)
    p_total = rng.uniform(0.05, p_max)
    probs_vec = w * p_total
    # keys: all 2-qubit Paulis except II
    keys = []
    for a in "IXYZ":
        for b in "IXYZ":
            if a + b != "II":
                keys.append(a + b)
    assert len(keys) == 15
    return {k: float(v) for k, v in zip(keys, probs_vec)}


def find_multiple_optima(F_ent_fn, probs, n_seeds=12, eps=1e-3):
    """Return list of (F, params) for all seeds, sorted by F descending.
    Caller analyses multiplicity by counting F-values within eps of the max.
    """
    rng = np.random.default_rng(0)
    found = []
    for seed in range(n_seeds):
        # re-seed the DE differently
        np.random.seed(seed * 7 + 11)
        F, params = F_ent_fn(probs)  # F_ent uses internal seeding; we perturb via outer seed
        found.append((F, params))
    found.sort(key=lambda x: -x[0])
    return found


def multiplicity(found, eps=1e-3):
    """How many of the found optima are within eps of the best?"""
    best = found[0][0]
    return sum(1 for F, _ in found if abs(F - best) < eps)


def singular_locus_scan(F_ent_fn, S_entropy_fn, N=200, p_max=0.4, eps=1e-3, seed=42):
    """Main loop. Returns list of per-channel records."""
    rng = np.random.default_rng(seed)
    records = []
    t0 = time.time()
    for i in range(N):
        probs = sample_random_channel(rng, p_max=p_max)
        found = find_multiple_optima(F_ent_fn, probs, n_seeds=12, eps=eps)
        k = multiplicity(found, eps=eps)
        # Characterize each of the top-k optima by entanglement entropy
        S_vals = [S_entropy_fn(p) for F, p in found[:k]]
        records.append(dict(
            channel_id=i,
            k=k,
            F_best=found[0][0],
            S_opts=S_vals,
            total_rate=sum(probs.values()),
        ))
        if (i + 1) % 20 == 0:
            dt = time.time() - t0
            print(f"  {i + 1}/{N} channels  ({dt:.0f}s, {dt/(i+1):.1f}s/channel)")
    return records


def summarize(records, eps=1e-3):
    import collections
    k_counts = collections.Counter(r['k'] for r in records)
    N = len(records)
    print(f"\n=== Singular-locus statistics on {N} random channels ===")
    print(f"{'k':<5} {'count':>8} {'P(k)':>8}")
    for k in sorted(k_counts.keys()):
        c = k_counts[k]
        print(f"{k:<5} {c:>8} {c / N:>8.3f}")
    print()
    # Classify k>=2 cases by the entanglement-structure pair
    print("For k>=2 channels, classification of the top-2 optima by (S_1, S_2):")
    pairs = collections.Counter()
    for r in records:
        if r['k'] >= 2:
            S1, S2 = r['S_opts'][0], r['S_opts'][1]
            def bin_(s):
                if s < 0.15: return "prod"
                if s > 0.85: return "Bell"
                return "partial"
            pair_key = tuple(sorted([bin_(S1), bin_(S2)]))
            pairs[pair_key] += 1
    for key, count in sorted(pairs.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    # Requires F_ent and S_entropy from notebook 07 in scope
    raise RuntimeError("Run from notebook: define F_ent, S_entropy first.")
