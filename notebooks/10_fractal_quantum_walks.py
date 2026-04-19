"""
10 — Beyond spectral dimension: a second invariant for quantum walks on fractals?
---------------------------------------------------------------------------------
Darázs et al. (Phys. Rev. E 90, 032113, 2014) showed that the spectral dimension
d_s alone does NOT determine the evolution of continuous-time quantum walks
(CTQW) on fractals -- unlike the classical case where d_s uniquely characterizes
the Polya-type return probability scaling.

Question: given that d_s is insufficient, is there a _second_ topological
invariant that, together with d_s, does predict the quantum-walk return
probability exponent alpha?  Candidates tested:
  - branching ratio (max degree / min degree on the fractal)
  - mean chemical distance (= mean shortest-path length / system size)
  - spectral gap (= lowest nonzero Laplacian eigenvalue)
  - node count / mean degree

Method
------
1. Build 4-6 small fractal lattices (generation 2 or 3) with various
   dimensions.
2. Compute graph Laplacian, construct CTQW propagator U(t) = exp(-i L t).
3. Starting from a corner node, evolve for T steps and measure |<0|psi(t)>|^2.
4. Fit P_return(t) ~ a t^(-alpha) + b in the long-time tail.
5. Correlate alpha with each candidate invariant via simple linear regression.
6. Report which (if any) invariant significantly improves prediction beyond d_s.

All on classical NumPy - no Braket needed. ~3-5 minutes runtime.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# -------- Fractal adjacency matrices --------

def sierpinski_gasket(generation):
    """Sierpinski triangle / gasket via recursive construction.
    Returns adjacency matrix and node count.
    """
    # Start with a single triangle (3 nodes, all connected)
    nodes = [(0, 0), (1, 0), (0.5, np.sqrt(3)/2)]
    edges = [(0, 1), (1, 2), (0, 2)]

    for _ in range(generation):
        # Scale existing + make 2 more copies translated
        scale = 0.5
        new_nodes = [(x*scale, y*scale) for (x, y) in nodes]
        base_len = len(nodes)
        # copy 2: shifted right
        offset_x = max(n[0] for n in new_nodes) - min(n[0] for n in new_nodes)
        new_nodes += [(x*scale + offset_x + scale, y*scale) for (x, y) in nodes]
        # copy 3: shifted up-right (centered on top vertex)
        offset_x2 = offset_x / 2 + scale / 2
        offset_y2 = (offset_x) * np.sqrt(3)/2
        new_nodes += [(x*scale + offset_x2, y*scale + offset_y2) for (x, y) in nodes]

        # Build new edge list (edges within each copy)
        new_edges = []
        for (a, b) in edges:
            new_edges.append((a, b))
            new_edges.append((a + base_len, b + base_len))
            new_edges.append((a + 2*base_len, b + 2*base_len))

        # Identify and merge touching corner nodes between copies
        # (for small generations, approximate merge by rounded coords)
        coord_map = {}
        merged_idx = []
        eps = 1e-6
        for i, (x, y) in enumerate(new_nodes):
            key = (round(x / scale * 1000) / 1000, round(y / scale * 1000) / 1000)
            if key in coord_map:
                merged_idx.append(coord_map[key])
            else:
                coord_map[key] = len(coord_map)
                merged_idx.append(coord_map[key])
        new_nodes_merged = list(set(merged_idx))
        # Simpler: dedup by rounding
        rounded = [(round(x * 1000) / 1000, round(y * 1000) / 1000) for (x, y) in new_nodes]
        unique = {}
        remap = []
        for i, p in enumerate(rounded):
            if p not in unique:
                unique[p] = len(unique)
            remap.append(unique[p])
        merged_edges = []
        seen = set()
        for (a, b) in new_edges:
            ra, rb = remap[a], remap[b]
            if ra == rb:
                continue
            key = (min(ra, rb), max(ra, rb))
            if key not in seen:
                seen.add(key)
                merged_edges.append(key)
        nodes = list(unique.keys())
        edges = merged_edges

    N = len(nodes)
    A = np.zeros((N, N))
    for (a, b) in edges:
        A[a, b] = A[b, a] = 1
    return A, nodes


def t_fractal(generation):
    """Simple T-fractal via recursive replacement of each edge with T shape."""
    # base: single edge
    nodes = [(0.0, 0.0), (1.0, 0.0)]
    edges = [(0, 1)]
    for _ in range(generation):
        new_nodes = list(nodes)
        new_edges = []
        for (a, b) in edges:
            # midpoint
            mx = (nodes[a][0] + nodes[b][0]) / 2
            my = (nodes[a][1] + nodes[b][1]) / 2
            # perpendicular direction
            dx = nodes[b][0] - nodes[a][0]
            dy = nodes[b][1] - nodes[a][1]
            L = np.hypot(dx, dy)
            px, py = -dy / L * L / 3, dx / L * L / 3
            m_idx = len(new_nodes)
            new_nodes.append((mx, my))
            stem_idx = len(new_nodes)
            new_nodes.append((mx + px, my + py))
            new_edges += [(a, m_idx), (m_idx, b), (m_idx, stem_idx)]
        nodes = new_nodes
        edges = new_edges
    N = len(nodes)
    A = np.zeros((N, N))
    for (a, b) in edges:
        A[a, b] = A[b, a] = 1
    return A, nodes


def vicsek_fractal(generation):
    """Vicsek fractal: each cross replaced by 5 smaller crosses."""
    # Start: 5 nodes in a plus
    nodes = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    for _ in range(generation):
        # replace each node with a shrunk Vicsek-pattern (simplified)
        # NOTE: simplified proxy - we just add a Vicsek attachment at each leaf
        new_nodes = list(nodes)
        new_edges = list(edges)
        leaves = [i for i in range(len(nodes)) if sum(1 for (a, b) in edges if a == i or b == i) == 1]
        for leaf in leaves:
            lx, ly = nodes[leaf]
            base = len(new_nodes)
            # add 4 satellite nodes around the leaf
            for dx, dy in [(0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3)]:
                new_nodes.append((lx + dx, ly + dy))
            for k in range(4):
                new_edges.append((leaf, base + k))
        nodes = new_nodes
        edges = new_edges
    N = len(nodes)
    A = np.zeros((N, N))
    for (a, b) in edges:
        A[a, b] = A[b, a] = 1
    return A, nodes


def cayley_tree(depth, branching=3):
    """Bethe-lattice-like, for contrast (tree with branching ratio b)."""
    nodes = [0]
    edges = []
    to_expand = [(0, depth)]
    next_id = 1
    while to_expand:
        parent, d = to_expand.pop()
        if d == 0:
            continue
        for _ in range(branching if parent == 0 else branching - 1):
            nodes.append(next_id)
            edges.append((parent, next_id))
            to_expand.append((next_id, d - 1))
            next_id += 1
    N = len(nodes)
    A = np.zeros((N, N))
    for (a, b) in edges:
        A[a, b] = A[b, a] = 1
    return A, nodes


def ring(N):
    """1D ring for reference."""
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i + 1) % N] = A[(i + 1) % N, i] = 1
    return A, [(np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N)) for i in range(N)]


# -------- Graph invariants --------

def graph_invariants(A):
    """Compute candidate predictors."""
    N = A.shape[0]
    degrees = A.sum(axis=1)
    mean_degree = float(degrees.mean())
    branching_ratio = float(degrees.max() / max(degrees.min(), 1))

    # Laplacian spectrum
    L = np.diag(degrees) - A
    eigs = np.linalg.eigvalsh(L)
    eigs = np.sort(np.real(eigs))
    spectral_gap = float(eigs[1])  # lowest non-zero
    spectral_dim_estimate = _estimate_spectral_dim(eigs, N)

    # BFS-based chemical (shortest-path) distance
    mean_chem_dist = _mean_shortest_path(A)

    return dict(
        N=N,
        mean_degree=mean_degree,
        branching_ratio=branching_ratio,
        spectral_gap=spectral_gap,
        d_s=spectral_dim_estimate,
        mean_chem_dist=mean_chem_dist,
    )


def _estimate_spectral_dim(eigs, N):
    """Rough d_s estimate from the low-lying density of Laplacian eigenvalues.
    For a d-dimensional graph, N(lambda) ~ lambda^(d/2) for small lambda.
    Fit log-log slope of cumulative distribution."""
    eigs_sorted = np.sort(np.real(eigs))
    eigs_pos = eigs_sorted[eigs_sorted > 1e-8]
    if len(eigs_pos) < 5:
        return 1.0
    low = eigs_pos[:len(eigs_pos) // 3]
    cum = np.arange(1, len(low) + 1, dtype=float)
    log_l = np.log(low + 1e-12)
    log_c = np.log(cum)
    slope, _ = np.polyfit(log_l, log_c, 1)
    return 2 * float(slope)


def _mean_shortest_path(A):
    """BFS from each node; average finite shortest path."""
    N = A.shape[0]
    dist_sum = 0
    pairs = 0
    for s in range(N):
        visited = {s: 0}
        q = [s]
        while q:
            v = q.pop(0)
            for u in range(N):
                if A[v, u] > 0 and u not in visited:
                    visited[u] = visited[v] + 1
                    q.append(u)
        for u, d in visited.items():
            if u != s:
                dist_sum += d
                pairs += 1
    return dist_sum / max(pairs, 1)


# -------- Quantum walk and return-probability analysis --------

def ctqw_return_probability(A, T_steps=200, t_max=50.0, start=0):
    """Evolve continuous-time quantum walk with Hamiltonian H = L.
    Returns times, |<start | psi(t)>|^2 array.
    """
    N = A.shape[0]
    degrees = A.sum(axis=1)
    H = np.diag(degrees) - A  # graph Laplacian
    psi0 = np.zeros(N, dtype=complex); psi0[start] = 1.0
    eigvals, eigvecs = np.linalg.eigh(H)
    c = eigvecs.T @ psi0

    ts = np.linspace(0.5, t_max, T_steps)
    ret = np.zeros(T_steps)
    for i, t in enumerate(ts):
        psi_t = eigvecs @ (np.exp(-1j * eigvals * t) * c)
        ret[i] = abs(psi_t[start]) ** 2
    return ts, ret


def return_decay_exponent(ts, ret):
    """Fit P_return(t) ~ a t^(-alpha) + b in the tail.
    Returns alpha from the log-log slope of the smoothed tail."""
    # Smooth oscillations (quantum walks oscillate heavily)
    window = max(len(ret) // 20, 3)
    ret_smooth = np.convolve(ret, np.ones(window) / window, mode='same')
    # Use the second half of the time series (long-time tail)
    tail_t = ts[len(ts) // 2:]
    tail_r = ret_smooth[len(ts) // 2:]
    tail_r = np.maximum(tail_r, 1e-8)
    log_t = np.log(tail_t)
    log_r = np.log(tail_r)
    slope, _ = np.polyfit(log_t, log_r, 1)
    return -float(slope)


# -------- Main experiment --------

def run_experiment(verbose=True):
    graphs = {}
    # generation choices tuned so each graph has 30-80 nodes
    graphs["ring_30"] = ring(30)
    graphs["sierpinski_g2"] = sierpinski_gasket(2)
    graphs["sierpinski_g3"] = sierpinski_gasket(3)
    graphs["t_fractal_g2"] = t_fractal(2)
    graphs["t_fractal_g3"] = t_fractal(3)
    graphs["vicsek_g2"] = vicsek_fractal(2)
    graphs["cayley_b3_d3"] = cayley_tree(3, 3)
    graphs["cayley_b4_d3"] = cayley_tree(3, 4)

    rows = []
    for name, (A, nodes) in graphs.items():
        inv = graph_invariants(A)
        ts, ret = ctqw_return_probability(A, T_steps=200, t_max=50.0, start=0)
        alpha = return_decay_exponent(ts, ret)
        row = dict(name=name, alpha=alpha, **inv)
        rows.append(row)
        if verbose:
            print(f"{name:18s}  N={inv['N']:3d}  d_s={inv['d_s']:.2f}  "
                  f"branching={inv['branching_ratio']:.2f}  chem_dist={inv['mean_chem_dist']:.2f}  "
                  f"spec_gap={inv['spectral_gap']:.3f}  alpha={alpha:+.3f}")
    return rows


def regression_analysis(rows):
    """Simple univariate correlations of alpha with each invariant."""
    from numpy import corrcoef
    names = [r['name'] for r in rows]
    alpha = np.array([r['alpha'] for r in rows])
    print("\n=== Correlations with alpha (decay exponent) ===")
    for feature in ['d_s', 'branching_ratio', 'mean_chem_dist', 'spectral_gap', 'mean_degree', 'N']:
        vals = np.array([r[feature] for r in rows])
        r = float(corrcoef(vals, alpha)[0, 1])
        print(f"  {feature:20s}  r = {r:+.3f}")

    # Try 2-feature combined prediction: d_s + X
    print("\n=== Residual correlation after partialling out d_s ===")
    ds = np.array([r['d_s'] for r in rows])
    # Linear regression alpha ~ a + b*d_s; compute residuals
    b1, a1 = np.polyfit(ds, alpha, 1)
    resid = alpha - (a1 + b1 * ds)
    for feature in ['branching_ratio', 'mean_chem_dist', 'spectral_gap', 'mean_degree']:
        vals = np.array([r[feature] for r in rows])
        r = float(corrcoef(vals, resid)[0, 1])
        print(f"  {feature:20s} residual r = {r:+.3f}")

    print("\nInterpretation:")
    print("  high |r| on first table -> direct correlate of quantum decay exponent")
    print("  high |r| on residual table -> feature gives NEW info beyond d_s")


if __name__ == "__main__":
    rows = run_experiment()
    regression_analysis(rows)

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in rows:
        # reconstruct one walk for plotting
        pass
    # Quick feature plot
    features = ['d_s', 'branching_ratio', 'mean_chem_dist', 'spectral_gap']
    alpha = np.array([r['alpha'] for r in rows])
    for i, feat in enumerate(['d_s', 'branching_ratio']):
        ax = axes[i]
        vals = np.array([r[feat] for r in rows])
        ax.scatter(vals, alpha, s=80)
        for j, r in enumerate(rows):
            ax.annotate(r['name'], (vals[j], alpha[j]), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel(feat); ax.set_ylabel("alpha (decay exponent)")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fractal_walks.png", dpi=130)
    print("\nSaved: fractal_walks.png")
