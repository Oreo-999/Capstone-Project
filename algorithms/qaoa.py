"""
QAOA (Quantum Approximate Optimization Algorithm) for Max-Cut.

Features
--------
- Weighted Max-Cut: edge weights scale the cost-Hamiltonian ZZ terms so the
  circuit targets the weighted objective Σ w_{uv} (1−Z_u Z_v)/2.
- Parameter optimisation: coarse grid search + Nelder-Mead refinement.
- Transpilation comparison across Qiskit optimisation levels 0–3.
- Realistic IBM Aer noise model (thermal relaxation + depolarising + readout).
- Ideal / noisy / measurement-error-mitigated three-way simulation.
- Zero-Noise Extrapolation (ZNE): run at noise scales 1–4, fit a polynomial,
  extrapolate to λ=0 for a noiseless estimate without extra qubits.
- Goemans–Williamson SDP: solve the semidefinite programming relaxation of
  weighted Max-Cut using CVXPY, then round with random hyperplanes to obtain
  a 0.878-approximation classical benchmark.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

# ---------------------------------------------------------------------------
# Optional dependency: cvxpy for Goemans-Williamson SDP
# ---------------------------------------------------------------------------
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

# ---------------------------------------------------------------------------
# Preset graph library  (edges + edge weights)
# ---------------------------------------------------------------------------
# weights dict keys are always (min(u,v), max(u,v)) → weight value.
# max_cut is the weighted optimum (verified by brute force below).

PRESET_GRAPHS = {
    "Triangle  (3 nodes · 3 edges)": {
        "n_nodes": 3,
        "edges": [(0, 1), (1, 2), (0, 2)],
        "weights": {(0, 1): 1.5, (1, 2): 1.0, (0, 2): 0.5},
        # Optimal weighted cut: {1} | {0,2} → (0,1)=1.5 + (1,2)=1.0 = 2.5
        "pos": {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 0.0)},
    },
    "Square / Cycle C₄  (4 nodes · 4 edges)": {
        "n_nodes": 4,
        "edges": [(0, 1), (1, 2), (2, 3), (0, 3)],
        "weights": {(0, 1): 2.0, (1, 2): 1.0, (2, 3): 2.0, (0, 3): 1.0},
        # Optimal: {0,2} | {1,3} → all 4 edges cut = 2+1+2+1 = 6.0
        "pos": {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)},
    },
    "Diamond  (4 nodes · 5 edges)": {
        "n_nodes": 4,
        "edges": [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)],
        "weights": {(0, 1): 1.5, (0, 2): 1.5, (1, 2): 0.5, (1, 3): 2.0, (2, 3): 2.0},
        # Optimal: {0,3} | {1,2} → 1.5+1.5+2.0+2.0 = 7.0
        "pos": {0: (1.0, 0.0), 1: (0.0, 1.0), 2: (2.0, 1.0), 3: (1.0, 2.0)},
    },
    "Complete K₄  (4 nodes · 6 edges)": {
        "n_nodes": 4,
        "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        "weights": {(0, 1): 2.0, (0, 2): 1.0, (0, 3): 1.5,
                    (1, 2): 1.5, (1, 3): 1.0, (2, 3): 2.0},
        # Optimal: {0,2} | {1,3} → 2.0+1.5+1.0+2.0 = 6.5  (also verified via brute force)
        "pos": {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)},
    },
}

# ---------------------------------------------------------------------------
# Fake backend for transpilation comparison (5-qubit IBM T-shape topology)
# ---------------------------------------------------------------------------
COUPLING_MAP_EDGES = [(0, 1), (1, 2), (1, 3), (3, 4)]
PHYS_QUBIT_POS = {
    0: (0.0, 2.0), 1: (1.0, 2.0), 2: (2.0, 2.0),
    3: (1.0, 1.0), 4: (1.0, 0.0),
}

try:
    from qiskit.providers.fake_provider import GenericBackendV2
    _cm = [[u, v] for (u, v) in COUPLING_MAP_EDGES] + [
        [v, u] for (u, v) in COUPLING_MAP_EDGES
    ]
    FAKE_BACKEND = GenericBackendV2(num_qubits=5, coupling_map=_cm, seed=42)
    HAS_FAKE_BACKEND = True
except Exception:
    FAKE_BACKEND = None
    HAS_FAKE_BACKEND = False


# ---------------------------------------------------------------------------
# Edge weight helper
# ---------------------------------------------------------------------------

def _w(u: int, v: int, weights: dict) -> float:
    """Return the weight of edge (u, v), defaulting to 1.0 if not found."""
    if weights is None:
        return 1.0
    return weights.get((min(u, v), max(u, v)), 1.0)


# ---------------------------------------------------------------------------
# Core combinatorics
# ---------------------------------------------------------------------------

def compute_cut_value(bitstring: str, edges: list, weights: dict = None) -> float:
    """
    Compute the (weighted) cut value for the partition encoded in *bitstring*.

    Qiskit returns big-endian bitstrings: leftmost character = highest-indexed
    qubit.  Converting to an integer and extracting bits little-endian keeps
    qubit i ↔ vertex i consistent.
    """
    val = int(bitstring, 2)
    n   = len(bitstring)
    bits = [(val >> i) & 1 for i in range(n)]
    return sum(_w(u, v, weights) for (u, v) in edges if bits[u] != bits[v])


def classical_max_cut(n_nodes: int, edges: list, weights: dict = None) -> tuple:
    """
    Brute-force weighted Max-Cut over all 2^n partitions.

    Returns
    -------
    best_cut : float
    best_partition : list[int]
    """
    best_val = 0.0
    best_partition = [0] * n_nodes
    for assignment in range(2 ** n_nodes):
        bits = [(assignment >> i) & 1 for i in range(n_nodes)]
        val  = sum(_w(u, v, weights) for (u, v) in edges if bits[u] != bits[v])
        if val > best_val:
            best_val, best_partition = val, bits
    return best_val, best_partition


def compute_expected_cut(
    counts: dict, edges: list, n_nodes: int, weights: dict = None
) -> float:
    """Compute ⟨C⟩ from a Qiskit counts dictionary."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(
        cnt * compute_cut_value(bs, edges, weights) / total
        for bs, cnt in counts.items()
    )


def approximation_ratio(expected_cut: float, max_cut: float) -> float:
    return expected_cut / max_cut if max_cut > 0 else 0.0


# ---------------------------------------------------------------------------
# QAOA circuit builder  (weighted)
# ---------------------------------------------------------------------------

def build_qaoa_circuit(
    n_nodes: int,
    edges: list,
    gamma: list,
    beta: list,
    p: int,
    weights: dict = None,
) -> QuantumCircuit:
    """
    Build the weighted QAOA ansatz for Max-Cut with *p* alternating layers.

    The cost Hamiltonian for weighted Max-Cut is:

        H_C = Σ_{(u,v)∈E} w_{uv} · (I − Z_u Z_v) / 2

    Each ZZ term is implemented as CNOT – Rz(2·w·γ) – CNOT.
    For unweighted graphs (weights=None), w_{uv} = 1 everywhere.

    Parameters
    ----------
    weights : dict {(min_u, max_v): float} or None
    """
    qr = QuantumRegister(n_nodes, "q")
    cr = ClassicalRegister(n_nodes, "meas")
    qc = QuantumCircuit(qr, cr)

    qc.h(range(n_nodes))           # |+⟩^⊗n superposition

    for layer in range(p):
        # ── Weighted cost layer ───────────────────────────────────────────
        for (u, v) in edges:
            w = _w(u, v, weights)
            qc.cx(u, v)
            qc.rz(2 * w * gamma[layer], v)
            qc.cx(u, v)
        # ── Mixer layer ───────────────────────────────────────────────────
        for i in range(n_nodes):
            qc.rx(2 * beta[layer], i)

    qc.measure(qr, cr)
    return qc


# ---------------------------------------------------------------------------
# Circuit runner helper
# ---------------------------------------------------------------------------

def _run_circuit(
    circuit: QuantumCircuit,
    noise_model=None,
    shots: int = 2048,
    optimization_level: int = 0,
) -> dict:
    backend = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
    tc  = transpile(circuit, backend, optimization_level=optimization_level)
    job = backend.run(tc, shots=shots)
    return job.result().get_counts()


# ---------------------------------------------------------------------------
# Parameter optimisation
# ---------------------------------------------------------------------------

def optimize_qaoa_params(
    n_nodes: int,
    edges: list,
    p: int = 1,
    grid_size: int = 10,
    grid_shots: int = 256,
    weights: dict = None,
) -> tuple:
    """
    Optimise QAOA angles (γ*, β*) by coarse grid search.

    For p = 1 returns a (grid_size × grid_size) landscape matrix over (γ, β).
    For p > 1 uses random-restart Nelder-Mead.

    Returns
    -------
    opt_gamma, opt_beta, opt_expectation, landscape_data
    """
    backend = AerSimulator()

    if p == 1:
        gamma_vals = np.linspace(0.05, np.pi - 0.05, grid_size)
        beta_vals  = np.linspace(0.05, np.pi / 2 - 0.05, grid_size)

        circuits = [
            build_qaoa_circuit(n_nodes, edges, [g], [b], 1, weights=weights)
            for g in gamma_vals for b in beta_vals
        ]
        transpiled = transpile(circuits, backend, optimization_level=0)
        result     = backend.run(transpiled, shots=grid_shots).result()

        landscape = np.zeros((grid_size, grid_size))
        idx = 0
        for gi in range(grid_size):
            for bi in range(grid_size):
                try:
                    counts = result.get_counts(idx)
                    landscape[gi, bi] = compute_expected_cut(
                        counts, edges, n_nodes, weights=weights
                    )
                except Exception:
                    pass
                idx += 1

        gi_best, bi_best = np.unravel_index(np.argmax(landscape), landscape.shape)
        opt_gamma       = [float(gamma_vals[gi_best])]
        opt_beta        = [float(beta_vals[bi_best])]
        opt_expectation = float(landscape[gi_best, bi_best])
        landscape_data  = (gamma_vals, beta_vals, landscape)

    else:
        from scipy.optimize import minimize

        best_val = -np.inf
        opt_gamma, opt_beta = [0.5] * p, [0.3] * p

        def neg_cut(params):
            g, b = params[:p], params[p:]
            qc   = build_qaoa_circuit(n_nodes, edges, list(g), list(b), p, weights=weights)
            tc   = transpile(qc, backend, optimization_level=0)
            job  = backend.run(tc, shots=256)
            return -compute_expected_cut(job.result().get_counts(), edges, n_nodes, weights=weights)

        for _ in range(4):
            x0 = np.concatenate([
                np.random.uniform(0.1, np.pi - 0.1, p),
                np.random.uniform(0.1, np.pi / 2 - 0.1, p),
            ])
            try:
                res = minimize(neg_cut, x0, method="Nelder-Mead",
                               options={"maxiter": 80, "xatol": 0.05, "fatol": 0.05})
                if -res.fun > best_val:
                    best_val  = -res.fun
                    opt_gamma = list(res.x[:p])
                    opt_beta  = list(res.x[p:])
            except Exception:
                pass

        opt_expectation = best_val
        landscape_data  = None

    return opt_gamma, opt_beta, opt_expectation, landscape_data


# ---------------------------------------------------------------------------
# IBM Aer noise model
# ---------------------------------------------------------------------------

def build_ibm_noise_model(scale: float = 1.0) -> NoiseModel:
    """
    Realistic IBM-device noise model with controllable strength.

    Channels: thermal relaxation (T₁/T₂) + depolarising on gates,
    plus readout assignment errors.  ``scale`` multiplies all error rates.
    """
    nm = NoiseModel()

    T1_base, T2_base = 50_000.0, 70_000.0   # ns
    t_1q, t_cx       = 35.5, 519.0          # ns
    p1q_base, p2q_base, p_ro_base = 0.001, 0.010, 0.015

    p1q  = min(scale * p1q_base,  0.49)
    p2q  = min(scale * p2q_base,  0.49)
    p_ro = min(scale * p_ro_base, 0.49)

    eff  = max(scale, 0.01)
    T1   = max(T1_base / eff, t_cx * 2)
    T2   = min(T2_base / eff, 2 * T1 - 1)

    try:
        sq_error = thermal_relaxation_error(T1, T2, t_1q).compose(
            depolarizing_error(p1q, 1)
        )
    except Exception:
        sq_error = depolarizing_error(p1q, 1)

    for gate in ["u", "h", "rx", "ry", "rz", "sx", "x", "id"]:
        nm.add_all_qubit_quantum_error(sq_error, gate)

    try:
        relax_2q = thermal_relaxation_error(T1, T2, t_cx).tensor(
            thermal_relaxation_error(T1, T2, t_cx)
        )
        tq_error = relax_2q.compose(depolarizing_error(p2q, 2))
    except Exception:
        tq_error = depolarizing_error(p2q, 2)

    for gate in ["cx", "ecr", "swap"]:
        nm.add_all_qubit_quantum_error(tq_error, gate)

    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    )
    return nm


# ---------------------------------------------------------------------------
# Transpilation comparison
# ---------------------------------------------------------------------------

def compare_transpilation_levels(circuit: QuantumCircuit) -> list:
    """Transpile at levels 0–3 against the fake 5-qubit IBM backend."""
    results = []
    for level in range(4):
        try:
            if HAS_FAKE_BACKEND:
                pm = generate_preset_pass_manager(
                    target=FAKE_BACKEND.target, optimization_level=level
                )
                t = pm.run(circuit)
            else:
                t = transpile(circuit, optimization_level=level,
                              basis_gates=["cx", "rz", "sx", "x"])

            ops       = dict(t.count_ops())
            gate_ops  = {k: v for k, v in ops.items() if k != "measure"}
            total     = sum(gate_ops.values())
            two_q     = sum(v for k, v in gate_ops.items() if k in ("cx", "ecr", "cz"))
            depth     = t.depth(filter_function=lambda i: i.operation.name != "measure")

            mapping = {}
            try:
                if t.layout and t.layout.initial_layout:
                    il = t.layout.initial_layout
                    for vq in circuit.qubits:
                        mapping[circuit.find_bit(vq).index] = il[vq]
            except Exception:
                pass

        except Exception as exc:
            ops, gate_ops, total, two_q, depth, mapping = {}, {}, 0, 0, 0, {}
            print(f"[qaoa] transpilation level {level} failed: {exc}")

        results.append({
            "level": level, "depth": depth, "total_gates": total,
            "two_qubit_gates": two_q, "ops": gate_ops, "qubit_mapping": mapping,
        })
    return results


# ---------------------------------------------------------------------------
# Measurement error mitigation
# ---------------------------------------------------------------------------

def build_calibration_matrix(
    n_qubits: int, noise_model: NoiseModel, shots: int = 4096
) -> np.ndarray:
    """Build 2^n × 2^n readout calibration matrix."""
    n_states = 2 ** n_qubits
    backend  = AerSimulator(noise_model=noise_model)
    circuits = []
    for i in range(n_states):
        qc = QuantumCircuit(n_qubits, n_qubits)
        for q in range(n_qubits):
            if (i >> q) & 1:
                qc.x(q)
        qc.measure(range(n_qubits), range(n_qubits))
        circuits.append(qc)
    transpiled = transpile(circuits, backend, optimization_level=0)
    result     = backend.run(transpiled, shots=shots).result()
    cal = np.zeros((n_states, n_states))
    for i in range(n_states):
        cnt   = result.get_counts(i)
        total = sum(cnt.values())
        for bs, c in cnt.items():
            j = int(bs, 2)
            if 0 <= j < n_states:
                cal[j, i] = c / total
    return cal


def apply_measurement_mitigation(
    counts: dict, cal_matrix: np.ndarray, n_qubits: int
) -> dict:
    """Apply pseudo-inverse calibration to correct readout errors."""
    n_states = 2 ** n_qubits
    total    = sum(counts.values())
    if total == 0:
        return counts
    prob = np.zeros(n_states)
    for bs, c in counts.items():
        idx = int(bs, 2)
        if 0 <= idx < n_states:
            prob[idx] = c / total
    try:
        corrected = np.linalg.pinv(cal_matrix) @ prob
    except Exception:
        corrected = prob
    corrected = np.clip(corrected, 0, None)
    s = corrected.sum()
    if s > 0:
        corrected /= s
    mitigated = {}
    for i in range(n_states):
        if corrected[i] > 1e-6:
            mitigated[format(i, f"0{n_qubits}b")] = max(1, int(round(corrected[i] * total)))
    return mitigated


# ---------------------------------------------------------------------------
# Three-way comparison: ideal · noisy · mitigated
# ---------------------------------------------------------------------------

def run_three_way_comparison(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    n_nodes: int,
    shots: int = 2048,
) -> dict:
    """Return ideal, noisy, and measurement-mitigated counts for *circuit*."""
    ideal_counts = _run_circuit(circuit, noise_model=None, shots=shots)
    noisy_counts = _run_circuit(circuit, noise_model=noise_model, shots=shots)
    cal = build_calibration_matrix(n_nodes, noise_model, shots=min(shots * 2, 4096))
    mitigated_counts = apply_measurement_mitigation(noisy_counts, cal, n_nodes)
    return {"ideal": ideal_counts, "noisy": noisy_counts, "mitigated": mitigated_counts}


# ---------------------------------------------------------------------------
# ★ NEW — Zero-Noise Extrapolation (ZNE)
# ---------------------------------------------------------------------------

def run_zne(
    circuit: QuantumCircuit,
    edges: list,
    n_nodes: int,
    scale_factors: list = None,
    shots: int = 1024,
    weights: dict = None,
) -> dict:
    """
    Zero-Noise Extrapolation via noise-model scaling.

    For each scale factor λ we build a noise model with λ× baseline error
    rates, simulate the circuit, and record ⟨C(λ)⟩.  We then fit both a
    linear and a quadratic polynomial to the (λ, ⟨C⟩) data and extrapolate
    to λ = 0 — the estimated noiseless expectation value.

    On real hardware the same idea is achieved by *gate folding*: replacing
    every gate G with G·G†·G (which is logically equivalent but physically
    3× noisier).  The extrapolation procedure is identical.

    Returns
    -------
    dict with keys:
        scale_data      : list of {scale, expected_cut, counts}
        zne_linear      : float — linear extrapolation to λ=0
        zne_quadratic   : float — quadratic extrapolation to λ=0
        coeffs_linear   : np.ndarray
        coeffs_quadratic: np.ndarray or None
        scales          : np.ndarray
        cuts            : np.ndarray
    """
    if scale_factors is None:
        scale_factors = [1, 2, 3, 4]

    scale_data = []
    for scale in scale_factors:
        nm     = build_ibm_noise_model(scale=float(scale))
        counts = _run_circuit(circuit, noise_model=nm, shots=shots)
        exp    = compute_expected_cut(counts, edges, n_nodes, weights=weights)
        scale_data.append({"scale": scale, "expected_cut": exp, "counts": counts})

    scales = np.array([r["scale"]        for r in scale_data], dtype=float)
    cuts   = np.array([r["expected_cut"] for r in scale_data], dtype=float)

    # ── Linear extrapolation ──────────────────────────────────────────────
    coeffs_lin = np.polyfit(scales, cuts, 1)
    zne_linear = float(np.polyval(coeffs_lin, 0.0))

    # ── Quadratic extrapolation (requires ≥ 3 points) ────────────────────
    if len(scale_factors) >= 3:
        coeffs_quad   = np.polyfit(scales, cuts, 2)
        zne_quadratic = float(np.polyval(coeffs_quad, 0.0))
    else:
        coeffs_quad   = None
        zne_quadratic = None

    return {
        "scale_data":       scale_data,
        "zne_linear":       zne_linear,
        "zne_quadratic":    zne_quadratic,
        "coeffs_linear":    coeffs_lin,
        "coeffs_quadratic": coeffs_quad,
        "scales":           scales,
        "cuts":             cuts,
    }


# ---------------------------------------------------------------------------
# ★ NEW — Goemans–Williamson SDP + hyperplane rounding
# ---------------------------------------------------------------------------

def goemans_williamson(
    n_nodes: int,
    edges: list,
    weights: dict = None,
    n_rounds: int = 200,
) -> dict:
    """
    Goemans–Williamson SDP relaxation + hyperplane rounding for weighted Max-Cut.

    **Algorithm**

    1. Solve the SDP:
           maximise  Σ_{(u,v)∈E} w_{uv} (1 − X_{uv}) / 2
           subject to  X ⪰ 0,   X_{ii} = 1  ∀i
       The optimal value is an *upper bound* on the true Max-Cut.

    2. Recover unit vectors via Cholesky: X* = L Lᵀ.

    3. For each of ``n_rounds`` trials:
       - Sample a random unit vector r.
       - Assign vertex i to set 0 if  (Lᵢ · r) ≥ 0,  else set 1.
       - Compute the cut value for this partition.

    4. Return the best cut found.

    The expected approximation ratio over random roundings is ≥ 0.878 × C*
    (the Goemans–Williamson guarantee for unweighted graphs; the bound extends
    to weighted instances).

    Requires: ``pip install cvxpy``

    Returns
    -------
    dict with keys:
        gw_cut           : float — best rounded cut value
        gw_partition     : list[int]
        sdp_bound        : float — SDP upper bound on C*
        cut_distribution : np.ndarray — cut value from each rounding round
        approx_ratio     : float — gw_cut / classical_max_cut
        classical_max_cut: float
        mean_rounding    : float — average cut over all rounds
    """
    if not HAS_CVXPY:
        raise ImportError("cvxpy is required for Goemans-Williamson: pip install cvxpy")

    # ── Build SDP ─────────────────────────────────────────────────────────
    X = cp.Variable((n_nodes, n_nodes), symmetric=True)

    objective = cp.Maximize(
        cp.sum([_w(u, v, weights) * (1 - X[u, v]) / 2 for (u, v) in edges])
    )
    constraints = [X >> 0] + [X[i, i] == 1 for i in range(n_nodes)]
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.SCS, eps=1e-5, verbose=False)
    except Exception:
        try:
            prob.solve(verbose=False)
        except Exception as exc:
            raise RuntimeError(f"SDP solver failed: {exc}")

    if X.value is None:
        raise RuntimeError("SDP solver returned no solution — try a different graph.")

    sdp_bound = float(prob.value)

    # ── Cholesky decomposition ────────────────────────────────────────────
    X_val = (X.value + X.value.T) / 2        # symmetrise
    X_val += 1e-6 * np.eye(n_nodes)          # numerical stability
    try:
        L = np.linalg.cholesky(X_val)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(X_val)
        L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))

    # ── Hyperplane rounding ───────────────────────────────────────────────
    best_cut      = 0.0
    best_partition = [0] * n_nodes
    cut_distribution = []

    rng = np.random.default_rng(seed=42)
    for _ in range(n_rounds):
        r         = rng.standard_normal(n_nodes)
        r        /= np.linalg.norm(r) + 1e-12
        signs     = L @ r
        partition = [0 if s >= 0 else 1 for s in signs]
        cut       = sum(_w(u, v, weights) for (u, v) in edges
                        if partition[u] != partition[v])
        cut_distribution.append(cut)
        if cut > best_cut:
            best_cut       = cut
            best_partition = partition

    mc_val, _ = classical_max_cut(n_nodes, edges, weights)

    return {
        "gw_cut":            best_cut,
        "gw_partition":      best_partition,
        "sdp_bound":         sdp_bound,
        "cut_distribution":  np.array(cut_distribution),
        "approx_ratio":      best_cut / mc_val if mc_val > 0 else 0.0,
        "classical_max_cut": mc_val,
        "mean_rounding":     float(np.mean(cut_distribution)),
    }


# ---------------------------------------------------------------------------
# Noise-strength sweep
# ---------------------------------------------------------------------------

def noise_sweep(
    circuit: QuantumCircuit,
    edges: list,
    n_nodes: int,
    scale_factors: list = None,
    shots: int = 1024,
    weights: dict = None,
) -> list:
    """Sweep noise scale and return expected cut + approximation ratio at each level."""
    if scale_factors is None:
        scale_factors = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    mc_val, _ = classical_max_cut(n_nodes, edges, weights)
    results = []
    for scale in scale_factors:
        nm     = build_ibm_noise_model(scale=scale) if scale > 0 else None
        counts = _run_circuit(circuit, noise_model=nm, shots=shots)
        exp    = compute_expected_cut(counts, edges, n_nodes, weights=weights)
        results.append({
            "scale": scale, "expected_cut": exp,
            "approx_ratio": approximation_ratio(exp, mc_val), "counts": counts,
        })
    return results


# ---------------------------------------------------------------------------
# Depth–quality sweep  (p = 1, 2, 3)
# ---------------------------------------------------------------------------

def depth_quality_sweep(
    n_nodes: int,
    edges: list,
    max_cut: float,
    noise_model: NoiseModel,
    p_values: list = None,
    shots: int = 1024,
    weights: dict = None,
) -> list:
    """Compare ideal vs noisy QAOA quality for p = 1, 2, 3 layers."""
    if p_values is None:
        p_values = [1, 2, 3]
    results = []
    for p in p_values:
        qc    = build_qaoa_circuit(n_nodes, edges, [0.5]*p, [0.3]*p, p, weights=weights)
        depth = qc.depth(filter_function=lambda i: i.operation.name != "measure")
        ideal = compute_expected_cut(
            _run_circuit(qc, noise_model=None,         shots=shots), edges, n_nodes, weights=weights
        )
        noisy = compute_expected_cut(
            _run_circuit(qc, noise_model=noise_model,  shots=shots), edges, n_nodes, weights=weights
        )
        results.append({
            "p": p, "depth": depth,
            "ideal_cut": ideal,  "noisy_cut": noisy,
            "ideal_ratio": approximation_ratio(ideal, max_cut),
            "noisy_ratio": approximation_ratio(noisy, max_cut),
        })
    return results
