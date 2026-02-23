"""
QAOA (Quantum Approximate Optimization Algorithm) for Max-Cut.

The Quantum Approximate Optimization Algorithm is a variational hybrid
quantum-classical algorithm designed to find approximate solutions to
combinatorial optimization problems. This module applies QAOA to the
Maximum Cut (Max-Cut) problem on small graphs, with comprehensive analysis:

  - Parameterised QAOA circuit construction (p layers)
  - Classical brute-force oracle for solution quality benchmarking
  - Grid-search + Nelder-Mead optimisation of circuit angles (γ, β)
  - Transpilation comparison across Qiskit optimisation levels 0–3
  - Qubit-mapping extraction from transpiled layouts
  - Realistic IBM Aer noise model (depolarising + thermal relaxation + readout)
  - Ideal vs. noisy vs. measurement-error-mitigated simulation
  - Noise-strength sweep and QAOA-depth sweep for performance profiling
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
# Preset graph library
# ---------------------------------------------------------------------------

PRESET_GRAPHS = {
    "Triangle  (3 nodes · 3 edges)": {
        "n_nodes": 3,
        "edges": [(0, 1), (1, 2), (0, 2)],
        "max_cut": 2,
        # 2-D positions for matplotlib drawing
        "pos": {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 0.0)},
    },
    "Square / Cycle C₄  (4 nodes · 4 edges)": {
        "n_nodes": 4,
        "edges": [(0, 1), (1, 2), (2, 3), (0, 3)],
        "max_cut": 4,
        "pos": {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)},
    },
    "Diamond  (4 nodes · 5 edges)": {
        "n_nodes": 4,
        "edges": [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)],
        "max_cut": 4,
        "pos": {0: (1.0, 0.0), 1: (0.0, 1.0), 2: (2.0, 1.0), 3: (1.0, 2.0)},
    },
    "Complete K₄  (4 nodes · 6 edges)": {
        "n_nodes": 4,
        "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        "max_cut": 4,
        "pos": {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)},
    },
}

# ---------------------------------------------------------------------------
# Fake backend for transpilation comparison (5-qubit IBM T-shape topology)
# Physical coupling:  0 ── 1 ── 2
#                          │
#                          3
#                          │
#                          4
# ---------------------------------------------------------------------------
COUPLING_MAP_EDGES = [(0, 1), (1, 2), (1, 3), (3, 4)]

# Physical (x, y) positions for the 5-qubit layout visualisation
PHYS_QUBIT_POS = {
    0: (0.0, 2.0),
    1: (1.0, 2.0),
    2: (2.0, 2.0),
    3: (1.0, 1.0),
    4: (1.0, 0.0),
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
# Core combinatorics helpers
# ---------------------------------------------------------------------------

def compute_cut_value(bitstring: str, edges: list) -> int:
    """
    Count the number of edges crossing the partition encoded in *bitstring*.

    Qiskit returns measurement results as big-endian strings where the
    leftmost character corresponds to the highest-indexed qubit.  We convert
    to an integer and then extract individual bits in little-endian order
    (bit i = qubit i), which is consistent with how the QAOA oracle labels
    graph vertices.

    Example: '0101' → integer 5 → bits[0]=1, bits[1]=0, bits[2]=1, bits[3]=0
    """
    val = int(bitstring, 2)
    n = len(bitstring)
    bits = [(val >> i) & 1 for i in range(n)]
    return sum(1 for (u, v) in edges if bits[u] != bits[v])


def classical_max_cut(n_nodes: int, edges: list) -> tuple:
    """
    Brute-force Max-Cut over all 2^n partitions.

    Returns
    -------
    best_cut_value : int
        The maximum number of edges that can be cut.
    best_partition : list[int]
        Vertex assignment (0 or 1) achieving the maximum cut.
    """
    best_val = 0
    best_partition = [0] * n_nodes
    for assignment in range(2 ** n_nodes):
        bits = [(assignment >> i) & 1 for i in range(n_nodes)]
        val = sum(1 for (u, v) in edges if bits[u] != bits[v])
        if val > best_val:
            best_val = val
            best_partition = bits
    return best_val, best_partition


def compute_expected_cut(counts: dict, edges: list, n_nodes: int) -> float:
    """Compute the expected cut value ⟨C⟩ from a Qiskit counts dictionary."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(
        count * compute_cut_value(bs, edges) / total
        for bs, count in counts.items()
    )


def approximation_ratio(expected_cut: float, max_cut: int) -> float:
    """Return ⟨C⟩ / C* — the QAOA approximation ratio (1.0 is optimal)."""
    return expected_cut / max_cut if max_cut > 0 else 0.0


# ---------------------------------------------------------------------------
# QAOA circuit builder
# ---------------------------------------------------------------------------

def build_qaoa_circuit(
    n_nodes: int,
    edges: list,
    gamma: list,
    beta: list,
    p: int,
) -> QuantumCircuit:
    """
    Build the QAOA ansatz for Max-Cut with *p* alternating layers.

    The circuit implements:

        |ψ(γ,β)⟩ = ∏_{l=1}^{p} U_B(βₗ) U_C(γₗ) |+⟩^⊗n

    where
        U_C(γ) = ∏_{(u,v)∈E} exp(−i γ (I − Zᵤ Zᵥ) / 2)
                = ∏_{(u,v)∈E} CNOT_{u→v} · Rz(2γ, v) · CNOT_{u→v}

        U_B(β)  = ∏_{u} exp(−i β Xᵤ) = ∏_{u} Rx(2β, u)

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes (= number of qubits).
    edges : list of (int, int)
        Graph edges.
    gamma : list of float, length p
        Cost-layer angles (one per QAOA layer).
    beta : list of float, length p
        Mixer-layer angles (one per QAOA layer).
    p : int
        Number of QAOA layers.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit with measurements appended.
    """
    qr = QuantumRegister(n_nodes, "q")
    cr = ClassicalRegister(n_nodes, "meas")
    qc = QuantumCircuit(qr, cr)

    # |+⟩^⊗n — equal superposition over all 2^n bitstrings
    qc.h(range(n_nodes))

    for layer in range(p):
        # ── Cost layer ────────────────────────────────────────────────────
        # For each edge (u, v): CNOT, Rz(2γ), CNOT implements e^{−iγ ZᵤZᵥ}
        for (u, v) in edges:
            qc.cx(u, v)
            qc.rz(2 * gamma[layer], v)
            qc.cx(u, v)

        # ── Mixer layer ───────────────────────────────────────────────────
        # Rx(2β) on every qubit implements e^{−iβ Xᵢ}
        for i in range(n_nodes):
            qc.rx(2 * beta[layer], i)

    qc.measure(qr, cr)
    return qc


# ---------------------------------------------------------------------------
# Parameter optimisation (grid search → Nelder-Mead)
# ---------------------------------------------------------------------------

def optimize_qaoa_params(
    n_nodes: int,
    edges: list,
    p: int = 1,
    grid_size: int = 10,
    grid_shots: int = 256,
) -> tuple:
    """
    Find optimal QAOA angles (γ*, β*) using a coarse grid search.

    For p = 1 the landscape is two-dimensional (γ, β), so a grid search is
    tractable.  All grid circuits are submitted in a single ``backend.run``
    call to minimise overhead.  The best grid point is returned together with
    the full landscape array for visualisation.

    Returns
    -------
    opt_gamma : list[float]
    opt_beta  : list[float]
    opt_expectation : float   — expected cut value at (γ*, β*)
    landscape_data  : tuple   — (gamma_vals, beta_vals, landscape_matrix) or None
    """
    backend = AerSimulator()

    if p == 1:
        gamma_vals = np.linspace(0.05, np.pi - 0.05, grid_size)
        beta_vals = np.linspace(0.05, np.pi / 2 - 0.05, grid_size)

        # Build all grid circuits
        circuits = []
        for g in gamma_vals:
            for b in beta_vals:
                qc = build_qaoa_circuit(n_nodes, edges, [g], [b], 1)
                circuits.append(qc)

        # Transpile once and run as a batch
        transpiled = transpile(circuits, backend, optimization_level=0)
        job = backend.run(transpiled, shots=grid_shots)
        result = job.result()

        # Fill landscape matrix
        landscape = np.zeros((grid_size, grid_size))
        idx = 0
        for gi in range(grid_size):
            for bi in range(grid_size):
                try:
                    counts = result.get_counts(idx)
                    landscape[gi, bi] = compute_expected_cut(counts, edges, n_nodes)
                except Exception:
                    pass
                idx += 1

        gi_best, bi_best = np.unravel_index(np.argmax(landscape), landscape.shape)
        opt_gamma = [float(gamma_vals[gi_best])]
        opt_beta = [float(beta_vals[bi_best])]
        opt_expectation = float(landscape[gi_best, bi_best])
        landscape_data = (gamma_vals, beta_vals, landscape)

    else:
        # For p > 1: multiple random restarts with Nelder-Mead
        from scipy.optimize import minimize

        best_val = -np.inf
        opt_gamma, opt_beta = [0.5] * p, [0.3] * p

        def neg_cut(params):
            g, b = params[:p], params[p:]
            qc = build_qaoa_circuit(n_nodes, edges, list(g), list(b), p)
            tc = transpile(qc, backend, optimization_level=0)
            job = backend.run(tc, shots=256)
            counts = job.result().get_counts()
            return -compute_expected_cut(counts, edges, n_nodes)

        for _ in range(4):
            x0 = np.concatenate([
                np.random.uniform(0.1, np.pi - 0.1, p),
                np.random.uniform(0.1, np.pi / 2 - 0.1, p),
            ])
            try:
                res = minimize(neg_cut, x0, method="Nelder-Mead",
                               options={"maxiter": 80, "xatol": 0.05, "fatol": 0.05})
                if -res.fun > best_val:
                    best_val = -res.fun
                    opt_gamma = list(res.x[:p])
                    opt_beta = list(res.x[p:])
            except Exception:
                pass

        opt_expectation = best_val
        landscape_data = None

    return opt_gamma, opt_beta, opt_expectation, landscape_data


# ---------------------------------------------------------------------------
# IBM Aer noise model
# ---------------------------------------------------------------------------

def build_ibm_noise_model(scale: float = 1.0) -> NoiseModel:
    """
    Build a realistic IBM-device-like noise model with controllable strength.

    Reference device parameters (IBM Nairobi, ~2023):
      - T1  = 50 µs  (energy relaxation time)
      - T2  = 70 µs  (dephasing time)
      - Single-qubit gate duration = 35.5 ns   → error ≈ 0.1 %
      - CX gate duration           = 519.0 ns   → error ≈ 1.0 %
      - Readout assignment error                → 1.5 %

    ``scale`` multiplies all error rates (and divides T1/T2) so callers can
    sweep from near-ideal (scale→0) to highly noisy (scale = 5).

    Each gate error is a composition of thermal relaxation (amplitude
    damping + dephasing) and depolarising noise — both of which occur on
    real superconducting devices.

    Parameters
    ----------
    scale : float
        Noise scale factor.  1.0 = baseline IBM parameters.

    Returns
    -------
    NoiseModel
    """
    noise_model = NoiseModel()

    # ── Hardware constants ─────────────────────────────────────────────────
    T1_base = 50_000.0   # ns
    T2_base = 70_000.0   # ns
    t_1q    = 35.5       # ns — single-qubit gate
    t_cx    = 519.0      # ns — CX gate
    p1q_base = 0.001     # single-qubit depolarising error
    p2q_base = 0.010     # two-qubit  depolarising error
    p_ro_base = 0.015    # readout assignment error

    # Scale error rates; cap to avoid unphysical values
    p1q  = min(scale * p1q_base, 0.49)
    p2q  = min(scale * p2q_base, 0.49)
    p_ro = min(scale * p_ro_base, 0.49)

    # Coherence times shorten as noise increases (simulates worse devices)
    eff_scale = max(scale, 0.01)
    T1 = max(T1_base / eff_scale, t_cx * 2)
    T2 = min(T2_base / eff_scale, 2 * T1 - 1)

    # ── Single-qubit gate errors ──────────────────────────────────────────
    try:
        relax_1q = thermal_relaxation_error(T1, T2, t_1q)
        depol_1q = depolarizing_error(p1q, 1)
        sq_error = relax_1q.compose(depol_1q)
    except Exception:
        sq_error = depolarizing_error(p1q, 1)

    for gate in ["u", "h", "rx", "ry", "rz", "sx", "x", "id"]:
        noise_model.add_all_qubit_quantum_error(sq_error, gate)

    # ── Two-qubit gate errors ─────────────────────────────────────────────
    try:
        relax_q0 = thermal_relaxation_error(T1, T2, t_cx)
        relax_q1 = thermal_relaxation_error(T1, T2, t_cx)
        relax_2q = relax_q0.tensor(relax_q1)
        depol_2q = depolarizing_error(p2q, 2)
        tq_error = relax_2q.compose(depol_2q)
    except Exception:
        tq_error = depolarizing_error(p2q, 2)

    for gate in ["cx", "ecr", "swap"]:
        noise_model.add_all_qubit_quantum_error(tq_error, gate)

    # ── Readout (measurement) errors ──────────────────────────────────────
    # [[P(measure 0 | true 0), P(measure 1 | true 0)],
    #  [P(measure 0 | true 1), P(measure 1 | true 1)]]
    ro_matrix = [[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]
    noise_model.add_all_qubit_readout_error(ReadoutError(ro_matrix))

    return noise_model


# ---------------------------------------------------------------------------
# Circuit runner helpers
# ---------------------------------------------------------------------------

def _run_circuit(
    circuit: QuantumCircuit,
    noise_model=None,
    shots: int = 2048,
    optimization_level: int = 0,
) -> dict:
    """Transpile and run a single circuit; return counts dict."""
    backend = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
    tc = transpile(circuit, backend, optimization_level=optimization_level)
    job = backend.run(tc, shots=shots)
    return job.result().get_counts()


# ---------------------------------------------------------------------------
# Transpilation comparison  (levels 0–3)
# ---------------------------------------------------------------------------

def compare_transpilation_levels(circuit: QuantumCircuit) -> list:
    """
    Transpile *circuit* at optimisation levels 0–3 against a realistic fake
    5-qubit IBM backend (T-shape coupling map, CX/RZ/SX basis gates) and
    collect metrics.

    Optimisation levels in Qiskit:
      0 — minimal: only basic decomposition, no routing optimisation
      1 — light:   layout + basic routing with sabre; light peephole rewrites
      2 — medium:  more aggressive 1Q/2Q optimisation passes
      3 — heavy:   full optimisation including commutativity analysis and
                   pulse-efficient two-qubit decompositions

    Returns
    -------
    list of dicts, one per level, each containing:
        level          : int
        depth          : int
        total_gates    : int
        two_qubit_gates: int — count of CX / ECR gates
        ops            : dict — {gate_name: count}
        qubit_mapping  : dict — {virtual_qubit_idx: physical_qubit_idx}
    """
    results = []

    for level in range(4):
        try:
            if HAS_FAKE_BACKEND:
                pm = generate_preset_pass_manager(
                    target=FAKE_BACKEND.target,
                    optimization_level=level,
                )
                transpiled = pm.run(circuit)
            else:
                transpiled = transpile(
                    circuit, optimization_level=level, basis_gates=["cx", "rz", "sx", "x"]
                )

            ops = dict(transpiled.count_ops())
            # Remove measurement from gate counts for fair comparison
            gate_ops = {k: v for k, v in ops.items() if k != "measure"}
            total_gates = sum(gate_ops.values())
            two_q = sum(v for k, v in gate_ops.items() if k in ("cx", "ecr", "cz"))
            depth = transpiled.depth(filter_function=lambda inst: inst.operation.name != "measure")

            # ── Extract qubit mapping ──────────────────────────────────────
            mapping = {}
            try:
                if transpiled.layout and transpiled.layout.initial_layout:
                    il = transpiled.layout.initial_layout
                    for virt_qubit in circuit.qubits:
                        phys = il[virt_qubit]
                        virt_idx = circuit.find_bit(virt_qubit).index
                        mapping[virt_idx] = phys
            except Exception:
                pass

        except Exception as exc:
            # Graceful fallback: report zeros if transpilation fails
            ops, gate_ops, total_gates, two_q, depth, mapping = {}, {}, 0, 0, 0, {}
            print(f"[qaoa] transpilation level {level} failed: {exc}")

        results.append(
            {
                "level": level,
                "depth": depth,
                "total_gates": total_gates,
                "two_qubit_gates": two_q,
                "ops": gate_ops,
                "qubit_mapping": mapping,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Measurement error mitigation (calibration matrix)
# ---------------------------------------------------------------------------

def build_calibration_matrix(
    n_qubits: int,
    noise_model: NoiseModel,
    shots: int = 4096,
) -> np.ndarray:
    """
    Build a (2^n × 2^n) measurement calibration matrix.

    Element A[j, i] = P(measuring state j  |  true state i).

    For each of the 2^n computational basis states we prepare it, measure it
    through the noisy channel, and record the resulting probability distribution.
    The matrix inverse (pseudo-inverse) is then applied to noisy counts to
    correct for readout assignment errors.

    Complexity: 2^n circuits → feasible for n ≤ 4.
    """
    n_states = 2 ** n_qubits
    backend = AerSimulator(noise_model=noise_model)

    circuits = []
    for i in range(n_states):
        qc = QuantumCircuit(n_qubits, n_qubits)
        for qubit in range(n_qubits):
            if (i >> qubit) & 1:
                qc.x(qubit)
        qc.measure(range(n_qubits), range(n_qubits))
        circuits.append(qc)

    transpiled = transpile(circuits, backend, optimization_level=0)
    job = backend.run(transpiled, shots=shots)
    result = job.result()

    cal_matrix = np.zeros((n_states, n_states))
    for i in range(n_states):
        counts = result.get_counts(i)
        total = sum(counts.values())
        for bitstring, count in counts.items():
            j = int(bitstring, 2)
            if 0 <= j < n_states:
                cal_matrix[j, i] = count / total

    return cal_matrix


def apply_measurement_mitigation(
    counts: dict,
    cal_matrix: np.ndarray,
    n_qubits: int,
) -> dict:
    """
    Apply measurement error mitigation using the pseudo-inverse of the
    calibration matrix.

    The corrected probability vector is clipped to [0, ∞) and renormalised
    to ensure it remains a valid probability distribution.

    Returns
    -------
    dict — mitigated counts {bitstring: int}
    """
    n_states = 2 ** n_qubits
    total = sum(counts.values())
    if total == 0:
        return counts

    # Build observed probability vector
    prob_vec = np.zeros(n_states)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        if 0 <= idx < n_states:
            prob_vec[idx] = count / total

    # Invert: corrected = A⁻¹ · observed
    try:
        cal_inv = np.linalg.pinv(cal_matrix)
        corrected = cal_inv @ prob_vec
    except Exception:
        corrected = prob_vec

    # Enforce non-negativity and normalise
    corrected = np.clip(corrected, 0.0, None)
    s = corrected.sum()
    if s > 0:
        corrected /= s

    # Convert back to counts dict (Qiskit big-endian format)
    mitigated = {}
    for i in range(n_states):
        if corrected[i] > 1e-6:
            bs = format(i, f"0{n_qubits}b")
            mitigated[bs] = max(1, int(round(corrected[i] * total)))

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
    """
    Run the same QAOA circuit under three simulation conditions.

    1. **Ideal**     — AerSimulator with no noise (statevector-accurate).
    2. **Noisy**     — AerSimulator with the provided noise model.
    3. **Mitigated** — Noisy simulation with post-hoc measurement-error
                       mitigation (calibration-matrix inversion).

    Returns a dict with keys ``ideal``, ``noisy``, ``mitigated``, each
    holding ``{counts, expected_cut, approx_ratio}``.
    """
    # ── Ideal ─────────────────────────────────────────────────────────────
    ideal_counts = _run_circuit(circuit, noise_model=None, shots=shots)

    # ── Noisy ─────────────────────────────────────────────────────────────
    noisy_counts = _run_circuit(circuit, noise_model=noise_model, shots=shots)

    # ── Measurement-error mitigated ───────────────────────────────────────
    cal_matrix = build_calibration_matrix(n_nodes, noise_model, shots=min(shots * 2, 4096))
    mitigated_counts = apply_measurement_mitigation(noisy_counts, cal_matrix, n_nodes)

    return {
        "ideal": ideal_counts,
        "noisy": noisy_counts,
        "mitigated": mitigated_counts,
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
) -> list:
    """
    Simulate the circuit across a range of noise strengths and return the
    expected Max-Cut value (and approximation ratio) for each.

    ``scale_factors`` controls how many times stronger the noise is relative
    to the baseline IBM model.  scale=0 corresponds to ideal (no noise).

    Returns
    -------
    list of dicts, each with keys:
        scale, expected_cut, approx_ratio, counts
    """
    if scale_factors is None:
        scale_factors = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

    results = []
    for scale in scale_factors:
        if scale == 0.0:
            counts = _run_circuit(circuit, noise_model=None, shots=shots)
        else:
            nm = build_ibm_noise_model(scale=scale)
            counts = _run_circuit(circuit, noise_model=nm, shots=shots)

        exp_cut = compute_expected_cut(counts, edges, n_nodes)
        # Approximate classical Max-Cut without re-running brute force every time
        mc_val, _ = classical_max_cut(n_nodes, edges)
        results.append(
            {
                "scale": scale,
                "expected_cut": exp_cut,
                "approx_ratio": approximation_ratio(exp_cut, mc_val),
                "counts": counts,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Depth–quality sweep  (vary QAOA layers p = 1, 2, 3)
# ---------------------------------------------------------------------------

def depth_quality_sweep(
    n_nodes: int,
    edges: list,
    max_cut: int,
    noise_model: NoiseModel,
    p_values: list = None,
    shots: int = 1024,
) -> list:
    """
    Compare QAOA approximation quality vs circuit depth for p = 1, 2, 3.

    For each p we use fixed heuristic angles (γ = 0.5, β = 0.3 repeated p
    times) to keep comparison fair — the goal is to show depth vs noise
    trade-off, not optimise each p independently.

    Returns
    -------
    list of dicts with keys:
        p, depth, ideal_cut, noisy_cut, ideal_ratio, noisy_ratio
    """
    if p_values is None:
        p_values = [1, 2, 3]

    results = []
    for p in p_values:
        gamma = [0.5] * p
        beta = [0.3] * p
        qc = build_qaoa_circuit(n_nodes, edges, gamma, beta, p)

        # Compute circuit depth (excluding measurements)
        depth = qc.depth(filter_function=lambda inst: inst.operation.name != "measure")

        ideal_counts = _run_circuit(qc, noise_model=None, shots=shots)
        noisy_counts = _run_circuit(qc, noise_model=noise_model, shots=shots)

        ideal_cut = compute_expected_cut(ideal_counts, edges, n_nodes)
        noisy_cut = compute_expected_cut(noisy_counts, edges, n_nodes)

        results.append(
            {
                "p": p,
                "depth": depth,
                "ideal_cut": ideal_cut,
                "noisy_cut": noisy_cut,
                "ideal_ratio": approximation_ratio(ideal_cut, max_cut),
                "noisy_ratio": approximation_ratio(noisy_cut, max_cut),
            }
        )

    return results
