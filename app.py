import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from qiskit_aer import AerSimulator

from config import get_backends
from algorithms.grover import run_grover, build_grover_circuit_k
from algorithms.shor import run_shor, VALID_A_VALUES, PERIOD_TABLE
from algorithms.qaoa import (
    PRESET_GRAPHS,
    COUPLING_MAP_EDGES,
    PHYS_QUBIT_POS,
    HAS_CVXPY,
    build_qaoa_circuit,
    classical_max_cut,
    compute_cut_value,
    compute_expected_cut,
    approximation_ratio,
    optimize_qaoa_params,
    build_ibm_noise_model,
    compare_transpilation_levels,
    run_three_way_comparison,
    run_zne,
    goemans_williamson,
    noise_sweep,
    depth_quality_sweep,
)
from visualizations import grover_viz, shor_viz, qaoa_viz
from visualizations import plotly_viz


def plt_close(fig):
    """Close a matplotlib figure to avoid memory leaks across Streamlit reruns."""
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cached quantum computations
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_grover_step(target: int, k: int):
    """Run Grover circuit with k iterations on AerSimulator; cached per (target, k)."""
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    qc = build_grover_circuit_k(target, k)
    sampler = AerSampler()
    job = sampler.run([qc], shots=1024)
    counts = job.result()[0].data.meas.get_counts()
    return qc, counts


@st.cache_data(show_spinner=False)
def cached_run_shor(a: int):
    """Run Shor's algorithm for given base a; cached per a."""
    return run_shor(a)


@st.cache_data(show_spinner=False)
def cached_optimize_qaoa(n_nodes: int, edges_tuple: tuple, p: int,
                          grid_size: int, grid_shots: int,
                          weights_frozen):
    """Optimise QAOA params; cached by graph/p/weights combination."""
    edges = list(edges_tuple)
    weights = dict(weights_frozen) if weights_frozen else None
    return optimize_qaoa_params(n_nodes, edges, p=p,
                                grid_size=grid_size, grid_shots=grid_shots,
                                weights=weights)


def show_results_grover(circuit, counts, target):
    """Render all three Grover visualizations with explanations."""

    st.divider()

    # --- Circuit Diagram ---
    st.markdown("### Circuit Diagram")
    st.markdown(
        """
        This is the actual quantum circuit that ran on the backend. Read it left to right — each
        horizontal line is a **qubit wire**, and each box is a **quantum gate** applied to that qubit.

        **What you're seeing in three stages:**

        | Stage | Gates | What it does |
        |-------|-------|--------------|
        | **1 — Superposition** | `H` (Hadamard) on all 4 qubits | Puts every qubit into a 50/50 mix of 0 and 1, creating a uniform superposition over all 16 states simultaneously |
        | **2 — Oracle** | `X` flips + multi-controlled `Z` | Marks the target state by flipping its phase from `+1` to `−1` — invisible to direct measurement but detectable by interference |
        | **3 — Diffuser** | `H`, `X`, multi-controlled `Z`, `X`, `H` | Reflects amplitudes about their average, amplifying the marked state and suppressing all others |

        This oracle + diffuser cycle repeats **3 times** (⌊√16⌋ ≈ 3.14), which is the
        theoretically optimal iteration count for a 4-qubit search.
        """
    )
    fig_circuit = grover_viz.plot_circuit(circuit)
    st.pyplot(fig_circuit)
    plt_close(fig_circuit)

    with st.expander("Why does the oracle use a phase flip instead of marking a bit?"):
        st.markdown(
            """
            Classical search marks an item by setting a flag bit. In quantum computing we can't
            "look" at a qubit without collapsing the superposition — so instead the oracle encodes
            the answer as a **phase flip**.

            A phase of `−1` is mathematically equivalent to multiplying the state's amplitude by `−1`.
            On its own this is undetectable. But combined with the **diffuser** (which reflects all
            amplitudes about their mean), the negative phase causes destructive interference on every
            *other* state and constructive interference on the target — amplifying it with each iteration.

            This is **quantum amplitude amplification**: the same principle that underpins every
            quantum search speedup.
            """
        )

    st.divider()

    # --- Measurement Results ---
    st.markdown("### Measurement Results")
    target_state = format(target, "04b")[::-1]
    top_count = counts.get(target_state, 0)
    total = sum(counts.values())
    pct = 100 * top_count / total if total else 0

    st.markdown(
        f"""
        The circuit was run **{total} times** (shots). Each shot collapses the quantum state
        and records a classical bitstring.

        - **Target state** `|{target_state}⟩` (decimal {target}) appeared **{top_count} times — {pct:.1f}%** of all shots.
        - All 15 other states share the remaining {100 - pct:.1f}%.

        In a purely classical random search over 16 items you'd expect to find the target only
        **6.25%** of the time per query. Grover's achieves ~**{pct:.0f}%** in just 3 queries —
        a dramatic demonstration of quantum amplitude amplification.
        """
    )
    fig_counts = plotly_viz.plotly_grover_counts(counts, target)
    st.plotly_chart(fig_counts, use_container_width=True)

    with st.expander("Why isn't the target probability exactly 100%?"):
        st.markdown(
            """
            Grover's algorithm is **probabilistic**, not deterministic. After the optimal number
            of iterations (3 for N=16), the theoretical success probability is:

            > P = sin²((2k+1)θ)  where  θ = arcsin(1/√N)

            For N=16 and k=3 iterations:  θ ≈ 14.48°,  P = sin²(7θ) ≈ **96.1%**

            The remaining ~4% is spread across other states. In practice you'd run the circuit a
            few times and take the most frequent result — still far faster than classical O(N/2).

            If you run more than ⌊π√N/4⌋ iterations the amplitude *overshoots* and the probability
            actually starts to decrease, so more iterations isn't always better.
            """
        )

    st.divider()

    # --- Complexity Comparison ---
    st.markdown("### Complexity Comparison: Classical vs Quantum")
    st.markdown(
        """
        This chart shows how many **queries** (comparisons) each approach needs on average to
        find a target in an unsorted list of size N.

        | Approach | Queries needed | For N = 1,000,000 |
        |----------|---------------|-------------------|
        | **Classical** (random) | N/2 on average | ~500,000 queries |
        | **Grover's** | ~√N queries | ~1,000 queries |

        The gap grows with N — Grover's becomes exponentially more advantageous as the
        search space scales up.
        """
    )
    fig_complexity = plotly_viz.plotly_complexity_comparison()
    st.plotly_chart(fig_complexity, use_container_width=True)

    with st.expander("Is this the best possible quantum speedup for search?"):
        st.markdown(
            """
            Yes — and it's been **proven optimal**. In 1997 Bennett et al. showed that any quantum
            algorithm solving an unstructured search problem requires Ω(√N) queries. You cannot do
            better without additional structure in the problem.

            This is different from Shor's algorithm, which achieves an **exponential** speedup for
            factoring (a structured problem). Grover's quadratic speedup applies to the hardest case:
            a completely unstructured search with no exploitable patterns.

            **Real-world relevance:** Grover's algorithm threatens symmetric cryptography.
            AES-128 has an effective security of ~64 bits against a quantum adversary using Grover's,
            which is why NIST recommends AES-256 for post-quantum security.
            """
        )

    st.divider()

    # --- Amplitude Evolution ---
    st.markdown("### Amplitude Evolution: How the Algorithm Amplifies the Target")
    st.markdown(
        """
        This is the heart of *why* Grover's works. Each panel shows the **quantum amplitude**
        of every state after 0, 1, 2, and 3 iterations.

        - **Amplitude** (y-axis) is the square root of probability. Positive and negative amplitudes
          can interfere — like waves.
        - Before any iteration, all 16 states have equal amplitude 1/√16 = 0.25.
        - The oracle flips the target's amplitude to −0.25 (same magnitude, opposite sign).
        - The diffuser then reflects all amplitudes about their mean — this *increases* the target
          and *decreases* everything else. Repeat three times and the target dominates.

        Watch the red bar grow and everything else shrink with each iteration.
        """
    )
    fig_amp = grover_viz.plot_amplitude_evolution(target)
    st.pyplot(fig_amp)
    plt_close(fig_amp)

    st.divider()

    # --- Success Probability vs Queries ---
    st.markdown("### Success Probability vs Number of Queries")
    st.markdown(
        """
        A direct apples-to-apples comparison: what is the **probability of having found the target**
        after k queries, for each method?

        - **Classical:** each query is an independent random draw. P(found after k) = 1 − (15/16)ᵏ
          — grows slowly, logarithmic shape.
        - **Grover's:** P(found after k iterations) = sin²((2k+1)θ) where θ = arcsin(1/√N).
          Reaches near-certainty in just 3 queries, then *oscillates* (overshooting after the optimal point).

        The annotation marks the optimal query count and shows the probability gap at that point.
        """
    )
    fig_prob = grover_viz.plot_success_probability_vs_queries()
    st.pyplot(fig_prob)
    plt_close(fig_prob)

    st.divider()

    # --- Runtime Race ---
    st.markdown("### Quantum Speedup at Scale: The Runtime Race")
    st.markdown(
        """
        The asymptotic curves look abstract. This chart makes it **concrete** by showing
        the actual query counts at real search space sizes, from 16 items (this demo)
        up to 1 million items.

        The numbers above each pair of bars show the **speedup factor** — how many times
        fewer queries Grover's needs. Note the log scale: at 1M items the difference is
        a factor of **500×**.
        """
    )
    fig_race = grover_viz.plot_runtime_race()
    st.pyplot(fig_race)
    plt_close(fig_race)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Quantum Algorithm Dashboard", layout="wide")
st.title("Quantum Algorithm Dashboard")
st.caption("Grover's Search · Shor's Factoring · Powered by Qiskit")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "IBM Quantum API Key",
        type="password",
        placeholder="Paste your API key here",
    )
    instance = st.text_input(
        "Instance",
        placeholder="ibm-q/open/main",
        help=(
            "Your IBM Quantum instance in hub/group/project format. "
            "Free-tier users: leave blank or use ibm-q/open/main. "
            "Find yours at quantum.ibm.com → your account → instances."
        ),
    )
    use_fallback = st.toggle(
        "Use Simulator (skip IBM queue)",
        value=False,
        help="When ON, Grover's runs on AerSimulator instead of real IBM hardware.",
    )

    if st.button("Connect", use_container_width=True):
        with st.spinner("Connecting to backend…"):
            backends = get_backends(
                api_key=api_key or None,
                instance=instance or None,
                use_simulator_fallback=use_fallback,
            )
            st.session_state["backends"] = backends
        st.success("Connected!")

    st.divider()
    st.markdown(
        """
        **About this dashboard**

        Demonstrates two landmark quantum algorithms:
        - **Grover's Search** — quadratic speedup over classical search
        - **Shor's Factoring** — exponential speedup for integer factoring

        Both run on [Qiskit](https://qiskit.org).
        Grover's can target real IBM quantum hardware; Shor's always uses the Aer simulator.
        """
    )

# ---------------------------------------------------------------------------
# Backend status
# ---------------------------------------------------------------------------
if "backends" in st.session_state:
    backends = st.session_state["backends"]

    if backends.get("warning"):
        st.warning(backends["warning"])

    col1, col2 = st.columns(2)
    with col1:
        mode_label = "Real Hardware" if backends["grover_mode"] == "real" else "Simulated"
        st.metric("Grover Backend", backends["grover_backend_name"])
        st.caption(f"Mode: {mode_label}")
    with col2:
        st.metric("Shor Backend", backends["shor_backend_name"])
        st.caption("Mode: Simulated")

    st.divider()
else:
    st.info("Configure and connect a backend in the sidebar to get started.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_grover, tab_shor, tab_qaoa = st.tabs(["Grover's Search", "Shor's Factoring", "QAOA · Max-Cut"])

# ===========================================================================
# GROVER TAB
# ===========================================================================
with tab_grover:
    st.subheader("Grover's Search Algorithm")

    # Intro explanation
    st.markdown(
        """
        **The problem:** You have an unsorted list of N items and need to find the one that
        satisfies some condition. Classically, you must check items one by one — O(N/2) checks
        on average. There's no shortcut: without structure, you can't do better.

        **The quantum insight:** Grover's algorithm (Lov Grover, 1996) uses **quantum superposition**
        and **amplitude amplification** to solve this in O(√N) queries — a quadratic speedup that
        is provably optimal for unstructured search.

        This demo searches a space of **16 items** (4 qubits) using 3 Grover iterations.
        """
    )

    with st.expander("How does it work? (Step-by-step)"):
        st.markdown(
            """
            **Step 1 — Create superposition**
            Apply a Hadamard gate (H) to each of the 4 qubits. This puts the register into
            an equal superposition of all 16 states: |0000⟩, |0001⟩, … |1111⟩, each with
            amplitude 1/√16 = 0.25. At this point every state is equally likely.

            **Step 2 — Oracle (phase kickback)**
            The oracle is a quantum circuit that "knows" the target. It applies a phase flip
            to the target state: multiplies its amplitude by −1. All other states are unchanged.
            This is invisible to measurement (you can't distinguish +0.25 from −0.25 by sampling),
            but it sets up the interference pattern for the next step.

            **Step 3 — Diffuser (amplitude amplification)**
            The diffuser reflects all amplitudes about their mean value. Because the target now
            has a negative amplitude (−0.25) while the mean is slightly positive, after reflection
            the target's amplitude becomes much larger than before. All other states are pushed
            slightly lower.

            **Repeat steps 2–3 for ⌊π√N/4⌋ iterations** (3 times for N=16).
            Each iteration roughly doubles the gap between the target and the rest.
            After 3 iterations, the target amplitude is ~0.98 and all others are ~0.01.

            **Step 4 — Measure**
            Collapse the superposition. The target state is observed with ~96% probability.
            """
        )

    st.divider()

    # ── Mode selector ──────────────────────────────────────────────────────
    grover_mode = st.radio(
        "Mode",
        ["Full Run (3 iterations)", "Step-by-Step Explorer"],
        horizontal=True,
        key="grover_mode_radio",
        help="Full Run executes 3 Grover iterations at once. Step-by-Step lets you advance one iteration at a time and watch the amplitude change.",
    )

    target = st.slider(
        "Select target item to search for (0 – 15)",
        min_value=0, max_value=15, value=7,
        help="This is the item Grover's algorithm will amplify. The binary representation is shown in the results."
    )
    target_bits = format(target, "04b")[::-1]
    st.caption(f"Target {target} in binary (little-endian): `|{target_bits}⟩`")

    # ── Step-by-Step Explorer ──────────────────────────────────────────────
    if grover_mode == "Step-by-Step Explorer":
        # Session state initialisation
        if "grover_iteration" not in st.session_state:
            st.session_state["grover_iteration"] = 0
        if "grover_target_locked" not in st.session_state:
            st.session_state["grover_target_locked"] = target

        # Reset when target changes
        if st.session_state["grover_target_locked"] != target:
            st.session_state["grover_iteration"] = 0
            st.session_state["grover_target_locked"] = target

        k = st.session_state["grover_iteration"]
        N_space = 16
        theta = np.arcsin(1.0 / np.sqrt(N_space))

        # Metric row
        grover_prob   = np.sin((2 * k + 1) * theta) ** 2
        classical_prob = 1 - (15 / 16) ** max(k, 1) if k > 0 else 1 / 16

        st.divider()
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Iterations used", k, help="Number of oracle + diffuser cycles applied")
        mc2.metric(
            "Grover P(success)",
            f"{grover_prob:.1%}",
            f"{grover_prob - classical_prob:+.1%} vs classical" if k > 0 else None,
        )
        mc3.metric("Classical P(found)", f"{classical_prob:.1%}",
                   help="Probability of finding by random sampling after same # of queries")

        # Navigation buttons
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        with btn_col1:
            if st.button("⏮ Reset", key="grover_reset"):
                st.session_state["grover_iteration"] = 0
                st.rerun()
        with btn_col2:
            if st.button("◀ Previous", key="grover_prev",
                         disabled=(k == 0)):
                st.session_state["grover_iteration"] = max(0, k - 1)
                st.rerun()
        with btn_col3:
            if st.button("Next Iteration ▶", key="grover_next",
                         disabled=(k >= 3), type="primary"):
                st.session_state["grover_iteration"] = min(3, k + 1)
                st.rerun()
        with btn_col4:
            if st.button("⏭ Jump to Optimal (3)", key="grover_jump",
                         disabled=(k == 3)):
                st.session_state["grover_iteration"] = 3
                st.rerun()

        # Amplitude chart (analytical — instantaneous, no quantum run needed)
        st.markdown("#### Amplitude Evolution")
        if k == 0:
            st.markdown(
                "All 16 states have **equal amplitude** 1/√16 ≈ 0.25 after the Hadamard layer. "
                "No state is preferred yet — click **Next Iteration** to apply the oracle and diffuser."
            )
        else:
            st.markdown(
                f"After **{k} iteration{'s' if k > 1 else ''}**: the target state amplitude has grown to "
                f"**{np.sin((2*k+1)*theta):.4f}** (probability **{grover_prob:.1%}**). "
                "Every other state is being suppressed by destructive interference."
            )
        amp_fig = plotly_viz.plotly_amplitude_step(target, k)
        st.plotly_chart(amp_fig, use_container_width=True)

        # Actual measurement counts (cached quantum circuit)
        if k > 0:
            with st.spinner(f"Running Grover circuit (k={k}) on AerSimulator…"):
                _, step_counts = cached_grover_step(target, k)
            target_state_bs = format(target, "04b")[::-1]
            found = step_counts.get(target_state_bs, 0)
            total = sum(step_counts.values())
            st.markdown(
                f"**Measurement simulation** ({total} shots): target `|{target_state_bs}⟩` "
                f"found **{found} times ({100*found/total:.1f}%)**"
            )
            counts_fig = plotly_viz.plotly_grover_counts(step_counts, target)
            st.plotly_chart(counts_fig, use_container_width=True)
        else:
            st.info("No measurement yet — apply at least 1 iteration to run the circuit.")

        if k == 3:
            st.success(
                f"Optimal! After 3 iterations, Grover's finds the target with **{grover_prob:.1%}** probability "
                f"— classical random search achieves only **{classical_prob:.1%}** in the same 3 queries."
            )

        # Complexity comparison (always visible)
        st.divider()
        st.markdown("#### Classical vs Quantum: Queries to Succeed")
        comp_fig = plotly_viz.plotly_complexity_comparison()
        st.plotly_chart(comp_fig, use_container_width=True)

    else:
        # ── Full Run mode ──────────────────────────────────────────────────
        run_grover_btn = st.button("Run Grover's Algorithm", key="run_grover", type="primary")

        if run_grover_btn:
            if "backends" not in st.session_state:
                st.error("Please connect to a backend first (use the sidebar).")
            else:
                backends = st.session_state["backends"]
                grover_backend = backends["grover_backend"]

                if backends["grover_mode"] == "real":
                    st.info(
                        f"Job submitted to **{backends['grover_backend_name']}**. "
                        "Waiting in IBM queue — this may take a few minutes…"
                    )

                try:
                    with st.spinner("Running Grover's algorithm…"):
                        circuit, counts = run_grover(target, grover_backend)
                    st.success(f"Job complete on **{backends['grover_backend_name']}**!")
                    show_results_grover(circuit, counts, target)

                except Exception as e:
                    st.error(f"Error on real hardware: {e}")
                    st.info("Auto-switching to AerSimulator fallback…")
                    try:
                        with st.spinner("Retrying on AerSimulator…"):
                            circuit, counts = run_grover(target, AerSimulator())
                        st.success("Fallback run complete on AerSimulator!")
                        show_results_grover(circuit, counts, target)
                    except Exception as e2:
                        st.error(f"Fallback also failed: {e2}")

# ===========================================================================
# SHOR TAB
# ===========================================================================
with tab_shor:
    st.subheader("Shor's Factoring Algorithm")

    # Intro explanation
    st.markdown(
        """
        **The problem:** Given a large integer N, find its prime factors.
        Classical algorithms (trial division, general number field sieve) scale
        **super-polynomially** — RSA-2048 would take longer than the age of the universe
        on the best classical hardware.

        **The quantum insight:** Shor's algorithm (Peter Shor, 1994) reduces factoring to
        **period finding**, which quantum computers can solve in polynomial time using
        the **Quantum Fourier Transform (QFT)**. This is an *exponential* speedup over
        the best known classical algorithms — and is the reason quantum computers threaten RSA.

        This demo factors **N = 15** into **3 × 5** using 8 qubits (4 counting + 4 work).
        """
    )

    with st.expander("How does it work? (Step-by-step)"):
        st.markdown(
            """
            **The key mathematical insight:**
            To factor N, pick a random integer `a` coprime to N. The function
            `f(x) = aˣ mod N` is **periodic** — it repeats with some period `r`.
            Once you know `r`, you can compute:

            > factor₁ = gcd(a^(r/2) − 1, N)
            > factor₂ = gcd(a^(r/2) + 1, N)

            The hard part is finding `r`. Classically this takes exponential time.
            Quantum computers can do it efficiently using the QFT.

            ---

            **The quantum circuit (this demo: a = 2, N = 15):**

            **Step 1 — Initialize**
            - 4 counting qubits → put into superposition with Hadamard gates
            - 4 work qubits → initialized to |0001⟩ (representing the value 1)

            **Step 2 — Controlled modular exponentiation**
            Apply controlled-Uᵏ gates, where U = "multiply by 2 mod 15".
            Each counting qubit k controls U^(2ᵏ). This entangles the counting register
            with the work register in a way that encodes all values of 2ˣ mod 15 simultaneously.

            The multiplication cycle for a=2, N=15 is:
            `1 → 2 → 4 → 8 → 1 → 2 → ...` (period r = 4)

            **Step 3 — Inverse Quantum Fourier Transform**
            Apply the inverse QFT to the counting register. This converts the periodic
            pattern in the work register into sharp peaks at multiples of 2ⁿ/r in the
            counting register. Measuring the counting register reveals phase information
            from which r can be recovered.

            **Step 4 — Classical post-processing**
            Use the **continued fractions algorithm** to extract r from the measured phase,
            then compute the factors via gcd.
            """
        )

    st.divider()

    # ── Parameter exploration ──────────────────────────────────────────────
    col_shor_a, col_shor_info = st.columns([1, 2])
    with col_shor_a:
        a_value = st.selectbox(
            "Base a (coprime to 15)",
            VALID_A_VALUES,
            index=0,
            help="The random base for modular exponentiation. Different values give different circuits but all factor 15 = 3 × 5.",
        )
    with col_shor_info:
        expected_r = PERIOD_TABLE[a_value]
        st.markdown(
            f"**Selected:** a = {a_value}  |  Expected period: **r = {expected_r}**  |  "
            f"{a_value}^r mod 15 = {pow(a_value, expected_r, 15)} ✓"
        )

    with st.expander("Period table for all valid bases"):
        rows = "| a | Period r | aʳ mod 15 | Gives factors? |\n|---|----------|-----------|----------------|\n"
        for av in VALID_A_VALUES:
            r_ = PERIOD_TABLE[av]
            ok = "✓" if r_ % 2 == 0 else "✗ (odd period)"
            rows += f"| {av} | {r_} | {pow(av, r_, 15)} | {ok} |\n"
        st.markdown(rows)
        st.caption(
            "Bases with odd period (11, 14) don't directly yield factors via gcd — "
            "Shor's algorithm would retry with a different a. Bases 2, 4, 7, 8, 13 all work."
        )

    run_shor_btn = st.button("Run on Simulator", key="run_shor", type="primary")

    if run_shor_btn:
        try:
            with st.spinner(f"Running Shor's algorithm  (a={a_value}, N=15)  on AerSimulator (2048 shots)…"):
                circuit, counts, period, factors = cached_run_shor(a_value)

            st.success(
                f"Factoring complete!  a = {a_value}  →  Detected period **r = {period}**  →  "
                f"**15 = {factors[0]} × {factors[1]}**"
            )

            st.divider()

            # --- Circuit ---
            st.markdown("### Circuit Diagram")
            st.markdown(
                f"""
                This 8-qubit circuit is the full Shor's algorithm for N=15, **a={a_value}**.

                **Top 4 wires — counting register (`count`):**
                These start in superposition (H gates) and their final measurement reveals the
                phase of the modular exponentiation. After the inverse QFT, peaks appear at
                positions that are multiples of 2⁴/r, which encodes the period r.

                **Bottom 4 wires — work register (`work`):**
                Initialized to |0001⟩. The controlled-U gates compute `{a_value}^x mod 15`
                coherently. The work register is *not* measured — it entangles with the
                counting register as a quantum scratchpad.

                **Controlled-U gates:** Each one represents "multiply the work register by
                {a_value} mod 15, controlled on the corresponding counting qubit."

                **QFT⁻¹ block:** The inverse Quantum Fourier Transform converts the periodic
                entanglement into measurable phase peaks.
                """
            )
            fig_circuit = shor_viz.plot_circuit(circuit)
            st.pyplot(fig_circuit)
            plt_close(fig_circuit)

            st.divider()

            # --- Counts ---
            st.markdown("### Phase Estimation Measurement Results")
            st.markdown(
                """
                After the inverse QFT, measuring the 4-qubit counting register should
                produce peaks at **multiples of 2ⁿ/r**, where n=4 counting qubits and r is the period.

                For N=15, a=2, the true period is **r = 4**, so peaks appear at:
                > 2⁴/4 = **4** → binary `0100` (decimal 4)
                > 2×(2⁴/4) = **8** → binary `1000` (decimal 8)
                > 3×(2⁴/4) = **12** → binary `1100` (decimal 12)
                > 4×(2⁴/4) = **0** → binary `0000` (decimal 0, trivial)

                **Look for roughly equal peaks at `0000`, `0100`, `1000`, `1100`** — four peaks
                for a period-4 function. The peaks won't be perfectly equal due to quantum noise
                and finite sampling, but the pattern is clear.
                """
            )
            fig_counts = plotly_viz.plotly_shor_phases(counts, a=a_value)
            st.plotly_chart(fig_counts, use_container_width=True)

            with st.expander("How do we get the period from these measurements?"):
                st.markdown(
                    f"""
                    Take any non-zero peak measurement, say `0100` = decimal **4**.

                    The measured value encodes a **phase** φ = measured / 2ⁿ = 4/16 = **1/4**.

                    Apply the **continued fractions algorithm** to find the best rational
                    approximation p/q of φ with q ≤ N:

                    > φ = 1/4  →  p/q = 1/4  →  **r = q = 4**

                    Verify: 2⁴ mod 15 = 16 mod 15 = **1** ✓  (confirms r = 4)

                    Similarly, peak `1000` = 8 gives φ = 8/16 = 1/2 → p/q = 1/2 → r = 2.
                    But 2² mod 15 = 4 ≠ 1, so r=2 fails the check — we'd try the next peak.
                    Peak `1100` = 12 gives φ = 12/16 = 3/4 → p/q = 3/4 → r = 4 ✓ (same result).

                    In the code, `_extract_period()` in `algorithms/shor.py` does exactly this:
                    iterates through peaks by frequency and returns the first r that satisfies
                    `a^r mod N == 1`.
                    """
                )

            st.divider()

            # --- Derivation ---
            st.markdown("### Factor Derivation")
            if period % 2 == 0:
                half_power = a_value ** (period // 2)
                st.markdown(
                    f"""
                    With the period **r = {period}** confirmed, the final step is pure classical math
                    using the **greatest common divisor (gcd)**:

                    The key theorem (Euler's): if `a^r ≡ 1 (mod N)` then `(a^(r/2))² ≡ 1 (mod N)`,
                    which means `N` divides `(a^(r/2) − 1)(a^(r/2) + 1)`.
                    So the factors of N are "hiding" in gcd(a^(r/2) ± 1, N).

                    | Computation | Value |
                    |-------------|-------|
                    | a = {a_value}, r = {period}, so a^(r/2) = {a_value}^{period//2} | = **{half_power}** |
                    | gcd({half_power} − 1, 15) = gcd({half_power-1}, 15) | = **{factors[0]}** |
                    | gcd({half_power} + 1, 15) = gcd({half_power+1}, 15) | = **{factors[1]}** |
                    | Result | **15 = {factors[0]} × {factors[1]}** ✓ |
                    """
                )
            else:
                st.markdown(
                    f"""
                    Period **r = {period}** is **odd** for a = {a_value}.
                    This means the standard gcd trick doesn't apply directly — in a full implementation
                    of Shor's algorithm the circuit would retry with a different random base.
                    The known factors 3 and 5 are returned as the result.

                    | Result | **15 = {factors[0]} × {factors[1]}** |
                    """
                )
            fig_deriv = shor_viz.plot_derivation(period, factors)
            st.pyplot(fig_deriv)
            plt_close(fig_deriv)

            with st.expander("Why does this threaten RSA encryption?"):
                st.markdown(
                    """
                    **RSA security** relies entirely on the fact that factoring large numbers is
                    computationally infeasible classically. An RSA-2048 key is the product of two
                    ~1024-bit primes. The best classical algorithm (GNFS) would take ~10¹⁵ years.

                    Shor's algorithm running on a fault-tolerant quantum computer with ~4,000 logical
                    qubits (millions of physical qubits after error correction) could factor RSA-2048
                    in **hours to days**.

                    This is why NIST finalized post-quantum cryptography standards in 2024
                    (CRYSTALS-Kyber for key exchange, CRYSTALS-Dilithium for signatures) — both
                    based on lattice problems that even quantum computers can't efficiently solve
                    with known algorithms.

                    **Current reality:** Today's quantum hardware (NISQ era) can only run this demo
                    on N=15 — not anywhere near RSA-scale. We need millions of physical qubits with
                    low error rates to threaten real cryptography.
                    """
                )

            st.divider()

            # --- Period Function ---
            st.markdown("### Why Period Finding = Factoring")
            st.markdown(
                """
                Before showing the complexity gap, here's the geometric intuition behind *why*
                finding the period of f(x) = aˣ mod N lets you factor N.

                **Left panel:** Plot f(x) = 2ˣ mod 15 for x = 0…15. The output cycles through
                {1, 2, 4, 8} and repeats — the period r = 4 is visually obvious. Each color
                represents one phase of the cycle.

                **Right panel:** Apply the Discrete Fourier Transform to those same values.
                Sharp peaks appear at k = 0, 4, 8, 12 — multiples of 2⁴/r = 4. This is exactly
                what the **Quantum Fourier Transform** produces when it runs on the counting register.
                The QFT turns a periodic time-domain signal into isolated frequency peaks,
                and measuring those peaks reveals the period.

                This is the quantum analogue of how audio software detects musical pitch — but
                running on quantum superposition, evaluating all x simultaneously.
                """
            )
            fig_period = shor_viz.plot_period_function()
            st.pyplot(fig_period)
            plt_close(fig_period)

            st.divider()

            # --- Factoring Complexity ---
            st.markdown("### Complexity Gap: Classical GNFS vs Shor's Algorithm")
            st.markdown(
                """
                This is where the **exponential vs polynomial** difference becomes visceral.

                - **Classical (GNFS):** The best classical factoring algorithm. Complexity grows as
                  exp(c · n^(1/3) · log(n)^(2/3)) where n = number of bits in N.
                  It's sub-exponential but still astronomically large for real key sizes.
                - **Shor's:** Polynomial — O(n³) in the number of bits. The curve barely rises.

                The shaded region is where quantum wins. The dashed horizontal lines show
                reference scales: how many operations a 1 GHz classical computer can do per second,
                and the operations-per-second equivalent of the age of the universe.

                RSA-2048 sits firmly in the "older than the universe" zone classically,
                and well within the polynomial curve for Shor's.
                """
            )
            fig_complexity = shor_viz.plot_factoring_complexity()
            st.pyplot(fig_complexity)
            plt_close(fig_complexity)

            st.divider()

            # --- RSA Time to Break ---
            st.markdown("### Time to Break RSA: Classical vs Quantum")
            st.markdown(
                """
                The most concrete way to see the threat: **how long would it actually take**
                to break each RSA key size?

                The bars show log₁₀(seconds) — each unit is a factor of 10 in time.
                Classical times (red) stretch far off-screen for large keys;
                quantum times (purple) barely budge as key size grows.

                Note: the quantum estimates assume a **large-scale fault-tolerant** quantum
                computer that doesn't exist yet. Today's NISQ hardware can't run Shor's
                on anything larger than N=15. But this is the trajectory the field is on.
                """
            )
            fig_rsa = shor_viz.plot_rsa_time_to_break()
            st.pyplot(fig_rsa)
            plt_close(fig_rsa)

        except Exception as e:
            st.error(f"Error running Shor's algorithm: {e}")


# ===========================================================================
# QAOA TAB
# ===========================================================================
with tab_qaoa:
    st.subheader("QAOA: Quantum Approximate Optimization Algorithm")

    st.markdown(
        """
        **The problem:** Maximum Cut (Max-Cut) — partition vertices of a graph so that the
        number (or total weight) of edges crossing between partitions is maximised.
        It is **NP-hard** in general and appears in logistics, finance, drug discovery,
        energy-grid optimisation, and social-network analysis.

        **The quantum insight:** QAOA (Farhi, Goldstone & Gutmann, 2014) is a *variational
        hybrid* algorithm that uses a parameterised quantum circuit to encode approximate
        solutions and a classical optimiser to tune the circuit angles.

        **New in this version:**
        - **Weighted Max-Cut** — edges carry real-valued weights; the circuit scales ZZ rotation
          angles accordingly.
        - **Zero-Noise Extrapolation (ZNE)** — run at noise scales λ=1–4, fit a polynomial,
          extrapolate to λ=0 for a noiseless estimate.
        - **Goemans–Williamson SDP** — classical 0.878-approximation benchmark solved via
          semidefinite programming + hyperplane rounding, providing a direct quantum vs. classical
          comparison.
        """
    )

    with st.expander("How does QAOA work?"):
        st.markdown(
            """
            **Weighted cost Hamiltonian:**

            > H_C = Σ_{(u,v)∈E} w_{uv} · (I − Z_u Z_v) / 2

            Each ZZ term is implemented as **CNOT – Rz(2·w·γ) – CNOT**.  Heavier edges
            rotate further, naturally prioritising high-weight cuts.

            **Circuit layers (p):**
            - Cost layer: apply exp(−i γ H_C) — phase-encodes the objective
            - Mixer layer: apply exp(−i β Σ Xᵢ) — mixes amplitudes to explore the space

            **Classical outer loop:** measure ⟨C⟩, adjust (γ, β) to maximise it.
            """
        )

    with st.expander("What is Zero-Noise Extrapolation?"):
        st.markdown(
            """
            ZNE is an *error mitigation* technique that doesn't require extra qubits.

            1. Run the circuit at noise levels λ=1, 2, 3, 4 (achieved by scaling the noise
               model, or on real hardware by *gate folding*: replace G → G·G†·G).
            2. Fit a polynomial to ⟨C(λ)⟩.
            3. Extrapolate to λ=0 — the estimated noiseless value.

            ZNE is used in production by IBM Quantum and Google for NISQ experiments and is
            far more powerful than measurement error mitigation alone, since it also addresses
            gate errors.
            """
        )

    with st.expander("What is the Goemans–Williamson algorithm?"):
        st.markdown(
            """
            GW (1995) is the best known *classical* polynomial-time approximation algorithm
            for Max-Cut.  It achieves ≥ **0.878 × C*** in expectation.

            **Steps:**
            1. Solve the SDP relaxation: assign a unit vector **v**ᵢ ∈ ℝⁿ to each vertex so
               that the objective Σ w_{uv}(1−**v**ᵤ·**v**ᵥ)/2 is maximised.
            2. Choose a random hyperplane (unit vector **r**).
            3. Assign vertex i to set 0 if **v**ᵢ·**r** ≥ 0, else set 1.
            4. Repeat 200 times, take the best cut.

            Comparing QAOA against GW directly answers the core question:
            *"Does the quantum circuit beat the best classical polynomial-time method?"*
            """
        )

    with st.expander("Societal impact of quantum optimisation"):
        st.markdown(
            """
            **Logistics:** Max-Cut and its relatives (QUBO, Ising models) underpin vehicle
            routing, warehouse zone assignment, and supply-chain balancing.

            **Finance:** Risk-minimising portfolio diversification maps to finding minimum-weight
            cuts in asset-correlation graphs.

            **Energy:** Splitting power grids into balanced partitions reduces inter-zone
            transmission losses — critical for integrating renewable energy sources.

            **Drug discovery:** Protein folding and docking problems reduce to QUBO, the same
            class as weighted Max-Cut.

            **Machine learning:** Quantum-enhanced clustering and graph neural networks use
            QAOA-like circuits as subroutines.

            **Current reality:** NISQ hardware (50–1000 qubits, ~0.5–1% CX error) can run
            QAOA at p=1–3 for graphs up to ~20 nodes.  Fault-tolerant machines are estimated
            5–15 years away but will unlock industrially relevant problem sizes.
            """
        )

    st.divider()

    # ── Controls ────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        graph_choice = st.selectbox("Select graph", list(PRESET_GRAPHS.keys()))
    with col_b:
        p_layers = st.selectbox("QAOA layers (p)", [1, 2, 3], index=0)
    with col_c:
        use_weights = st.toggle("Weighted edges", value=True,
                                help="Use non-uniform edge weights in the cost Hamiltonian.")

    graph_data = PRESET_GRAPHS[graph_choice]
    n_nodes    = graph_data["n_nodes"]
    edges      = graph_data["edges"]
    pos        = graph_data["pos"]
    weights    = graph_data.get("weights") if use_weights else None

    # Compute max_cut dynamically (works for both weighted and unweighted)
    max_cut, best_partition = classical_max_cut(n_nodes, edges, weights)

    st.caption(
        f"**{n_nodes} nodes · {len(edges)} edges** · "
        f"{'Weighted' if use_weights else 'Unweighted'} · "
        f"Classical optimum C* = **{max_cut:.2f}** · "
        f"{n_nodes} qubits"
    )

    # ── Guess the Cut Challenge ────────────────────────────────────────────
    st.divider()
    st.markdown("### Challenge: Guess the Maximum Cut")
    st.markdown(
        "Assign each node to **Set 0 (blue)** or **Set 1 (red)** to maximise the number of "
        "cut edges. Then submit your guess to see how it compares to QAOA and the classical optimum!"
    )

    # Reset challenge state when graph or weights change
    _challenge_key = f"{graph_choice}_{use_weights}"
    if st.session_state.get("qaoa_guess_graph_key") != _challenge_key:
        st.session_state["qaoa_guess_graph_key"]  = _challenge_key
        st.session_state["qaoa_guess_partition"]  = {i: 0 for i in range(n_nodes)}
        st.session_state["qaoa_guess_submitted"]  = False
        st.session_state["qaoa_challenge_result"] = None

    # Node coloring controls
    node_cols = st.columns(n_nodes)
    for i, col in enumerate(node_cols):
        with col:
            choice = st.radio(
                f"Node {i}",
                options=[0, 1],
                format_func=lambda x: "Blue S₀" if x == 0 else "Red S₁",
                key=f"qaoa_node_{i}_{_challenge_key}",
                index=st.session_state["qaoa_guess_partition"].get(i, 0),
            )
            st.session_state["qaoa_guess_partition"][i] = choice

    # Live cut value
    user_partition = [st.session_state["qaoa_guess_partition"].get(i, 0) for i in range(n_nodes)]
    user_cut = sum(
        (weights.get((u, v)) or weights.get((v, u)) or 1.0) if weights else 1.0
        for (u, v) in edges
        if user_partition[u] != user_partition[v]
    )

    uc1, uc2 = st.columns(2)
    uc1.metric(
        "Your current cut value",
        f"{user_cut:.2f}",
        f"{user_cut / max_cut:.0%} of optimal C* = {max_cut:.2f}",
    )

    # Live graph preview updates as user toggles nodes
    with uc2:
        preview_fig = qaoa_viz.plot_graph(
            n_nodes, edges, pos,
            partition=user_partition,
            weights=weights,
            title=f"Your Partition (cut = {user_cut:.2f})",
        )
        st.pyplot(preview_fig)
        plt_close(preview_fig)

    # Submit button
    submit_btn = st.button("Submit My Cut & Run QAOA", key="qaoa_submit_guess", type="primary")

    if submit_btn:
        st.session_state["qaoa_guess_submitted"] = True
        st.session_state["qaoa_guess_user_cut"]  = user_cut
        st.session_state["qaoa_guess_partition_snap"] = user_partition[:]

    # Show challenge results if submitted
    if st.session_state.get("qaoa_guess_submitted"):
        locked_user_cut = st.session_state.get("qaoa_guess_user_cut", user_cut)
        locked_partition = st.session_state.get("qaoa_guess_partition_snap", user_partition)

        # Run QAOA optimisation (cached)
        with st.spinner("Running QAOA optimisation…"):
            try:
                edges_tuple   = tuple(tuple(e) for e in edges)
                weights_frozen = frozenset(weights.items()) if weights else None
                opt_gamma_c, opt_beta_c, opt_exp_c, _ = cached_optimize_qaoa(
                    n_nodes, edges_tuple, 1, 10, 256, weights_frozen
                )
                qaoa_cut = opt_exp_c
            except Exception:
                qaoa_cut = None

        if not HAS_CVXPY:
            gw_cut = None
        else:
            with st.spinner("Running Goemans-Williamson…"):
                try:
                    gw_res = goemans_williamson(n_nodes, edges, weights=weights, n_rounds=100)
                    gw_cut = gw_res["gw_cut"]
                except Exception:
                    gw_cut = None

        st.divider()
        st.markdown("#### Your Results vs Algorithms")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Your Cut",          f"{locked_user_cut:.2f}",
                   f"{locked_user_cut/max_cut:.0%} of C*")
        rc2.metric("QAOA Ideal ⟨C⟩",   f"{qaoa_cut:.3f}" if qaoa_cut is not None else "—",
                   f"{qaoa_cut/max_cut:.0%} of C*" if qaoa_cut else None)
        rc3.metric("Goemans-Williamson", f"{gw_cut:.2f}" if gw_cut is not None else "—",
                   f"{gw_cut/max_cut:.0%} of C*" if gw_cut else None)
        rc4.metric("Classical C*",       f"{max_cut:.2f}", "100%", delta_color="off")

        if qaoa_cut is not None and locked_user_cut >= qaoa_cut:
            st.success(
                f"You beat QAOA! Your cut ({locked_user_cut:.2f}) ≥ QAOA ({qaoa_cut:.3f}). "
                "Human intuition wins this round — try a harder graph!"
            )
        elif gw_cut is not None and locked_user_cut >= gw_cut:
            st.info(
                f"You matched or beat the Goemans-Williamson algorithm ({gw_cut:.2f})! "
                "That's impressive — GW is the best known classical approximation."
            )
        elif qaoa_cut is not None:
            st.warning(
                f"QAOA ({qaoa_cut:.3f}) beat your cut ({locked_user_cut:.2f}). "
                "Try recolouring the nodes — can you do better?"
            )

        st.caption("Scroll down to run the full QAOA analysis and see all the details.")

    st.divider()

    run_btn = st.button("Run Full QAOA Analysis", type="primary", key="run_qaoa")

    if run_btn:
        # ── 1. Classical solution ──────────────────────────────────────────
        best_cut_val, best_partition = classical_max_cut(n_nodes, edges, weights)

        st.divider()
        st.markdown("### Graph and Classical Optimal Partition")
        weight_note = " — edge labels show weights" if use_weights else ""
        st.markdown(
            f"Brute-force optimal: **{best_cut_val:.2f}** weighted edges cut{weight_note}. "
            "Blue = set S₀, red = set S₁.  Orange edges are cut; dashed grey edges are not."
        )
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig = qaoa_viz.plot_graph(n_nodes, edges, pos, weights=weights, title="Input Graph")
            st.pyplot(fig); plt_close(fig)
        with col_g2:
            fig = qaoa_viz.plot_graph(n_nodes, edges, pos, partition=best_partition,
                                      weights=weights, title=f"Optimal Cut (C*={best_cut_val:.2f})")
            st.pyplot(fig); plt_close(fig)

        with st.expander("Why does edge weight change the QAOA circuit?"):
            st.markdown(
                """
                In the weighted cost Hamiltonian H_C = Σ w_{uv}(I−Z_u Z_v)/2, each term
                contributes proportionally to its weight.  The corresponding ZZ rotation angle
                becomes **Rz(2·w·γ)** — heavier edges rotate further, encoding more phase
                information about that edge's importance.  The optimiser then learns angles that
                simultaneously maximise the total weighted cut.

                For unweighted graphs (all w=1), the circuit reduces to the standard QAOA formulation.
                Weighted QAOA is strictly more general and handles all real-world problem instances.
                """
            )

        st.divider()

        # ── 2. Parameter optimisation ──────────────────────────────────────
        st.markdown(f"### Parameter Optimisation  (p = {p_layers})")
        weight_str = "weighted " if use_weights else ""
        st.markdown(
            f"Grid search over {10}×{10} (γ,β) pairs maximising {weight_str}⟨C⟩ (256 shots each), "
            "then Nelder-Mead refinement."
        )

        with st.spinner("Optimising parameters…"):
            try:
                opt_gamma, opt_beta, opt_exp, ld = optimize_qaoa_params(
                    n_nodes, edges, p=p_layers, grid_size=10, grid_shots=256, weights=weights
                )
                st.success(
                    f"γ* = {[f'{g:.3f}' for g in opt_gamma]}, "
                    f"β* = {[f'{b:.3f}' for b in opt_beta]}  →  "
                    f"⟨C⟩ = **{opt_exp:.3f}** / C* = {max_cut:.2f}  "
                    f"(ratio = {opt_exp/max_cut:.3f})"
                )
            except Exception as e:
                st.error(f"Optimisation failed: {e}")
                opt_gamma, opt_beta, opt_exp, ld = [0.5]*p_layers, [0.3]*p_layers, 0.0, None

        if ld is not None and p_layers == 1:
            fig = plotly_viz.plotly_optimization_landscape(
                ld[0], ld[1], ld[2], opt_gamma[0], opt_beta[0],
                graph_choice.split("(")[0].strip()
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── 3. Circuit ─────────────────────────────────────────────────────
        opt_circuit = build_qaoa_circuit(n_nodes, edges, opt_gamma, opt_beta, p_layers, weights=weights)
        gate_ops    = opt_circuit.count_ops()
        depth       = opt_circuit.depth(filter_function=lambda i: i.operation.name != "measure")

        st.markdown(f"### QAOA Circuit  (p = {p_layers})")
        st.markdown(
            f"Depth = **{depth}** · "
            f"Total gates = **{sum(gate_ops.values()) - gate_ops.get('measure',0)}** · "
            f"CX = **{gate_ops.get('cx',0)}**"
            + (" · Rz angles scaled by edge weights" if use_weights else "")
        )
        fig = qaoa_viz.plot_qaoa_circuit(opt_circuit)
        st.pyplot(fig); plt_close(fig)

        st.divider()

        # ── 4. Transpilation ───────────────────────────────────────────────
        st.markdown("### Transpilation Analysis  (Levels 0–3)")
        st.markdown(
            "Transpilation maps the abstract circuit to IBM native gates (CX, RZ, SX, X) on a "
            "5-qubit T-shape coupling map.  Higher optimisation levels reduce 2-qubit gate count, "
            "directly lowering error accumulation on real hardware."
        )
        with st.spinner("Transpiling at levels 0–3…"):
            try:
                transp_results = compare_transpilation_levels(opt_circuit)
            except Exception as e:
                st.error(f"Transpilation failed: {e}"); transp_results = []

        if transp_results:
            fig = qaoa_viz.plot_transpilation_comparison(transp_results)
            st.pyplot(fig); plt_close(fig)

            st.markdown("#### Qubit Mapping")
            fig = qaoa_viz.plot_qubit_mapping(
                transp_results, n_nodes, COUPLING_MAP_EDGES, PHYS_QUBIT_POS
            )
            st.pyplot(fig); plt_close(fig)

        st.divider()

        # ── 5. Ideal / Noisy / Mitigated ──────────────────────────────────
        st.markdown("### Ideal · Noisy · Measurement-Error-Mitigated")
        st.markdown(
            "Three simulation runs: **Ideal** (no noise), **Noisy** (full IBM model), "
            "**Mitigated** (noisy + calibration-matrix readout correction).  "
            "Mitigation corrects readout errors but not gate errors."
        )
        with st.spinner("Running ideal / noisy / mitigated (2048 shots each)…"):
            try:
                baseline_nm = build_ibm_noise_model(scale=1.0)
                comparison  = run_three_way_comparison(opt_circuit, baseline_nm, n_nodes, shots=2048)
            except Exception as e:
                st.error(f"Three-way comparison failed: {e}"); comparison = None

        if comparison:
            fig = qaoa_viz.plot_ideal_vs_noisy_vs_mitigated(
                comparison, edges, n_nodes, max_cut, weights=weights
            )
            st.pyplot(fig); plt_close(fig)
            fig = qaoa_viz.plot_solution_distribution(
                comparison, edges, n_nodes, max_cut, weights=weights
            )
            st.pyplot(fig); plt_close(fig)

            ideal_cut = compute_expected_cut(comparison["ideal"],     edges, n_nodes, weights=weights)
            noisy_cut = compute_expected_cut(comparison["noisy"],     edges, n_nodes, weights=weights)
            mitig_cut = compute_expected_cut(comparison["mitigated"], edges, n_nodes, weights=weights)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ideal ⟨C⟩",     f"{ideal_cut:.3f}", f"Ratio {ideal_cut/max_cut:.3f}")
            c2.metric("Noisy ⟨C⟩",     f"{noisy_cut:.3f}", f"−{ideal_cut-noisy_cut:.3f}")
            c3.metric("Mitigated ⟨C⟩", f"{mitig_cut:.3f}", f"+{mitig_cut-noisy_cut:.3f} vs noisy")
            c4.metric("Classical C*",   f"{max_cut:.2f}")

            st.divider()

            # ── 6. Noise sweep ─────────────────────────────────────────────
            st.markdown("### Performance vs Noise Strength")
            with st.spinner("Running noise sweep…"):
                try:
                    sweep = noise_sweep(opt_circuit, edges, n_nodes,
                                        scale_factors=[0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
                                        shots=1024, weights=weights)
                    fig = plotly_viz.plotly_noise_sweep(sweep, max_cut)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Noise sweep failed: {e}")

            st.divider()

            # ── 7. Depth–quality sweep ─────────────────────────────────────
            st.markdown("### Circuit Depth vs Solution Quality  (p = 1, 2, 3)")
            st.markdown(
                "More layers → better ideal approximation, but deeper circuits accumulate more "
                "noise.  This tension is the central challenge of NISQ-era quantum optimisation."
            )
            with st.spinner("Running depth–quality sweep…"):
                try:
                    dq = depth_quality_sweep(n_nodes, edges, max_cut, baseline_nm,
                                             p_values=[1,2,3], shots=1024, weights=weights)
                    fig = qaoa_viz.plot_depth_quality(dq, max_cut)
                    st.pyplot(fig); plt_close(fig)
                except Exception as e:
                    st.error(f"Depth–quality sweep failed: {e}")

            st.divider()

            # ── 8. ★ Zero-Noise Extrapolation (ZNE) ───────────────────────
            st.markdown("### Zero-Noise Extrapolation (ZNE)")
            st.markdown(
                """
                ZNE runs the circuit at noise levels **λ = 1, 2, 3, 4** (achieved here by
                scaling IBM noise model error rates by λ; on real hardware, *gate folding* is
                used instead).  A **linear** and a **quadratic** polynomial are fitted to the
                measured ⟨C(λ)⟩ values and extrapolated to λ=0 — the estimated noiseless result.

                This technique requires no extra qubits, works on any circuit, and is used in
                production by IBM Quantum and Google for their leading NISQ experiments.
                Comparing ZNE to the ideal simulation shows how close extrapolation gets to
                the true noiseless value.
                """
            )

            with st.spinner("Running ZNE (4 noise levels × 1024 shots)…"):
                try:
                    zne_result = run_zne(
                        opt_circuit, edges, n_nodes,
                        scale_factors=[1, 2, 3, 4],
                        shots=1024, weights=weights,
                    )
                    fig = qaoa_viz.plot_zne(zne_result, ideal_cut, max_cut)
                    st.pyplot(fig); plt_close(fig)

                    zl  = zne_result["zne_linear"]
                    zq  = zne_result["zne_quadratic"]
                    z1  = zne_result["scale_data"][0]["expected_cut"]  # noisy at λ=1

                    cz1, cz2, cz3, cz4 = st.columns(4)
                    cz1.metric("Noisy ⟨C⟩  (λ=1)", f"{z1:.3f}")
                    cz2.metric("ZNE Linear",         f"{zl:.3f}",
                               f"+{zl-z1:.3f} vs noisy")
                    if zq is not None:
                        cz3.metric("ZNE Quadratic",  f"{zq:.3f}",
                                   f"+{zq-z1:.3f} vs noisy")
                    cz4.metric("Ideal (target)",     f"{ideal_cut:.3f}")

                    with st.expander("Why does ZNE work?"):
                        st.markdown(
                            f"""
                            In the low-noise limit, the expected value ⟨C(λ)⟩ can be
                            Taylor-expanded around λ=0:

                            > ⟨C(λ)⟩ ≈ ⟨C⟩_ideal + a·λ + b·λ² + …

                            By evaluating this function at λ=1,2,3,4 we obtain a system of
                            equations that lets us back-solve for ⟨C⟩_ideal = ⟨C(0)⟩.

                            For this circuit:
                            - **Noisy baseline** (λ=1): ⟨C⟩ = {z1:.3f}
                            - **ZNE linear estimate**: {zl:.3f}  (improvement: +{zl-z1:.3f})
                            {'- **ZNE quadratic estimate**: ' + f'{zq:.3f}' if zq is not None else ''}
                            - **Ideal** (ground truth): {ideal_cut:.3f}

                            ZNE works best when the noise is truly weakly-coupled (each gate
                            introduces a small independent error).  It degrades if the circuit
                            is already deeply in the noise-dominated regime (scale > 3×).
                            """
                        )
                except Exception as e:
                    st.error(f"ZNE failed: {e}")

            st.divider()

            # ── 9. ★ Goemans–Williamson SDP ───────────────────────────────
            st.markdown("### Goemans–Williamson SDP Comparison")
            st.markdown(
                """
                The **Goemans–Williamson** (1995) algorithm is the best known classical
                polynomial-time approximation algorithm for weighted Max-Cut, achieving ≥ **0.878 × C***
                in expectation.  It solves a semidefinite programme (SDP) to find optimal unit vectors
                for each vertex, then uses **200 random hyperplane rounding** trials.

                Plotting GW alongside QAOA provides a direct classical benchmark:
                if QAOA exceeds the GW rounded cut on a hard graph, it demonstrates a form
                of quantum advantage.
                """
            )

            if not HAS_CVXPY:
                st.warning("cvxpy not installed — run `pip install cvxpy` to enable Goemans–Williamson.")
            else:
                with st.spinner("Solving SDP and running hyperplane rounding (200 rounds)…"):
                    try:
                        gw_result = goemans_williamson(n_nodes, edges, weights=weights, n_rounds=200)
                        fig = qaoa_viz.plot_gw_comparison(gw_result, ideal_cut, max_cut, graph_choice)
                        st.pyplot(fig); plt_close(fig)

                        gw_cut   = gw_result["gw_cut"]
                        sdp_bd   = gw_result["sdp_bound"]
                        gw_ratio = gw_result["approx_ratio"]
                        qa_ratio = ideal_cut / max_cut if max_cut > 0 else 0

                        cg1, cg2, cg3, cg4 = st.columns(4)
                        cg1.metric("Classical C*",       f"{max_cut:.3f}")
                        cg2.metric("GW SDP Bound",       f"{sdp_bd:.3f}",
                                   f"{sdp_bd/max_cut:.3f}×C*")
                        cg3.metric("GW Rounded",         f"{gw_cut:.3f}",
                                   f"{gw_ratio:.3f}×C*")
                        cg4.metric("QAOA (Ideal)",       f"{ideal_cut:.3f}",
                                   f"{qa_ratio:.3f}×C*")

                        if ideal_cut >= gw_cut:
                            st.success(
                                f"QAOA ({ideal_cut:.3f}) ≥ GW rounded ({gw_cut:.3f}) — "
                                "the quantum circuit matches or beats the classical approximation!"
                            )
                        else:
                            st.info(
                                f"GW rounded ({gw_cut:.3f}) > QAOA ({ideal_cut:.3f}) — "
                                "GW wins here.  Increasing p or optimising parameters further "
                                "could close the gap."
                            )

                        with st.expander("Understanding the SDP bound vs rounded cut"):
                            st.markdown(
                                f"""
                                The **SDP bound** ({sdp_bd:.3f}) is an *upper bound* on C* —
                                the SDP relaxation allows fractional solutions (unit vectors) rather
                                than binary partitions.  It is always ≥ the true Max-Cut.

                                The **GW rounded** cut ({gw_cut:.3f}) is what hyperplane rounding
                                achieves in the best of 200 trials.  The gap between the SDP bound
                                and the rounded cut is the *integrality gap* — an inherent feature
                                of rounding continuous solutions to binary ones.

                                The **GW guarantee** (0.878×C* = {0.878*max_cut:.3f} here) is a
                                worst-case lower bound over all graphs.  On many practical instances
                                GW performs significantly better.

                                QAOA explores the solution space via quantum superposition and
                                interference rather than SDP + rounding.  For some graph families
                                (e.g. dense random graphs at large p) QAOA can outperform GW — this
                                is an active research area and one of the most exciting open questions
                                in quantum optimisation.
                                """
                            )
                    except Exception as e:
                        st.error(f"Goemans–Williamson failed: {e}")

        st.divider()
        st.markdown("### Summary")
        st.markdown(
            f"""
            **Graph:** {graph_choice} · **p = {p_layers}** · {'Weighted' if use_weights else 'Unweighted'}

            | Method | ⟨C⟩ / Value | Ratio vs C*={max_cut:.2f} |
            |--------|------------|--------------------------|
            | Classical brute force | {max_cut:.3f} | 1.000 |
            | QAOA ideal | {opt_exp:.3f} | {opt_exp/max_cut:.3f} |
            | QAOA noisy (IBM model) | — | see plots |
            | ZNE extrapolated | — | see plots |
            | Goemans–Williamson | — | ≥ 0.878 (guaranteed) |

            **Key lessons:**
            - Weighted QAOA naturally handles non-uniform edge importance via scaled Rz angles.
            - ZNE recovers significant signal from noisy circuits with no extra qubits.
            - GW is a powerful classical benchmark — quantum advantage requires beating it consistently.
            - Deeper circuits (larger p) improve the ideal approximation but suffer more from noise,
              making error mitigation and fault-tolerant hardware essential for large-scale QAOA.
            """
        )
