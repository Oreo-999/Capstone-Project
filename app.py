import matplotlib.pyplot as plt
import streamlit as st
from qiskit_aer import AerSimulator

from config import get_backends
from algorithms.grover import run_grover
from algorithms.shor import run_shor
from algorithms.qaoa import (
    PRESET_GRAPHS,
    COUPLING_MAP_EDGES,
    PHYS_QUBIT_POS,
    build_qaoa_circuit,
    classical_max_cut,
    compute_expected_cut,
    approximation_ratio,
    optimize_qaoa_params,
    build_ibm_noise_model,
    compare_transpilation_levels,
    run_three_way_comparison,
    noise_sweep,
    depth_quality_sweep,
)
from visualizations import grover_viz, shor_viz, qaoa_viz


def plt_close(fig):
    """Close a matplotlib figure to avoid memory leaks across Streamlit reruns."""
    plt.close(fig)


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
    fig_counts = grover_viz.plot_counts(counts, target)
    st.pyplot(fig_counts)
    plt_close(fig_counts)

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
    fig_complexity = grover_viz.plot_complexity()
    st.pyplot(fig_complexity)
    plt_close(fig_complexity)

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
    use_fallback = st.toggle(
        "Use Simulator (skip IBM queue)",
        value=False,
        help="When ON, Grover's runs on AerSimulator instead of real IBM hardware.",
    )

    if st.button("Connect", use_container_width=True):
        with st.spinner("Connecting to backend…"):
            backends = get_backends(
                api_key=api_key or None, use_simulator_fallback=use_fallback
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

    target = st.slider(
        "Select target item to search for (0 – 15)",
        min_value=0, max_value=15, value=7,
        help="This is the item Grover's algorithm will amplify. The binary representation is shown in the results."
    )
    target_bits = format(target, "04b")[::-1]
    st.caption(f"Target {target} in binary (little-endian): `|{target_bits}⟩`")

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

    run_shor_btn = st.button("Run on Simulator", key="run_shor", type="primary")

    if run_shor_btn:
        try:
            with st.spinner("Running Shor's algorithm on AerSimulator (2048 shots)…"):
                circuit, counts, period, factors = run_shor()

            st.success(
                f"Factoring complete!  Detected period **r = {period}**  →  "
                f"**15 = {factors[0]} × {factors[1]}**"
            )

            st.divider()

            # --- Circuit ---
            st.markdown("### Circuit Diagram")
            st.markdown(
                """
                This 8-qubit circuit is the full Shor's algorithm for N=15, a=2.

                **Top 4 wires — counting register (`count`):**
                These start in superposition (H gates) and their final measurement reveals the
                phase of the modular exponentiation. After the inverse QFT, peaks appear at
                positions that are multiples of 2⁴/r = 4, which encodes the period r.

                **Bottom 4 wires — work register (`work`):**
                Initialized to |0001⟩. The controlled-U gates act on this register to compute
                `a^x mod N` coherently. The work register is *not* measured — it's traced out
                after the computation, acting as a "scratchpad" that entangles with the counting register.

                **Controlled-U gates:** Each one represents "multiply the work register by 2 mod 15,
                controlled on the corresponding counting qubit." Internally implemented as
                a sequence of SWAP gates (cyclic left shift = multiply by 2 in binary for this case).

                **QFT⁻¹ block:** The inverse Quantum Fourier Transform at the end. This is the
                heart of the speedup — it converts the periodic entanglement into measurable phase peaks.
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
            fig_counts = shor_viz.plot_counts(counts)
            st.pyplot(fig_counts)
            plt_close(fig_counts)

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
            st.markdown(
                f"""
                With the period **r = {period}** confirmed, the final step is pure classical math
                using the **greatest common divisor (gcd)**:

                The key theorem (Euler's): if `a^r ≡ 1 (mod N)` then `(a^(r/2))² ≡ 1 (mod N)`,
                which means `N` divides `(a^(r/2) − 1)(a^(r/2) + 1)`.
                So the factors of N are "hiding" in gcd(a^(r/2) ± 1, N).

                | Computation | Value |
                |-------------|-------|
                | a = 2, r = {period}, so a^(r/2) = 2^{period//2} | = **{2**(period//2)}** |
                | gcd(2^{period//2} − 1, 15) = gcd({2**(period//2)-1}, 15) | = **{factors[0]}** |
                | gcd(2^{period//2} + 1, 15) = gcd({2**(period//2)+1}, 15) | = **{factors[1]}** |
                | Result | **15 = {factors[0]} × {factors[1]}** ✓ |
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
# QAOA TAB — Quantum Approximate Optimization Algorithm · Max-Cut
# ===========================================================================
with tab_qaoa:
    st.subheader("QAOA: Quantum Approximate Optimization Algorithm")

    # ── Introduction ────────────────────────────────────────────────────────
    st.markdown(
        """
        **The problem:** Maximum Cut (Max-Cut) asks you to partition the vertices of a graph
        into two sets so that the number of edges crossing between them is as large as possible.
        It is **NP-hard** in general — no polynomial-time classical algorithm is known.
        Yet it appears naturally in hundreds of real-world settings: logistics routing,
        portfolio optimisation, VLSI layout, community detection in social networks, and more.

        **The quantum insight:** QAOA (Farhi, Goldstone & Gutmann, 2014) is a *variational
        hybrid* algorithm — it uses a parameterised quantum circuit to prepare a quantum state
        that encodes approximate solutions, then uses classical optimisation to tune the
        circuit angles toward the optimal partition.

        Unlike Grover's or Shor's (which are fixed, exact algorithms), QAOA is *approximate*
        and *heuristic*: its quality improves as the number of circuit layers **p** grows,
        but so does its susceptibility to noise. This makes QAOA one of the most important
        algorithms for studying the **near-term quantum advantage** question on NISQ hardware.
        """
    )

    with st.expander("How does QAOA work? (Step-by-step)"):
        st.markdown(
            """
            **The Max-Cut cost function:**
            For a graph G = (V, E), assign each vertex a spin ±1.  The number of edges cut
            equals ½ Σ_{(u,v)∈E} (1 − zᵤ zᵥ), where zᵢ ∈ {+1, −1}.  In quantum notation,
            replace zᵢ with the Pauli Z operator; the cost Hamiltonian is:

            > H_C = ½ Σ_{(u,v)∈E} (I − Zᵤ Zᵥ)

            **The QAOA ansatz (p layers):**
            Starting from the equal superposition |+⟩^⊗n, alternate two unitary operators:

            | Operator | Formula | Role |
            |----------|---------|------|
            | **Cost** U_C(γ) | exp(−i γ H_C) | Encodes the objective — edges with high cost accumulate phase |
            | **Mixer** U_B(β) | exp(−i β Σᵢ Xᵢ) | Mixes amplitudes between states — prevents getting trapped |

            Each U_C layer is implemented as CNOT–Rz(2γ)–CNOT per edge.
            Each U_B layer is Rx(2β) on every qubit.

            **Classical optimisation loop:**
            Measure the circuit to estimate ⟨C⟩ = ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩.
            Adjust angles (γ, β) using a classical optimiser (grid search → Nelder-Mead) to
            maximise ⟨C⟩.  The best bitstring observed is the candidate solution.

            **Approximation guarantee (p=1):**
            For any graph, QAOA with p=1 achieves at least **0.6924 × C*** (the optimal cut)
            — better than the best known classical poly-time algorithm in some regimes.
            With p→∞, QAOA converges to exact (adiabatic theorem).
            """
        )

    with st.expander("Why is Max-Cut important for society?"):
        st.markdown(
            """
            **Logistics & Supply Chain**
            Graph partitioning underlies vehicle routing, warehouse zone assignment, and
            network traffic splitting. A near-optimal partition can reduce fuel costs and
            delivery times at scale — millions of dollars of savings for large operators.

            **Energy Grid Optimisation**
            Splitting a power network into balanced partitions minimises inter-zone power flow,
            reducing transmission losses.  Quantum-assisted grid optimisation could accelerate
            the transition to renewable energy by making distributed grids more efficient.

            **Finance & Portfolio Optimisation**
            Risk diversification in a portfolio is equivalent to finding a partition of assets
            that minimises correlation (cuts edges in a correlation graph).  Banks and hedge
            funds are actively investigating QAOA-like methods.

            **Drug Discovery & Molecular Simulation**
            Protein conformational analysis and drug-target binding problems map onto QUBO
            (Quadratic Unconstrained Binary Optimisation) — the same mathematical class as
            Max-Cut.  QAOA circuits could help identify candidate drug molecules faster.

            **Machine Learning**
            Training certain quantum neural networks and performing quantum clustering reduces
            to Max-Cut on similarity graphs.  Quantum-enhanced ML could speed up the training
            of large models.

            **Fundamental Science**
            QAOA is a testbed for *quantum advantage* on NISQ hardware.  Demonstrating that
            a quantum circuit beats all classical algorithms — even on a small graph — would
            be a landmark result in computer science.

            **Current Reality & Honest Limitations**
            Today's NISQ devices (50–1000 qubits, error rate ~0.1–1% per gate) can run QAOA
            circuits with p=1–3 on graphs of up to ~20 nodes.  Classical simulation remains
            competitive for small instances.  Fault-tolerant quantum computers (requiring
            ~1,000× more physical qubits per logical qubit) are needed for genuine quantum
            advantage on industrially relevant problem sizes.  The scientific community
            estimates this is 5–15 years away — but the algorithmic and hardware groundwork
            being laid today is critical.
            """
        )

    st.divider()

    # ── User Controls ────────────────────────────────────────────────────────
    col_cfg1, col_cfg2 = st.columns([1, 1])
    with col_cfg1:
        graph_choice = st.selectbox(
            "Select graph",
            list(PRESET_GRAPHS.keys()),
            index=0,
            help="The quantum register has one qubit per graph vertex.",
        )
    with col_cfg2:
        p_layers = st.selectbox(
            "QAOA layers  (p)",
            [1, 2, 3],
            index=0,
            help="More layers → better approximation but deeper (noisier) circuit.",
        )

    graph_data = PRESET_GRAPHS[graph_choice]
    n_nodes = graph_data["n_nodes"]
    edges   = graph_data["edges"]
    max_cut = graph_data["max_cut"]
    pos     = graph_data["pos"]

    st.caption(
        f"Graph: **{n_nodes} nodes**, **{len(edges)} edges** · "
        f"Optimal Max-Cut: **C* = {max_cut}** · "
        f"QAOA circuit uses **{n_nodes} qubits**"
    )

    run_qaoa_btn = st.button("Run Full QAOA Analysis", type="primary", key="run_qaoa")

    if run_qaoa_btn:
        # ── 1. Classical optimal solution ──────────────────────────────────
        with st.spinner("Computing classical optimal solution…"):
            best_cut_val, best_partition = classical_max_cut(n_nodes, edges)

        st.divider()
        st.markdown("### The Graph and Its Optimal Max-Cut")
        st.markdown(
            f"""
            The graph below has **{n_nodes} vertices** and **{len(edges)} edges**.
            The classical brute-force solution (feasible only for small graphs) finds
            the optimal partition that cuts **{best_cut_val}** edges — this is the
            benchmark against which we measure QAOA's performance.

            **Left** — the unpartitioned graph.
            **Right** — the optimal two-colouring.
            *Blue* vertices are in set S₀, *red* vertices in set S₁.
            *Orange* edges are cut; *grey dashed* edges are inside the same partition.
            """
        )

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_g_plain = qaoa_viz.plot_graph(n_nodes, edges, pos, title="Input Graph")
            st.pyplot(fig_g_plain)
            plt_close(fig_g_plain)
        with col_g2:
            fig_g_cut = qaoa_viz.plot_graph(
                n_nodes, edges, pos, partition=best_partition,
                title=f"Optimal Max-Cut (C* = {best_cut_val})"
            )
            st.pyplot(fig_g_cut)
            plt_close(fig_g_cut)

        with st.expander("Why is Max-Cut NP-hard?"):
            st.markdown(
                """
                For a graph with n vertices there are 2^(n−1) distinct partitions (up to
                reflection).  Checking all of them takes exponential time.  No polynomial-time
                classical algorithm is known that guarantees the optimum (unless P = NP),
                so the best classical algorithms are *approximation* algorithms.

                The **Goemans-Williamson** (1995) SDP-based algorithm achieves a 0.878 × C*
                approximation guarantee, the best known for general graphs.  QAOA with p=1
                achieves 0.6924 × C*, but the quantum approach has different scaling properties
                and may outperform classical methods on specific graph families.
                """
            )

        st.divider()

        # ── 2. Parameter optimisation ───────────────────────────────────────
        st.markdown("### Optimising QAOA Parameters  (γ, β)")
        st.markdown(
            f"""
            QAOA requires finding the angles **γ** (cost-layer rotation) and **β**
            (mixer-layer rotation) that maximise the expected cut value ⟨C⟩.

            For **p = {p_layers}** {'layer' if p_layers == 1 else 'layers'} there are
            **{2 * p_layers}** angles to optimise.  We use a **{10}×{10} grid search**
            followed by **Nelder-Mead** local refinement.  All grid circuits are run
            simultaneously in a single batch on the Aer simulator (256 shots each).
            """
        )

        with st.spinner("Running parameter optimisation (grid search + refinement)…"):
            try:
                opt_gamma, opt_beta, opt_exp, landscape_data = optimize_qaoa_params(
                    n_nodes, edges, p=p_layers, grid_size=10, grid_shots=256
                )
                st.success(
                    f"Optimal parameters found — γ* = {[f'{g:.3f}' for g in opt_gamma]}, "
                    f"β* = {[f'{b:.3f}' for b in opt_beta]}  →  "
                    f"⟨C⟩ = **{opt_exp:.3f}**  (C* = {max_cut}, "
                    f"ratio = {opt_exp/max_cut:.3f})"
                )
            except Exception as e:
                st.error(f"Optimisation failed: {e}")
                opt_gamma, opt_beta = [0.5] * p_layers, [0.3] * p_layers
                opt_exp = 0.0
                landscape_data = None

        if landscape_data is not None and p_layers == 1:
            gamma_vals, beta_vals, landscape = landscape_data
            fig_landscape = qaoa_viz.plot_optimization_landscape(
                gamma_vals, beta_vals, landscape,
                opt_gamma[0], opt_beta[0], graph_choice.split("(")[0].strip()
            )
            st.pyplot(fig_landscape)
            plt_close(fig_landscape)

            with st.expander("Reading the optimisation landscape"):
                st.markdown(
                    """
                    The colour encodes ⟨C⟩ — brighter = more edges cut on average.  You can see:

                    - **Smooth structure**: the landscape is not random.  QAOA theory predicts a
                      periodicity in γ (period 2π) and β (period π/2) for Max-Cut.
                    - **Multiple local optima**: the landscape has several bright regions.  A poor
                      starting point for gradient descent can get trapped; grid search avoids this.
                    - **Quantum interference at work**: the pattern of light and dark regions reflects
                      constructive/destructive interference between the cost and mixer unitaries.
                      The star marks the global maximum found — this is where our QAOA circuit runs.
                    """
                )

        st.divider()

        # ── 3. Build circuit and show diagram ────────────────────────────────
        st.markdown(f"### QAOA Circuit  (p = {p_layers})")
        st.markdown(
            f"""
            The optimised QAOA circuit for this graph has **{n_nodes} qubits** and
            **{p_layers}** alternating cost-mixer layer{'s' if p_layers > 1 else ''}.

            **Reading the circuit:**
            - **H gates** — initialise each qubit in the |+⟩ state (equal superposition)
            - **CNOT–Rz–CNOT triplets** — implement e^(−iγ ZᵤZᵥ) for each edge (u, v)
            - **Rx gates** — implement e^(−iβ Xᵢ) on each qubit (mixer layer)
            - **Measurement** — collapse the quantum state to a classical bitstring

            Each bitstring is a candidate partition.  After many shots, the most frequent
            bitstring is the QAOA solution.
            """
        )

        opt_circuit = build_qaoa_circuit(n_nodes, edges, opt_gamma, opt_beta, p_layers)
        fig_circ = qaoa_viz.plot_qaoa_circuit(opt_circuit)
        st.pyplot(fig_circ)
        plt_close(fig_circ)

        circuit_depth = opt_circuit.depth(
            filter_function=lambda inst: inst.operation.name != "measure"
        )
        circuit_gates = opt_circuit.count_ops()
        cx_count = circuit_gates.get("cx", 0)
        st.caption(
            f"Circuit stats: depth = **{circuit_depth}**, total gates = "
            f"**{sum(circuit_gates.values()) - circuit_gates.get('measure', 0)}**, "
            f"CX gates = **{cx_count}**"
        )

        st.divider()

        # ── 4. Transpilation comparison ──────────────────────────────────────
        st.markdown("### Transpilation Analysis  (Optimisation Levels 0–3)")
        st.markdown(
            """
            Real quantum hardware has restricted connectivity (not every qubit can interact
            with every other) and only supports a small set of **native basis gates**
            (typically CX, RZ, SX, X on IBM devices).  *Transpilation* maps the abstract
            QAOA circuit onto these hardware constraints.

            Qiskit offers four optimisation levels that balance compilation speed against
            circuit quality:

            | Level | Strategy | Use case |
            |-------|----------|----------|
            | **0** | Trivial layout + no optimisation | Fast prototyping |
            | **1** | SABRE routing + basic peephole rewrites | Default |
            | **2** | More aggressive 1Q/2Q gate merging | Production |
            | **3** | Full transpiler stack: commutativity, pulse-efficient 2Q | Best quality |

            We transpile to a **simulated 5-qubit IBM backend** with a T-shape coupling map
            (matching real IBM Tenerife/Nairobi topology).  The key metric is the
            **2-qubit gate count** — each CX gate introduces ~1% error on real hardware.
            """
        )

        with st.spinner("Running transpilation comparison at levels 0–3…"):
            try:
                transp_results = compare_transpilation_levels(opt_circuit)
            except Exception as e:
                st.error(f"Transpilation comparison failed: {e}")
                transp_results = []

        if transp_results:
            fig_transp = qaoa_viz.plot_transpilation_comparison(transp_results)
            st.pyplot(fig_transp)
            plt_close(fig_transp)

            with st.expander("Why does gate count matter so much?"):
                st.markdown(
                    """
                    Every gate applied to a real quantum chip accumulates error:
                    - **1-qubit gates**: ~0.05–0.1% error rate
                    - **2-qubit (CX) gates**: ~0.5–1.5% error rate — roughly **10× worse**

                    For a QAOA circuit with d CX gates, the expected fidelity scales roughly as
                    (1 − p_2q)^d.  At p_2q = 1% and d = 20 gates, fidelity ≈ (0.99)^20 ≈ 82%.
                    At d = 40, it drops to ~67%.  Optimisation level 3 minimises d, directly
                    improving the probability of getting the correct answer on real hardware.

                    **SWAP overhead:** Qubits on real chips can only directly interact with
                    neighbours.  For non-neighbouring pairs, the transpiler must insert SWAP gates
                    (each costs 3 CX gates).  Better routing algorithms (levels 2–3) find more
                    efficient paths, reducing SWAP overhead.
                    """
                )

            st.divider()

            # ── 5. Qubit mapping ─────────────────────────────────────────────
            st.markdown("### Qubit Mapping  (Virtual → Physical Qubits)")
            st.markdown(
                """
                The coupling-map diagram below shows the 5-qubit IBM T-shape topology.
                Each circle is a physical qubit; colour shows which virtual (circuit) qubit
                it is assigned to.  Grey nodes are unused.

                **Why does the mapping matter?**
                If two circuit qubits that need to interact are mapped to non-adjacent physical
                qubits, the transpiler must insert SWAP chains — each adding 3 CX gates and
                ~3% extra error.  A clever layout (levels 2–3) places frequently-interacting
                qubits next to each other, minimising SWAPs.
                """
            )
            fig_mapping = qaoa_viz.plot_qubit_mapping(
                transp_results, n_nodes, COUPLING_MAP_EDGES, PHYS_QUBIT_POS
            )
            st.pyplot(fig_mapping)
            plt_close(fig_mapping)

            # Show mapping table
            for r in transp_results:
                if r["qubit_mapping"]:
                    mapping_str = ", ".join(
                        f"v{v} → P{p}" for v, p in sorted(r["qubit_mapping"].items())
                    )
                    st.caption(f"Level {r['level']}: {mapping_str}")

        st.divider()

        # ── 6. Noise model explanation ───────────────────────────────────────
        st.markdown("### IBM Aer Noise Model")
        st.markdown(
            """
            To simulate realistic performance on IBM quantum hardware, we build a composite
            noise model with three error channels:

            | Error channel | Rate | Physical origin |
            |---------------|------|-----------------|
            | **Thermal relaxation** (T₁/T₂) | T₁ = 50 µs, T₂ = 70 µs | Qubit decays to ground state; dephasing from environment |
            | **Depolarising** (single-qubit) | ~0.1% per gate | Gate imperfections, calibration drift |
            | **Depolarising** (CX gate) | ~1.0% per gate | Cross-resonance control imperfections |
            | **Readout assignment** | ~1.5% per qubit | Measurement crosstalk, threshold errors |

            **Thermal relaxation** is the dominant long-timescale error: if a qubit waits
            idle for too long (relative to T₁), it spontaneously emits a photon and flips
            from |1⟩ to |0⟩.  T₂ captures dephasing — random phase kicks that destroy
            quantum coherence without changing energy.

            **Depolarising noise** replaces the intended gate output with a random Pauli
            (X, Y, Z, I) with equal probability p/4 each.  This is a standard model for
            gate crosstalk and pulse imperfections.

            **Readout error** is a classical error: the qubit is in state |1⟩ but the
            measurement electronics report 0 (or vice versa).  This can be corrected
            post-measurement using *measurement error mitigation*.

            The noise scale slider (used in later plots) multiplies all error rates while
            simultaneously dividing T₁/T₂ — modelling a range from near-ideal (scale → 0)
            to heavily noisy (scale = 5×).
            """
        )

        # ── 7. Ideal vs Noisy vs Mitigated ──────────────────────────────────
        st.markdown("### Ideal · Noisy · Measurement-Error-Mitigated Comparison")
        st.markdown(
            """
            We run the same optimised QAOA circuit under three conditions:

            1. **Ideal** — Aer statevector-accurate simulation (no noise).
               This is the best possible QAOA answer given the circuit angles.
            2. **Noisy** — Aer simulation with the full IBM noise model (gate errors +
               thermal relaxation + readout errors).  This approximates what you would
               see running on real IBM hardware today.
            3. **Mitigated** — Noisy simulation with *measurement error mitigation* (MEM)
               applied.  MEM uses 2ⁿ calibration circuits to estimate the readout error
               matrix, then inverts it to correct the observed counts.

            Note: MEM corrects *readout* errors only.  Gate errors (the larger contribution)
            remain.  The mitigated result is better than raw noisy but still below ideal.
            """
        )

        with st.spinner("Running ideal / noisy / mitigated simulations (2048 shots each)…"):
            try:
                baseline_nm = build_ibm_noise_model(scale=1.0)
                comparison = run_three_way_comparison(
                    opt_circuit, baseline_nm, n_nodes, shots=2048
                )
            except Exception as e:
                st.error(f"Three-way comparison failed: {e}")
                comparison = None

        if comparison:
            fig_compare = qaoa_viz.plot_ideal_vs_noisy_vs_mitigated(
                comparison, edges, n_nodes, max_cut
            )
            st.pyplot(fig_compare)
            plt_close(fig_compare)

            ideal_cut  = compute_expected_cut(comparison["ideal"],     edges, n_nodes)
            noisy_cut  = compute_expected_cut(comparison["noisy"],     edges, n_nodes)
            mitig_cut  = compute_expected_cut(comparison["mitigated"], edges, n_nodes)
            noise_penalty = ideal_cut - noisy_cut
            mitig_gain    = mitig_cut - noisy_cut

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Ideal ⟨C⟩",     f"{ideal_cut:.3f}",  f"Ratio {ideal_cut/max_cut:.3f}")
            col_m2.metric("Noisy ⟨C⟩",     f"{noisy_cut:.3f}",  f"−{noise_penalty:.3f} vs ideal")
            col_m3.metric("Mitigated ⟨C⟩", f"{mitig_cut:.3f}",  f"+{mitig_gain:.3f} vs noisy")
            col_m4.metric("Classical C*",   str(max_cut),         "Optimum")

            with st.expander("How does measurement error mitigation work?"):
                st.markdown(
                    f"""
                    **Step 1 — Calibration:**
                    Prepare each of the **2^{n_nodes} = {2**n_nodes}** computational basis
                    states (|000…0⟩, |000…1⟩, …, |111…1⟩) and measure them through the noisy
                    device.  Record the **calibration matrix** A where A[j, i] = P(measuring
                    bitstring j | we prepared state i).

                    **Step 2 — Inversion:**
                    For the actual QAOA measurement vector **p_noisy**, compute:
                    > **p_corrected** = A⁻¹ · **p_noisy**

                    **Step 3 — Post-process:**
                    Clip negative entries (which arise from statistical fluctuations or
                    ill-conditioning) and renormalise to get a valid probability distribution.

                    **Limitations:**
                    - Calibration itself is noisy (4096 shots used here).
                    - Gate errors are **not** corrected — only readout.
                    - For n > 7 qubits, the 2^n circuits become expensive.
                      Methods like M3 (matrix-free measurement mitigation) scale to ~127 qubits
                      by exploiting sparsity in the calibration matrix.
                    """
                )

            st.divider()

            # ── 8. Solution distribution ─────────────────────────────────────
            st.markdown("### Solution Probability Distribution")
            st.markdown(
                """
                Each bar represents a candidate partition (bitstring).  Bitstrings are ordered
                left → right by increasing cut value, so the **rightmost columns** are the best
                solutions.  Three bars per bitstring show how noise redistributes probability
                mass away from optimal solutions.

                **What to look for:**
                - In the ideal run, probability concentrates on high-cut bitstrings.
                - In the noisy run, the distribution flattens — noise causes random bit flips
                  that make suboptimal partitions appear more probable.
                - Mitigation partially restores the ideal shape (readout correction), but gate
                  errors still cause bleed toward lower-cut bitstrings.
                """
            )
            fig_dist = qaoa_viz.plot_solution_distribution(
                comparison, edges, n_nodes, max_cut
            )
            st.pyplot(fig_dist)
            plt_close(fig_dist)

            st.divider()

            # ── 9. Noise sweep ───────────────────────────────────────────────
            st.markdown("### Performance vs. Noise Strength")
            st.markdown(
                """
                We now vary the noise scale factor from **0** (ideal) to **5×** the baseline
                IBM noise and measure how QAOA quality degrades.

                This sweep answers the key engineering question:
                *"How bad can the noise get before QAOA stops being useful?"*

                The **random baseline** (⟨C⟩ = C*/2) is the expected cut from a random
                partition — the minimum bar for any algorithm to be useful.  When QAOA's
                expected cut drops below this, the circuit is too noisy to provide any signal.
                """
            )

            with st.spinner("Running noise sweep (7 noise levels × 1024 shots each)…"):
                try:
                    sweep_results = noise_sweep(
                        opt_circuit, edges, n_nodes,
                        scale_factors=[0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
                        shots=1024,
                    )
                except Exception as e:
                    st.error(f"Noise sweep failed: {e}")
                    sweep_results = []

            if sweep_results:
                fig_sweep = qaoa_viz.plot_noise_sweep(sweep_results, max_cut)
                st.pyplot(fig_sweep)
                plt_close(fig_sweep)

                # Find the noise threshold
                threshold_scale = None
                for r in sweep_results:
                    if r["approx_ratio"] < 0.5:
                        threshold_scale = r["scale"]
                        break

                if threshold_scale:
                    st.info(
                        f"QAOA advantage is lost at noise scale ≈ **{threshold_scale}×** "
                        "— beyond this, a random partition does equally well."
                    )
                else:
                    st.info(
                        "QAOA maintains advantage over random guessing across all tested "
                        "noise levels — a sign of a robust circuit for this graph."
                    )

                with st.expander("Interpreting the noise sweep"):
                    st.markdown(
                        """
                        **The noise scale factor is defined as:**
                        - Scale 0.0 → no noise (ideal simulation)
                        - Scale 1.0 → baseline IBM hardware noise (T₁=50µs, ~1% CX error)
                        - Scale 2.0 → twice as noisy (shorter coherence, higher gate error rates)
                        - Scale 5.0 → severely noisy — approximates early NISQ prototypes or
                          very long, deep circuits running at the edge of coherence

                        **Why does quality degrade linearly at first, then catastrophically?**
                        For small noise, errors introduce small perturbations to the probability
                        distribution.  The expected cut decreases slowly.  At a critical noise
                        level, decoherence destroys the quantum coherence that QAOA relies on —
                        the circuit output approaches a maximally mixed state, giving random
                        results.  This transition is the **noise threshold** for this circuit.

                        **Implications for hardware roadmaps:**
                        To run QAOA circuits with p=3 on 50-qubit graphs — the scale needed for
                        commercial relevance — error rates must drop below ~0.1% per CX gate.
                        Current IBM devices are at ~0.5–1%, so roughly a 5–10× improvement is
                        needed.  IBM's roadmap targets this with error-corrected logical qubits
                        by ~2029.
                        """
                    )

                st.divider()

            # ── 10. Depth–quality sweep ──────────────────────────────────────
            st.markdown("### Circuit Depth vs. Solution Quality  (p = 1, 2, 3 Layers)")
            st.markdown(
                """
                Deeper circuits (more QAOA layers) should improve the approximation ratio —
                with p→∞, QAOA converges to the exact optimum.  But on noisy hardware,
                deeper circuits accumulate more gate errors, potentially *worsening* performance.

                This trade-off — the **noise–depth dilemma** — is one of the central
                challenges of NISQ-era quantum computing.  The optimal p depends on:
                - Hardware error rates (lower error → can use larger p)
                - Graph structure (some graphs benefit more from deeper circuits)
                - Coherence times (T₁, T₂ must exceed circuit duration)

                **We use fixed angles** (γ = 0.5, β = 0.3) across all p to isolate the
                depth effect — note that proper per-p optimisation would improve all bars.
                """
            )

            with st.spinner("Running depth–quality sweep (p = 1, 2, 3)…"):
                try:
                    dq_results = depth_quality_sweep(
                        n_nodes, edges, max_cut,
                        baseline_nm,
                        p_values=[1, 2, 3],
                        shots=1024,
                    )
                except Exception as e:
                    st.error(f"Depth–quality sweep failed: {e}")
                    dq_results = []

            if dq_results:
                fig_dq = qaoa_viz.plot_depth_quality(dq_results, max_cut)
                st.pyplot(fig_dq)
                plt_close(fig_dq)

                with st.expander("The noise–depth dilemma in detail"):
                    st.markdown(
                        """
                        **Why does the ideal curve improve with p but the noisy curve may not?**

                        In the *ideal* case, each additional QAOA layer applies an extra round
                        of cost + mixer unitaries.  This is analogous to more Trotter steps in
                        adiabatic evolution — the quantum state gets progressively closer to the
                        ground state of H_C (the optimal solution).

                        In the *noisy* case, each additional layer also adds:
                        - 2|E| new CX gates (cost layer: one CX–Rz–CX triplet per edge)
                        - n new Rx gates (mixer layer)
                        - Additional idle time (more decoherence)

                        The signal-to-noise ratio of each layer diminishes as the circuit
                        gets longer.  At some critical depth p*, the noise benefit outweighs
                        the algorithmic benefit, and the noisy expected cut *decreases*.

                        **What this means for quantum error correction (QEC):**
                        The only way to run QAOA at large p without noise destroying the
                        result is to use **fault-tolerant quantum gates** — gates with logical
                        error rates below ~10⁻⁶ achieved via quantum error correction codes
                        (e.g., the surface code).  This requires ~1000 physical qubits per
                        logical qubit.  Once we have fault-tolerant machines, the depth limit
                        on QAOA is removed, and the algorithm can run to large p for
                        genuinely hard problem instances.
                        """
                    )

            st.divider()

        # ── Summary ──────────────────────────────────────────────────────────
        st.markdown("### Summary: Key Takeaways")
        st.markdown(
            f"""
            This analysis demonstrated QAOA on the **{graph_choice}** graph with p = {p_layers} layer(s):

            **Algorithm:**
            - QAOA is a variational hybrid quantum-classical algorithm for combinatorial optimisation.
            - The circuit alternates cost (ZZ) and mixer (X) layers, parameterised by angles (γ, β).
            - Optimising (γ, β) is a classical outer-loop problem solved here by grid search.

            **Transpilation:**
            - Optimisation level 3 reduces the 2-qubit gate count by routing qubits more intelligently.
            - Fewer CX gates → lower accumulated error → higher probability of correct answer on hardware.

            **Noise effects:**
            - Even at baseline IBM noise (scale = 1×), QAOA quality degrades measurably.
            - Measurement error mitigation recovers some of the lost quality at low cost (2^n extra circuits).
            - Gate errors — the dominant noise source — require more powerful techniques
              (zero-noise extrapolation, probabilistic error cancellation, or full error correction).

            **Depth–quality trade-off:**
            - More QAOA layers improve the theoretical approximation ratio (closer to C* = {max_cut}).
            - But on noisy hardware, deeper circuits lose more to decoherence.
            - This tension is the central open problem in NISQ-era quantum optimisation.

            **Broader impact:**
            - QAOA is a prototype for the class of variational quantum algorithms that will likely
              be the first to demonstrate practical quantum advantage.
            - As hardware improves (lower error rates, longer coherence, more qubits), the problems
              QAOA can tackle will scale up — from toy graphs to industrially relevant instances
              with hundreds of variables.
            - This is why every gate saved by the transpiler, every error mitigated by post-processing,
              and every qubit preserved by better materials brings the field closer to a genuine
              quantum advantage over classical computing.
            """
        )
