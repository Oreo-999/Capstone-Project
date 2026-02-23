# Quantum Algorithm Dashboard

An interactive, educational dashboard for exploring three landmark quantum algorithms — built with [Qiskit](https://qiskit.org), [Streamlit](https://streamlit.io), and Plotly. Every visualization is interactive: hover for values, zoom in, toggle data series, and step through algorithms one iteration at a time.

> Run circuits on a real IBM quantum computer or locally on AerSimulator — no quantum hardware required to get started.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Algorithms](#algorithms)
  - [Grover's Search](#grovers-search)
  - [Shor's Factoring](#shors-factoring)
  - [QAOA · Weighted Max-Cut](#qaoa--weighted-max-cut)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
  - [Connecting to IBM Quantum Hardware](#connecting-to-ibm-quantum-hardware)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Architecture Notes](#architecture-notes)

---

## Overview

This dashboard makes quantum computing tangible. Instead of reading about algorithms in a textbook, you run them, see the circuits, watch amplitudes evolve, tune parameters, and compare quantum performance against the best classical alternatives — all in a browser.

It is aimed at students, researchers, and anyone curious about quantum computing who wants to go beyond "Hello, World" circuits and understand what these algorithms actually do and why they matter.

---

## Features

### Interactive Visualizations
All key charts are built with Plotly — hover over any bar or data point to see exact values, zoom in on regions of interest, and toggle data series on and off via the legend. Static matplotlib is used only for circuit diagrams (where Qiskit's native renderer is best).

### Step-by-Step Grover Explorer
Switch Grover's tab into **Step-by-Step Explorer** mode to advance the algorithm one oracle+diffuser iteration at a time. Watch the amplitude chart update live, track the growing quantum vs. classical success probability gap with each step, and navigate forward and backward freely. Results are cached so revisiting steps is instant.

### Guess the Cut Challenge
Before running QAOA, color each graph node blue or red to attempt your own maximum cut partition. The graph redraws live after every toggle, showing which edges you've cut and your running cut value. Submit your partition to see a scorecard comparing your score against QAOA, the Goemans–Williamson classical approximation, and the brute-force optimum.

### Shor's Base Explorer
Choose any of the 7 valid bases coprime to N=15 (2, 4, 7, 8, 11, 13, 14) and see how the circuit changes, how the period r varies, and how the classical post-processing derives the factors. A full period table with explanations is included.

### IBM Quantum Hardware Support
Grover's algorithm can run on real IBM quantum hardware. Connect with your IBM Quantum API key to submit jobs to the least-busy available backend. Falls back to AerSimulator automatically if the hardware connection fails.

---

## Algorithms

### Grover's Search

**What it solves:** Unstructured search — find a target item in an unsorted list of N items.

**Classical baseline:** O(N/2) queries on average. For N=1,000,000 that's ~500,000 comparisons.

**Quantum speedup:** O(√N) queries. For N=1,000,000 that's ~1,000 queries — a **500× speedup**, and provably optimal for unstructured search.

**This demo:** 4 qubits, search space of 16 items, 3 Grover iterations. Target is user-selectable (0–15). Achieves ~96.1% success probability — the theoretical maximum for 3 iterations on N=16.

**How it works:**
1. Apply Hadamard gates to all qubits — creates a uniform superposition over all 16 states simultaneously
2. **Oracle** — marks the target state with a phase flip (multiplies amplitude by −1) without collapsing the superposition
3. **Diffuser** — reflects all amplitudes about their mean, amplifying the target and suppressing every other state
4. Repeat steps 2–3 for ⌊π√N/4⌋ iterations (3 for N=16), then measure

**Visualizations:**
- Circuit diagram with labeled stages (Superposition / Oracle / Diffuser)
- Interactive measurement histogram with per-state hover (count + probability)
- Amplitude evolution across 0–3 iterations (live in step-by-step mode)
- Interactive complexity comparison: Classical O(N/2) vs Grover O(√N)
- Success probability vs queries: cumulative probability for both methods
- Quantum speedup at scale: concrete query counts from N=16 to N=1,000,000

---

### Shor's Factoring

**What it solves:** Integer factorization — find the prime factors of a large composite number N.

**Classical baseline:** General Number Field Sieve (GNFS), sub-exponential but practically infeasible for RSA key sizes. RSA-2048 would take longer than the age of the universe.

**Quantum speedup:** Polynomial time O(n³) in the number of bits n. A fault-tolerant quantum computer could factor RSA-2048 in hours to days — the reason quantum computing threatens RSA encryption.

**This demo:** N=15, 8 qubits (4 counting + 4 work). Choose from any of the 7 valid bases a ∈ {2, 4, 7, 8, 11, 13, 14} — each produces a different circuit and period, all yielding the factors 3 × 5.

**How it works:**
1. Initialize counting register in superposition, work register to |0001⟩
2. Apply controlled-U^(2^k) gates — U = "multiply by a mod 15" — entangling the counting and work registers with all values of a^x mod 15 simultaneously
3. Apply the Inverse Quantum Fourier Transform to the counting register — converts the periodic pattern into sharp measurement peaks
4. Measure the counting register — peaks appear at multiples of 2^n/r, revealing the period r
5. Classical post-processing: use continued fractions to extract r, then compute gcd(a^(r/2) ± 1, N)

**Visualizations:**
- Circuit diagram (8-qubit, labeled registers)
- Interactive phase estimation bar chart with hover showing bitstring, decimal, phase value, and annotated expected peaks
- Factor derivation with step-by-step gcd calculation
- Period function f(x) = a^x mod N with discrete Fourier transform
- Complexity gap: Classical GNFS vs Shor's polynomial curve
- Time to break RSA: Classical vs quantum for RSA-512 through RSA-4096

---

### QAOA · Weighted Max-Cut

**What it solves:** Maximum Cut (Max-Cut) — partition vertices of a graph into two sets to maximize the total weight of edges crossing between them. NP-hard in general, with applications in logistics, finance, drug discovery, energy grid optimization, and machine learning.

**Approach:** QAOA (Quantum Approximate Optimization Algorithm, Farhi et al. 2014) is a variational hybrid algorithm. A parameterized quantum circuit encodes approximate solutions, and a classical optimizer tunes the circuit angles to maximize the expected cut value.

**This demo:** Four preset graphs (Triangle, Square/C4, Diamond, Complete K4), 1–3 QAOA layers (p), optional edge weights. Full analysis pipeline covering optimization, noise, error mitigation, and classical benchmarking.

#### Preset Graphs

| Graph | Nodes | Edges | Weights | Optimal C* |
|-------|-------|-------|---------|------------|
| Triangle | 3 | 3 | Yes | 2.50 |
| Square / C4 | 4 | 4 | Yes | 6.00 |
| Diamond | 4 | 5 | Yes | 7.00 |
| Complete K4 | 4 | 6 | Yes | 6.50 |

#### Analysis Pipeline

**1. Classical optimal partition**
Brute-force over all 2^n partitions to find the true optimum C*. Shown as a two-panel graph: input graph and optimal partition with colored nodes and cut edges highlighted.

**2. Parameter optimisation**
10×10 grid search over (γ, β) angle pairs maximising ⟨C⟩ (256 shots each), followed by Nelder-Mead refinement for p>1. For p=1, the full optimization landscape is shown as an interactive Plotly heatmap — hover any cell to see the exact (γ, β, ⟨C⟩) triple and the optimal point is marked with a star.

**3. QAOA circuit**
The optimized circuit with labeled gate counts, depth (excluding measurements), and CX count. Rz rotation angles are scaled by edge weights in weighted mode.

**4. Transpilation analysis**
Maps the abstract QAOA circuit to IBM native gates (CX, RZ, SX, X) on a realistic 5-qubit T-shape coupling map at optimization levels 0–3. Shows gate count, circuit depth, and routing overhead per level, plus a visual qubit mapping diagram showing virtual → physical qubit assignments.

**5. Ideal · Noisy · Mitigated comparison**
Three simulation runs using a realistic IBM noise model (thermal relaxation T₁=50µs / T₂=70µs, depolarizing errors, 1.5% readout error):
- **Ideal** — noiseless SamplerV2
- **Noisy** — full IBM Aer noise model
- **Mitigated** — noisy + calibration-matrix measurement error correction

Results shown as side-by-side bar charts of ⟨C⟩ and approximation ratio, plus a grouped solution distribution histogram across all 2^n basis states.

**6. Noise sweep**
Runs the circuit at noise scale factors λ ∈ {0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0}. Interactive two-panel Plotly chart shows how ⟨C⟩ and approximation ratio degrade as noise increases, with shaded quantum advantage region.

**7. Depth–quality sweep**
Compares ideal vs noisy performance for p=1, 2, 3 QAOA layers — directly illustrating the central NISQ-era tension: more layers improve the ideal approximation ratio but accumulate more noise.

**8. Zero-Noise Extrapolation (ZNE)**
Runs the circuit at noise levels λ=1, 2, 3, 4. Fits linear and quadratic polynomials to ⟨C(λ)⟩ and extrapolates to λ=0 — the estimated noiseless result without requiring extra qubits. Used in production by IBM Quantum and Google for NISQ experiments.

**9. Goemans–Williamson SDP comparison**
The best known classical polynomial-time approximation algorithm for weighted Max-Cut, guaranteeing ≥0.878×C* in expectation. Implemented via CVXPY semidefinite programming + 200 rounds of random hyperplane rounding. Compared directly against QAOA in a four-way bar chart (C* / SDP bound / GW rounded / QAOA ideal) and a rounding-distribution histogram.

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- pip
- (Optional) An [IBM Quantum](https://quantum.cloud.ibm.com) account to run Grover's on real hardware

### Installation

```bash
git clone https://github.com/Oreo-999/Capstone-Project.git
cd Capstone-Project
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The app will be available immediately — no backend connection required to use the simulator.

### Connecting to IBM Quantum Hardware

Grover's algorithm can optionally run on real IBM quantum hardware.

1. Create a free account at [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com)
2. Go to your account profile and copy your **API key**
3. In the dashboard sidebar:
   - Paste your API key into the **IBM Quantum API Key** field
   - Leave **Instance CRN** blank (auto-selects your first available instance), or paste a specific instance CRN from your Instances page
   - Make sure the **Use Simulator** toggle is OFF
   - Click **Connect**

If the hardware connection fails for any reason (queue, network, permissions), the app automatically falls back to AerSimulator and shows a warning.

> **Note:** Shor's algorithm always runs on AerSimulator regardless of connection status — the 8-qubit circuit with custom controlled-U gates requires transpilation that isn't optimized for short queue times.

---

## Project Structure

```
Capstone-Project/
├── app.py                        Main Streamlit application (3 tabs + sidebar)
├── config.py                     Backend selector: IBM hardware or AerSimulator
├── requirements.txt
├── .streamlit/
│   └── config.toml               Disables usage-stats prompt for headless startup
├── algorithms/
│   ├── grover.py                 4-qubit Grover's circuit (full run + k-iteration variant)
│   ├── shor.py                   8-qubit Shor's circuit (parameterized base a, N=15)
│   └── qaoa.py                   QAOA: circuit builder, optimizer, noise model,
│                                 ZNE, GW SDP, transpilation comparison
└── visualizations/
    ├── grover_viz.py             Grover matplotlib plots
    ├── shor_viz.py               Shor matplotlib plots
    ├── qaoa_viz.py               QAOA matplotlib plots
    └── plotly_viz.py             Interactive Plotly charts (histogram, heatmap,
                                  noise sweep, phase estimation, amplitude step)
```

### Key Design Decisions

**Visualization split:** Static matplotlib is used for circuit diagrams (Qiskit's `draw("mpl")` renderer) and complex multi-panel figures. Plotly is used for all charts where interactivity adds value — histograms, line charts, heatmaps.

**Figure lifecycle:** All matplotlib figures are passed back to the caller and closed with `plt.close(fig)` after `st.pyplot(fig)` to prevent memory leaks across Streamlit reruns.

**Caching:** Expensive quantum computations are wrapped in `@st.cache_data` with hashable arguments (lists → tuples, dicts → frozensets). This makes the step-by-step Grover explorer feel instant and prevents re-running QAOA optimization on every widget interaction.

**Session state:** Used for the step-by-step Grover explorer (current iteration, locked target) and the Guess the Cut challenge (node partition, submission state, cached result). Resets automatically when graph or target changes.

**Bitstring endianness:** Qiskit returns big-endian bitstrings. All cut value and partition logic uses `(val >> i) & 1` indexing to map qubit i → vertex i consistently in little-endian order.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `qiskit` | ≥1.0, <2.0 | Quantum circuit construction and transpilation |
| `qiskit-aer` | ≥0.15 | Local quantum simulation with noise models |
| `qiskit-ibm-runtime` | ≥0.25 | IBM Quantum hardware access |
| `streamlit` | ≥1.30 | Web application framework |
| `plotly` | ≥5.18 | Interactive charts |
| `matplotlib` | ≥3.8 | Circuit diagrams and static plots |
| `numpy` | ≥1.24 | Numerical computing |
| `scipy` | ≥1.10 | Nelder-Mead optimization, continued fractions |
| `cvxpy` | ≥1.4 | Semidefinite programming for Goemans–Williamson |
| `pylatexenc` | ≥2.10 | LaTeX rendering in Qiskit circuit diagrams |

---

## Architecture Notes

### Noise Model
The IBM noise model is built in `algorithms/qaoa.py` using `qiskit_aer.noise`:
- **Thermal relaxation** on every gate: T₁=50µs, T₂=70µs, with gate times of 50ns (1-qubit) and 300ns (2-qubit)
- **Depolarizing error** composed with thermal relaxation: p=0.001 (1-qubit), p=0.01 (2-qubit)
- **Readout error**: 1.5% per qubit
- Scale factor λ multiplies all error rates for the noise sweep and ZNE runs

### Measurement Error Mitigation
A 2^n × 2^n calibration matrix is built by preparing each computational basis state and measuring it. The noisy counts vector is corrected by multiplying by the pseudoinverse (`np.linalg.pinv`) of the calibration matrix, then clipping negative probabilities and renormalizing.

### Goemans–Williamson Implementation
1. Solve the SDP relaxation using CVXPY: assign a unit vector **v**ᵢ ∈ ℝⁿ to each vertex, maximizing Σ w_{uv}(1−**v**ᵤ·**v**ᵥ)/2
2. Sample 200 random hyperplanes (unit vectors **r** ∈ ℝⁿ)
3. Assign vertex i to set 0 if **v**ᵢ·**r** ≥ 0, else set 1
4. Return the best cut across all 200 rounds

### Shor's Circuit — Controlled-U Gates
For each base a ∈ {2, 4, 7, 8, 11, 13, 14}, the controlled-U^power gate implements "multiply the 4-qubit work register by a^power mod 15" as an explicit sequence of SWAP gates derived from the permutation cycle structure of a on {1, 2, 4, 8}. The effective power is reduced modulo the period r before constructing the gate.
