# Quantum Algorithm Dashboard

An interactive Streamlit dashboard demonstrating three landmark quantum algorithms, powered by [Qiskit](https://qiskit.org) and IBM Aer.

## Algorithms

### Grover's Search
Quadratic speedup over classical unstructured search. Searches a space of **16 items** (4 qubits) using amplitude amplification, finding the target with ~96% probability in just 3 iterations vs. O(N/2) classically. Supports real IBM quantum hardware or local simulation.

### Shor's Factoring
Exponential speedup for integer factorisation. Factors **N = 15** into 3 × 5 using 8 qubits (4 counting + 4 work) via modular exponentiation and the Quantum Fourier Transform — the algorithm that threatens RSA encryption.

### QAOA · Max-Cut
The Quantum Approximate Optimization Algorithm applied to the Maximum Cut problem on small graphs. Includes a full analysis pipeline:

- **Parameter optimisation** — 10×10 grid search over (γ, β) angles with a landscape heatmap
- **Transpilation comparison** — gate count, circuit depth, and 2-qubit gate overhead at optimisation levels 0–3 against a realistic IBM 5-qubit T-shape topology
- **Qubit mapping** — visual layout of virtual → physical qubit assignments on the coupling map
- **Noise simulation** — realistic IBM Aer noise model (thermal relaxation T₁=50µs/T₂=70µs, depolarising, readout errors)
- **Ideal vs Noisy vs Mitigated** — side-by-side comparison with measurement error mitigation (calibration matrix inversion)
- **Noise sweep** — solution quality vs noise strength from ideal (0×) to severely noisy (5×)
- **Depth–quality sweep** — approximation ratio vs circuit depth for p = 1, 2, 3 QAOA layers

## Getting Started

### Prerequisites
- Python 3.10+
- (Optional) IBM Quantum account for running Grover's on real hardware

### Installation

```bash
git clone https://github.com/Oreo-999/Capstone-Project.git
cd Capstone-Project
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
app.py                    — Streamlit UI (3 tabs)
config.py                 — Backend selector (IBM hardware or AerSimulator)
algorithms/
  grover.py               — 4-qubit Grover's search circuit & runner
  shor.py                 — 8-qubit Shor's factoring circuit & runner
  qaoa.py                 — QAOA Max-Cut: circuit, optimiser, noise model, sweeps
visualizations/
  grover_viz.py           — Grover plots (circuit, counts, amplitude evolution, complexity)
  shor_viz.py             — Shor plots (circuit, counts, period function, complexity)
  qaoa_viz.py             — QAOA plots (graph, landscape, transpilation, noise, distribution)
requirements.txt
```

## Requirements

```
qiskit >= 1.0, < 2.0
qiskit-ibm-runtime >= 0.25
qiskit-aer >= 0.15
streamlit >= 1.30
matplotlib >= 3.8
numpy >= 1.24
scipy >= 1.10
pylatexenc >= 2.10
```
