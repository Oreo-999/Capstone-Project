import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from qiskit import QuantumCircuit


def plot_circuit(circuit: QuantumCircuit):
    """Return matplotlib figure of the Grover circuit diagram."""
    fig = circuit.draw("mpl")
    return fig


def plot_counts(counts: dict, target: int):
    """
    Bar chart of all 16 basis states with the target state highlighted.

    Args:
        counts: dict of bitstring -> count
        target: integer (0-15) representing the searched state
    """
    n = 4
    all_states = [format(i, f"0{n}b") for i in range(2 ** n)]
    values = [counts.get(state, 0) for state in all_states]

    # Qiskit little-endian: target state bitstring is reversed
    target_state = format(target, f"0{n}b")[::-1]
    colors = ["#ff6b6b" if s == target_state else "#4ecdc4" for s in all_states]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(all_states, values, color=colors)
    ax.set_xlabel("Basis State (little-endian)", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(
        f"Grover's Search Results — Target: |{target_state}⟩ (decimal {target})",
        fontsize=13,
    )
    plt.xticks(rotation=45, ha="right")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ff6b6b", label=f"Target |{target_state}⟩"),
        Patch(facecolor="#4ecdc4", label="Other states"),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    return fig


def plot_complexity():
    """
    Line chart comparing classical O(N/2) vs Grover's O(sqrt(N)) search complexity
    for list sizes 2 to 256.
    """
    sizes = np.arange(2, 257)
    classical = sizes / 2
    quantum = np.sqrt(sizes)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, classical, label="Classical O(N/2)", linewidth=2, color="#e17055")
    ax.plot(sizes, quantum, label="Grover's O(√N)", linewidth=2, color="#6c5ce7")
    ax.set_xlabel("Search Space Size (N)", fontsize=12)
    ax.set_ylabel("Expected Queries", fontsize=12)
    ax.set_title("Search Complexity: Classical vs Quantum", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_amplitude_evolution(target: int, n: int = 4):
    """
    4-panel figure showing how amplitudes across all 2^n states evolve
    iteration by iteration (0 through 3 Grover iterations).
    Demonstrates amplitude amplification visually.
    """
    N = 2 ** n
    target_state = format(target, f"0{n}b")[::-1]
    target_idx = int(target_state, 2)
    theta = np.arcsin(1 / np.sqrt(N))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle(
        f"Amplitude Evolution Across Grover Iterations  (target = |{target_state}⟩)",
        fontsize=13, fontweight="bold"
    )

    for k, ax in enumerate(axes):
        # Theoretical amplitude after k iterations
        amp_target = np.sin((2 * k + 1) * theta)
        amp_other = np.cos((2 * k + 1) * theta) / np.sqrt(N - 1)
        prob_target = amp_target ** 2

        amplitudes = np.full(N, amp_other)
        amplitudes[target_idx] = amp_target

        colors = ["#ff6b6b" if i == target_idx else "#4ecdc4" for i in range(N)]
        ax.bar(range(N), amplitudes, color=colors, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"{'Initial' if k == 0 else f'Iteration {k}'}\nP(target) = {prob_target:.1%}",
            fontsize=10
        )
        ax.set_xlabel("State index", fontsize=9)
        if k == 0:
            ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_ylim(-0.4, 1.05)
        ax.set_xticks([0, 7, 15])
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(labelsize=8)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor="#ff6b6b", label=f"Target |{target_state}⟩"),
            Patch(facecolor="#4ecdc4", label="All other states"),
        ],
        loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.05)
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def plot_success_probability_vs_queries(n: int = 4):
    """
    Line chart comparing the cumulative probability of finding the target
    after k queries: classical random search vs Grover's algorithm.
    """
    N = 2 ** n
    theta = np.arcsin(1 / np.sqrt(N))
    max_queries = 20

    k_vals = np.arange(0, max_queries + 1)

    # Classical: probability of finding target in k independent random draws
    p_classical = 1 - ((N - 1) / N) ** k_vals

    # Grover: P = sin²((2k+1)θ) after k full iterations
    p_grover = np.sin((2 * k_vals + 1) * theta) ** 2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_vals, p_classical * 100, label="Classical random search", color="#e17055",
            linewidth=2.5, marker="o", markersize=4)
    ax.plot(k_vals, p_grover * 100, label="Grover's algorithm", color="#6c5ce7",
            linewidth=2.5, marker="s", markersize=4)

    # Annotate the crossover advantage
    grover_optimal_k = int(np.floor(np.pi / (4 * theta)))
    p_at_optimal = np.sin((2 * grover_optimal_k + 1) * theta) ** 2
    classical_at_optimal = 1 - ((N - 1) / N) ** grover_optimal_k
    ax.annotate(
        f"Grover reaches {p_at_optimal:.0%}\nClassical only {classical_at_optimal:.0%}\n(at {grover_optimal_k} queries)",
        xy=(grover_optimal_k, p_at_optimal * 100),
        xytext=(grover_optimal_k + 3, p_at_optimal * 100 - 20),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7", alpha=0.8)
    )

    ax.set_xlabel("Number of Queries (k)", fontsize=12)
    ax.set_ylabel("Probability of Finding Target (%)", fontsize=12)
    ax.set_title(
        f"Success Probability vs Queries  (N = {N} states, {n} qubits)",
        fontsize=13
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, max_queries)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_runtime_race():
    """
    Grouped bar chart showing classical vs Grover query counts at concrete N values,
    making the speedup viscerally clear.
    """
    Ns = [16, 64, 256, 1024, 4096, 65536, 1_048_576]
    labels = ["16\n(4 qubits)", "64\n(6 qubits)", "256\n(8 qubits)",
              "1K\n(10 qubits)", "4K\n(12 qubits)", "65K\n(16 qubits)", "1M\n(20 qubits)"]
    classical = [N / 2 for N in Ns]
    grover = [int(np.floor(np.pi * np.sqrt(N) / 4)) for N in Ns]

    x = np.arange(len(Ns))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_c = ax.bar(x - width / 2, classical, width, label="Classical O(N/2)",
                    color="#e17055", alpha=0.9)
    bars_q = ax.bar(x + width / 2, grover, width, label="Grover's O(√N)",
                    color="#6c5ce7", alpha=0.9)

    # Speedup annotations
    for i, (c, q) in enumerate(zip(classical, grover)):
        speedup = c / q
        ax.text(x[i], max(c, q) * 1.05, f"{speedup:.0f}×",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2d3436")

    ax.set_yscale("log")
    ax.set_xlabel("Search Space Size N", fontsize=12)
    ax.set_ylabel("Queries Required (log scale)", fontsize=12)
    ax.set_title("Quantum Speedup: Queries Required to Find Target", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0.99, 0.97, "Numbers above bars = quantum speedup factor",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#636e72")
    plt.tight_layout()
    return fig
