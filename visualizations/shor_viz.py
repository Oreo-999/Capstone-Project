import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from math import gcd


def plot_circuit(circuit: QuantumCircuit):
    """Return matplotlib figure of the Shor circuit diagram."""
    fig = circuit.draw("mpl")
    return fig


def plot_counts(counts: dict):
    """
    Bar chart of counting register measurements showing periodic peaks.

    Args:
        counts: dict of bitstring -> count from the counting register
    """
    sorted_states = sorted(counts.keys())
    values = [counts[s] for s in sorted_states]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(sorted_states, values, color="#6c5ce7")
    ax.set_xlabel("Counting Register Measurement", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(
        "Shor's Algorithm — Phase Estimation Results (N=15, a=2)", fontsize=13
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_derivation(period: int, factors: tuple):
    """
    Matplotlib figure with a plain-text breakdown of how period r leads to factors.

    Args:
        period: detected period r
        factors: (factor1, factor2) tuple
    """
    a = 2
    N = 15
    half_r = period // 2

    lines = [
        "Shor's Algorithm: Factoring N = 15",
        "",
        f"  Step 1 — Choose a = {a}  (coprime to {N}, gcd({a},{N}) = {gcd(a,N)})",
        f"  Step 2 — Quantum period finding:",
        f"            Find r such that a^r ≡ 1 (mod N)",
        f"            Measured period: r = {period}",
        f"            Verify: {a}^{period} mod {N} = {pow(a, period, N)} ✓",
        "",
        f"  Step 3 — Compute factor candidates (r = {period} is even ✓):",
        f"            x = a^(r/2) = {a}^{half_r} = {a**half_r}",
        "",
        f"            gcd(x - 1, N) = gcd({a**half_r - 1}, {N}) = {factors[0]}",
        f"            gcd(x + 1, N) = gcd({a**half_r + 1}, {N}) = {factors[1]}",
        "",
        f"  Result:   {N} = {factors[0]} × {factors[1]}",
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    text = "\n".join(lines)
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#ffeaa7", alpha=0.8),
    )
    ax.set_title("Factor Derivation", fontsize=14, pad=10)
    plt.tight_layout()
    return fig


def plot_period_function():
    """
    Two-panel figure: left shows f(x) = 2^x mod 15 highlighting the period,
    right shows the same values in the frequency domain (DFT magnitude)
    to show why QFT reveals the period as sharp peaks.
    """
    a, N = 2, 15
    x_vals = np.arange(0, 16)
    f_vals = np.array([pow(a, int(x), N) for x in x_vals])
    period = 4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("How Period Finding Works: f(x) = 2ˣ mod 15", fontsize=13, fontweight="bold")

    # Left: the periodic function
    colors = [f"#{['ff6b6b','fdcb6e','6c5ce7','00b894'][int(x) % period]}ff" for x in x_vals]
    ax1.bar(x_vals, f_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("x (exponent)", fontsize=11)
    ax1.set_ylabel("2ˣ mod 15", fontsize=11)
    ax1.set_title("The Periodic Function\n(same color = same phase in the period)", fontsize=10)
    ax1.set_xticks(x_vals)
    ax1.set_yticks([0, 1, 2, 4, 8])
    ax1.grid(True, axis="y", alpha=0.3)

    # Annotate the repeating cycle
    cycle_vals = [pow(a, i, N) for i in range(period)]
    ax1.annotate("", xy=(period, 1.2), xytext=(0, 1.2),
                 arrowprops=dict(arrowstyle="<->", color="#e17055", lw=2))
    ax1.text(period / 2, 1.5, f"r = {period}", ha="center", fontsize=11,
             color="#e17055", fontweight="bold")

    from matplotlib.patches import Patch
    cycle_colors = [f"#{['ff6b6b','fdcb6e','6c5ce7','00b894'][i]}ff" for i in range(period)]
    legend_handles = [
        Patch(facecolor=c, label=f"Phase {i}: f={cycle_vals[i]}")
        for i, c in enumerate(cycle_colors)
    ]
    ax1.legend(handles=legend_handles, fontsize=9, loc="upper right")

    # Right: Discrete Fourier Transform magnitude — shows peaks at multiples of N/r
    n_count = 16  # 4 counting qubits → 2^4 = 16 states
    dft_mag = np.zeros(n_count)
    for k in range(n_count):
        s = sum(f_vals[x] * np.exp(-2j * np.pi * k * x / n_count) for x in range(n_count))
        dft_mag[k] = abs(s)

    peak_indices = [0, n_count // period, 2 * n_count // period, 3 * n_count // period]
    bar_colors = ["#ff6b6b" if i in peak_indices else "#b2bec3" for i in range(n_count)]
    ax2.bar(range(n_count), dft_mag, color=bar_colors)
    ax2.set_xlabel("Frequency bin k  (= measured state)", fontsize=11)
    ax2.set_ylabel("|DFT(f)|  ≈  QFT measurement probability", fontsize=10)
    ax2.set_title(
        "Frequency Domain (what QFT reveals)\nPeaks at k = 0, 4, 8, 12  →  period r = 4",
        fontsize=10
    )
    ax2.set_xticks(range(n_count))
    ax2.grid(True, axis="y", alpha=0.3)

    for idx in peak_indices:
        if idx > 0:
            ax2.annotate(
                f"k={idx}\nφ={idx}/{n_count}",
                xy=(idx, dft_mag[idx]),
                xytext=(idx + 0.6, dft_mag[idx] * 0.85),
                fontsize=8, color="#d63031"
            )

    plt.tight_layout()
    return fig


def plot_factoring_complexity():
    """
    Log-scale line chart comparing classical (GNFS) vs Shor's algorithm
    complexity as a function of the number of bits in N.
    """
    bits = np.linspace(8, 4096, 500)

    # Classical best: General Number Field Sieve
    # Complexity ≈ exp(1.9 * (bits * ln2)^(1/3) * (ln(bits * ln2))^(2/3))
    # We plot log10 of operations for readability
    ln2 = np.log(2)
    n_classical = bits * ln2  # ln(N) where N has 'bits' bits
    log10_classical = (1.9 * n_classical ** (1 / 3) * np.log(n_classical) ** (2 / 3)) / np.log(10)

    # Shor's: O(n^3) polynomial in number of bits
    log10_shor = 3 * np.log10(bits)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bits, log10_classical, label="Classical (GNFS) — sub-exponential",
            color="#e17055", linewidth=2.5)
    ax.plot(bits, log10_shor, label="Shor's Algorithm — polynomial O(n³)",
            color="#6c5ce7", linewidth=2.5)

    # RSA key size markers
    rsa_sizes = [512, 1024, 2048, 4096]
    for rsa in rsa_sizes:
        ln_rsa = rsa * ln2
        log10_c = (1.9 * ln_rsa ** (1 / 3) * np.log(ln_rsa) ** (2 / 3)) / np.log(10)
        log10_s = 3 * np.log10(rsa)
        ax.axvline(rsa, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(rsa + 20, ax.get_ylim()[0] + 2 if ax.get_ylim()[0] > 0 else 2,
                f"RSA-{rsa}", fontsize=8, color="gray", rotation=90, va="bottom")

    # Shade the advantage region
    ax.fill_between(bits, log10_classical, log10_shor,
                    where=log10_classical > log10_shor,
                    alpha=0.12, color="#6c5ce7", label="Quantum advantage region")

    ax.set_xlabel("Key Size (bits in N)", fontsize=12)
    ax.set_ylabel("log₁₀(Operations Required)", fontsize=12)
    ax.set_title("Factoring Complexity: Classical vs Shor's Algorithm", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Reference lines for operations scale
    ax.axhline(9, color="#00b894", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(4200, 9.3, "10⁹ ops/sec\n(1 GHz CPU)", fontsize=8, color="#00b894", ha="right")
    ax.axhline(26, color="#fdcb6e", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(4200, 26.3, "~age of universe\nat 10⁹ ops/sec", fontsize=8,
            color="#fdcb6e", ha="right")

    plt.tight_layout()
    return fig


def plot_rsa_time_to_break():
    """
    Horizontal bar chart: estimated time to break RSA keys of various sizes
    with classical (GNFS) vs Shor's algorithm on a hypothetical fault-tolerant
    quantum computer, to make the threat concrete.
    """
    rsa_labels = ["RSA-512", "RSA-1024", "RSA-2048", "RSA-4096"]

    # Classical times (approximate, sourced from cryptographic literature)
    classical_times = [
        "~2 months",
        "~100 years",
        "~10¹⁵ years",
        "~10³⁴ years",
    ]
    classical_log = [np.log10(60 * 24 * 60),        # 2 months in seconds
                     np.log10(100 * 365 * 24 * 3600),
                     np.log10(1e15 * 365 * 24 * 3600),
                     np.log10(1e34 * 365 * 24 * 3600)]

    # Shor's times on hypothetical large-scale quantum computer
    # Estimated circuit execution: O(n³) gates at ~1 µs per gate
    quantum_times = [
        "~seconds",
        "~minutes",
        "~hours",
        "~days",
    ]
    quantum_log = [np.log10(10),          # ~10 seconds
                   np.log10(600),          # ~10 minutes
                   np.log10(3 * 3600),     # ~3 hours
                   np.log10(2 * 24 * 3600)]  # ~2 days

    fig, ax = plt.subplots(figsize=(12, 5))
    y = np.arange(len(rsa_labels))
    height = 0.35

    bars_c = ax.barh(y + height / 2, classical_log, height, label="Classical (GNFS)",
                     color="#e17055", alpha=0.9)
    bars_q = ax.barh(y - height / 2, quantum_log, height, label="Shor's Algorithm (fault-tolerant QC)",
                     color="#6c5ce7", alpha=0.9)

    # Labels on bars
    for i, (bar, txt) in enumerate(zip(bars_c, classical_times)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                txt, va="center", fontsize=9, color="#e17055", fontweight="bold")
    for i, (bar, txt) in enumerate(zip(bars_q, quantum_times)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                txt, va="center", fontsize=9, color="#6c5ce7", fontweight="bold")

    # Reference lines
    ax.axvline(np.log10(3600), color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(np.log10(3600), ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 3.5,
            "1 hour", fontsize=8, color="gray", ha="center")
    ax.axvline(np.log10(365 * 24 * 3600), color="gray", linestyle="--",
               linewidth=1, alpha=0.6)
    ax.text(np.log10(365 * 24 * 3600), ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 3.5,
            "1 year", fontsize=8, color="gray", ha="center")

    ax.set_yticks(y)
    ax.set_yticklabels(rsa_labels, fontsize=11)
    ax.set_xlabel("log₁₀(Time in seconds)", fontsize=11)
    ax.set_title("Time to Break RSA: Classical vs Shor's Algorithm\n"
                 "(Shor's assumes large-scale fault-tolerant quantum computer)",
                 fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig
