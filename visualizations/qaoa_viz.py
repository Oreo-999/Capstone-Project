"""
QAOA visualisation suite.

All functions return a matplotlib Figure object that Streamlit renders with
``st.pyplot(fig)``.  Call ``plt.close(fig)`` after rendering to avoid memory
leaks across Streamlit reruns.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Consistent colour palette across all plots
_PALETTE = {
    "ideal":     "#2196F3",   # blue
    "noisy":     "#F44336",   # red
    "mitigated": "#4CAF50",   # green
    "max_cut":   "#FF9800",   # amber
    "optimal":   "#9C27B0",   # purple
    "set0":      "#42A5F5",   # light blue  (partition 0)
    "set1":      "#EF5350",   # light red   (partition 1)
    "edge_cut":  "#FF9800",
    "edge_uncut":"#BDBDBD",
}

_LEVEL_COLORS = ["#B3E5FC", "#4FC3F7", "#0288D1", "#01579B"]


# ---------------------------------------------------------------------------
# 1. Graph visualisation
# ---------------------------------------------------------------------------

def plot_graph(
    n_nodes: int,
    edges: list,
    pos: dict,
    partition: list = None,
    title: str = "Max-Cut Graph",
) -> plt.Figure:
    """
    Draw the graph.

    Parameters
    ----------
    partition : list[int] or None
        Vertex colouring (0 or 1).  If provided, edges are highlighted as
        cut (coloured) or uncut (grey), and nodes are coloured by partition.
    """
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Edges ──────────────────────────────────────────────────────────────
    for (u, v) in edges:
        xu, yu = pos[u]
        xv, yv = pos[v]
        if partition is not None:
            is_cut = partition[u] != partition[v]
            color = _PALETTE["edge_cut"] if is_cut else _PALETTE["edge_uncut"]
            lw = 3.0 if is_cut else 1.5
            ls = "-" if is_cut else "--"
        else:
            color, lw, ls = "#90A4AE", 2.0, "-"
        ax.plot([xu, xv], [yu, yv], color=color, linewidth=lw, linestyle=ls, zorder=1)

    # ── Nodes ──────────────────────────────────────────────────────────────
    node_r = 0.18
    for node, (x, y) in pos.items():
        if partition is not None:
            face = _PALETTE["set0"] if partition[node] == 0 else _PALETTE["set1"]
        else:
            face = "#546E7A"
        circle = plt.Circle((x, y), node_r, color=face, zorder=2, ec="white", linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, str(node), ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=3)

    if partition is not None:
        n_cut = sum(1 for (u, v) in edges if partition[u] != partition[v])
        title = f"{title}  —  Cut = {n_cut} edges"

    ax.set_title(title, color="white", fontsize=13, pad=8)

    # Padding
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. QAOA circuit diagram
# ---------------------------------------------------------------------------

def plot_qaoa_circuit(circuit, title: str = "QAOA Circuit") -> plt.Figure:
    """Render the QAOA QuantumCircuit using Qiskit's matplotlib drawer."""
    try:
        fig = circuit.draw(
            output="mpl",
            style={"backgroundcolor": "#0E1117", "textcolor": "white",
                   "gatefacecolor": "#1565C0", "gatetextcolor": "white",
                   "barrierfacecolor": "#333333"},
            fold=60,
        )
        fig.set_facecolor("#0E1117")
        fig.suptitle(title, color="white", fontsize=12)
    except Exception:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0E1117")
        ax.text(0.5, 0.5, "Circuit diagram unavailable\n(install pylatexenc)",
                ha="center", va="center", color="white", fontsize=12)
        ax.axis("off")
    return fig


# ---------------------------------------------------------------------------
# 3. Optimisation landscape
# ---------------------------------------------------------------------------

def plot_optimization_landscape(
    gamma_vals: np.ndarray,
    beta_vals: np.ndarray,
    landscape: np.ndarray,
    opt_gamma: float,
    opt_beta: float,
    graph_name: str = "",
) -> plt.Figure:
    """
    Heatmap of expected Max-Cut value as a function of QAOA angles γ and β.

    The red star marks the optimal (γ*, β*) found by the grid search.
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    im = ax.imshow(
        landscape.T,
        origin="lower",
        extent=[gamma_vals[0], gamma_vals[-1], beta_vals[0], beta_vals[-1]],
        aspect="auto",
        cmap="plasma",
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Expected Cut Value ⟨C⟩", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Mark the optimum
    ax.scatter(
        [opt_gamma], [opt_beta],
        marker="*", s=300, color="#FF5252", zorder=5, label=f"γ*={opt_gamma:.2f}, β*={opt_beta:.2f}"
    )
    ax.legend(loc="upper right", fontsize=9, facecolor="#1A1A2E", labelcolor="white")

    ax.set_xlabel("γ  (cost-layer angle)", color="white", fontsize=11)
    ax.set_ylabel("β  (mixer-layer angle)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    ttl = f"QAOA p=1 Parameter Landscape"
    if graph_name:
        ttl += f" — {graph_name}"
    ax.set_title(ttl, color="white", fontsize=13, pad=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Transpilation comparison  (gate count + depth bar charts)
# ---------------------------------------------------------------------------

def plot_transpilation_comparison(results: list) -> plt.Figure:
    """
    Side-by-side bar charts showing how gate count and circuit depth vary
    across Qiskit optimisation levels 0–3.
    """
    levels = [r["level"] for r in results]
    depths = [r["depth"] for r in results]
    totals = [r["total_gates"] for r in results]
    two_q  = [r["two_qubit_gates"] for r in results]
    one_q  = [t - q for t, q in zip(totals, two_q)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="#0E1117")

    def _style_ax(ax, title):
        ax.set_facecolor("#111827")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_xticks(levels)
        ax.set_xticklabels([f"Level {l}" for l in levels], color="white")

    x = np.array(levels)
    bar_w = 0.35

    # ── Panel 1: Total gate count (stacked 1Q / 2Q) ──────────────────────
    ax0 = axes[0]
    b1 = ax0.bar(x, one_q, bar_w, label="1-qubit gates", color="#1976D2")
    b2 = ax0.bar(x, two_q, bar_w, bottom=one_q, label="2-qubit gates (CX)", color="#E53935")
    ax0.set_ylabel("Gate count", color="white", fontsize=10)
    ax0.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    _style_ax(ax0, "Total Gate Count")
    for i, (o, t) in enumerate(zip(one_q, totals)):
        ax0.text(i, totals[i] + 0.5, str(totals[i]), ha="center", va="bottom",
                 color="white", fontsize=9, fontweight="bold")

    # ── Panel 2: Circuit depth ────────────────────────────────────────────
    ax1 = axes[1]
    bars = ax1.bar(x, depths, 0.5,
                   color=_LEVEL_COLORS[:len(levels)], edgecolor="#EEEEEE", linewidth=0.5)
    ax1.set_ylabel("Circuit depth", color="white", fontsize=10)
    _style_ax(ax1, "Circuit Depth")
    for bar, d in zip(bars, depths):
        ax1.text(bar.get_x() + bar.get_width() / 2, d + 0.3, str(d),
                 ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")

    # ── Panel 3: 2-qubit gate count (routing efficiency) ─────────────────
    ax2 = axes[2]
    bars2 = ax2.bar(x, two_q, 0.5, color="#E53935", edgecolor="#EEEEEE", linewidth=0.5)
    ax2.set_ylabel("CX / ECR gate count", color="white", fontsize=10)
    _style_ax(ax2, "2-Qubit Gates  (routing overhead)")
    for bar, q in zip(bars2, two_q):
        ax2.text(bar.get_x() + bar.get_width() / 2, q + 0.1, str(q),
                 ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Transpilation Analysis — IBM 5-Qubit Backend (T-shape topology)",
        color="white", fontsize=13, y=1.02
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Qubit-mapping visualisation
# ---------------------------------------------------------------------------

def plot_qubit_mapping(
    transpilation_results: list,
    n_virtual: int,
    coupling_map_edges: list,
    phys_qubit_pos: dict,
) -> plt.Figure:
    """
    Draw the 5-qubit coupling-map topology for each transpilation level,
    highlighting which physical qubit each virtual qubit maps to.
    """
    cmap_nodes = matplotlib.cm.get_cmap("Set1", n_virtual + 1)
    virt_colors = [mcolors.to_hex(cmap_nodes(i)) for i in range(n_virtual)]

    n_levels = len(transpilation_results)
    fig, axes = plt.subplots(1, n_levels, figsize=(3.5 * n_levels, 4), facecolor="#0E1117")
    if n_levels == 1:
        axes = [axes]

    n_phys = max(phys_qubit_pos.keys()) + 1

    for ax, res in zip(axes, transpilation_results):
        ax.set_facecolor("#111827")
        ax.set_aspect("equal")
        ax.axis("off")

        mapping = res["qubit_mapping"]  # {virt_idx: phys_idx}
        phys_to_virt = {v: k for k, v in mapping.items()}

        # Draw coupling edges
        for (u, v) in coupling_map_edges:
            if u in phys_qubit_pos and v in phys_qubit_pos:
                xu, yu = phys_qubit_pos[u]
                xv, yv = phys_qubit_pos[v]
                ax.plot([xu, xv], [yu, yv], color="#555555", linewidth=2.5, zorder=1)

        # Draw physical qubits
        for phys, (x, y) in phys_qubit_pos.items():
            if phys in phys_to_virt:
                virt = phys_to_virt[phys]
                face = virt_colors[virt]
                label = f"q{virt}"
                alpha = 1.0
            else:
                face = "#2D3748"
                label = f"P{phys}"
                alpha = 0.5

            circ = plt.Circle((x, y), 0.22, color=face, zorder=2,
                               ec="white", linewidth=1.5, alpha=alpha)
            ax.add_patch(circ)
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white", zorder=3)
            ax.text(x, y - 0.32, f"P{phys}", ha="center", va="top",
                    fontsize=7, color="#AAAAAA", zorder=3)

        ax.set_xlim(-0.5, 2.7)
        ax.set_ylim(-0.5, 2.7)
        ax.set_title(f"Opt. Level {res['level']}", color="white", fontsize=10, pad=5)

    # Legend: which colour = which virtual qubit
    legend_patches = [
        mpatches.Patch(color=virt_colors[i], label=f"Virtual q{i}")
        for i in range(n_virtual)
    ]
    legend_patches.append(mpatches.Patch(color="#2D3748", label="Unused physical qubit"))
    fig.legend(handles=legend_patches, loc="lower center", ncol=min(n_virtual + 1, 4),
               facecolor="#1A1A2E", labelcolor="white", fontsize=9,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        "Qubit Mapping: Virtual → Physical Qubits on IBM 5-Qubit Topology",
        color="white", fontsize=12, y=1.03
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Ideal · Noisy · Mitigated comparison
# ---------------------------------------------------------------------------

def plot_ideal_vs_noisy_vs_mitigated(
    comparison: dict,
    edges: list,
    n_nodes: int,
    max_cut: int,
) -> plt.Figure:
    """
    Two-panel figure:
      Left  — expected cut value bar chart for each simulation mode.
      Right — approximation ratio (⟨C⟩ / C*) with 1.0 reference line.
    """
    from algorithms.qaoa import compute_expected_cut, approximation_ratio

    labels = ["Ideal", "Noisy", "Mitigated"]
    colors = [_PALETTE["ideal"], _PALETTE["noisy"], _PALETTE["mitigated"]]
    keys   = ["ideal", "noisy", "mitigated"]

    exp_cuts = [
        compute_expected_cut(comparison[k], edges, n_nodes) for k in keys
    ]
    ratios = [approximation_ratio(e, max_cut) for e in exp_cuts]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="#0E1117")

    def _style(ax):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_edgecolor("#444444")

    x = np.arange(3)

    # ── Left: expected cut value ──────────────────────────────────────────
    bars0 = ax0.bar(x, exp_cuts, 0.5, color=colors, edgecolor="white", linewidth=0.8)
    ax0.axhline(max_cut, color=_PALETTE["max_cut"], linewidth=2, linestyle="--",
                label=f"Classical optimum (C* = {max_cut})")
    for bar, val in zip(bars0, exp_cuts):
        ax0.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=11)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, color="white", fontsize=11)
    ax0.set_ylabel("Expected Cut Value ⟨C⟩", color="white", fontsize=10)
    ax0.set_title("Solution Quality by Simulation Mode", color="white", fontsize=11, pad=8)
    ax0.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax0.set_ylim(0, max_cut * 1.3)
    _style(ax0)

    # ── Right: approximation ratio ────────────────────────────────────────
    bars1 = ax1.bar(x, ratios, 0.5, color=colors, edgecolor="white", linewidth=0.8)
    ax1.axhline(1.0, color=_PALETTE["max_cut"], linewidth=2, linestyle="--",
                label="Perfect approximation (ratio = 1.0)")
    for bar, r in zip(bars1, ratios):
        ax1.text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                 f"{r:.3f}", ha="center", va="bottom", color="white", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color="white", fontsize=11)
    ax1.set_ylabel("Approximation Ratio ⟨C⟩ / C*", color="white", fontsize=10)
    ax1.set_title("Approximation Ratio vs Classical Optimum", color="white", fontsize=11, pad=8)
    ax1.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax1.set_ylim(0, 1.2)
    _style(ax1)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Solution distribution histogram
# ---------------------------------------------------------------------------

def plot_solution_distribution(
    comparison: dict,
    edges: list,
    n_nodes: int,
    max_cut: int,
) -> plt.Figure:
    """
    Bar chart showing the probability of each bitstring outcome, coloured by
    its cut value.  Three overlay groups: ideal, noisy, mitigated.
    Shows how noise spreads probability mass away from optimal solutions.
    """
    from algorithms.qaoa import compute_cut_value

    n_states = 2 ** n_nodes

    def _counts_to_probs(counts):
        total = max(sum(counts.values()), 1)
        return {bs: c / total for bs, c in counts.items()}

    probs = {k: _counts_to_probs(comparison[k]) for k in ("ideal", "noisy", "mitigated")}

    # Enumerate all bitstrings sorted by cut value, then lexicographically
    all_bs = [format(i, f"0{n_nodes}b") for i in range(n_states)]
    all_bs.sort(key=lambda b: (compute_cut_value(b, edges), b))
    cut_vals = [compute_cut_value(b, edges) for b in all_bs]

    # Colour map by cut value
    cut_cmap = matplotlib.cm.get_cmap("RdYlGn", max_cut + 1)

    fig, ax = plt.subplots(figsize=(max(10, n_states * 0.9), 5), facecolor="#0E1117")
    ax.set_facecolor("#111827")

    x = np.arange(n_states)
    w = 0.26
    keys   = ["ideal", "noisy", "mitigated"]
    labels = ["Ideal", "Noisy", "Mitigated"]
    offsets = [-w, 0, w]
    hatches = ["", "//", ".."]

    for key, label, offset, hatch in zip(keys, labels, offsets, hatches):
        heights = [probs[key].get(bs, 0.0) for bs in all_bs]
        # Edge colour by cut value
        for xi, (h, cv) in enumerate(zip(heights, cut_vals)):
            edge_c = mcolors.to_hex(cut_cmap(cv / max_cut)) if max_cut > 0 else "#888888"
            ax.bar(xi + offset, h, w,
                   color=_PALETTE[key], alpha=0.75, edgecolor=edge_c, linewidth=1.5,
                   hatch=hatch, label=label if xi == 0 else "")

    # Annotate cut values below x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b}\n(cut={c})" for b, c in zip(all_bs, cut_vals)],
        color="white", fontsize=7, rotation=0
    )

    # Vertical lines separating cut value groups
    prev_cv = -1
    for xi, cv in enumerate(cut_vals):
        if cv != prev_cv and xi > 0:
            ax.axvline(xi - 0.5, color="#444444", linewidth=1, linestyle="--")
        prev_cv = cv

    ax.set_ylabel("Probability", color="white", fontsize=11)
    ax.set_title(
        "Solution Distribution: Ideal vs Noisy vs Mitigated\n"
        "(bitstrings sorted by Max-Cut value; colour = cut value)",
        color="white", fontsize=11, pad=10
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Legend for modes
    mode_patches = [
        mpatches.Patch(color=_PALETTE[k], alpha=0.75, label=l)
        for k, l in zip(keys, labels)
    ]
    # Legend for cut-value colour scale
    cut_patches = [
        mpatches.Patch(color=mcolors.to_hex(cut_cmap(cv / max_cut)),
                       label=f"Cut = {cv}")
        for cv in range(max_cut + 1)
    ] if max_cut <= 6 else []

    ax.legend(handles=mode_patches + cut_patches,
              loc="upper left", facecolor="#1A1A2E", labelcolor="white", fontsize=8,
              ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Noise-strength sweep
# ---------------------------------------------------------------------------

def plot_noise_sweep(
    sweep_results: list,
    max_cut: int,
) -> plt.Figure:
    """
    Dual-axis plot showing expected cut value and approximation ratio as a
    function of noise scale factor.

    The dashed orange line marks the classical optimum (C*).
    The shaded region shows where QAOA still exceeds a random guess (C*/2).
    """
    scales = [r["scale"] for r in sweep_results]
    exp_cuts = [r["expected_cut"] for r in sweep_results]
    ratios = [r["approx_ratio"] for r in sweep_results]
    random_guess = max_cut / 2  # expected value of random partition

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), facecolor="#0E1117", sharex=True)

    def _style(ax):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_edgecolor("#444444")
        ax.yaxis.label.set_color("white")

    # ── Top: expected cut value ───────────────────────────────────────────
    ax0.plot(scales, exp_cuts, "o-", color=_PALETTE["ideal"], linewidth=2.5,
             markersize=8, label="QAOA ⟨C⟩")
    ax0.axhline(max_cut, color=_PALETTE["max_cut"], linewidth=2, linestyle="--",
                label=f"Classical optimum C* = {max_cut}")
    ax0.axhline(random_guess, color="#888888", linewidth=1.5, linestyle=":",
                label=f"Random guess = {random_guess:.1f}")
    ax0.fill_between(scales, random_guess, exp_cuts,
                     where=[e >= random_guess for e in exp_cuts],
                     alpha=0.15, color=_PALETTE["ideal"], label="QAOA advantage")
    ax0.set_ylabel("Expected Cut Value ⟨C⟩", fontsize=11)
    ax0.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax0.set_title("QAOA Solution Quality vs. Noise Strength", color="white", fontsize=13)
    _style(ax0)

    # ── Bottom: approximation ratio ───────────────────────────────────────
    ax1.plot(scales, ratios, "s-", color=_PALETTE["mitigated"], linewidth=2.5,
             markersize=8, label="Approximation ratio ⟨C⟩/C*")
    ax1.axhline(1.0, color=_PALETTE["max_cut"], linewidth=2, linestyle="--",
                label="Perfect (ratio = 1.0)")
    ax1.axhline(0.5, color="#888888", linewidth=1.5, linestyle=":",
                label="Random baseline (ratio = 0.5)")
    ax1.fill_between(scales, 0.5, ratios,
                     where=[r >= 0.5 for r in ratios],
                     alpha=0.15, color=_PALETTE["mitigated"])
    ax1.set_ylabel("Approximation Ratio ⟨C⟩ / C*", fontsize=11)
    ax1.set_xlabel("Noise Scale Factor  (1.0 = baseline IBM noise)", color="white", fontsize=11)
    ax1.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax1.set_ylim(0.0, 1.15)
    _style(ax1)
    ax1.tick_params(axis="x", colors="white")
    ax1.set_xlabel("Noise Scale Factor  (1.0 = baseline IBM noise)", color="white", fontsize=11)

    # Annotate scale = 1.0 (real device)
    idx_1 = next((i for i, r in enumerate(sweep_results) if abs(r["scale"] - 1.0) < 0.01), None)
    if idx_1 is not None:
        for ax in (ax0, ax1):
            ax.axvline(1.0, color="#FFFFFF", linewidth=1, linestyle="-.", alpha=0.5)
        ax0.annotate("Real IBM\ndevice", xy=(1.0, exp_cuts[idx_1]),
                     xytext=(1.2, exp_cuts[idx_1] * 0.9),
                     color="white", fontsize=8,
                     arrowprops=dict(arrowstyle="->", color="white"))

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. Depth–quality sweep  (p layers)
# ---------------------------------------------------------------------------

def plot_depth_quality(depth_quality_results: list, max_cut: int) -> plt.Figure:
    """
    Three-panel figure showing, for each QAOA depth (p = 1, 2, 3):
      1. Circuit depth (gate layers)
      2. Ideal vs noisy expected cut value
      3. Ideal vs noisy approximation ratio
    """
    ps      = [r["p"]            for r in depth_quality_results]
    depths  = [r["depth"]        for r in depth_quality_results]
    ideal_c = [r["ideal_cut"]    for r in depth_quality_results]
    noisy_c = [r["noisy_cut"]    for r in depth_quality_results]
    ideal_r = [r["ideal_ratio"]  for r in depth_quality_results]
    noisy_r = [r["noisy_ratio"]  for r in depth_quality_results]

    x = np.arange(len(ps))
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="#0E1117")

    def _style(ax, title):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"p={p}" for p in ps], color="white", fontsize=10)
        for s in ax.spines.values():
            s.set_edgecolor("#444444")

    # ── Panel 1: circuit depth ────────────────────────────────────────────
    ax0.bar(x, depths, 0.5, color="#7E57C2", edgecolor="white", linewidth=0.8)
    for xi, d in zip(x, depths):
        ax0.text(xi, d + 0.3, str(d), ha="center", va="bottom", color="white",
                 fontsize=11, fontweight="bold")
    ax0.set_ylabel("Gate depth (excl. measurement)", color="white", fontsize=10)
    _style(ax0, "Circuit Depth vs QAOA Layers")

    # ── Panel 2: expected cut value ───────────────────────────────────────
    w = 0.3
    ax1.bar(x - w / 2, ideal_c, w, color=_PALETTE["ideal"], label="Ideal", edgecolor="white")
    ax1.bar(x + w / 2, noisy_c, w, color=_PALETTE["noisy"], label="Noisy (IBM model)", edgecolor="white")
    ax1.axhline(max_cut, color=_PALETTE["max_cut"], linestyle="--", linewidth=2,
                label=f"Optimum C*={max_cut}")
    for xi, (ic, nc) in enumerate(zip(ideal_c, noisy_c)):
        ax1.text(xi - w / 2, ic + 0.03, f"{ic:.2f}", ha="center", va="bottom",
                 color=_PALETTE["ideal"], fontsize=8)
        ax1.text(xi + w / 2, nc + 0.03, f"{nc:.2f}", ha="center", va="bottom",
                 color=_PALETTE["noisy"], fontsize=8)
    ax1.set_ylabel("Expected Cut Value ⟨C⟩", color="white", fontsize=10)
    ax1.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax1.set_ylim(0, max_cut * 1.3)
    _style(ax1, "Solution Quality: Ideal vs Noisy")

    # ── Panel 3: approximation ratio ──────────────────────────────────────
    ax2.plot(x, ideal_r, "o-", color=_PALETTE["ideal"], linewidth=2.5,
             markersize=9, label="Ideal")
    ax2.plot(x, noisy_r, "s--", color=_PALETTE["noisy"], linewidth=2.5,
             markersize=9, label="Noisy")
    ax2.axhline(1.0, color=_PALETTE["max_cut"], linestyle="--", linewidth=2,
                label="Optimal (1.0)")
    # Shade the ideal–noisy gap
    ax2.fill_between(x, noisy_r, ideal_r, alpha=0.25, color="#FF8A65",
                     label="Noise penalty")
    ax2.set_ylabel("Approximation Ratio ⟨C⟩ / C*", color="white", fontsize=10)
    ax2.set_ylim(0.0, 1.2)
    ax2.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    _style(ax2, "Approx. Ratio: Ideal vs Noisy")

    fig.suptitle(
        "QAOA Performance vs. Circuit Depth  (p = number of layers)",
        color="white", fontsize=13, y=1.02
    )
    fig.tight_layout()
    return fig
