"""
QAOA visualisation suite — all functions return a matplotlib Figure.

Call plt.close(fig) after st.pyplot(fig) to avoid memory leaks.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

_PALETTE = {
    "ideal":     "#2196F3",
    "noisy":     "#F44336",
    "mitigated": "#4CAF50",
    "zne_lin":   "#FF9800",
    "zne_quad":  "#00BCD4",
    "max_cut":   "#FF9800",
    "gw":        "#AB47BC",
    "set0":      "#42A5F5",
    "set1":      "#EF5350",
    "edge_cut":  "#FF9800",
    "edge_uncut":"#607D8B",
}
_LEVEL_COLORS = ["#B3E5FC", "#4FC3F7", "#0288D1", "#01579B"]


# ---------------------------------------------------------------------------
# 1. Graph
# ---------------------------------------------------------------------------

def plot_graph(
    n_nodes, edges, pos,
    partition=None, weights=None, title="Max-Cut Graph",
) -> plt.Figure:
    """Draw the graph, optionally with partition colouring and edge-weight labels."""
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")
    ax.set_aspect("equal")
    ax.axis("off")

    for (u, v) in edges:
        xu, yu = pos[u]; xv, yv = pos[v]
        if partition is not None:
            is_cut = partition[u] != partition[v]
            color, lw, ls = (
                (_PALETTE["edge_cut"], 3.0, "-") if is_cut
                else (_PALETTE["edge_uncut"], 1.5, "--")
            )
        else:
            color, lw, ls = "#90A4AE", 2.0, "-"
        ax.plot([xu, xv], [yu, yv], color=color, linewidth=lw, linestyle=ls, zorder=1)

        # ── Edge weight label ────────────────────────────────────────────
        if weights is not None:
            w = weights.get((min(u, v), max(u, v)), 1.0)
            mx, my = (xu + xv) / 2, (yu + yv) / 2
            # Slight perpendicular offset so label doesn't sit on the line
            dx, dy = xv - xu, yv - yu
            norm   = max((dx**2 + dy**2)**0.5, 1e-6)
            ox, oy = -dy / norm * 0.18, dx / norm * 0.18
            ax.text(mx + ox, my + oy, f"{w:.1f}", ha="center", va="center",
                    fontsize=8, color="#FFD54F", fontweight="bold", zorder=4)

    node_r = 0.18
    for node, (x, y) in pos.items():
        face = (_PALETTE["set0"] if partition is not None and partition[node] == 0
                else _PALETTE["set1"] if partition is not None
                else "#546E7A")
        ax.add_patch(plt.Circle((x, y), node_r, color=face, zorder=2,
                                ec="white", linewidth=1.5))
        ax.text(x, y, str(node), ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=3)

    if partition is not None:
        n_cut = sum(1 for (u, v) in edges if partition[u] != partition[v])
        wt_cut = (sum(weights.get((min(u, v), max(u, v)), 1.0)
                      for (u, v) in edges if partition[u] != partition[v])
                  if weights else n_cut)
        title = f"{title}  (cut = {wt_cut:.2g})"

    ax.set_title(title, color="white", fontsize=13, pad=8)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Circuit diagram
# ---------------------------------------------------------------------------

def plot_qaoa_circuit(circuit, title="QAOA Circuit") -> plt.Figure:
    try:
        fig = circuit.draw(
            output="mpl", fold=60,
            style={"backgroundcolor": "#0E1117", "textcolor": "white",
                   "gatefacecolor": "#1565C0", "gatetextcolor": "white",
                   "barrierfacecolor": "#333333"},
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
    gamma_vals, beta_vals, landscape, opt_gamma, opt_beta, graph_name=""
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")
    im = ax.imshow(landscape.T, origin="lower", cmap="plasma", interpolation="bilinear",
                   aspect="auto",
                   extent=[gamma_vals[0], gamma_vals[-1], beta_vals[0], beta_vals[-1]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("⟨C⟩ (expected weighted cut)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.scatter([opt_gamma], [opt_beta], marker="*", s=300, color="#FF5252", zorder=5,
               label=f"γ*={opt_gamma:.2f}, β*={opt_beta:.2f}")
    ax.legend(loc="upper right", fontsize=9, facecolor="#1A1A2E", labelcolor="white")
    ax.set_xlabel("γ  (cost-layer angle)", color="white", fontsize=11)
    ax.set_ylabel("β  (mixer-layer angle)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_edgecolor("#555555")
    ttl = "QAOA p=1 Optimisation Landscape"
    if graph_name: ttl += f"  —  {graph_name}"
    ax.set_title(ttl, color="white", fontsize=13, pad=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Transpilation comparison
# ---------------------------------------------------------------------------

def plot_transpilation_comparison(results) -> plt.Figure:
    levels = [r["level"] for r in results]
    depths = [r["depth"] for r in results]
    totals = [r["total_gates"] for r in results]
    two_q  = [r["two_qubit_gates"] for r in results]
    one_q  = [t - q for t, q in zip(totals, two_q)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="#0E1117")

    def _style(ax, title):
        ax.set_facecolor("#111827"); ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        for s in ax.spines.values(): s.set_edgecolor("#444444")
        ax.set_xticks(levels)
        ax.set_xticklabels([f"Level {l}" for l in levels], color="white")

    x = np.array(levels)
    ax0 = axes[0]
    ax0.bar(x, one_q, 0.35, label="1-qubit gates", color="#1976D2")
    ax0.bar(x, two_q, 0.35, bottom=one_q, label="2-qubit (CX)", color="#E53935")
    ax0.set_ylabel("Gate count", color="white", fontsize=10)
    ax0.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    _style(ax0, "Total Gate Count")
    for i, t in enumerate(totals):
        ax0.text(i, t + 0.5, str(t), ha="center", color="white", fontsize=9, fontweight="bold")

    ax1 = axes[1]
    bars = ax1.bar(x, depths, 0.5, color=_LEVEL_COLORS[:len(levels)], edgecolor="#EEE", linewidth=0.5)
    ax1.set_ylabel("Circuit depth", color="white", fontsize=10)
    _style(ax1, "Circuit Depth")
    for bar, d in zip(bars, depths):
        ax1.text(bar.get_x() + bar.get_width()/2, d + 0.3, str(d),
                 ha="center", color="white", fontsize=10, fontweight="bold")

    ax2 = axes[2]
    bars2 = ax2.bar(x, two_q, 0.5, color="#E53935", edgecolor="#EEE", linewidth=0.5)
    ax2.set_ylabel("CX / ECR count", color="white", fontsize=10)
    _style(ax2, "2-Qubit Gates (routing overhead)")
    for bar, q in zip(bars2, two_q):
        ax2.text(bar.get_x() + bar.get_width()/2, q + 0.1, str(q),
                 ha="center", color="white", fontsize=10, fontweight="bold")

    fig.suptitle("Transpilation Analysis — IBM 5-Qubit Backend (T-shape)",
                 color="white", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Qubit mapping
# ---------------------------------------------------------------------------

def plot_qubit_mapping(transpilation_results, n_virtual, coupling_map_edges, phys_qubit_pos) -> plt.Figure:
    cmap = matplotlib.cm.get_cmap("Set1", n_virtual + 1)
    virt_colors = [mcolors.to_hex(cmap(i)) for i in range(n_virtual)]
    n_levels = len(transpilation_results)
    fig, axes = plt.subplots(1, n_levels, figsize=(3.5 * n_levels, 4), facecolor="#0E1117")
    if n_levels == 1: axes = [axes]

    for ax, res in zip(axes, transpilation_results):
        ax.set_facecolor("#111827"); ax.set_aspect("equal"); ax.axis("off")
        mapping = res["qubit_mapping"]
        phys_to_virt = {v: k for k, v in mapping.items()}
        for (u, v) in coupling_map_edges:
            if u in phys_qubit_pos and v in phys_qubit_pos:
                xu, yu = phys_qubit_pos[u]; xv, yv = phys_qubit_pos[v]
                ax.plot([xu, xv], [yu, yv], color="#555555", linewidth=2.5, zorder=1)
        for phys, (x, y) in phys_qubit_pos.items():
            if phys in phys_to_virt:
                virt = phys_to_virt[phys]; face = virt_colors[virt]; label = f"q{virt}"; alpha = 1.0
            else:
                face = "#2D3748"; label = f"P{phys}"; alpha = 0.5
            ax.add_patch(plt.Circle((x, y), 0.22, color=face, zorder=2, ec="white", linewidth=1.5, alpha=alpha))
            ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=3)
            ax.text(x, y - 0.32, f"P{phys}", ha="center", va="top", fontsize=7, color="#AAAAAA", zorder=3)
        ax.set_xlim(-0.5, 2.7); ax.set_ylim(-0.5, 2.7)
        ax.set_title(f"Opt. Level {res['level']}", color="white", fontsize=10, pad=5)

    patches = [mpatches.Patch(color=virt_colors[i], label=f"Virtual q{i}") for i in range(n_virtual)]
    patches.append(mpatches.Patch(color="#2D3748", label="Unused physical"))
    fig.legend(handles=patches, loc="lower center", ncol=min(n_virtual + 1, 4),
               facecolor="#1A1A2E", labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Qubit Mapping: Virtual → Physical on IBM 5-Qubit Topology",
                 color="white", fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Ideal · Noisy · Mitigated comparison
# ---------------------------------------------------------------------------

def plot_ideal_vs_noisy_vs_mitigated(
    comparison, edges, n_nodes, max_cut, weights=None
) -> plt.Figure:
    from algorithms.qaoa import compute_expected_cut, approximation_ratio

    labels = ["Ideal", "Noisy", "Mitigated"]
    colors = [_PALETTE["ideal"], _PALETTE["noisy"], _PALETTE["mitigated"]]
    keys   = ["ideal", "noisy", "mitigated"]

    exp_cuts = [compute_expected_cut(comparison[k], edges, n_nodes, weights=weights) for k in keys]
    ratios   = [approximation_ratio(e, max_cut) for e in exp_cuts]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="#0E1117")

    def _style(ax):
        ax.set_facecolor("#111827"); ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_edgecolor("#444444")

    x = np.arange(3)
    bars0 = ax0.bar(x, exp_cuts, 0.5, color=colors, edgecolor="white", linewidth=0.8)
    ax0.axhline(max_cut, color=_PALETTE["max_cut"], linewidth=2, linestyle="--",
                label=f"Optimum C* = {max_cut:.2f}")
    for bar, val in zip(bars0, exp_cuts):
        ax0.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                 f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=11)
    ax0.set_xticks(x); ax0.set_xticklabels(labels, color="white", fontsize=11)
    ax0.set_ylabel("Expected Cut Value ⟨C⟩", color="white", fontsize=10)
    ax0.set_title("Solution Quality by Simulation Mode", color="white", fontsize=11, pad=8)
    ax0.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax0.set_ylim(0, max_cut * 1.3)
    _style(ax0)

    bars1 = ax1.bar(x, ratios, 0.5, color=colors, edgecolor="white", linewidth=0.8)
    ax1.axhline(1.0, color=_PALETTE["max_cut"], linewidth=2, linestyle="--", label="Perfect (ratio = 1.0)")
    for bar, r in zip(bars1, ratios):
        ax1.text(bar.get_x() + bar.get_width()/2, r + 0.01,
                 f"{r:.3f}", ha="center", va="bottom", color="white", fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, color="white", fontsize=11)
    ax1.set_ylabel("Approximation Ratio ⟨C⟩ / C*", color="white", fontsize=10)
    ax1.set_title("Approximation Ratio vs Classical Optimum", color="white", fontsize=11, pad=8)
    ax1.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax1.set_ylim(0, 1.2)
    _style(ax1)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Solution distribution
# ---------------------------------------------------------------------------

def plot_solution_distribution(comparison, edges, n_nodes, max_cut, weights=None) -> plt.Figure:
    from algorithms.qaoa import compute_cut_value

    n_states = 2 ** n_nodes

    def _probs(counts):
        total = max(sum(counts.values()), 1)
        return {bs: c / total for bs, c in counts.items()}

    probs = {k: _probs(comparison[k]) for k in ("ideal", "noisy", "mitigated")}

    all_bs = [format(i, f"0{n_nodes}b") for i in range(n_states)]
    all_bs.sort(key=lambda b: (compute_cut_value(b, edges, weights), b))
    cut_vals = [compute_cut_value(b, edges, weights) for b in all_bs]
    max_cv   = max(cut_vals) if cut_vals else 1.0
    cut_cmap = matplotlib.cm.get_cmap("RdYlGn", max(int(max_cv) + 2, 3))

    fig, ax = plt.subplots(figsize=(max(10, n_states * 0.9), 5), facecolor="#0E1117")
    ax.set_facecolor("#111827")
    x = np.arange(n_states); w = 0.26
    for key, label, offset, hatch in zip(
        ["ideal", "noisy", "mitigated"], ["Ideal", "Noisy", "Mitigated"],
        [-w, 0, w], ["", "//", ".."]
    ):
        heights = [probs[key].get(bs, 0.0) for bs in all_bs]
        for xi, (h, cv) in enumerate(zip(heights, cut_vals)):
            ec = mcolors.to_hex(cut_cmap(cv / max_cv)) if max_cv > 0 else "#888"
            ax.bar(xi + offset, h, w, color=_PALETTE[key], alpha=0.75,
                   edgecolor=ec, linewidth=1.5, hatch=hatch,
                   label=label if xi == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n({cv:.2g})" for b, cv in zip(all_bs, cut_vals)],
                       color="white", fontsize=7)
    prev_cv = -1
    for xi, cv in enumerate(cut_vals):
        if cv != prev_cv and xi > 0:
            ax.axvline(xi - 0.5, color="#444", linewidth=1, linestyle="--")
        prev_cv = cv
    ax.set_ylabel("Probability", color="white", fontsize=11)
    ax.set_title("Solution Distribution — Ideal / Noisy / Mitigated\n"
                 "(sorted by weighted cut value; colour = cut value)",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_edgecolor("#444")
    mode_patches = [mpatches.Patch(color=_PALETTE[k], alpha=0.75, label=l)
                    for k, l in zip(["ideal","noisy","mitigated"],["Ideal","Noisy","Mitigated"])]
    ax.legend(handles=mode_patches, loc="upper left",
              facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Noise sweep
# ---------------------------------------------------------------------------

def plot_noise_sweep(sweep_results, max_cut) -> plt.Figure:
    scales   = [r["scale"]        for r in sweep_results]
    exp_cuts = [r["expected_cut"] for r in sweep_results]
    ratios   = [r["approx_ratio"] for r in sweep_results]
    random_guess = max_cut / 2

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), facecolor="#0E1117", sharex=True)

    def _style(ax):
        ax.set_facecolor("#111827"); ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_edgecolor("#444")
        ax.yaxis.label.set_color("white")

    ax0.plot(scales, exp_cuts, "o-", color=_PALETTE["ideal"], linewidth=2.5, markersize=8)
    ax0.axhline(max_cut,      color=_PALETTE["max_cut"], linewidth=2, linestyle="--", label=f"C* = {max_cut:.2f}")
    ax0.axhline(random_guess, color="#888", linewidth=1.5, linestyle=":", label=f"Random = {random_guess:.2f}")
    ax0.fill_between(scales, random_guess, exp_cuts,
                     where=[e >= random_guess for e in exp_cuts],
                     alpha=0.15, color=_PALETTE["ideal"], label="QAOA advantage")
    ax0.set_ylabel("Expected Cut Value ⟨C⟩", fontsize=11)
    ax0.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax0.set_title("QAOA Quality vs Noise Strength", color="white", fontsize=13)
    _style(ax0)

    ax1.plot(scales, ratios, "s-", color=_PALETTE["mitigated"], linewidth=2.5, markersize=8)
    ax1.axhline(1.0, color=_PALETTE["max_cut"], linewidth=2, linestyle="--", label="Perfect (1.0)")
    ax1.axhline(0.5, color="#888", linewidth=1.5, linestyle=":", label="Random (0.5)")
    ax1.fill_between(scales, 0.5, ratios, where=[r >= 0.5 for r in ratios],
                     alpha=0.15, color=_PALETTE["mitigated"])
    ax1.set_ylabel("Approximation Ratio", fontsize=11)
    ax1.set_xlabel("Noise Scale Factor  (1.0 = baseline IBM)", color="white", fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    _style(ax1); ax1.tick_params(axis="x", colors="white")

    idx_1 = next((i for i, r in enumerate(sweep_results) if abs(r["scale"] - 1.0) < 0.01), None)
    if idx_1 is not None:
        for ax in (ax0, ax1):
            ax.axvline(1.0, color="white", linewidth=1, linestyle="-.", alpha=0.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. Depth–quality sweep
# ---------------------------------------------------------------------------

def plot_depth_quality(depth_quality_results, max_cut) -> plt.Figure:
    ps      = [r["p"]           for r in depth_quality_results]
    depths  = [r["depth"]       for r in depth_quality_results]
    ideal_c = [r["ideal_cut"]   for r in depth_quality_results]
    noisy_c = [r["noisy_cut"]   for r in depth_quality_results]
    ideal_r = [r["ideal_ratio"] for r in depth_quality_results]
    noisy_r = [r["noisy_ratio"] for r in depth_quality_results]

    x = np.arange(len(ps))
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="#0E1117")

    def _style(ax, title):
        ax.set_facecolor("#111827"); ax.tick_params(colors="white")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xticks(x); ax.set_xticklabels([f"p={p}" for p in ps], color="white", fontsize=10)
        for s in ax.spines.values(): s.set_edgecolor("#444")

    ax0.bar(x, depths, 0.5, color="#7E57C2", edgecolor="white", linewidth=0.8)
    for xi, d in zip(x, depths):
        ax0.text(xi, d + 0.3, str(d), ha="center", color="white", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Gate depth", color="white", fontsize=10)
    _style(ax0, "Circuit Depth vs p")

    w = 0.3
    ax1.bar(x - w/2, ideal_c, w, color=_PALETTE["ideal"],  label="Ideal",  edgecolor="white")
    ax1.bar(x + w/2, noisy_c, w, color=_PALETTE["noisy"],  label="Noisy",  edgecolor="white")
    ax1.axhline(max_cut, color=_PALETTE["max_cut"], linestyle="--", linewidth=2, label=f"C*={max_cut:.2f}")
    ax1.set_ylabel("Expected Cut Value ⟨C⟩", color="white", fontsize=10)
    ax1.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax1.set_ylim(0, max_cut * 1.3)
    _style(ax1, "Solution Quality: Ideal vs Noisy")

    ax2.plot(x, ideal_r, "o-", color=_PALETTE["ideal"],  linewidth=2.5, markersize=9, label="Ideal")
    ax2.plot(x, noisy_r, "s--", color=_PALETTE["noisy"], linewidth=2.5, markersize=9, label="Noisy")
    ax2.axhline(1.0, color=_PALETTE["max_cut"], linestyle="--", linewidth=2, label="Optimal (1.0)")
    ax2.fill_between(x, noisy_r, ideal_r, alpha=0.25, color="#FF8A65", label="Noise penalty")
    ax2.set_ylabel("Approximation Ratio", color="white", fontsize=10)
    ax2.set_ylim(0, 1.2)
    ax2.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    _style(ax2, "Approx. Ratio: Ideal vs Noisy")

    fig.suptitle("QAOA Performance vs Circuit Depth  (p layers)",
                 color="white", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ★ 10. Zero-Noise Extrapolation
# ---------------------------------------------------------------------------

def plot_zne(zne_result, ideal_cut, max_cut) -> plt.Figure:
    """
    Left panel  — measured ⟨C⟩ at each noise scale, fit lines, and ZNE estimates.
    Right panel — bar comparison: noisy / ZNE-linear / ZNE-quadratic / ideal.
    """
    scales     = zne_result["scales"]
    cuts       = zne_result["cuts"]
    zne_lin    = zne_result["zne_linear"]
    zne_quad   = zne_result["zne_quadratic"]
    c_lin      = zne_result["coeffs_linear"]
    c_quad     = zne_result["coeffs_quadratic"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0E1117")

    # ── Left: fits and extrapolation ──────────────────────────────────────
    ax0.set_facecolor("#111827")
    x_ext = np.linspace(0, max(scales) * 1.1, 300)

    # Linear fit
    ax0.plot(x_ext, np.polyval(c_lin, x_ext), "--",
             color=_PALETTE["zne_lin"], linewidth=2, label="Linear fit")
    ax0.scatter([0], [zne_lin], marker="*", s=350, color=_PALETTE["zne_lin"], zorder=7,
                label=f"ZNE linear = {zne_lin:.3f}")

    # Quadratic fit
    if c_quad is not None:
        ax0.plot(x_ext, np.polyval(c_quad, x_ext), "-",
                 color=_PALETTE["zne_quad"], linewidth=2, label="Quadratic fit")
        ax0.scatter([0], [zne_quad], marker="*", s=350, color=_PALETTE["zne_quad"], zorder=7,
                    label=f"ZNE quadratic = {zne_quad:.3f}")

    # Data points
    ax0.scatter(scales, cuts, color=_PALETTE["noisy"], s=120, zorder=6,
                label="Measured ⟨C⟩ (noisy)")

    # Reference lines
    ax0.axhline(ideal_cut, color=_PALETTE["ideal"], linewidth=2, linestyle=":",
                label=f"Ideal = {ideal_cut:.3f}")
    ax0.axhline(max_cut,   color=_PALETTE["max_cut"], linewidth=2, linestyle="--",
                label=f"C* = {max_cut:.2f}")

    # Extrapolation zone shading
    ax0.axvspan(0, min(scales), alpha=0.08, color="white")
    ax0.axvline(min(scales), color="white", linewidth=1, linestyle="-.", alpha=0.4)
    ax0.text(min(scales) * 0.45, cuts.mean(), "← extrapolation", color="#AAAAAA",
             fontsize=8, ha="center", rotation=90, va="center")

    ax0.set_xlabel("Noise Scale Factor λ", color="white", fontsize=11)
    ax0.set_ylabel("Expected Cut Value ⟨C⟩", color="white", fontsize=11)
    ax0.set_title("ZNE: Polynomial Fit and Extrapolation to λ=0",
                  color="white", fontsize=12, pad=8)
    ax0.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=8, loc="lower left")
    ax0.tick_params(colors="white")
    for s in ax0.spines.values(): s.set_edgecolor("#444")

    # ── Right: comparison bars ────────────────────────────────────────────
    ax1.set_facecolor("#111827")

    bar_labels = ["Noisy\n(λ=1)", "ZNE\nLinear"]
    bar_vals   = [cuts[0], zne_lin]
    bar_colors = [_PALETTE["noisy"], _PALETTE["zne_lin"]]

    if zne_quad is not None:
        bar_labels.append("ZNE\nQuadratic")
        bar_vals.append(zne_quad)
        bar_colors.append(_PALETTE["zne_quad"])

    bar_labels.append("Ideal")
    bar_vals.append(ideal_cut)
    bar_colors.append(_PALETTE["ideal"])

    xb = np.arange(len(bar_labels))
    bars = ax1.bar(xb, bar_vals, 0.5, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax1.axhline(max_cut,   color=_PALETTE["max_cut"], linewidth=2, linestyle="--",
                label=f"C* = {max_cut:.2f}")
    ax1.axhline(ideal_cut, color=_PALETTE["ideal"],   linewidth=1.5, linestyle=":",
                label=f"Ideal = {ideal_cut:.3f}")

    for bar, val in zip(bars, bar_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=10)

    ax1.set_xticks(xb)
    ax1.set_xticklabels(bar_labels, color="white", fontsize=10)
    ax1.set_ylabel("Expected Cut Value ⟨C⟩", color="white", fontsize=11)
    ax1.set_title("ZNE vs Noisy vs Ideal", color="white", fontsize=12, pad=8)
    ax1.set_ylim(0, max_cut * 1.3)
    ax1.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax1.tick_params(colors="white")
    for s in ax1.spines.values(): s.set_edgecolor("#444")

    fig.suptitle("Zero-Noise Extrapolation (ZNE)", color="white", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ★ 11. Goemans–Williamson comparison
# ---------------------------------------------------------------------------

def plot_gw_comparison(gw_result, qaoa_cut, max_cut, graph_name="") -> plt.Figure:
    """
    Left panel  — bar chart: Classical C* / SDP bound / GW rounded / QAOA.
    Right panel — histogram of cut values from all hyperplane-rounding rounds.
    """
    classical  = gw_result["classical_max_cut"]
    gw_cut     = gw_result["gw_cut"]
    sdp_bound  = gw_result["sdp_bound"]
    cut_dist   = gw_result["cut_distribution"]
    mean_round = gw_result["mean_rounding"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#0E1117")

    # ── Left: algorithm comparison ────────────────────────────────────────
    ax0.set_facecolor("#111827")
    labels = ["Classical\nBrute Force\n(C*)", "GW SDP\nBound\n(upper)", "GW\nRounded", "QAOA\n(Optimised)"]
    values = [classical, sdp_bound, gw_cut, qaoa_cut]
    colors = ["#9C27B0", "#FF9800", "#F44336", "#2196F3"]

    xb   = np.arange(len(labels))
    bars = ax0.bar(xb, values, 0.5, color=colors, edgecolor="white", linewidth=0.8)
    ax0.axhline(classical, color="#9C27B0", linewidth=1.5, linestyle="--", alpha=0.5)

    for bar, val in zip(bars, values):
        ax0.text(bar.get_x() + bar.get_width()/2, val + max(values)*0.02,
                 f"{val:.3f}", ha="center", va="bottom", color="white",
                 fontsize=10, fontweight="bold")

    # Approximation-ratio annotations inside bars
    for i, val in enumerate(values):
        ratio = val / classical if classical > 0 else 0
        label_str = f"{ratio:.3f}×C*"
        ax0.text(i, val * 0.45, label_str,
                 ha="center", va="center", color="white", fontsize=8, alpha=0.9)

    ax0.set_xticks(xb); ax0.set_xticklabels(labels, color="white", fontsize=10)
    ax0.set_ylabel("Cut Value", color="white", fontsize=11)
    ax0.set_title("Algorithm Comparison", color="white", fontsize=12, pad=8)
    ax0.set_ylim(0, max(values) * 1.3)
    ax0.tick_params(colors="white")
    for s in ax0.spines.values(): s.set_edgecolor("#444")

    # GW guarantee annotation
    gw_guarantee = 0.878 * classical
    ax0.axhline(gw_guarantee, color="#AB47BC", linewidth=1.5, linestyle="-.",
                label=f"0.878 × C* = {gw_guarantee:.2f}  (GW guarantee)")
    ax0.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=8)

    # ── Right: rounding distribution ─────────────────────────────────────
    ax1.set_facecolor("#111827")

    # Histogram of rounding outcomes
    unique_vals, counts = np.unique(cut_dist, return_counts=True)
    probs = counts / len(cut_dist)
    bar_w = max(0.08, (float(unique_vals.max()) - float(unique_vals.min())) * 0.6 / max(len(unique_vals), 1))
    ax1.bar(unique_vals, probs, width=bar_w, color="#F44336", edgecolor="white",
            linewidth=0.8, alpha=0.8, label="GW rounding outcomes")

    ax1.axvline(gw_cut,     color="white",           linewidth=2.5, linestyle="-",
                label=f"Best GW cut = {gw_cut:.2f}")
    ax1.axvline(mean_round, color=_PALETTE["zne_lin"], linewidth=2, linestyle="--",
                label=f"Mean = {mean_round:.2f}")
    ax1.axvline(qaoa_cut,   color=_PALETTE["ideal"],  linewidth=2, linestyle=":",
                label=f"QAOA ⟨C⟩ = {qaoa_cut:.2f}")
    ax1.axvline(classical,  color="#9C27B0",           linewidth=2, linestyle="-.",
                label=f"C* = {classical:.2f}")

    ax1.set_xlabel("Cut Value", color="white", fontsize=11)
    ax1.set_ylabel("Fraction of rounding rounds", color="white", fontsize=11)
    ax1.set_title(f"GW Hyperplane Rounding Distribution\n({len(cut_dist)} random rounds)",
                  color="white", fontsize=11, pad=8)
    ax1.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax1.tick_params(colors="white")
    for s in ax1.spines.values(): s.set_edgecolor("#444")

    ttl = "Goemans–Williamson SDP vs QAOA"
    if graph_name: ttl += f"  —  {graph_name.split('(')[0].strip()}"
    fig.suptitle(ttl, color="white", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
