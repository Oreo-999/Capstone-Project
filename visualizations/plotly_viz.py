"""
Interactive Plotly chart functions for the Quantum Algorithm Dashboard.
All figures use plotly_dark template with matching background colors.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PAPER_BG = "#0E1117"
_PLOT_BG  = "#111827"
_FONT_CLR = "#E0E0E0"

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=_PAPER_BG,
    plot_bgcolor=_PLOT_BG,
    font=dict(color=_FONT_CLR),
)


# ---------------------------------------------------------------------------
# Grover – measurement histogram
# ---------------------------------------------------------------------------

def plotly_grover_counts(counts: dict, target: int, n: int = 4) -> go.Figure:
    """
    Interactive bar chart of Grover measurement counts.
    Hover shows count + probability. Target bar highlighted in red.
    """
    all_states = [format(i, f"0{n}b") for i in range(2 ** n)]
    target_state = format(target, f"0{n}b")[::-1]   # little-endian
    total = sum(counts.values())

    values = [counts.get(s, 0) for s in all_states]
    probs  = [v / total if total else 0 for v in values]
    colors = ["#ff6b6b" if s == target_state else "#4ecdc4" for s in all_states]

    fig = go.Figure(go.Bar(
        x=all_states,
        y=values,
        marker_color=colors,
        customdata=list(zip(probs, [target_state] * len(all_states))),
        hovertemplate=(
            "<b>State |%{x}⟩</b><br>"
            "Count: %{y}<br>"
            "Probability: %{customdata[0]:.2%}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(
            text=f"Grover Measurement Results — Target |{target_state}⟩ (decimal {target})",
            font=dict(size=15),
        ),
        xaxis=dict(title="Basis State (little-endian)", tickangle=-45),
        yaxis=dict(title="Counts"),
        showlegend=False,
        height=400,
    )

    # Annotation marking target bar
    target_idx = all_states.index(target_state) if target_state in all_states else -1
    if target_idx >= 0:
        fig.add_annotation(
            x=target_state,
            y=values[target_idx],
            text=f"Target<br>{probs[target_idx]:.1%}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ff6b6b",
            font=dict(color="#ff6b6b", size=12),
            yshift=10,
        )

    return fig


# ---------------------------------------------------------------------------
# Grover – complexity comparison
# ---------------------------------------------------------------------------

def plotly_complexity_comparison() -> go.Figure:
    """
    Interactive line chart: Classical O(N/2) vs Grover O(√N).
    Legend items are clickable to toggle lines.
    """
    N_vals = np.arange(2, 257)
    classical = N_vals / 2
    grover    = np.sqrt(N_vals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=N_vals, y=classical,
        mode="lines",
        name="Classical  O(N/2)",
        line=dict(color="#e17055", width=2.5),
        hovertemplate="N=%{x}<br>Classical queries: %{y:.1f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=N_vals, y=grover,
        mode="lines",
        name="Grover  O(√N)",
        line=dict(color="#6c5ce7", width=2.5),
        hovertemplate="N=%{x}<br>Grover queries: %{y:.1f}<extra></extra>",
    ))

    # Speedup annotation at N=256
    fig.add_annotation(
        x=256, y=np.sqrt(256),
        text=f"Speedup at N=256:<br>{256/2 / np.sqrt(256):.0f}×",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#a29bfe",
        font=dict(color="#a29bfe", size=12),
        bgcolor=_PAPER_BG,
        bordercolor="#a29bfe",
        ax=-80, ay=-30,
    )

    fig.update_layout(
        **_DARK_LAYOUT,
        title="Search Algorithm Complexity: Classical vs Grover's",
        xaxis=dict(title="Search Space Size (N)"),
        yaxis=dict(title="Expected Queries"),
        legend=dict(
            bgcolor=_PAPER_BG,
            bordercolor="#444",
            borderwidth=1,
        ),
        height=420,
    )

    return fig


# ---------------------------------------------------------------------------
# Grover – step-by-step amplitude chart
# ---------------------------------------------------------------------------

def plotly_amplitude_step(target: int, k: int, n: int = 4) -> go.Figure:
    """
    Single-panel amplitude bar chart for Grover iteration k (0 = just H gates).
    Hover shows amplitude and probability (amplitude²).
    """
    N = 2 ** n
    target_bits = format(target, f"0{n}b")[::-1]
    target_idx  = int(target_bits, 2)

    theta = np.arcsin(1.0 / np.sqrt(N))
    amp_target = np.sin((2 * k + 1) * theta)
    amp_other  = np.cos((2 * k + 1) * theta) / np.sqrt(N - 1)

    amplitudes = [amp_target if i == target_idx else amp_other for i in range(N)]
    probs      = [a ** 2 for a in amplitudes]
    colors     = ["#ff6b6b" if i == target_idx else "#4ecdc4" for i in range(N)]

    labels = [format(i, f"0{n}b") for i in range(N)]

    fig = go.Figure(go.Bar(
        x=labels,
        y=amplitudes,
        marker_color=colors,
        customdata=probs,
        hovertemplate=(
            "<b>State |%{x}⟩</b><br>"
            "Amplitude: %{y:.4f}<br>"
            "Probability: %{customdata:.2%}"
            "<extra></extra>"
        ),
    ))

    if k == 0:
        title_str = "Initial State (after H gates) — Uniform superposition"
    else:
        title_str = f"After Iteration {k} — P(target) = {amp_target**2:.1%}"

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=title_str, font=dict(size=15)),
        xaxis=dict(title="Basis State (little-endian)", tickangle=-45),
        yaxis=dict(title="Amplitude", range=[-0.1, 1.1]),
        showlegend=False,
        height=380,
    )

    # Dashed zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)

    # Annotation for target
    fig.add_annotation(
        x=labels[target_idx],
        y=amp_target,
        text=f"Target<br>P={amp_target**2:.1%}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ff6b6b",
        font=dict(color="#ff6b6b", size=12),
        yshift=8,
    )

    return fig


# ---------------------------------------------------------------------------
# QAOA – parameter optimisation landscape
# ---------------------------------------------------------------------------

def plotly_optimization_landscape(
    gamma_vals, beta_vals, landscape, opt_gamma: float, opt_beta: float,
    graph_name: str = ""
) -> go.Figure:
    """
    Interactive heatmap of the p=1 QAOA optimisation landscape.
    Hover shows exact (γ, β, ⟨C⟩). Star marker at optimal angles.
    """
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=landscape.T,
        x=gamma_vals,
        y=beta_vals,
        colorscale="Plasma",
        hovertemplate=(
            "γ = %{x:.3f}<br>"
            "β = %{y:.3f}<br>"
            "⟨C⟩ = %{z:.4f}"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="⟨C⟩", font=dict(color=_FONT_CLR)),
            tickfont=dict(color=_FONT_CLR),
        ),
    ))

    fig.add_trace(go.Scatter(
        x=[opt_gamma],
        y=[opt_beta],
        mode="markers",
        marker=dict(symbol="star", size=18, color="#FF5252",
                    line=dict(color="white", width=1)),
        name=f"Optimum γ*={opt_gamma:.3f}, β*={opt_beta:.3f}",
        hovertemplate=(
            f"<b>Optimum</b><br>γ* = {opt_gamma:.3f}<br>β* = {opt_beta:.3f}"
            "<extra></extra>"
        ),
    ))

    title = "QAOA p=1 Optimisation Landscape"
    if graph_name:
        title += f" — {graph_name}"

    fig.update_layout(
        **_DARK_LAYOUT,
        title=title,
        xaxis=dict(title="γ (cost-layer angle)"),
        yaxis=dict(title="β (mixer-layer angle)"),
        legend=dict(bgcolor=_PAPER_BG, bordercolor="#444", borderwidth=1),
        height=460,
    )

    return fig


# ---------------------------------------------------------------------------
# QAOA – noise sweep line chart
# ---------------------------------------------------------------------------

def plotly_noise_sweep(sweep: list, max_cut: float) -> go.Figure:
    """
    Interactive two-panel chart: ⟨C⟩ and approximation ratio vs noise scale.
    Includes reference lines, shaded quantum advantage region, and range slider.
    """
    scales    = [d["scale"] for d in sweep]
    cuts      = [d["expected_cut"] for d in sweep]
    ratios    = [d["expected_cut"] / max_cut if max_cut > 0 else 0 for d in sweep]
    random_c  = max_cut / 2

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Expected Cut ⟨C⟩ vs Noise Scale",
                        "Approximation Ratio vs Noise Scale"),
        vertical_spacing=0.12,
    )

    # --- Top panel: ⟨C⟩ ---
    fig.add_trace(go.Scatter(
        x=scales, y=cuts,
        mode="lines+markers",
        name="⟨C⟩",
        line=dict(color="#4ecdc4", width=2.5),
        marker=dict(size=8),
        hovertemplate="λ=%{x:.2f}<br>⟨C⟩=%{y:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=max_cut, line_dash="dash", line_color="#ff6b6b",
                  annotation_text=f"C*={max_cut:.2f}", annotation_position="right",
                  row=1, col=1)
    fig.add_hline(y=random_c, line_dash="dot", line_color="#888",
                  annotation_text="Random", annotation_position="right",
                  row=1, col=1)

    # Shaded advantage region
    fig.add_trace(go.Scatter(
        x=scales + scales[::-1],
        y=[random_c] * len(scales) + cuts[::-1],
        fill="toself",
        fillcolor="rgba(78,205,196,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=1)

    # --- Bottom panel: ratio ---
    fig.add_trace(go.Scatter(
        x=scales, y=ratios,
        mode="lines+markers",
        name="Approx ratio",
        line=dict(color="#a29bfe", width=2.5),
        marker=dict(symbol="square", size=8),
        hovertemplate="λ=%{x:.2f}<br>Ratio=%{y:.4f}<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color="#ff6b6b",
                  annotation_text="Perfect (1.0)", annotation_position="right",
                  row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="#888",
                  annotation_text="Random (0.5)", annotation_position="right",
                  row=2, col=1)

    fig.update_layout(
        **_DARK_LAYOUT,
        title="QAOA Performance vs Noise Strength",
        xaxis2=dict(title="Noise Scale Factor λ  (1.0 = baseline IBM)"),
        yaxis=dict(title="⟨C⟩"),
        yaxis2=dict(title="Approx ratio"),
        height=560,
        showlegend=True,
        legend=dict(bgcolor=_PAPER_BG, bordercolor="#444", borderwidth=1),
    )

    return fig


# ---------------------------------------------------------------------------
# Shor – phase estimation bar chart
# ---------------------------------------------------------------------------

def plotly_shor_phases(counts: dict, a: int = 2, n_count: int = 4) -> go.Figure:
    """
    Interactive bar chart of Shor phase estimation counts.
    Hover shows bitstring, decimal value, phase, and count.
    """
    total = sum(counts.values())
    items = sorted(counts.items(), key=lambda x: int(x[0], 2))

    bitstrings = [bs for bs, _ in items]
    decimals   = [int(bs, 2) for bs in bitstrings]
    vals       = [c for _, c in items]
    phases     = [d / (2 ** n_count) for d in decimals]

    fig = go.Figure(go.Bar(
        x=bitstrings,
        y=vals,
        marker_color="#6c5ce7",
        customdata=list(zip(decimals, phases, [v / total for v in vals])),
        hovertemplate=(
            "<b>|%{x}⟩</b><br>"
            "Decimal: %{customdata[0]}<br>"
            "Phase φ = %{customdata[1]:.4f}<br>"
            "Count: %{y}<br>"
            "Fraction: %{customdata[2]:.2%}"
            "<extra></extra>"
        ),
    ))

    # Mark expected peaks for current a value
    from math import gcd
    N = 15
    r = 1
    for candidate in range(1, N + 1):
        if pow(a, candidate, N) == 1:
            r = candidate
            break
    expected_peaks = [
        format(int(k * (2 ** n_count) / r), f"0{n_count}b")
        for k in range(r)
        if int(k * (2 ** n_count) / r) < 2 ** n_count
    ]
    for peak_bs in expected_peaks:
        if peak_bs in bitstrings:
            idx = bitstrings.index(peak_bs)
            fig.add_annotation(
                x=peak_bs,
                y=vals[idx],
                text=f"k·2ⁿ/r",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#fdcb6e",
                font=dict(color="#fdcb6e", size=10),
                yshift=8,
            )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=f"Shor Phase Estimation Results  (a={a}, N=15, r={r})",
        xaxis=dict(title="Counting Register State", tickangle=-45),
        yaxis=dict(title="Counts"),
        height=420,
        showlegend=False,
    )

    return fig
