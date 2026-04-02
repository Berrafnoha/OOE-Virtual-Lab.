
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import io
import base64
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📡 Antenna Leakage Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #0d1117; }
  .block-container { padding: 1.5rem 2rem 2rem; max-width: 1600px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
  }
  [data-testid="stSidebar"] .stSlider > label { color: #a0c4e8 !important; font-size: 13px; }
  [data-testid="stSidebar"] h2 { color: #4fc3f7 !important; font-size: 16px; }

  /* ── Metric cards ── */
  .metric-card {
    background: linear-gradient(135deg, #0f1e2d 0%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    transition: all 0.2s;
  }
  .metric-card:hover { border-color: #2e6da4; transform: translateY(-2px); }
  .metric-card .label { font-size: 11px; color: #5a8ab0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .metric-card .value { font-size: 28px; font-weight: 700; color: #e8f4fd; line-height: 1.2; }
  .metric-card .sub   { font-size: 11px; color: #4a7a9b; margin-top: 4px; }

  /* Risk badges */
  .badge-low    { color: #2ecc71; }
  .badge-medium { color: #f39c12; }
  .badge-high   { color: #e74c3c; }

  /* ── Section headers ── */
  .section-header {
    font-size: 13px;
    font-weight: 600;
    color: #4fc3f7;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem;
  }

  /* ── Alert overrides ── */
  .stAlert { border-radius: 10px; font-size: 14px; }

  /* ── Tables ── */
  .styled-table {
    width: 100%; border-collapse: collapse;
    font-size: 13px; color: #c8dbe8;
    background: #0a1628; border-radius: 10px; overflow: hidden;
  }
  .styled-table th {
    background: #0f2035; color: #4fc3f7;
    padding: 10px 14px; text-align: left;
    font-weight: 600; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.8px;
  }
  .styled-table td { padding: 9px 14px; border-top: 1px solid #1a2f45; }
  .styled-table tr:hover td { background: #0f2035; }

  /* ── Dividers ── */
  hr { border-color: #1e3a5f; margin: 1.5rem 0; }

  /* Plotly chart bg */
  .js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_radiation_pattern(
    theta: np.ndarray,
    main_lobe_sharpness: float,
    side_lobe_strength: float,
    num_side_lobes: int,
    noise_level: float,
    steering_angle: float = 0.0,
    optimized: bool = False,
) -> np.ndarray:
    """
    Compute the antenna gain G(θ) in linear scale.

    Parameters
    ----------
    theta               : angles in radians (0 … 2π)
    main_lobe_sharpness : higher → narrower main beam
    side_lobe_strength  : relative amplitude of side lobes (0–1)
    num_side_lobes      : integer count of side lobes
    noise_level         : std-dev of additive Gaussian noise
    steering_angle      : beam-steering offset in degrees
    optimized           : if True, apply Chebyshev-style taper (−30 dB SLL)
    """
    offset = np.deg2rad(steering_angle)
    # Main lobe: raised-cosine profile
    main = np.power(np.cos(theta - offset), main_lobe_sharpness)
    main = np.clip(main, 0, None)

    if optimized:
        # Chebyshev taper drastically reduces side lobes
        side_lobe_strength *= 0.12

    # Side lobes: symmetrically distributed harmonics
    side = np.zeros_like(theta)
    angular_spacing = 2 * np.pi / max(num_side_lobes * 2, 2)
    for k in range(1, num_side_lobes + 1):
        phase = k * angular_spacing
        amp   = side_lobe_strength * np.exp(-0.3 * k)   # decay with index
        side += amp * np.power(np.abs(np.cos(theta - phase - offset)), 2)
        side += amp * np.power(np.abs(np.cos(theta + phase - offset)), 2)

    # Additive noise (real-world imperfections)
    rng   = np.random.default_rng(42)
    noise = noise_level * np.abs(rng.normal(0, 1, len(theta)))

    gain = main + side + noise
    gain = np.clip(gain, 0, None)
    return gain


def detect_leakage_zones(
    theta: np.ndarray,
    gain: np.ndarray,
    threshold: float,
    min_angular_sep: float,
) -> list[dict]:
    """
    Identify local maxima in G(θ) that exceed `threshold` and
    are not the global maximum (= main lobe).

    Returns a list of dicts with keys: angle_deg, gain, gain_dB, risk.
    """
    main_idx = int(np.argmax(gain))

    leakage_zones = []
    n = len(gain)

    for i in range(n):
        if gain[i] < threshold:
            continue
        # Local maximum check (wrap-around)
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        if gain[i] <= gain[prev_i] or gain[i] <= gain[next_i]:
            continue
        # Skip if this IS the main lobe
        angle_diff = abs(np.rad2deg(theta[i] - theta[main_idx]))
        if min(angle_diff, 360 - angle_diff) < min_angular_sep:
            continue

        g_db = 20 * np.log10(gain[i] / gain[main_idx] + 1e-12)
        # Risk classification based on SLL
        if g_db > -10:
            risk = "🔴 HIGH"
        elif g_db > -20:
            risk = "🟠 MEDIUM"
        else:
            risk = "🟢 LOW"

        leakage_zones.append({
            "angle_deg": np.rad2deg(theta[i]) % 360,
            "gain":      gain[i],
            "gain_dB":   g_db,
            "risk":      risk,
        })

    return sorted(leakage_zones, key=lambda x: x["gain"], reverse=True)


def compute_metrics(
    theta: np.ndarray,
    gain: np.ndarray,
    leakage_zones: list[dict],
    threshold: float,
) -> dict:
    """
    Compute energy metrics from the radiation pattern.
    """
    total_energy   = float(np.trapezoid(gain, theta))
    main_idx       = int(np.argmax(gain))
    main_gain      = float(gain[main_idx])
    main_gain_dB   = 10 * np.log10(main_gain + 1e-12)

    # Main lobe energy: integrate ±HPBW/2 region
    half_power    = main_gain / 2
    in_main       = gain >= half_power
    main_energy   = float(np.trapezoid(gain * in_main, theta))

    leaked_energy = float(sum(z["gain"] for z in leakage_zones))
    efficiency    = (main_energy / (total_energy + 1e-12)) * 100

    # Risk score (0–100)
    if not leakage_zones:
        risk_score = 0
    else:
        max_sll = max(z["gain_dB"] for z in leakage_zones)  # least negative = worst
        count_w = min(len(leakage_zones) / 10, 1.0)
        sll_w   = min((max_sll + 40) / 40, 1.0)             # maps [−40,0] → [0,1]
        risk_score = int((0.6 * sll_w + 0.4 * count_w) * 100)

    if risk_score < 30:
        risk_label, risk_class = "LOW",    "badge-low"
    elif risk_score < 65:
        risk_label, risk_class = "MEDIUM", "badge-medium"
    else:
        risk_label, risk_class = "HIGH",   "badge-high"

    # HPBW approximation
    above_half = np.where(in_main)[0]
    if len(above_half) >= 2:
        hpbw = np.rad2deg(theta[above_half[-1]] - theta[above_half[0]])
    else:
        hpbw = 0.0

    return {
        "main_gain":       main_gain,
        "main_gain_dB":    main_gain_dB,
        "total_energy":    total_energy,
        "main_energy":     main_energy,
        "leaked_energy":   leaked_energy,
        "efficiency":      efficiency,
        "leakage_count":   len(leakage_zones),
        "risk_score":      risk_score,
        "risk_label":      risk_label,
        "risk_class":      risk_class,
        "hpbw":            hpbw,
        "main_angle_deg":  np.rad2deg(theta[main_idx]) % 360,
    }


def spy_antenna_analysis(
    theta: np.ndarray,
    gain: np.ndarray,
    spy_angle_deg: float,
) -> dict:
    """
    Evaluate the signal strength captured by a hypothetical spy antenna
    placed at spy_angle_deg.
    """
    spy_rad    = np.deg2rad(spy_angle_deg % 360)
    idx        = int(np.argmin(np.abs(theta - spy_rad)))
    spy_gain   = float(gain[idx])
    main_gain  = float(gain[np.argmax(gain)])
    sll_dB     = 20 * np.log10(spy_gain / main_gain + 1e-12)
    intercepted = spy_gain > (main_gain * 0.05)   # >−26 dB threshold

    return {
        "spy_angle":    spy_angle_deg,
        "spy_gain":     spy_gain,
        "sll_dB":       sll_dB,
        "intercepted":  intercepted,
        "pct_main":     (spy_gain / main_gain) * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

DARK = "#0d1117"
PANEL = "#0f1e2d"
GRID  = "#1a3050"

def polar_plot(
    theta: np.ndarray,
    gain: np.ndarray,
    gain_opt: np.ndarray | None,
    leakage_zones: list[dict],
    metrics: dict,
    spy: dict | None,
    show_comparison: bool,
    threshold: float,
) -> go.Figure:
    """Build the main polar radiation pattern figure."""
    theta_deg = np.rad2deg(theta)
    gain_norm = gain / (gain.max() + 1e-12)
    gain_dB   = 20 * np.log10(gain_norm + 1e-12)

    fig = go.Figure()

    # ── Threshold ring ──
    thresh_dB = 20 * np.log10(threshold / (gain.max() + 1e-12) + 1e-12)
    fig.add_trace(go.Scatterpolar(
        r=np.full_like(theta_deg, thresh_dB),
        theta=theta_deg,
        mode="lines",
        line=dict(color="#f39c12", width=1.2, dash="dot"),
        name=f"Threshold ({thresh_dB:.1f} dB)",
        hoverinfo="skip",
    ))

    # ── Optimized pattern (comparison mode) ──
    if show_comparison and gain_opt is not None:
        gain_opt_norm = gain_opt / (gain_opt.max() + 1e-12)
        gain_opt_dB   = 20 * np.log10(gain_opt_norm + 1e-12)
        fig.add_trace(go.Scatterpolar(
            r=gain_opt_dB,
            theta=theta_deg,
            mode="lines",
            fill="toself",
            fillcolor="rgba(39,174,96,0.12)",
            line=dict(color="#2ecc71", width=1.5),
            name="✅ Optimized",
        ))

    # ── Main pattern ──
    fig.add_trace(go.Scatterpolar(
        r=gain_dB,
        theta=theta_deg,
        mode="lines",
        fill="toself",
        fillcolor="rgba(79,195,247,0.10)",
        line=dict(color="#4fc3f7", width=2),
        name="📡 Gain G(θ)",
        hovertemplate="θ = %{theta:.1f}°<br>Gain = %{r:.2f} dB<extra></extra>",
    ))

    # ── Main lobe marker ──
    main_a = metrics["main_angle_deg"]
    fig.add_trace(go.Scatterpolar(
        r=[gain_dB[int(np.argmax(gain))]],
        theta=[main_a],
        mode="markers",
        marker=dict(color="#f1c40f", size=14, symbol="star", line=dict(color="#fff", width=1)),
        name=f"⭐ Main Lobe ({main_a:.1f}°)",
    ))

    # ── Leakage zone markers ──
    if leakage_zones:
        leak_r, leak_t, leak_txt = [], [], []
        for z in leakage_zones:
            idx = int(np.argmin(np.abs(theta_deg - z["angle_deg"])))
            leak_r.append(gain_dB[idx])
            leak_t.append(z["angle_deg"])
            leak_txt.append(f"θ={z['angle_deg']:.1f}°  {z['gain_dB']:.1f} dB  {z['risk']}")
        fig.add_trace(go.Scatterpolar(
            r=leak_r, theta=leak_t,
            mode="markers",
            marker=dict(color="#e74c3c", size=10, symbol="circle",
                        line=dict(color="#fff", width=1)),
            name="⚠️ Leakage Zones",
            hovertext=leak_txt,
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    # ── Spy antenna ──
    if spy:
        idx_s = int(np.argmin(np.abs(theta_deg - spy["spy_angle"])))
        color_s = "#e74c3c" if spy["intercepted"] else "#2ecc71"
        sym_s   = "triangle-up" if spy["intercepted"] else "triangle-down"
        label_s = "🕵️ SPY (INTERCEPTING!)" if spy["intercepted"] else "🕵️ SPY (safe)"
        fig.add_trace(go.Scatterpolar(
            r=[gain_dB[idx_s]],
            theta=[spy["spy_angle"]],
            mode="markers",
            marker=dict(color=color_s, size=16, symbol=sym_s,
                        line=dict(color="#fff", width=1.5)),
            name=label_s,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=PANEL,
            angularaxis=dict(
                tickfont=dict(size=11, color="#5a8ab0"),
                linecolor=GRID, gridcolor=GRID,
                rotation=90, direction="clockwise",
            ),
            radialaxis=dict(
                range=[-45, 3],
                tickfont=dict(size=10, color="#5a8ab0"),
                linecolor=GRID, gridcolor=GRID,
                ticksuffix=" dB",
            ),
        ),
        paper_bgcolor=DARK,
        plot_bgcolor=DARK,
        legend=dict(
            bgcolor="#0a1628", bordercolor="#1e3a5f", borderwidth=1,
            font=dict(color="#a0c4e8", size=12),
            orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5,
        ),
        margin=dict(t=20, b=60, l=20, r=20),
        height=540,
    )
    return fig


def energy_bar_chart(
    theta: np.ndarray,
    gain: np.ndarray,
    n_bins: int = 36,
) -> go.Figure:
    """Energy distribution across angular sectors (bar chart)."""
    sector_size = 360 / n_bins
    edges  = np.linspace(0, 2 * np.pi, n_bins + 1)
    labels = [f"{int(i * sector_size)}°" for i in range(n_bins)]
    energies = [
        float(np.trapezoid(
            gain[(theta >= edges[i]) & (theta < edges[i + 1])],
            theta[(theta >= edges[i]) & (theta < edges[i + 1])],
        ))
        for i in range(n_bins)
    ]

    norm_e = np.array(energies) / (max(energies) + 1e-12)
    colors = [
        f"rgba({int(255*(1-v))},{int(100+100*v)},{int(255*v)},0.85)"
        for v in norm_e
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=energies,
        marker_color=colors,
        hovertemplate="Sector: %{x}<br>Energy: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=PANEL,
        xaxis=dict(title="Angular Sector", tickfont=dict(color="#5a8ab0"), gridcolor=GRID,
                   tickangle=45, tickmode="array",
                   tickvals=labels[::4], ticktext=labels[::4]),
        yaxis=dict(title="Energy (a.u.)", tickfont=dict(color="#5a8ab0"), gridcolor=GRID),
        margin=dict(t=10, b=60, l=60, r=10),
        height=280,
        showlegend=False,
    )
    return fig


def poynting_heatmap(
    theta: np.ndarray,
    gain: np.ndarray,
) -> go.Figure:
    """2-D Poynting-vector magnitude map in Cartesian (x–y) plane."""
    n     = 200
    r_max = 1.2
    xs    = np.linspace(-r_max, r_max, n)
    ys    = np.linspace(-r_max, r_max, n)
    XX, YY = np.meshgrid(xs, ys)

    R    = np.sqrt(XX**2 + YY**2) + 1e-9
    ANGL = (np.arctan2(YY, XX)) % (2 * np.pi)

    GAIN_MAP = np.interp(ANGL.ravel(), theta, gain).reshape(n, n)
    S_MAP    = GAIN_MAP / (R + 0.1)          # |S| ∝ G(θ)/r²  (near-field approximation)

    fig = go.Figure(go.Heatmap(
        z=S_MAP,
        x=xs, y=ys,
        colorscale=[
            [0,   "#0d1117"],
            [0.2, "#0a2a4a"],
            [0.5, "#1565c0"],
            [0.75,"#ef9a27"],
            [1,   "#e74c3c"],
        ],
        colorbar=dict(
            tickfont=dict(color="#a0c4e8"),
            title=dict(text="|S| (a.u.)", font=dict(color="#a0c4e8")),
        ),
        hovertemplate="x=%{x:.2f}  y=%{y:.2f}<br>|S|=%{z:.3f}<extra></extra>",
    ))
    # Antenna symbol
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(symbol="star", size=16, color="#f1c40f"),
        text=["📡"], textposition="top center",
        showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        xaxis=dict(title="x (λ)", showgrid=False, zeroline=False,
                   tickfont=dict(color="#5a8ab0")),
        yaxis=dict(title="y (λ)", showgrid=False, zeroline=False,
                   tickfont=dict(color="#5a8ab0"), scaleanchor="x"),
        margin=dict(t=10, b=50, l=60, r=10),
        height=400,
    )
    return fig


def time_simulation_chart(
    theta: np.ndarray,
    base_gain: np.ndarray,
    n_frames: int = 40,
) -> go.Figure:
    """Animated polar plot showing time-varying gain (e.g., vibration / scanning)."""
    frames = []
    for t_idx in range(n_frames):
        phase   = t_idx / n_frames * 2 * np.pi
        jitter  = 0.05 * np.sin(3 * theta + phase)
        g_frame = np.clip(base_gain + jitter, 0, None)
        g_norm  = g_frame / (g_frame.max() + 1e-12)
        g_dB    = 20 * np.log10(g_norm + 1e-12)
        frames.append(go.Frame(data=[
            go.Scatterpolar(r=g_dB, theta=np.rad2deg(theta),
                            mode="lines", fill="toself",
                            fillcolor="rgba(79,195,247,0.12)",
                            line=dict(color="#4fc3f7", width=2))
        ], name=str(t_idx)))

    g_norm_init = base_gain / (base_gain.max() + 1e-12)
    g_dB_init   = 20 * np.log10(g_norm_init + 1e-12)

    fig = go.Figure(
        data=[go.Scatterpolar(r=g_dB_init, theta=np.rad2deg(theta),
                              mode="lines", fill="toself",
                              fillcolor="rgba(79,195,247,0.12)",
                              line=dict(color="#4fc3f7", width=2))],
        frames=frames,
    )
    fig.update_layout(
        polar=dict(
            bgcolor=PANEL,
            angularaxis=dict(tickfont=dict(size=11, color="#5a8ab0"),
                             linecolor=GRID, gridcolor=GRID,
                             rotation=90, direction="clockwise"),
            radialaxis=dict(range=[-45, 3], tickfont=dict(size=10, color="#5a8ab0"),
                            linecolor=GRID, gridcolor=GRID, ticksuffix=" dB"),
        ),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=-0.08, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True)]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
            bgcolor="#0f2035", bordercolor="#1e3a5f",
            font=dict(color="#a0c4e8"),
        )],
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        margin=dict(t=20, b=80, l=20, r=20),
        height=460,
    )
    return fig


def sll_gauge(risk_score: int, risk_label: str) -> go.Figure:
    """Gauge chart for the overall security risk score."""
    color = "#2ecc71" if risk_label == "LOW" else ("#f39c12" if risk_label == "MEDIUM" else "#e74c3c")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        number=dict(font=dict(color=color, size=42)),
        delta=dict(reference=50, font=dict(size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#5a8ab0")),
            bar=dict(color=color, thickness=0.25),
            bgcolor=PANEL,
            borderwidth=0,
            steps=[
                dict(range=[0, 30],  color="#0d2e1a"),
                dict(range=[30, 65], color="#2e1e06"),
                dict(range=[65, 100],color="#2e0a0a"),
            ],
            threshold=dict(line=dict(color="#fff", width=3), value=risk_score),
        ),
        title=dict(text=f"Security Risk Score<br><b>{risk_label}</b>",
                   font=dict(color="#a0c4e8", size=14)),
    ))
    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        margin=dict(t=30, b=20, l=20, r=20),
        height=250,
        font=dict(color="#a0c4e8"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def export_data(metrics: dict, leakage_zones: list[dict], params: dict) -> str:
    """Serialize simulation results to a JSON string."""
    payload = {
        "timestamp":    datetime.now().isoformat(),
        "parameters":   params,
        "metrics":      {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                         for k, v in metrics.items()},
        "leakage_zones": [
            {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in z.items()}
            for z in leakage_zones
        ],
    }
    return json.dumps(payload, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def build_sidebar() -> dict:
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 0.5rem 0 1.2rem;'>
          <div style='font-size:42px'>📡</div>
          <div style='font-size:16px; font-weight:700; color:#4fc3f7;
                      letter-spacing:0.5px;'>Antenna Leakage</div>
          <div style='font-size:11px; color:#4a7a9b; margin-top:2px;
                      text-transform:uppercase;letter-spacing:1px;'>
                      Detection System</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 🎛 Antenna Parameters")

        p = {}

        p["main_lobe_sharpness"] = st.slider(
            "Main Lobe Sharpness", 1, 30, 8,
            help="Higher → narrower, more focused beam")
        p["side_lobe_strength"] = st.slider(
            "Side Lobe Strength", 0.0, 1.0, 0.35, 0.01,
            help="Relative amplitude of side lobes")
        p["num_side_lobes"] = st.slider(
            "Number of Side Lobes", 1, 10, 4,
            help="How many side lobe peaks to simulate")
        p["noise_level"] = st.slider(
            "Noise / Randomness", 0.0, 0.4, 0.05, 0.01,
            help="Gaussian noise (real-world imperfections)")
        p["steering_angle"] = st.slider(
            "Beam Steering θ₀ (°)", -90, 90, 0,
            help="Electronic steering angle of the main beam")

        st.markdown("---")
        st.markdown("## 🔍 Detection Settings")

        p["threshold_pct"] = st.slider(
            "Leakage Threshold (%)", 1, 50, 15,
            help="% of main lobe peak to flag as leakage")
        p["min_angular_sep"] = st.slider(
            "Min Angular Sep. (°)", 5, 60, 20,
            help="Minimum angular distance from main lobe")

        st.markdown("---")
        st.markdown("## 🕵️ Spy Antenna")

        p["enable_spy"] = st.toggle("Enable Spy Antenna", True)
        p["spy_angle"]  = st.slider(
            "Spy Angle (°)", 0, 359, 45,
            help="Direction where the spy antenna is placed")

        st.markdown("---")
        st.markdown("## ⚙️ Modes")

        p["show_comparison"] = st.toggle("Comparison Mode (Before/After)", False)
        p["show_time_sim"]   = st.toggle("Time Animation Mode", False)
        p["show_poynting"]   = st.toggle("Poynting Vector Map", True)
        p["show_energy_bar"] = st.toggle("Energy Distribution Bar", True)
        p["show_time_simulation"] = st.toggle("Animated Time Simulation", False)

        st.markdown("---")
        p["n_points"] = st.select_slider(
            "Angular Resolution", options=[360, 720, 1440, 2880], value=720)

        st.markdown("""
        <div style='margin-top:2rem; padding:.75rem;
                    background:#0a1628; border-radius:8px;
                    border:1px solid #1e3a5f; font-size:11px; color:#4a7a9b;'>
          📐 Based on Poynting Vector S⃗ = E⃗ × H⃗<br>
          📊 G(θ) = cos^n(θ) + Σ side lobes<br>
          🔐 SLL threshold for risk classification
        </div>
        """, unsafe_allow_html=True)

    return p


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    params = build_sidebar()

    # ── Page title ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:.5rem 0 1rem;'>
      <h1 style='font-size:28px; font-weight:700; color:#e8f4fd; margin:0;'>
        📡 Antenna Signal Leakage Detection System
      </h1>
      <p style='color:#5a8ab0; font-size:13px; margin:.3rem 0 0;'>
        Real-time simulation of electromagnetic energy distribution &amp;
        security risk analysis via the Poynting Vector
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Run simulation ───────────────────────────────────────────────────────
    theta    = np.linspace(0, 2 * np.pi, params["n_points"], endpoint=False)
    gain     = generate_radiation_pattern(
        theta,
        main_lobe_sharpness = params["main_lobe_sharpness"],
        side_lobe_strength  = params["side_lobe_strength"],
        num_side_lobes      = params["num_side_lobes"],
        noise_level         = params["noise_level"],
        steering_angle      = params["steering_angle"],
        optimized           = False,
    )
    gain_opt = generate_radiation_pattern(
        theta,
        main_lobe_sharpness = params["main_lobe_sharpness"],
        side_lobe_strength  = params["side_lobe_strength"],
        num_side_lobes      = params["num_side_lobes"],
        noise_level         = params["noise_level"],
        steering_angle      = params["steering_angle"],
        optimized           = True,
    ) if params["show_comparison"] else None

    threshold     = params["threshold_pct"] / 100 * gain.max()
    leakage_zones = detect_leakage_zones(
        theta, gain, threshold, params["min_angular_sep"])
    metrics       = compute_metrics(theta, gain, leakage_zones, threshold)
    spy           = spy_antenna_analysis(theta, gain, params["spy_angle"]) \
                    if params["enable_spy"] else None

    # ── TOP ALERTS ───────────────────────────────────────────────────────────
    if metrics["risk_label"] == "HIGH":
        st.error(f"🚨 **HIGH SECURITY RISK** — {metrics['leakage_count']} leakage zone(s) detected. "
                 f"Efficiency: {metrics['efficiency']:.1f}%. Immediate hardening recommended.")
    elif metrics["risk_label"] == "MEDIUM":
        st.warning(f"⚠️ **MEDIUM RISK** — {metrics['leakage_count']} leakage zone(s) detected. "
                   f"Efficiency: {metrics['efficiency']:.1f}%. Consider applying side lobe reduction.")
    else:
        st.success(f"✅ **LOW RISK** — Antenna pattern is well-controlled. "
                   f"Efficiency: {metrics['efficiency']:.1f}%.")

    if spy and spy["intercepted"]:
        st.error(f"🕵️ **SPY ANTENNA ALERT** — Interception at {spy['spy_angle']:.0f}°! "
                 f"Captured {spy['pct_main']:.1f}% of main lobe power ({spy['sll_dB']:.1f} dB).")
    elif spy:
        st.info(f"🕵️ Spy antenna at {spy['spy_angle']:.0f}° — signal too weak to intercept "
                f"({spy['sll_dB']:.1f} dB, {spy['pct_main']:.2f}% of main lobe).")

    # ── METRIC CARDS ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    risk_emoji = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}[metrics["risk_label"]]

    cards = [
        (c1, "🛡 Risk Score",      f"{metrics['risk_score']}", "/100",           metrics["risk_class"]),
        (c2, "⚡ Efficiency",      f"{metrics['efficiency']:.1f}", "%",           ""),
        (c3, "⚠️ Leakage Zones",  f"{metrics['leakage_count']}",  "detected",    ""),
        (c4, "📐 HPBW",           f"{metrics['hpbw']:.1f}", "°",                ""),
        (c5, "🎯 Main Gain",       f"{metrics['main_gain_dB']:.1f}", "dBi",      ""),
        (c6, "🔄 Main Angle",      f"{metrics['main_angle_deg']:.1f}", "°",      ""),
    ]
    for col, label, val, sub, cls in cards:
        with col:
            badge_style = f"class='{cls}'" if cls else ""
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value {cls}">{val}</div>
              <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 1: Polar + Gauge ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📡 Radiation Pattern Analysis</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 1])
    with col_left:
        fig_polar = polar_plot(
            theta, gain, gain_opt, leakage_zones, metrics, spy,
            params["show_comparison"], threshold,
        )
        st.plotly_chart(fig_polar, use_container_width=True)

    with col_right:
        st.plotly_chart(sll_gauge(metrics["risk_score"], metrics["risk_label"]),
                        use_container_width=True)

        # Quick stats
        st.markdown("""<div style='background:#0a1628;border:1px solid #1e3a5f;
                        border-radius:10px;padding:14px 16px;font-size:13px;'>""",
                    unsafe_allow_html=True)
        stats = [
            ("Total Energy",  f"{metrics['total_energy']:.3f}"),
            ("Main Energy",   f"{metrics['main_energy']:.3f}"),
            ("Leaked Energy", f"{metrics['leaked_energy']:.4f}"),
            ("Steering θ₀",   f"{params['steering_angle']}°"),
            ("Elements N",    f"{params['num_side_lobes']*2+1}"),
            ("Freq Band",     "2.4 GHz"),
        ]
        for k, v in stats:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid #1a3050;'>"
                f"<span style='color:#5a8ab0;'>{k}</span>"
                f"<span style='color:#c8dbe8;font-weight:500;'>{v}</span></div>",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ROW 2: Poynting Heatmap + Energy Bar ─────────────────────────────────
    show_both = params["show_poynting"] and params["show_energy_bar"]
    col_a, col_b = st.columns(2) if show_both else (st.columns(1)[0], None), st.columns(1)[0]

    # Reset — simpler approach
    if params["show_poynting"] or params["show_energy_bar"]:
        st.markdown('<div class="section-header">🔥 Field Intensity & Energy Distribution</div>',
                    unsafe_allow_html=True)
        c_left2, c_right2 = st.columns(2)
        if params["show_poynting"]:
            with c_left2:
                st.markdown("**⚡ Poynting Vector Magnitude |S⃗| Map**")
                st.plotly_chart(poynting_heatmap(theta, gain), use_container_width=True)
        if params["show_energy_bar"]:
            with c_right2:
                st.markdown("**📊 Angular Energy Distribution**")
                st.plotly_chart(energy_bar_chart(theta, gain), use_container_width=True)

    # ── ROW 3: Time Simulation ────────────────────────────────────────────────
    if params["show_time_simulation"]:
        st.markdown('<div class="section-header">⏱ Time-Domain Simulation (Animated)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(time_simulation_chart(theta, gain), use_container_width=True)

    # ── ROW 4: Leakage Table ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔐 Leakage Zone Report</div>',
                unsafe_allow_html=True)

    if leakage_zones:
        table_rows = ""
        for i, z in enumerate(leakage_zones, 1):
            row_bg = "#150505" if "HIGH" in z["risk"] else (
                     "#15100a" if "MEDIUM" in z["risk"] else "#050f08")
            table_rows += f"""
            <tr style='background:{row_bg};'>
              <td>{i}</td>
              <td>{z['angle_deg']:.1f}°</td>
              <td>{z['gain']:.4f}</td>
              <td>{z['gain_dB']:.2f} dB</td>
              <td>{z['risk']}</td>
            </tr>"""

        st.markdown(f"""
        <table class="styled-table">
          <thead>
            <tr>
              <th>#</th><th>Direction θ</th><th>Gain (linear)</th>
              <th>SLL (dB)</th><th>Risk Level</th>
            </tr>
          </thead>
          <tbody>{table_rows}</tbody>
        </table>""", unsafe_allow_html=True)
    else:
        st.info("✅ No leakage zones detected above the current threshold.")

    # ── ROW 5: Comparison + Spy Details ──────────────────────────────────────
    if params["show_comparison"] or (spy and spy["intercepted"]):
        st.markdown('<div class="section-header">🔬 Deep Analysis</div>',
                    unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)

        if params["show_comparison"]:
            with col_c1:
                lz_opt = detect_leakage_zones(theta, gain_opt, threshold, params["min_angular_sep"])
                m_opt  = compute_metrics(theta, gain_opt, lz_opt, threshold)
                st.markdown("**📊 Before vs After Optimization**")
                df_cmp = pd.DataFrame({
                    "Metric":        ["Leakage Zones", "Efficiency (%)", "Risk Score",
                                      "Max SLL (dB)", "HPBW (°)"],
                    "Before":        [metrics["leakage_count"], f"{metrics['efficiency']:.1f}",
                                      metrics["risk_score"], f"{min(z['gain_dB'] for z in leakage_zones) if leakage_zones else 0:.1f}",
                                      f"{metrics['hpbw']:.1f}"],
                    "After (opt.)":  [m_opt["leakage_count"], f"{m_opt['efficiency']:.1f}",
                                      m_opt["risk_score"],
                                      f"{min(z['gain_dB'] for z in lz_opt) if lz_opt else 0:.1f}",
                                      f"{m_opt['hpbw']:.1f}"],
                })
                st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        if spy:
            with col_c2:
                st.markdown("**🕵️ Spy Antenna Analysis**")
                intercept_status = "⛔ INTERCEPTING" if spy["intercepted"] else "✅ Safe (below threshold)"
                col_s1, col_s2 = st.columns(2)
                col_s1.metric("Status",          intercept_status)
                col_s2.metric("Signal Strength", f"{spy['sll_dB']:.1f} dB")
                col_s1.metric("Spy Angle",       f"{spy['spy_angle']:.0f}°")
                col_s2.metric("% of Main Lobe",  f"{spy['pct_main']:.2f}%")

                if spy["intercepted"]:
                    st.error("⚠️ The spy antenna can intercept the signal. "
                             "Consider null-steering or increasing sharpness.")
                else:
                    st.success("Signal at spy position is below interception threshold.")

    # ── HARDENING RECOMMENDATIONS ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🛡 Security Hardening Recommendations</div>',
                unsafe_allow_html=True)

    recs = [
        ("📐 Chebyshev/Taylor Tapering",
         "Apply amplitude tapering weights across array elements to reduce first SLL from −13 dB to −30 dB."),
        ("🎯 Null Steering",
         "Inject a controlled null in the direction of detected leakage or known eavesdropper positions."),
        ("📻 Frequency Hopping (FHSS)",
         "Randomize transmission frequency so side lobe angular positions constantly shift."),
        ("⬇️ Transmit Power Minimization",
         "Reduce power to minimum required for the link budget; side lobe intercept range scales as √P."),
        ("🔊 Artificial Noise Injection",
         "Transmit structured noise in the null-space of the legitimate channel (MIMO systems)."),
        ("🔄 Beam Randomization",
         "Randomly dither θ₀ within ±5° to prevent coherent integration by passive eavesdroppers."),
    ]

    cols_r = st.columns(3)
    for i, (title, desc) in enumerate(recs):
        with cols_r[i % 3]:
            st.markdown(f"""
            <div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;
                        padding:14px 16px;margin-bottom:12px;min-height:110px;'>
              <div style='font-weight:600;color:#4fc3f7;font-size:13px;margin-bottom:6px;'>{title}</div>
              <div style='color:#8ab4cc;font-size:12px;line-height:1.6;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── EXPORT ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💾 Export Results</div>',
                unsafe_allow_html=True)
    col_e1, col_e2, col_e3 = st.columns(3)

    export_params = {k: int(v) if isinstance(v, (np.integer,)) else
                       float(v) if isinstance(v, (np.floating,)) else v
                     for k, v in params.items()}
    json_str = export_data(metrics, leakage_zones, export_params)

    with col_e1:
        st.download_button(
            label="📥 Download JSON Report",
            data=json_str,
            file_name=f"antenna_leakage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_e2:
        df_exp = pd.DataFrame(leakage_zones) if leakage_zones else pd.DataFrame(
            columns=["angle_deg", "gain", "gain_dB", "risk"])
        csv_str = df_exp.to_csv(index=False)
        st.download_button(
            label="📥 Download Leakage CSV",
            data=csv_str,
            file_name=f"leakage_zones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_e3:
        gain_norm_export = gain / gain.max()
        df_pat = pd.DataFrame({
            "angle_deg": np.rad2deg(theta),
            "gain_linear": gain,
            "gain_normalized": gain_norm_export,
            "gain_dB": 20 * np.log10(gain_norm_export + 1e-12),
        })
        st.download_button(
            label="📥 Download Pattern CSV",
            data=df_pat.to_csv(index=False),
            file_name=f"radiation_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <hr>
    <div style='text-align:center;color:#2a4a6a;font-size:12px;padding:.5rem 0 1rem;'>
      📡 Antenna Signal Leakage Detection System &nbsp;|&nbsp;
      Poynting Vector: S⃗ = E⃗ × H⃗ &nbsp;|&nbsp;
      G(θ) = cosⁿ(θ) + Σ side lobes &nbsp;|&nbsp;
      Built with Streamlit &amp; Plotly
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


