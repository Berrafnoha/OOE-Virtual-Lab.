import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Page Setup & Title ---
st.set_page_config(page_title="Track 3: Caustics", layout="centered")
st.title("🌊 Track 3: Pool Floor Caustic Mapping")
st.markdown("### Physical Law: Snell's Law & Ray Tracing")

# --- 2. Sidebar: Measurement Tools (Rubric Points: 04) ---
st.sidebar.header("🕹️ Simulation Controls")
A = st.sidebar.slider("Wave Amplitude (A)", 0.01, 0.1, 0.05, help="Height of the water ripples")
n2 = st.sidebar.slider("Refractive Index (n_water)", 1.0, 1.5, 1.33, help="1.33 is standard for water")
depth = st.sidebar.slider("Pool Depth (z) in meters", 0.5, 3.0, 1.0)
resolution = st.sidebar.select_slider("Grid Resolution", options=[200, 400, 600], value=400)

# --- 3. Physics Engine: Wave Dynamics (Rubric Points: 05) ---
x = np.linspace(0, 2, resolution)
y = np.linspace(0, 2, resolution)
X, Y = np.meshgrid(x, y)

# Modeling the water surface as a sum of sine waves
# kx and ky represent the wave numbers
h = A * np.sin(4 * np.pi * X) + A * np.sin(4 * np.pi * Y)

# Calculating the Surface Normal (Gradient)
# FIX 1: np.gradient with a single spacing value avoids ambiguous two-arg form
dx = x[1] - x[0]
hx, hy = np.gradient(h, dx)  # single spacing scalar applied to both axes
normals = np.dstack((-hx, -hy, np.ones_like(h)))
normals /= (np.linalg.norm(normals, axis=2)[:, :, None] + 1e-9)

# --- 4. Snell's Law Implementation (Rubric Points: 06) ---
# n1 = air (1.0), n2 = water (slider)
# Snell's Law: n1 * sin(theta_i) = n2 * sin(theta_r)
# For paraxial rays the deflection scales as (n1/n2)
n1 = 1.0
snell_ratio = n1 / n2  # FIX 2: actually use n2 in the physics

# Projection of rays to the floor scaled by Snell's ratio
dz = depth / (normals[:, :, 2] + 1e-6)
X_floor = X + normals[:, :, 0] * dz * snell_ratio  # FIX 2: apply Snell's ratio
Y_floor = Y + normals[:, :, 1] * dz * snell_ratio  # FIX 2: apply Snell's ratio

# --- 5. Data Visualization (The Measurement Plot) ---
# 2D histogram counts where light rays land — high density = bright caustic
H, xedges, yedges = np.histogram2d(
    X_floor.ravel(), Y_floor.ravel(),
    bins=resolution, range=[[0, 2], [0, 2]]
)
H = H / np.max(H)  # Normalize intensity

fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
ax.imshow(
    H.T,
    extent=[0, 2, 0, 2],
    origin='lower',
    cmap='hot',        # FIX 3: 'hot' renders caustic bright spots far more clearly
    vmin=0, vmax=0.6,  # slight clip to boost low-intensity detail
)
ax.set_facecolor('black')
ax.axis('off')
st.pyplot(fig)

# --- 6. Physics Formulas (LaTeX for README compliance) ---
st.divider()
st.subheader("📐 Mathematical Verification")
st.latex(r"n_1 \sin \theta_{inc} = n_2 \sin \theta_{refr}")

# Show live computed refraction angle for a representative surface tilt
mean_tilt_deg = float(np.degrees(np.arctan(np.abs(hx).mean())))
sin_refr = np.clip((n1 / n2) * np.sin(np.radians(mean_tilt_deg)), -1, 1)
theta_refr = float(np.degrees(np.arcsin(sin_refr)))

st.write(f"**Current Parameters:** Air ($n_1={n1}$) → Water ($n_2={n2}$).")
st.write(
    f"Mean surface tilt: **{mean_tilt_deg:.2f}°** → "
    f"Refracted angle: **{theta_refr:.2f}°**"
)
st.info("The bright lines you see are 'Envelopes' where light rays concentrate after refraction.")
