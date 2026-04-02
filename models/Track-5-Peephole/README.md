
# Track 5: Interactive Diverging Lens Explorer 👁️

## 🔬 Module Overview
This module is a custom-engineered **Optical Ray-Tracing Simulator**. It was designed to demonstrate the diverse applications of diverging lenses ($f' < 0$) in modern technology, moving from simple door security to advanced medical laser systems.

## 🛠️ Key Applications Simulated
The explorer allows users to interactively build ray diagrams for four critical scenarios:

1.  **The Door Peephole:** Modeling how a diverging lens transforms a 180° wide-angle exterior view into a compressed virtual image visible through a small aperture.
2.  **Myopia Correction:** Visualizing the "pre-spreading" of incident light to shift the focal point from the vitreous humor exactly onto the **retina**.
3.  **The Galilean Telescope:** Demonstrating the use of a diverging eyepiece to intercept converging rays, resulting in an upright, magnified image.
4.  **Laser Beam Expander:** Analyzing the expansion of high-intensity narrow beams into wide, safe, collimated paths for surgical and industrial use.

---

## 📐 Physics Engine & Logic
The simulation engine is built on the **Thin Lens Equation** and the **Cartesian Sign Convention**:

$$\frac{1}{v} - \frac{1}{u} = \frac{1}{f'}$$

### Implementation Highlights:
* **Dynamic Ray Tracing:** Instead of static images, the canvas calculates ray vectors in real-time based on the lens position.
* **Virtual Image Extrapolation:** Uses dashed-line rendering to show the geometric origin of virtual images, helping students distinguish between real and virtual light paths.
* **Step-by-Step Pedagogy:** The interface is designed as a "Guided Lab," breaking down complex diagrams into 4 logical steps (Scene Setup, Refraction, Focal Alignment, and Final Image).

---

## 💻 Technical Stack
* **Frontend:** HTML5 / CSS3 (Custom Grid System)
* **Rendering:** JavaScript Canvas API
* **Typography:** Integrated Google Fonts (*DM Serif Display* & *DM Sans*) for a clean, academic aesthetic.
* **Responsiveness:** Fully adaptive UI for both desktop and mobile viewing.

---

## 🎯 Lab Results (Mission 5)
Based on the specific parameters provided in the "Scout Mission":
* **Object Distance ($u$):** $-50$ cm
* **Focal Length ($f'$):** $-10$ cm
* **Calculated Image Position ($v$):** $\approx -8.33$ cm (Virtual, Upright)
* **Magnification ($M$):** $+0.167$

---
**Status:** ✅ UI/UX Optimized | ✅ Physics Logic Verified | ✅ Mobile Responsive
