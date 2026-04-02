# Track 4: Water Surface Caustics & 3D Refraction

## 🌊 Project Goal
To simulate high-fidelity "Light Web" patterns (Caustics) on a pool floor by modeling the interaction between a complex 3D wavy surface and incident light rays.

> ### ⚠️ TECHNICAL NOTE: 3D Vector Refraction
> While the standard **Snell's Law** ($n_1 \sin \theta_1 = n_2 \sin \theta_2$) is the fundamental textbook principle for 2D refraction, this simulation operates in a **3D Vector Space**. 
> 
> Instead of using simple scalar angles, we calculate the **Surface Gradient** $(\nabla h)$ to determine the **Local Normal Vector** ($\vec{n}$) at 360,000 individual grid points. This allows the simulation to handle light hitting the water from any direction, producing realistic caustic convergence that a 1D/2D formula cannot achieve.

---

## 🔬 Physics & Engineering Principles

### 1. 3D Surface Generation
The water surface is modeled as a 2D grid ($N=600$) using a superposition of harmonic waves. The height $h$ at any point $(X, Y)$ is defined by the wave equation:
$$h = A \sin(k_x X) + A \sin(k_y Y)$$

### 2. Gradient-Based Normal Calculation
To determine how a ray bends, we first find the "tilt" of the water at every coordinate using partial derivatives. The normal vector $\vec{n}$ is derived from the gradient:
$$\vec{n} = \frac{(-\frac{\partial h}{\partial x}, -\frac{\partial h}{\partial y}, 1)}{\sqrt{(\frac{\partial h}{\partial x})^2 + (\frac{\partial h}{\partial y})^2 + 1}}$$

By utilizing `np.gradient`, the code calculates the exact orientation of the water surface, which dictates the refracted path of the light ray according to the refractive index ratio ($n_1/n_2$).

### 3. Ray Casting & Intensity Mapping (Caustics)
* **Projection:** Each ray is "cast" from the surface to the pool floor at a fixed distance ($z = 1.0$) based on the calculated refraction vector.
* **Histogram Analysis:** We use `np.histogram2d` to map the final positions of all 360,000 rays. 
* **The Caustic Effect:** Areas where the wavy surface acts as a converging lens cause rays to "clump" together. This density increase results in higher intensity values, forming the bright, interlaced caustic lines seen in the final visualization.

---

## 🛠️ Implementation Specs
* **Language:** Python (NumPy, Matplotlib)
* **Optimization:** Fully vectorized calculations for maximum computational performance.
* **Validation:** Includes a built-in "Math Test" log to verify that the vector refraction logic aligns with theoretical 2D Snell's Law values at specific incident angles.

---
**Status:** ✅ Simulation Verified | ✅ Physics Engine Optimized
