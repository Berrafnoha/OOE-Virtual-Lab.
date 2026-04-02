# 📡 Integrated Engineering Laboratory: Unified Mission
### **Instructor:** Dr. CHEGGOU | **Group Project 2026**

---

## 🌐 Live Project Dashboard

* **Live Simulation (Streamlit Cloud For Image Constraction Track5 Diverging Lenses):** https://berrafnoha.github.io/OOE-Virtual-Lab/
* **Geometrical optics Track4 (Caustics simulation)** https://ooe-virtual-lab-gafcvg9wmnucxedby6q4ta.streamlit.app/
* **Wave Optics (Track5 Antenna)** https://ooe-virtual-lab-cjaefehwzhztnf8cxaezpj.streamlit.app/

---

## 📖 Project Overview
This repository contains a suite of three interactive engineering simulations. Our mission is to bridge the gap between **Geometrical Optics** and **Electromagnetic Signal Security**, specifically focusing on wave divergence, energy flux via the Poynting Vector, and the formation of optical caustics.

---

## 👥 The Engineering Team
* **BERRAF Noha** | **KHELIL Ikram** | **BAKIRI Soundous**
* **ABDENOUZ Khadidja** | **MOUHEB Maya**

---

## 📂 Laboratory Tracks
# Here are the links of the Tracks and every Track has it's own README.md:

### 🌊 [Track 1: Pool Caustics Simulation](./Track-4-Caustics/)
* **Physics:** Study of light ray envelopes formed by refraction through wavy water surfaces.
* **Implementation:** Python/Matplotlib ray-tracing engine.
* **Formula:** $n_1 \sin \theta_1 = n_2 \sin \theta_2$ (Snell's Law applied to surface gradients).

### 👁️ [Track 2: Diverging Lens Explorer](./Track-5-Peephole/)
* **Physics:** Analysis of virtual image formation and focal lengths ($f' < 0$).
* **Implementation:** Interactive HTML5/JS engine.
* **Formula:** $\frac{1}{v} - \frac{1}{u} = \frac{1}{f'}$ (Thin Lens Equation).

### 📡 [Track 3: Spy Antenna & Poynting Flux](./Track-5-Antenna/)
* **Physics:** EM Security and Signal Leakage Detection.
* **Implementation:** Python Streamlit Dashboard.
* **Formula:** $\vec{S} = \vec{E} \times \vec{H}$ (Poynting Vector for energy flux).

---

## 🛠️ Execution Guide

### 🌐 Web Modules (HTML/JS)
Simply open the **GitHub Pages** link above or run `index.html` locally.

### 🐍 Python Modules (Streamlit)
To run locally, ensure you have the dependencies installed:
```bash
pip install -r requirements.txt
streamlit run app.py
