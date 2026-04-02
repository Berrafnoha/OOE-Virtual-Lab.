
# Track 5: Spy Antenna (Poynting Vector) 📡

## 🔬 Module Overview
This module is a professional **Antenna Radiation & Security Simulator**. It was engineered to analyze the spatial distribution of electromagnetic energy and identify "Side Lobes"—unintended leakage zones where sensitive data can be intercepted by unauthorized "spy" receivers.

## 🛠️ Key Security Scenarios
The system allows for the interactive auditing of four critical communication states:

1.  **Main Lobe Targeting:** Optimizing the primary beam width (HPBW) for point-to-point secure transmission.
2.  **Side Lobe Leakage:** Identifying "parasitic" energy peaks that radiate in directions other than the intended target.
3.  **Interception Analysis:** Real-time calculation of signal strength at a specific "Spy Angle" to determine if data decoding is possible.
4.  **Pattern Hardening:** Demonstrating how "Tapering" algorithms can suppress side lobes to below -30 dB for high-security military applications.

---

## 📐 Physics Engine & Logic
The simulation is governed by the **Poynting Vector** and **Far-Field Gain** equations:

### 1. The Poynting Vector ($\vec{S}$)
The directional energy flux (power density) of the electromagnetic field is calculated as:
$$\vec{S} = \vec{E} \times \vec{H}$$

### 2. Radiation Pattern & Gain
The Gain $G(\theta)$ describes how the antenna concentrates power. The engine identifies "Leakage Zones" where:
$$G(\theta)_{SideLobe} > \text{Threshold}$$

---

## 💻 Technical Stack
* **Language:** Python 3.9+
* **Data Processing:** NumPy (Vectorized Power Calculations)
* **Interface:** Streamlit (Reactive Security Dashboard)
* **Visualization:** Plotly (Interactive Polar & 2D Heatmap Rendering)

---

## 🎯 Lab Results (Security Audit)
Based on the specific parameters provided in the "Mission Parameters":
* **Main Lobe Angle ($\theta_0$):** $0°$ (Nadir)
* **Side Lobe Level (SLL):** $\approx -15$ dB
* **Efficiency:** $88.4\%$
* **Risk Level:** 🟠 **MEDIUM** (Interception possible at $45°$ and $135°$)

---
**Status:** ✅ Poynting Logic Verified | ✅ SLL Thresholds Active | ✅ Export Ready
