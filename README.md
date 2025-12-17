# Virtual Cell–Guided Optimization of NIR–Nanoparticle Therapy for Psoriasis

This repository contains the **virtual cell modeling framework** used in our study to quantitatively integrate **photothermal physics, nanoparticle-mediated drug delivery, inflammatory signaling, and therapeutic efficacy–safety balance** for non-invasive treatment optimization.

The code implements a **mechanism-informed, data-augmented virtual cell** that bridges **NIR photothermal parameters** and **UVB-equivalent biological outcomes**, enabling rational design and closed-loop optimization of therapeutic conditions.
<img width="3670" height="1340" alt="Untitled-31" src="https://github.com/user-attachments/assets/a539bb9b-8456-4e88-8361-bdae24c70e57" />

---

## 📌 Key Features

- **Multi-stage virtual cell simulation**
  - Physical response (photothermal heating, drug release)
  - Cellular uptake and intracellular drug accumulation
  - NF-κB–centered inflammatory pathway modulation
  - Therapeutic efficacy and safety assessment

- **Dual-light (NIR–UVB) equivalence modeling**
  - Converts NIR-induced effects into *biologically interpretable UVB-equivalent doses*

- **Mechanism-informed machine learning**
  - Random Forest and regression models constrained by biological priors
  - Avoids purely black-box optimization

- **Comprehensive parameter optimization**
  - Identifies optimal NIR power, irradiation time, nanoparticle size, drug loading, and targeting efficiency
  - Enforces clinical safety and efficacy thresholds

- **Publication-ready visualizations**
  - Model architecture
  - Training convergence
  - Mechanistic pathway diagrams
  - Therapeutic balance analysis

---
<img width="4808" height="3696" alt="Untitled-3" src="https://github.com/user-attachments/assets/1f90813e-1fa4-44e7-9745-f235d1a7c7e6" />

## 📁 Repository Structure

```text
.
├── psoriasis_virtual_cell.py      # Main virtual cell framework
├── README.md                      # This file
└── (generated outputs)
    ├── *.png                      # Figures generated during simulation
