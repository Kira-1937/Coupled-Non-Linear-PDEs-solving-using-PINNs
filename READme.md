# 🔬 Physics-Informed Neural Networks (PINNs) for Solving Burger’s Equation

This repository presents a collection of experiments and implementations using **Physics-Informed Neural Networks (PINNs)** to solve both **1D and coupled Burgers’ equations**, which are fundamental in modeling nonlinear transport and convection-diffusion phenomena.

---

## 📁 Project Structure

├── Articles/
│ ├── Research papers and reference materials
│
├── PINN_solver_couple_burger_equation/
│ └── PINN_solver_for_coupled_Burger_equation.ipynb
│
├── PINN_solver_for_1D_Burger_eqn/
│ └── PINN_Solver_Burger_eq.ipynb
│
├── results/
│ ├── A Numerical Algorithm Based on PINN for
| |    Simulating the Burger’s Equation.pdf
│ ├── PINN_Solver_Burger_eq_example_{1-11}.ipynb
│ └── PINN_Solver_Burger_eq.ipynb




---

## 🧠 PINN for Coupled Burgers’ Equations

**Notebook:** `PINN_solver_for_coupled_Burger_equation.ipynb`

This implementation extends the classical PINN formulation to **time-dependent coupled nonlinear PDEs**, simulating convection-diffusion interactions across multiple fields.

### 🔍 Key Features:
- Solves systems of the form:
  \[
  \begin{aligned}
  u_t + uu_x &= \epsilon u_{xx} \\
  v_t + uv_x &= \epsilon v_{xx}
  \end{aligned}
  \]
- Integrates **Fourier Feature Mapping** to improve learning of high-frequency components.
- Benchmarked against analytical solutions for various **Reynolds numbers (Re = 1/ε)**.
- Shows strong agreement with ground truth over varying viscosity regimes.

---

## 🌊 PINN for 1D Burgers’ Equation

**Notebook:** `PINN_Solver_Burger_eq.ipynb`

This classic example demonstrates how PINNs can accurately solve the **viscous 1D Burgers’ equation**, a canonical problem in nonlinear PDE modeling.

### 🔍 Key Features:
- Focused on:
  \[
  u_t + uu_x = \nu u_{xx}
  \]
- Multiple curated examples available in the `results/` directory.
- Covers:
  - Boundary/initial condition variations
  - Viscosity-dependent behaviors
  - Comparison with analytical solutions
  - Visualization of predicted vs actual solutions and loss trends

---

## 📚 Reference Materials

The `Articles/` folder contains:
- Seminar and research papers on PINNs
- Network architecture illustrations
- Notes on solving nonlinear and high-dimensional PDEs via deep learning
- Summaries of PINN-based approaches for complex PDEs

---

## 🛠️ Installation & Dependencies

Ensure the following Python libraries are installed:

```bash
pip install torch numpy scipy matplotlib notebook
