# 📦 PPI_GaMD_Tools
Python tools for Gaussian accelerated Molecular Dynamics (GaMD) reweighting and free-energy analysis of protein–protein interactions.

PPI_GaMD_Tools is a Python package for processing GaMD simulation data, reconstructing reweighted potential of mean force (PMF) profiles, visualizing free-energy landscapes, and analysing thermodynamic and kinetic properties of protein–protein interactions.

The package provides reusable functions for common analysis steps, with the aim of making PPI-GaMD analysis easier to reproduce, extend, and integrate into Python workflows.

Note the package have been developed based on syntax and output from the AmberMD implementation of PPI-GaMD for log-files and CPPTRAJ for reaction coordinates.

Features

## 🌟 Functionalities

- **Simulation data processing:** Load GaMD log files and reaction-coordinate data from cpptraj output.
Reaction-coordinate construction: Extract distance-, contact-, RMSD-, and Cartesian-coordinate-based reaction coordinates.
- **PMF reweighting:** Calculate biased and cumulant-reweighted one-, two-, and three-dimensional PMFs using the GaMD boost potential.
- **Free-energy visualization:** Plot biased and reweighted PMFs, sampled populations, and identified bound and unbound states.
- **Thermodynamic analysis:** Estimate binding free energies from three-dimensional Cartesian PMFs.
- **Kinetic analysis:** Identify PMF minima and barriers, estimate local PMF curvature, analyse residence times, and solve a one-dimensional Smoluchowski equation.

## ✍️ Authors

I'm Isabella K. Sundenæs, this package and the theory in the README have been created by me in relation with my masters thesis at the Chemistry at the Interface to Biology Group, Department of Chemistry, at the Technical University of Denmark, DTU.


## 🚀 Usage

The following example illustrates the main steps in a one-dimensional GaMD reweighting workflow.

```Python
import matplotlib.pyplot as plt

from ppi_gamd_reweighting import (
    calc_anharmonicity,
    calc_pmf_1D,
    load_gamd_log,
    load_reaction_coord,
    plot_pmf_1D,
)

# Input files
log_data = load_gamd_log("GaMD.log")
coord = load_reaction_coord(
    "dist.dat",
    coord_type="distance",
)

# Calculate the total boost potential
delta_V = log_data["deltaVp"] + log_data["deltaVd"]

# Analyse the boost-potential distribution
temperature = 300.0
gamma = calc_anharmonicity(delta_V, temperature)
print(f"Boost-potential anharmonicity: {gamma:.3f}")

# Calculate biased and reweighted PMFs
F_star, F, bin_centers = calc_pmf_1D(
    coord=coord,
    boost=delta_V,
    cutoff=10,
    bin_size=1.0,
    temperature=temperature,
)

# Visualize the PMFs
fig, ax = plot_pmf_1D(
    F_star=F_star,
    F=F,
    bin_centers=bin_centers,
    coord_type="distance",
)

plt.show()
```
The input files must correspond to the same trajectory frames and use compatible coordinate and boost-potential conventions. The example requires the GaMD log and reaction-coordinate files to be supplied by the user.

This and examples of implementations of the remaining functions in the package can be found in the examples folder along with sample data.


## ⬇️ Installation

Clone the repository and navigate to its root directory:

```bash
git clone https://github.com/isasunde/PPI_GaMD_Tools.git
cd PPI_GaMD_Tools
```

Create and activate a virtual environment:
```bash
python -m venv .venv
```
On Windows:
```bash
.venv\Scripts\Activate.ps1
```

On Linux or macOS:
```bash
source .venv/bin/activate
```
Install the package in editable mode:
```bash
python -m pip install -e .
```

**Dependencies:** The package uses NumPy, pandas, Matplotlib, SciPy, and tqdm.


## ℹ️ Scientific methods

`PPI_GaMD_Tools` combines energetic reweighting, free-energy analysis, and kinetic modelling to characterize protein–protein interactions from Gaussian accelerated Molecular Dynamics (GaMD) simulations.

### Energetic reweighting

GaMD enhances conformational sampling by adding a boost potential, $\Delta U$, to the system's potential energy. The resulting biased probability distribution can be reweighted to estimate the unbiased distribution along a selected reaction coordinate.

This package implements a second-order cumulant expansion of the boost-potential distribution to estimate reweighted potential of mean force (PMF) profiles. Both one-dimensional and multidimensional PMFs can be calculated, enabling analysis of free-energy landscapes along distance-based, structural, and Cartesian reaction coordinates.

The degree of anharmonicity of the boost-potential distribution can also be evaluated to help assess the suitability of the cumulant approximation.

For the mathematical derivations and implementation details, see [`docs/reweighting_theory.md`](docs/reweighting_theory.md).

### Thermodynamic analysis

Three-dimensional Cartesian PMFs can be used to estimate binding free energies by integrating Boltzmann-weighted configurations over bound and unbound regions. The calculation accounts for the configurational volume of the bound state and the standard-state volume.

This approach provides a PMF-based estimate of binding thermodynamics, subject to adequate sampling, consistent PMF normalization, and an appropriate definition of the bound and unbound regions.

For the mathematical derivations and implementation details, see [`docs/energy_theory.md`](docs/energy_theory.md).

### Kinetic analysis

The package includes tools for analysing residence times and estimating local PMF curvature near energy minima and barriers using quadratic fits. These quantities are used in a one-dimensional kinetic framework based on the Smoluchowski equation and Kramers' rate theory.

The Smoluchowski solver propagates a probability distribution along a PMF and estimates a model transition rate from its survival probability. Comparing this rate with a simulation-derived transition rate provides an estimate of the apparent diffusion coefficient along the chosen reaction coordinate, which can then be used to calculate Kramers-based association and dissociation rates.

These kinetic estimates depend on the selected reaction coordinate, PMF, boundary conditions, diffusion calibration, and assumptions of the underlying model.

For the mathematical derivations and implementation details, see [`docs/kinetic_theory.md`](docs/kinetic_theory.md).

### References

1. Miao, Y. & McCammon, J. A. *Gaussian Accelerated Molecular Dynamics: Theory, Implementation, and Applications.* Annual Reports in Computational Chemistry, 13, 231 (2017).
2. Miao, Y. et al. *Improved Reweighting of Accelerated Molecular Dynamics Simulations for Free Energy Calculation.* Journal of Chemical Theory and Computation, 10, 2677–2689 (2014).
3. Doudou, S., Burton, N. A. & Henchman, R. H. *Standard Free Energy of Binding from a One-Dimensional Potential of Mean Force.* Journal of Chemical Theory and Computation, 5, 909–918 (2009).
4. Miao, Y. *Acceleration of Biomolecular Kinetics in Gaussian Accelerated Molecular Dynamics.* Journal of Chemical Physics, 149 (2018).

## 💭 License

This project is distributed under the MIT License . See LICENSE for details.