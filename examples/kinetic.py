"""Example one-dimensional PPI-GaMD reweighting workflow."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ppi_gamd_reweighting import (
    calc_anharmonicity,
    calc_pmf_1D,
    load_gamd_log,
    load_reaction_coord,
    plot_pmf_1D,
    find_wells,
    find_curvature,
    find_residence_times,
    plot_minima_diagnostics,
    solve_smoluchowski,
)

BOLTZMANN_KCAL = 0.0019872041  # kcal mol^-1 K^-1
NA = 6.02214076e23

# Input files
example_dir = Path(__file__).parent / "data"

log_file = example_dir / "kinetic_GaMD.log"
coord_file = example_dir / "kinetic_dist.dat"

# Analysis parameters
coord_type = "min_distance"
bin_size = 1.0
cutoff = 10
temperature = 300.0
# Simulation box volume
Vol_A3 = 137.607**3 
Vol_L = Vol_A3 * 1e-27
conc = 1.0 / (NA * Vol_L)

# Kinetic parameters
rb = 10
ru = 15
curv_window = 3

# Load GaMD and reaction-coordinate data
log_data = load_gamd_log(log_file)

coord = load_reaction_coord(
    coord_file,
    coord_type,
)

# Total PPI-GaMD boost potential
delta_V = (
    log_data["deltaVp"]
    + log_data["deltaVd"]
)

# Evaluate boost-potential distribution
gamma = calc_anharmonicity(delta_V)

print(
    f"Boost-potential anharmonicity: {gamma:.3f}"
)

# Calculate biased and reweighted PMFs
F_star, F, bin_centers = calc_pmf_1D(
    coord=coord,
    boost=delta_V,
    cutoff=cutoff,
    bin_size=bin_size,
    temperature=temperature,
    progress=True,
)

# Plot PMFs
fig, ax = plot_pmf_1D(
    F_star=F_star,
    F=F,
    bin_centers=bin_centers,
    coord_type=coord_type,
)

plt.show()

# Should the PMF be trimmed??
# Smooth finite PMF only
#F_solver = gaussian_filter1d(F, sigma=smooth_sigma)
#F_solver -= np.nanmin(F_solver)
#dx = np.mean(np.diff(x_raw))

# ------------------------------------------------------------
# Identify bound and unbound minima
# ------------------------------------------------------------
bound, unbound, barrier = find_wells(
    F,
    bins=bin_centers,
    rb_cutoff=rb,
    ru_cutoff=ru,

)
print(f"Bound minimum at {bin_centers[bound]:.3f}, F = {F[bound]:.3f} kcal/mol")
print(f"Bound unbound minimum at {bin_centers[unbound]:.3f}, F = {F[unbound]:.3f} kcal/mol")

print(
        f"Free energy barrier of association: {F[barrier]-F[unbound]:.2e} kcal/mol, "
        f"& dissociation: {F[barrier]-F[bound]:.2e} kcal/mol"
    )

# ------------------------------------------------------------
# Local quadratic curvatures
# ------------------------------------------------------------
curvatures = {}
frequencies = {}

for name, idx in [
        ("bound", bound),
        ("barrier", barrier),
        ("unbound", unbound)
    ]:
    curvatures[name], frequencies[name] = find_curvature(
        F,
        bin_centers,
        idx,
        curv_window=curv_window
    )

Fpp_bound = curvatures["bound"]
Fpp_barrier = curvatures["barrier"]
Fpp_unbound = curvatures["unbound"]

if Fpp_bound <= 0:
    print("Warning: bound-state curvature is not positive.")
if Fpp_unbound <= 0:
    print("Warning: unbound-state curvature is not positive.")
if Fpp_barrier >= 0:
    print("Warning: barrier curvature is not negative. Barrier may not be well defined.")
    
print("Curvature of the free energy profile near:")
print(f"Bound: {Fpp_bound:.3e}")
print(f"Barrier: {Fpp_barrier:.3e}")
print(f"Unbound: {Fpp_unbound:.3e}")

# ------------------------------------------------------------
# Diagnostic PMF plot
# ------------------------------------------------------------
plt = plot_minima_diagnostics(
    bin_centers,
    F,
    F_star,
    bound,
    unbound,
    barrier,
    coord_type
)
plt.show()

# ------------------------------------------------------------
# Direct residence times from trajectory
# ------------------------------------------------------------
tau_b, tau_u = find_residence_times(
    F=F,
    boost=delta_V,
    x=coord,
    rb_cutoff=rb,
    ru_cutoff=ru,
    frame_dt=log_data["frame_dt"],
    min_event_duration=1.0,  # ns
)

# ------------------------------------------------------------
# Biased observed rates
# ------------------------------------------------------------
k_off_star = 1.0 / tau_b                 # s^-1
k_on1_star = 1.0 / tau_u                 # s^-1, first-order association
k_on_star = k_on1_star / conc            # M^-1 s^-1, second-order association

print(
    f"Biased association constant: {k_on_star:.2e} M^-1 s^-1 "
    f"& dissociation: {k_off_star:.2e} s^-1"
)

# ------------------------------------------------------------
# Bound-state interval for dissociation Smoluchowski solver
# ------------------------------------------------------------
bound_start = 0

for i in range(bound - 1, 0, -1):
    if F[i] > F[i - 1] and F[i] > F[i + 1]:
        bound_start = i
        break

result_off = solve_smoluchowski(
    F=F,
    x=bin_centers,
    well_start=bound_start,
    barrier=barrier,
    D=1.0,
    left_boundary="Reflective",
    right_boundary="Absorbing",
    nsteps=300000,
    output_stride=100,
    diagnostic=True
)

print(result_off["status"])
print("k_model off:", result_off["k_model"])
print("dt used:", result_off["dt"])

if not np.isfinite(result_off["k_model"]) or result_off["k_model"] <= 0:
    print(f"Invalid Smoluchowski off-rate model. Skipping.")


# ------------------------------------------------------------
# Unbound-state interval for association Smoluchowski solver
# ------------------------------------------------------------
unbound_end = len(F) - 1

for i in range(unbound + 1, len(F) - 1):
    if F[i] > F[i - 1] and F[i] > F[i + 1]:
        unbound_end = i
        break

result_on = solve_smoluchowski(
    F=F,
    x=bin_centers,
    well_start=barrier,
    barrier=unbound_end,
    D=1.0,
    left_boundary="Absorbing",
    right_boundary="Reflective",
    nsteps=300000,
    output_stride=100,
    diagnostic=True
)

print(result_on["status"])
print("k_model on:", result_on["k_model"])

if not np.isfinite(result_on["k_model"]) or result_on["k_model"] <= 0:
    print(f"Invalid Smoluchowski on-rate model. Skipping.")

# ------------------------------------------------------------
# Apparent diffusion coefficients
# ------------------------------------------------------------
D_off = k_off_star / result_off["k_model"]
D_on = k_on1_star / result_on["k_model"]

print(f"D_off: {D_off:.3e} Å²/s")
print(f"D_on:  {D_on:.3e} Å²/s")

kBT = BOLTZMANN_KCAL * temperature

xi_off = kBT/D_off
xi_on = kBT/D_on

# ------------------------------------------------------------
# Kramers corrected rates
# ------------------------------------------------------------
k_off = (
    D_off
    / (2.0 * np.pi * kBT)
        * np.sqrt(abs(Fpp_bound) * abs(Fpp_barrier))
        * np.exp(-F[barrier]-F[bound] / kBT)
    )

k_on1 = (
    D_on
    / (2.0 * np.pi * kBT)
    * np.sqrt(abs(Fpp_unbound) * abs(Fpp_barrier))
    * np.exp(-F[barrier]-F[unbound] / kBT)
)
k_on = k_on1 / conc
    
# Convert first-order association to second-order association
Kd_M = k_off / k_on
Ka_M_inv = k_on / k_off

print("\nFinal corrected kinetic estimates")
print(f"k_off: {k_off:.3e} s^-1")
print(f"k_on first-order: {k_on1:.3e} s^-1")
print(f"k_on: {k_on:.3e} M^-1 s^-1")
print(f"Kd: {Kd_M:.3e} M")
print(f"Ka: {Ka_M_inv:.3e} M^-1")