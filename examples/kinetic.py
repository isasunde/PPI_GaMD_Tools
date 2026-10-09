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
    calc_kinetic_params
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
time_step = dt = 0.002      # In nanoseconds

# Simulation box volume
Vol_A3 = 137.607**3 
Vol_L = Vol_A3 * 1e-27
conc = 1.0 / (NA * Vol_L)

# Kinetic parameters
rb = 20
ru = 25
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
gamma = calc_anharmonicity(delta_V, temperature)

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

# Evaluate boost-potential distribution on no. of bins from reweighting
gamma = calc_anharmonicity(delta_V, temperature, bins = len(bin_centers))

print(
    f"Boost-potential anharmonicity: {gamma:.3f}"
)

# Plot PMFs
fig, ax = plot_pmf_1D(
    F_star=F_star,
    F=F,
    bin_centers=bin_centers,
    coord_type=coord_type,
)
plt.show()

# ------------------------------------------------------------
# Identify bound and unbound minima
# ------------------------------------------------------------
bound, unbound, barrier = find_wells(
    F,
    bins=bin_centers,
    rb_cutoff=rb,
    ru_cutoff=ru,

)
print(f"\nBound minimum at {bin_centers[bound]:.3f}, F = {F[bound]:.3f} kcal/mol")
print(f"Bound unbound minimum at {bin_centers[unbound]:.3f}, F = {F[unbound]:.3f} kcal/mol")

print(
        f"Free energy barrier of association: {F[barrier]-F[unbound]:.2e} kcal/mol, "
        f"& dissociation: {F[barrier]-F[bound]:.2e} kcal/mol"
    )

# ------------------------------------------------------------
# Find curvatures through local quadratic curvatures
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
        fit_window=curv_window
    )

if curvatures["bound"] <= 0:
    print("Warning: bound-state curvature is not positive.")
if curvatures["unbound"] <= 0:
    print("Warning: unbound-state curvature is not positive.")
if curvatures["barrier"] >= 0:
    print("Warning: barrier curvature is not negative. Barrier may not be well defined.")
    
print("\nCurvature of the free energy profile near:")
print(f"Bound: {curvatures["bound"]:.3e}")
print(f"Barrier: {curvatures["barrier"]:.3e}")
print(f"Unbound: {curvatures["unbound"]:.3e}")

print("\nFrequencies of the free energy profile near:")
print(f"Bound: {frequencies['bound']:.3e}")
print(f"Barrier: {frequencies['barrier']:.3e}")
print(f"Unbound: {frequencies['unbound']:.3e}")
# ------------------------------------------------------------
# Diagnostic PMF plot
# ------------------------------------------------------------
fig, ax = plot_minima_diagnostics(
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
    boost=delta_V,
    x=coord,
    rb_cutoff=rb,
    ru_cutoff=ru,
    frame_dt=time_step,
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
print(f'\nBound minima found to start at bin {bound_start} corresponding to {bin_centers[bound_start]}')

result_off = solve_smoluchowski(
    F=F,
    x=bin_centers,
    well_start=bound_start,
    barrier=barrier,
    temperature=temperature,
    D=1.0,
    left_boundary="Reflective",
    right_boundary="Absorbing",
    nsteps=300000,
    output_stride=100,
    diagnostic=True
)

print('\n' + result_off["status"])
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
    temperature=temperature,
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
# Calculate kinetic parameters
# ------------------------------------------------------------
kinetics = calc_kinetic_params(
    F = F,
    barrier = barrier,
    bound = bound,
    unbound = unbound,
    w_b = frequencies['bound'],
    w_u = frequencies['unbound'],
    w_br = frequencies['barrier'],
    k_off_star = k_off_star,
    k_on1_star = k_on1_star,
    k_off_model = result_off["k_model"],
    k_on1_model = result_on["k_model"],
    temperature = temperature,
    conc = conc
)

print("\nFinal corrected kinetic estimates")
print(f"k_off: {kinetics['k_off']:.3e} s^-1")
print(f"k_on first-order: {kinetics['k_on1']:.3e} s^-1")
print(f"k_on: {kinetics['k_on']:.3e} M^-1 s^-1")
print(f"Kd: {kinetics['Kd_M']:.3e} M")
print(f"Ka: {kinetics['Ka_M_inv']:.3e} M^-1")