"""Example binding free energy calculation workflow."""

from pathlib import Path
import numpy as np

from ppi_gamd_reweighting import (
    calc_anharmonicity,
    calc_pmf_3D,
    plot_pmf_3D,
    load_gamd_log,
    load_reaction_coord,
    binding_free_energy,
)

# ---------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------
example_dir = Path(__file__).parent / "data"

log_file = example_dir / "energy_GaMD.log"

coord_file = example_dir / "energy_vector.dat"

# ---------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------
coord_type1 = "x-dist"
coord_type2 = "y-dist"
coord_type3 = "z-dist"

bin_size = 1.0
cutoff = 10
temperature = 300.0

# ---------------------------------------------------------------------
# Load GaMD and reaction-coordinate data
# ---------------------------------------------------------------------
print("\nLoading GaMD and reaction-coordinate data...")
log_data = load_gamd_log(log_file)

coord1 = load_reaction_coord(
    coord_file,
    coord_type1,
)

coord2 = load_reaction_coord(
    coord_file,
    coord_type2,
)

coord3 = load_reaction_coord(
    coord_file,
    coord_type3,
)
print("Done loading data.")

# ---------------------------------------------------------------------
# Construct total PPI-GaMD boost potential
# ---------------------------------------------------------------------
delta_U = (
    log_data["deltaVp"]
    + log_data["deltaVd"]
)

# ---------------------------------------------------------------------
# Evaluate boost-potential distribution
# ---------------------------------------------------------------------
print("\nEvaluating boost-potential distribution...")
gamma = calc_anharmonicity(delta_U, temperature)

print(
    f"Boost-potential anharmonicity: {gamma:.3f}"
)

# ---------------------------------------------------------------------
# Calculate biased and reweighted 3D PMFs
# ---------------------------------------------------------------------
print("\nCalculating biased and reweighted 3D PMFs...")
F_star, F, bin_centers1, bin_centers2, bin_centers3, counts = calc_pmf_3D(
    coord1=coord1,
    coord2=coord2,
    coord3=coord3,
    boost=delta_U,
    cutoff=cutoff,
    bin_size=bin_size,
    temperature=temperature,
    progress=True
)
print("Done calculating PMFs.")

plot_pmf_3D(
    F_star=F_star,
    F = F,
    bin_centers1 = bin_centers1,
    bin_centers2 = bin_centers2,
    bin_centers3 = bin_centers3,
    counts = counts,
    coord_type1 = coord_type1,
    coord_type2 = coord_type2,
    coord_type3 = coord_type3,
    )

# ---------------------------------------------------------------------
# Evaluate boost-potential distribution based on reweighting bins
# ---------------------------------------------------------------------
print("\nEvaluating boost-potential distribution...")
gamma = calc_anharmonicity(delta_U, temperature, bins = len(bin_centers1)*len(bin_centers2)*len(bin_centers3))

print(
    f"Boost-potential anharmonicity: {gamma:.3f}"
)

# ---------------------------------------------------------------------
# Calculate binding energetics for reweighted PMF
# ---------------------------------------------------------------------
print("\nCalculating binding energetics for reweighted PMF...")
dG, n_bound, n_unbound, V_bound, V_bound0, dW, V_unbound, V_unbound0, = binding_free_energy(
    pmf=F,
    rb=25.0,
    ru=30.0,
    x_bins=bin_centers1,
    y_bins=bin_centers2,
    z_bins=bin_centers3,
    bin_size=np.array((bin_size, bin_size, bin_size)),
    temperature=temperature,
)

# -----------------------------------------------------------------
# Print calculation summary.
# -----------------------------------------------------------------
print("Reweighted binding free energy calculation:")
if dG:
    print(
        f"n_bound             = {n_bound:d}"
    )
    print(
        f"n_unbound           = {n_unbound:d}"
    )
    print(
        f"dG                  = {dG:.3f} kcal/mol"
    )
    print(
        f"(V_bound, V_bound0) = "
        f"{V_bound:.3f}, {V_bound0:.3f}"
    )
    print(
        f"(dW, V_unbound, "
        f"V_unbound0)        = "
        f"{dW:.3f}, "
        f"{V_unbound:.3f}, "
        f"{V_unbound0:.3f}"
    )
else:
    print("Insufficient binding/unbinding events to calculate binding energetics")
    print(
        f"n_bound             = {n_bound:d}"
    )
    print(
        f"n_unbound           = {n_unbound:d}"
    )
    print(
        f"(V_bound, V_bound0) = "
        f"{V_bound:.3f}, {V_bound0:.3f}"
    )
    print(
        f"(V_unbound, V_unbound0)        = "
        f"{V_unbound:.3f}, "
        f"{V_unbound0:.3f}"
    )

# ---------------------------------------------------------------------
# Calculate binding energetics for biased PMF
# ---------------------------------------------------------------------
print("\nCalculating binding energetics for biased PMF...")
dG, n_bound, n_unbound, V_bound, V_bound0, dW, V_unbound, V_unbound0, = binding_free_energy(
    pmf=F_star,
    rb=25.0,
    ru=30.0,
    x_bins=bin_centers1,
    y_bins=bin_centers2,
    z_bins=bin_centers3,
    bin_size=np.array((bin_size, bin_size, bin_size)),
    temperature=temperature,
)

# -----------------------------------------------------------------
# Print calculation summary.
# -----------------------------------------------------------------
print("Biased binding free energy calculation:")
if dG:
    print(
        f"n_bound             = {n_bound:d}"
    )
    print(
        f"n_unbound           = {n_unbound:d}"
    )
    print(
        f"dG                  = {dG:.3f} kcal/mol"
    )
    print(
        f"(V_bound, V_bound0) = "
        f"{V_bound:.3f}, {V_bound0:.3f}"
    )
    print(
        f"(dW, V_unbound, "
        f"V_unbound0)        = "
        f"{dW:.3f}, "
        f"{V_unbound:.3f}, "
        f"{V_unbound0:.3f}"
    )
else:
    print("Insufficient binding/unbinding events to calculate binding energetics")
    print(
        f"n_bound             = {n_bound:d}"
    )
    print(
        f"n_unbound           = {n_unbound:d}"
    )
    print(
        f"(V_bound, V_bound0) = "
        f"{V_bound:.3f}, {V_bound0:.3f}"
    )
    print(
        f"(V_unbound, V_unbound0)        = "
        f"{V_unbound:.3f}, "
        f"{V_unbound0:.3f}"
    )