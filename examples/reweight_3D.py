"""Example three-dimensional PPI-GaMD reweighting workflow."""

from pathlib import Path

import matplotlib.pyplot as plt

from ppi_gamd_reweighting import (
    calc_anharmonicity,
    calc_pmf_3D,
    load_gamd_log,
    load_reaction_coord,
    plot_pmf_3D,
)

# ---------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------
example_dir = Path(__file__).parent / "data"

log_file = example_dir / "GaMD.log"

coord_file = example_dir / "vector_dist.dat"

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
gamma = calc_anharmonicity(delta_U, temperature)

print(
    f"Boost-potential anharmonicity: {gamma:.3f}"
)

# ---------------------------------------------------------------------
# Calculate biased and reweighted 3D PMFs
# ---------------------------------------------------------------------
F_star, F, bin_centers1, bin_centers2, bin_centers3, counts = calc_pmf_3D(
    coord1=coord1,
    coord2=coord2,
    coord3=coord3,
    boost=delta_U,
    cutoff=cutoff,
    bin_size=bin_size,
    temperature=temperature,
)

# ---------------------------------------------------------------------
# Evaluate boost-potential distribution on no. bins from PMF
# ---------------------------------------------------------------------
gamma = calc_anharmonicity(delta_U, temperature, bins = len(bin_centers1)*len(bin_centers2)*len(bin_centers3))

print(
    f"Boost-potential anharmonicity: {gamma:.3f}"
)

# ---------------------------------------------------------------------
# Plot PMFs and population map
# ---------------------------------------------------------------------
fig, axes = plot_pmf_3D(
    F_star=F_star,
    F=F,
    bin_centers1=bin_centers1,
    bin_centers2=bin_centers2,
    bin_centers3=bin_centers3,
    counts=counts,
    coord_type1=coord_type1,
    coord_type2=coord_type2,
    coord_type3=coord_type3,
)

plt.show()