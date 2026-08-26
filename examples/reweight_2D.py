"""Example two-dimensional PPI-GaMD reweighting workflow."""

from pathlib import Path

import matplotlib.pyplot as plt

from ppi_gamd_reweighting import (
    calc_anharmonicity,
    calc_pmf_2D,
    load_gamd_log,
    load_reaction_coord,
    plot_pmf_2D,
)


# ---------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------
example_dir = Path(__file__).parent / "data"

log_file = example_dir / "GaMD.log"

coord_file1 = example_dir / "dist.dat"
coord_file2 = example_dir / "rmsd.dat"


# ---------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------
coord_type1 = "distance"
coord_type2 = "rmsd"

bin_size1 = 1.0
bin_size2 = 0.5

cutoff = 10
temperature = 300.0


# ---------------------------------------------------------------------
# Load GaMD and reaction-coordinate data
# ---------------------------------------------------------------------
log_data = load_gamd_log(log_file)

coord1 = load_reaction_coord(
    coord_file1,
    coord_type1,
)

coord2 = load_reaction_coord(
    coord_file2,
    coord_type2,
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
gamma = calc_anharmonicity(delta_U)

print(
    f"Boost-potential anharmonicity: {gamma:.3f}"
)


# ---------------------------------------------------------------------
# Calculate biased and reweighted 2D PMFs
# ---------------------------------------------------------------------
F_star, F, bin_centers1, bin_centers2, counts = calc_pmf_2D(
    coord1=coord1,
    coord2=coord2,
    boost=delta_U,
    cutoff=cutoff,
    bin_size1=bin_size1,
    bin_size2=bin_size2,
    temperature=temperature,
)


# ---------------------------------------------------------------------
# Plot PMFs and population map
# ---------------------------------------------------------------------
fig, axes = plot_pmf_2D(
    F_star=F_star,
    F=F,
    bin_centers1=bin_centers1,
    bin_centers2=bin_centers2,
    counts=counts,
    coord_type1=coord_type1,
    coord_type2=coord_type2,
)

plt.show()