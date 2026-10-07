"""Example one-dimensional PPI-GaMD reweighting workflow."""
from pathlib import Path
import matplotlib.pyplot as plt
from ppi_gamd_reweighting import (
    calc_anharmonicity,
    calc_pmf_1D,
    load_gamd_log,
    load_reaction_coord,
    plot_pmf_1D,
)

# Input files
example_dir = Path(__file__).parent / "data"
log_file = example_dir / "GaMD.log"
coord_file = example_dir / "dist.dat"

# Analysis parameters
coord_type = "distance"
bin_size = 1.0
cutoff = 10
temperature = 300.0

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
)

# Evaluate boost-potential distribution with the no. of bins from the PMF
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