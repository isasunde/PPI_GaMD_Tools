"""Plotting functions for GaMD reweighting results."""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

from .coordinates import (
    COORD_LABELS,
    ReactionCoordType,
)
from .constants import BOLTZMANN_KCAL


def plot_pmf_1D(
    F_star: ArrayLike,
    F: ArrayLike,
    bin_centers: ArrayLike,
    coord_type: ReactionCoordType,
):
    """Plot biased and reweighted one-dimensional PMFs."""
    try:
        xlabel = COORD_LABELS[coord_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown reaction-coordinate type: {coord_type}"
        ) from exc

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(bin_centers, F_star, label="Biased PMF")
    ax.plot(bin_centers, F, label="Reweighted PMF",)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("PMF (kcal/mol)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()

    return fig, ax

def plot_pmf_2D(
    F_star: ArrayLike,
    F: ArrayLike,
    bin_centers1: ArrayLike,
    bin_centers2: ArrayLike,
    counts: ArrayLike,
    coord_type1: ReactionCoordType,
    coord_type2: ReactionCoordType,
    padding: int = 2,
):
    """Plot biased and reweighted two-dimensional PMFs.

    The displayed region is restricted to bins containing sampled
    configurations, with optional padding around the populated region.

    Parameters
    ----------
    F_star
        Biased two-dimensional PMF in kcal/mol.
    F
        Reweighted two-dimensional PMF in kcal/mol.
    bin_centers1
        Centers of bins along the first reaction coordinate.
    bin_centers2
        Centers of bins along the second reaction coordinate.
    counts
        Number of trajectory frames assigned to each two-dimensional bin.
    coord_type1
        Type of the first reaction coordinate.
    coord_type2
        Type of the second reaction coordinate.
    padding
        Number of additional bins shown around the populated region.

    Returns
    -------
    fig
        Matplotlib figure.
    axes
        Array of Matplotlib axes.
    """
    F_star = np.asarray(F_star, dtype=float)
    F = np.asarray(F, dtype=float)

    bin_centers1 = np.asarray(bin_centers1, dtype=float)
    bin_centers2 = np.asarray(bin_centers2, dtype=float)

    counts = np.asarray(counts)

    # Validate reaction-coordinate types.
    try:
        xlabel = COORD_LABELS[coord_type1]
        ylabel = COORD_LABELS[coord_type2]
    except KeyError as exc:
        raise ValueError(
            f"Unknown reaction-coordinate type: {exc.args[0]}"
        ) from exc

    # Validate PMF and population-map dimensions.
    if F_star.shape != F.shape:
        raise ValueError(
            "Biased and reweighted PMFs must have the same shape."
        )

    if counts.shape != F_star.shape:
        raise ValueError(
            "Population counts must have the same shape as the PMFs."
        )

    expected_shape = (len(bin_centers1), len(bin_centers2))

    if F_star.shape != expected_shape:
        raise ValueError(
            "PMF dimensions do not match the supplied bin centers. "
            f"Expected {expected_shape}, got {F_star.shape}."
        )

    if padding < 0:
        raise ValueError("Padding must be zero or greater.")

    # Identify the region containing sampled configurations.
    occupied = np.isfinite(F_star)

    if not np.any(occupied):
        raise ValueError("The PMF contains no populated bins.")

    rows = np.where(np.any(occupied, axis=1))[0]
    cols = np.where(np.any(occupied, axis=0))[0]

    xmin = rows[0]
    xmax = rows[-1] + 1

    ymin = cols[0]
    ymax = cols[-1] + 1

    # Restrict all data to the populated region.
    x = bin_centers1[xmin:xmax]
    y = bin_centers2[ymin:ymax]

    F_star_plot = F_star[xmin:xmax, ymin:ymax].copy()

    F_plot = F[xmin:xmax, ymin:ymax].copy()

    counts_plot = counts[xmin:xmax, ymin:ymax].copy()

    # Determine bin widths.
    bin_size1 = np.mean(np.diff(bin_centers1))
    bin_size2 = np.mean(np.diff(bin_centers2))

    # Pad the coordinate arrays beyond the sampled region.
    x_lower = x[0] - bin_size1 * np.arange(padding, 0, -1)
    x_upper = x[-1] + bin_size1 * np.arange(1, padding + 1)

    y_lower = y[0] - bin_size2 * np.arange(padding, 0, -1) 
    y_upper = y[-1] + bin_size2 * np.arange(1, padding + 1)

    x = np.concatenate([x_lower, x, x_upper])
    y = np.concatenate([y_lower, y, y_upper])

    # Pad PMF arrays with NaN. These will later be displayed
    # at the maximum plotting energy.
    F_star_plot = np.pad(
        F_star_plot,
        padding,
        mode="constant",
        constant_values=np.nan,
    )

    F_plot = np.pad(
        F_plot,
        padding,
        mode="constant",
        constant_values=np.nan,
    )

    # Pad population map with zero counts.
    counts_plot = np.pad(
        counts_plot,
        padding,
        mode="constant",
        constant_values=0,
    )

    # Construct a coordinate grid from the bin centers.
    X_grid, Y_grid = np.meshgrid(x, y, indexing="ij")

    # Use a shared free-energy scale for biased and reweighted PMFs.
    finite_energies = np.concatenate(
        [F_star_plot[np.isfinite(F_star_plot)],
            F_plot[np.isfinite(F_plot)],]
    )

    Emax = np.max(finite_energies) * 1.2

    if np.isclose(Emax, 0.0):
        Emax = 1.0

    levels = np.linspace(0.0, Emax, 31)

    F_star_plot[np.isnan(F_star_plot)] = Emax
    F_star_plot[F_star_plot > Emax] = Emax

    F_plot[np.isnan(F_plot)] = Emax
    F_plot[F_plot > Emax] = Emax

    # Initialize figure.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # Plot biased PMF.
    cf_star = axes[0].contourf(
        X_grid,
        Y_grid,
        F_star_plot,
        levels=levels,
        cmap="jet"
    )

    cs_star = axes[0].contour(
        X_grid,
        Y_grid,
        F_star_plot,
        levels=levels[::3],
        colors="black",
        linewidths=0.5,
    )
    
    axes[0].set_title("Biased PMF")

    # Plot reweighted PMF.
    cf = axes[1].contourf(
        X_grid,
        Y_grid,
        F_plot,
        levels=levels,
        cmap="jet"
    )

    cs = axes[1].contour(
        X_grid,
        Y_grid,
        F_plot,
        levels=levels[::3],
        colors="black",
        linewidths=0.5,
    )

    axes[1].set_title("Reweighted PMF")

    # Shared colorbar for the two PMFs.
    cbar = fig.colorbar(
        cf,
        ax=axes[:2],
        label="PMF (kcal/mol)",
    )

    # Plot sampled population across the reaction coordinates.
    population = axes[2].pcolormesh(
        x,
        y,
        counts_plot.T,
        shading="auto",
    )

    axes[2].set_title("Population map")

    fig.colorbar(
        population,
        ax=axes[2],
        label="Number of frames",
    )

    # Apply common reaction-coordinate labels.
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])

    return fig, axes

def plot_pmf_3D(
    F_star: ArrayLike,
    F: ArrayLike,
    bin_centers1: ArrayLike,
    bin_centers2: ArrayLike,
    bin_centers3: ArrayLike,
    counts: ArrayLike,
    coord_type1: ReactionCoordType,
    coord_type2: ReactionCoordType,
    coord_type3: ReactionCoordType,
    padding: int = 2,
    temperature: float = 300.0,
):
    """Plot two-dimensional projections of biased and reweighted 3D PMFs.

    The three possible two-dimensional projections are calculated by
    marginalizing over the remaining reaction coordinate:

    - coordinate 1 vs. coordinate 2
    - coordinate 1 vs. coordinate 3
    - coordinate 2 vs. coordinate 3

    Parameters
    ----------
    F_star
        Biased three-dimensional PMF in kcal/mol.
    F
        Reweighted three-dimensional PMF in kcal/mol.
    bin_centers1
        Centers of bins along the first reaction coordinate.
    bin_centers2
        Centers of bins along the second reaction coordinate.
    bin_centers3
        Centers of bins along the third reaction coordinate.
    counts
        Number of trajectory frames assigned to each three-dimensional bin.
    coord_type1
        Type of the first reaction coordinate.
    coord_type2
        Type of the second reaction coordinate.
    coord_type3
        Type of the third reaction coordinate.
    padding
        Number of additional bins displayed around the sampled region.
    temperature
        Simulation temperature in Kelvin.

    Returns
    -------
    fig
        Matplotlib figure.
    axes
        Array of Matplotlib axes.
    """
    F_star = np.asarray(F_star, dtype=float)
    F = np.asarray(F, dtype=float)

    bin_centers1 = np.asarray(bin_centers1, dtype=float)
    bin_centers2 = np.asarray(bin_centers2, dtype=float)
    bin_centers3 = np.asarray(bin_centers3, dtype=float)

    counts = np.asarray(counts, dtype=float)

    # Validate reaction-coordinate types.
    try:
        xlabel = COORD_LABELS[coord_type1]
        ylabel = COORD_LABELS[coord_type2]
        zlabel = COORD_LABELS[coord_type3]
    except KeyError as exc:
        raise ValueError(
            f"Unknown reaction-coordinate type: {exc.args[0]}"
        ) from exc

    # Validate temperature and padding.
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    if padding < 0:
        raise ValueError("Padding must be zero or greater.")

    # Validate PMF dimensions.
    expected_shape = (len(bin_centers1), len(bin_centers2), len(bin_centers3))

    if F_star.shape != expected_shape:
        raise ValueError(
            "Biased PMF dimensions do not match the supplied "
            f"bin centers. Expected {expected_shape}, "
            f"got {F_star.shape}."
        )

    if F.shape != expected_shape:
        raise ValueError(
            "Reweighted PMF dimensions do not match the supplied "
            f"bin centers. Expected {expected_shape}, "
            f"got {F.shape}."
        )

    if counts.shape != expected_shape:
        raise ValueError(
            "Population counts do not match the PMF dimensions. "
            f"Expected {expected_shape}, got {counts.shape}."
        )

    # -----------------------------------------------------------------
    # Constants
    # -----------------------------------------------------------------
    kBT = BOLTZMANN_KCAL * temperature

    # -----------------------------------------------------------------
    # Calculate two-dimensional population projections.
    # -----------------------------------------------------------------
    Nxy = np.sum(counts, axis=2)
    Nxz = np.sum(counts, axis=1)
    Nyz = np.sum(counts, axis=0)

    # -----------------------------------------------------------------
    # Convert 3D PMFs to probability distributions.
    #
    # P ∝ exp(-F / kBT) 
    # The arbitrary additive constant in the PMF therefore does not 
    # affect the resulting projected PMF after baseline shifting.
    # -----------------------------------------------------------------
    P_star = np.exp(-F_star / kBT)
    P = np.exp(-F / kBT)

    # -----------------------------------------------------------------
    # Marginalize probability distributions.
    # -----------------------------------------------------------------
    Pxy_star = np.nansum(P_star, axis=2)
    Pxz_star = np.nansum(P_star, axis=1,)
    Pyz_star = np.nansum(P_star, axis=0,)

    Pxy = np.nansum(P, axis=2)
    Pxz = np.nansum(P, axis=1)
    Pyz = np.nansum(P, axis=0)

    # -----------------------------------------------------------------
    # Convert projected probabilities back to PMFs.
    #
    # Zero-probability bins are assigned NaN because their PMF is
    # undefined rather than infinitely high.
    # -----------------------------------------------------------------
    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        Fxy_star = -kBT * np.log(Pxy_star)
        Fxz_star = -kBT * np.log(Pxz_star)
        Fyz_star = -kBT * np.log(Pyz_star)

        Fxy = -kBT * np.log(Pxy)
        Fxz = -kBT * np.log(Pxz)
        Fyz = -kBT * np.log(Pyz)

    Fxy_star[Pxy_star <= 0] = np.nan
    Fxz_star[Pxz_star <= 0] = np.nan
    Fyz_star[Pyz_star <= 0] = np.nan

    Fxy[Pxy <= 0] = np.nan
    Fxz[Pxz <= 0] = np.nan
    Fyz[Pyz <= 0] = np.nan

    # -----------------------------------------------------------------
    # Shift PMF baselines to zero.
    # -----------------------------------------------------------------
    for pmf in [
        Fxy_star,
        Fxz_star,
        Fyz_star,
        Fxy,
        Fxz,
        Fyz,
    ]:
        if np.any(np.isfinite(pmf)):
            pmf -= np.nanmin(pmf)

    # -----------------------------------------------------------------
    # Determine shared maximum plotting energy.
    # -----------------------------------------------------------------
    finite_energies = np.concatenate(
        [
            Fxy_star[np.isfinite(Fxy_star)],
            Fxz_star[np.isfinite(Fxz_star)],
            Fyz_star[np.isfinite(Fyz_star)],
            Fxy[np.isfinite(Fxy)],
            Fxz[np.isfinite(Fxz)],
            Fyz[np.isfinite(Fyz)],
        ]
    )

    if len(finite_energies) == 0:
        raise ValueError("No finite PMF values are available for plotting.")

    Emax = np.max(finite_energies) * 1.2

    if np.isclose(Emax, 0.0):
        Emax = 1.0

    # Replace unsampled regions with maximum plotting energy.
    for pmf in [
        Fxy_star,
        Fxz_star,
        Fyz_star,
        Fxy,
        Fxz,
        Fyz,
    ]:
        pmf[np.isnan(pmf)] = Emax
        pmf[pmf > Emax] = Emax

    # -----------------------------------------------------------------
    # Initialize figure.
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10, 10),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

    # -----------------------------------------------------------------
    # Define projections.
    # -----------------------------------------------------------------
    pairs = [
        (
            Nxy,
            Fxy_star,
            Fxy,
            bin_centers1,
            bin_centers2,
            xlabel,
            ylabel,
        ),
        (
            Nxz,
            Fxz_star,
            Fxz,
            bin_centers1,
            bin_centers3,
            xlabel,
            zlabel,
        ),
        (
            Nyz,
            Fyz_star,
            Fyz,
            bin_centers2,
            bin_centers3,
            ylabel,
            zlabel,
        ),
    ]

    titles = [
        "Coordinate 1 vs. Coordinate 2",
        "Coordinate 1 vs. Coordinate 3",
        "Coordinate 2 vs. Coordinate 3",
    ]

    # -----------------------------------------------------------------
    # Plot projections.
    # -----------------------------------------------------------------
    for row, (
        occupancy,
        pmf_star,
        pmf,
        centers1,
        centers2,
        label1,
        label2,
    ) in enumerate(pairs):

        # -------------------------------------------------------------
        # Determine populated region.
        # -------------------------------------------------------------
        occupied = occupancy > 0

        if not np.any(occupied):
            continue

        rows = np.where(np.any(occupied, axis=1))[0]
        cols = np.where(np.any(occupied, axis=0))[0]

        xmin = max(0, rows[0] - padding)
        xmax = min(occupancy.shape[0], rows[-1] + padding + 1)

        ymin = max(0, cols[0] - padding)
        ymax = min(occupancy.shape[1], cols[-1] + padding + 1)

        x = centers1[xmin:xmax]
        y = centers2[ymin:ymax]

        occupancy_plot = occupancy[xmin:xmax, ymin:ymax].copy()

        pmf_star_plot = pmf_star[xmin:xmax, ymin:ymax].copy()

        pmf_plot = pmf[xmin:xmax, ymin:ymax].copy()

        # -------------------------------------------------------------
        # Construct coordinate grid.
        # -------------------------------------------------------------
        X, Y = np.meshgrid(x, y, indexing="ij")

        # -------------------------------------------------------------
        # Occupancy plot.
        # -------------------------------------------------------------
        occupancy_plot = np.log10(occupancy_plot + 1)

        im1 = axes[row, 0].contourf(
            X,
            Y,
            occupancy_plot,
            levels=30,
            cmap="viridis",
        )

        # -------------------------------------------------------------
        # Biased PMF.
        # -------------------------------------------------------------
        im2 = axes[row, 1].contourf(
            X,
            Y,
            pmf_star_plot,
            levels=30,
            cmap="jet",
            vmin=0,
            vmax=Emax,
        )

        axes[row, 1].contour(
            X,
            Y,
            pmf_star_plot,
            levels=10,
            colors="black",
            linewidths=0.4,
        )

        # -------------------------------------------------------------
        # Reweighted PMF.
        # -------------------------------------------------------------
        im3 = axes[row, 2].contourf(
            X,
            Y,
            pmf_plot,
            levels=30,
            cmap="jet",
            vmin=0,
            vmax=Emax,
        )

        axes[row, 2].contour(
            X,
            Y,
            pmf_plot,
            levels=10,
            colors="black",
            linewidths=0.4,
        )

        # -------------------------------------------------------------
        # Axis labels.
        # -------------------------------------------------------------
        for col in range(3):
            axes[row, col].set_xlabel(label1)
            axes[row, col].set_ylabel(label2)
            axes[row, col].grid(alpha=0.2)

    axes[0, 0].set_title("Population")
    axes[0, 1].set_title("Biased PMF")
    axes[0, 2].set_title("Reweighted PMF")

    # -----------------------------------------------------------------
    # Colorbars.
    # -----------------------------------------------------------------
    cbar1 = fig.colorbar(
        im1,
        ax=axes[:, 0],
        fraction=0.11,
        pad=0.02,
        aspect = 30
    )
    cbar1.set_label("log$_{10}$(population + 1)")

    cbar2 = fig.colorbar(
        im3,
        ax=axes[:, 1:],
        fraction=0.05,
        pad=0.02,
        aspect = 30
    )
    cbar2.set_label("Free energy (kcal/mol)")

    return fig, axes

def plot_minima_diagnostics(
        x: ArrayLike,
        F: ArrayLike,
        F_star: ArrayLike,
        bound: int,
        unbound: int,
        barrier: int,
        coord_type: ReactionCoordType,
                
):
    """
    Plot diagnostics for identified bound and unbound minima, as well as the free-energy barrier. 
    """
    # Validate reaction-coordinate types.
    try:
        label = COORD_LABELS[coord_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown reaction-coordinate type: {exc.args[0]}"
        ) from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, F_star, label="Biased PMF", alpha=0.7)
    ax.plot(x, F, label="Reweighted PMF", alpha=0.7)
    #plt.plot(x, F_solver, label="Smoothed PMF", lw=2)
    
    ax.scatter(
        [x[bound], x[unbound], x[barrier]],
        [F[bound], F[unbound], F[barrier]],
        s=40,
        c=["green", "red", "orange"],
        zorder=5,
    )
   
    ax.set_xlabel(label)
    ax.set_ylabel("PMF (kcal/mol)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return plt