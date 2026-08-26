"""Free-energy reweighting functions."""

import numpy as np
from tqdm import tqdm
from numpy.typing import ArrayLike

from .constants import BOLTZMANN_KCAL


def calc_pmf_1D(
    coord: ArrayLike,
    boost: ArrayLike,
    cutoff: int,
    bin_size: float,
    temperature: float = 300.0,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate biased and cumulant-reweighted one-dimensional PMFs.

    Reweighting uses a second-order cumulant expansion of the GaMD
    boost potential within each reaction-coordinate bin.

    Parameters
    ----------
    coord
        Reaction-coordinate value for each trajectory frame.
    boost
        GaMD boost potential for each corresponding trajectory frame,
        in kcal/mol.
    cutoff
        Minimum number of frames required for a bin to be included.
    bin_size
        Width of reaction-coordinate bins.
    temperature
        Simulation temperature in Kelvin.
    progress
        Whether to display a progress bar during PMF calculation.

    Returns
    -------
    F_star
        Biased PMF in kcal/mol.
    F
        Reweighted PMF in kcal/mol.
    bin_centers
        Centers of the reaction-coordinate bins.
    """
    coord = np.asarray(
        coord,
        dtype=float,
    )
    delta_U = np.asarray(
        boost,
        dtype=float,
    )

    if coord.ndim != 1 or delta_U.ndim != 1:
        raise ValueError(
            "Coordinate and boost potential must be one-dimensional."
        )

    if len(coord) != len(delta_U):
        raise ValueError(
            "Coordinate and boost potential must contain the same "
            f"number of frames. Got {len(coord)} and "
            f"{len(delta_U)}."
        )

    if not np.all(np.isfinite(coord)):
        raise ValueError(
            "Reaction coordinate contains NaN or infinite values."
        )

    if not np.all(np.isfinite(delta_U)):
        raise ValueError(
            "Boost potential contains NaN or infinite values."
        )

    if cutoff < 1:
        raise ValueError(
            "Cutoff must be at least 1."
        )

    if bin_size <= 0:
        raise ValueError(
            "Bin size must be greater than zero."
        )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    beta = 1.0 / (
        BOLTZMANN_KCAL * temperature
    )

    total_frames = len(coord)

    lower_edge = (
        np.floor(coord.min() / bin_size)
        * bin_size
    )
    upper_edge = (
        np.ceil(coord.max() / bin_size)
        * bin_size
    )

    if np.isclose(
        lower_edge,
        upper_edge,
    ):
        upper_edge = lower_edge + bin_size

    edges = np.arange(
        lower_edge,
        upper_edge + bin_size,
        bin_size,
    )

    bin_centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    F_star = np.full(
        len(bin_centers),
        np.nan,
    )
    F = np.full(
        len(bin_centers),
        np.nan,
    )

    for i in tqdm(range(len(bin_centers)), desc="Reweighting") if progress else range(len(bin_centers)):
        if i == len(bin_centers) - 1:
            in_bin = (
                (coord >= edges[i])
                & (coord <= edges[i + 1])
            )
        else:
            in_bin = (
                (coord >= edges[i])
                & (coord < edges[i + 1])
            )

        n_frames = np.sum(in_bin)

        if n_frames < cutoff:
            continue

        p_star = n_frames / total_frames

        F_star_bin = (
            -BOLTZMANN_KCAL
            * temperature
            * np.log(p_star)
        )

        mean_delta_U = np.mean(
            delta_U[in_bin]
        )
        variance_delta_U = np.var(
            delta_U[in_bin]
        )

        # Second-order cumulant expansion:
        #
        # (1 / beta) ln <exp(beta ΔV)>
        # ≈ <ΔV> + beta / 2 Var(ΔV)
        correction = (
            mean_delta_U
            + (beta / 2.0)
            * variance_delta_U
        )

        F_star[i] = F_star_bin
        F[i] = F_star_bin - correction

    # Set the minimum of each PMF to zero.
    if np.any(np.isfinite(F_star)):
        F_star -= np.nanmin(F_star)

    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)

    return F_star, F, bin_centers

def calc_pmf_2D(
    coord1: ArrayLike,
    coord2: ArrayLike,
    boost: ArrayLike,
    cutoff: int,
    bin_size1: float,
    bin_size2: float,
    temperature: float = 300.0,
    progress: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calculate biased and cumulant-reweighted two-dimensional PMFs.

    Reweighting uses a second-order cumulant expansion of the GaMD
    boost potential within each two-dimensional reaction-coordinate bin.

    Parameters
    ----------
    coord1
        First reaction-coordinate value for each trajectory frame.
    coord2
        Second reaction-coordinate value for each trajectory frame.
    boost
        GaMD boost potential, ΔU, for each corresponding trajectory
        frame, in kcal/mol.
    cutoff
        Minimum number of frames required for a bin to be included.
    bin_size1
        Width of bins along the first reaction coordinate.
    bin_size2
        Width of bins along the second reaction coordinate.
    temperature
        Simulation temperature in Kelvin.
    progress
        Whether to display a progress bar during PMF calculation.

    Returns
    -------
    F_star
        Biased two-dimensional PMF in kcal/mol.
    F
        Reweighted two-dimensional PMF in kcal/mol.
    bin_centers1
        Centers of bins along the first reaction coordinate.
    bin_centers2
        Centers of bins along the second reaction coordinate.
    counts
        Number of frames in each bin.
    """
    X = np.asarray(
        coord1,
        dtype=float,
    )
    Y = np.asarray(
        coord2,
        dtype=float,
    )
    delta_U = np.asarray(
        boost,
        dtype=float,
    )

    # Validate input arrays.
    if X.ndim != 1 or Y.ndim != 1 or delta_U.ndim != 1:
        raise ValueError(
            "Coordinates and boost potential must be one-dimensional."
        )

    if not (
        len(X) == len(Y) == len(delta_U)
    ):
        raise ValueError(
            "Coordinates and boost potential must contain the same "
            f"number of frames. Got {len(X)}, {len(Y)}, and "
            f"{len(delta_U)}."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Reaction coordinate 1 contains NaN or infinite values."
        )

    if not np.all(np.isfinite(Y)):
        raise ValueError(
            "Reaction coordinate 2 contains NaN or infinite values."
        )

    if not np.all(np.isfinite(delta_U)):
        raise ValueError(
            "Boost potential contains NaN or infinite values."
        )

    if cutoff < 1:
        raise ValueError(
            "Cutoff must be at least 1."
        )

    if bin_size1 <= 0 or bin_size2 <= 0:
        raise ValueError(
            "Bin sizes must be greater than zero."
        )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    beta = 1.0 / (
        BOLTZMANN_KCAL * temperature
    )
    total_frames = len(delta_U)

    # Align bin edges with multiples of the requested bin sizes.
    lower_edge1 = (
        np.floor(X.min() / bin_size1)
        * bin_size1
    )
    upper_edge1 = (
        np.ceil(X.max() / bin_size1)
        * bin_size1
    )

    lower_edge2 = (
        np.floor(Y.min() / bin_size2)
        * bin_size2
    )
    upper_edge2 = (
        np.ceil(Y.max() / bin_size2)
        * bin_size2
    )

    if np.isclose(lower_edge1, upper_edge1):
        upper_edge1 = lower_edge1 + bin_size1

    if np.isclose(lower_edge2, upper_edge2):
        upper_edge2 = lower_edge2 + bin_size2

    edges1 = np.arange(
        lower_edge1,
        upper_edge1 + bin_size1,
        bin_size1,
    )
    edges2 = np.arange(
        lower_edge2,
        upper_edge2 + bin_size2,
        bin_size2,
    )

    bin_centers1 = 0.5 * (
        edges1[:-1] + edges1[1:]
    )
    bin_centers2 = 0.5 * (
        edges2[:-1] + edges2[1:]
    )

    F_star = np.full(
        (len(bin_centers1), len(bin_centers2)),
        np.nan,
    )
    F = np.full(
        (len(bin_centers1), len(bin_centers2)),
        np.nan,
    )

    counts = np.zeros(
        (len(bin_centers1), len(bin_centers2)),
        dtype=int,
    )

    for i in tqdm(range(len(bin_centers1)), desc="Reweighting") if progress else range(len(bin_centers1)):
        for j in range(len(bin_centers2)):
            # Include the upper edges in the final bins so that
            # maximum coordinate values are not discarded.
            if i == len(bin_centers1) - 1:
                in_bin1 = (
                    (X >= edges1[i])
                    & (X <= edges1[i + 1])
                )
            else:
                in_bin1 = (
                    (X >= edges1[i])
                    & (X < edges1[i + 1])
                )

            if j == len(bin_centers2) - 1:
                in_bin2 = (
                    (Y >= edges2[j])
                    & (Y <= edges2[j + 1])
                )
            else:
                in_bin2 = (
                    (Y >= edges2[j])
                    & (Y < edges2[j + 1])
                )

            in_bin = in_bin1 & in_bin2
            n_frames = np.sum(in_bin)
            counts[i, j] = n_frames

            if n_frames < cutoff:
                continue

            p_star = n_frames / total_frames

            F_star_bin = (
                -BOLTZMANN_KCAL
                * temperature
                * np.log(p_star)
            )

            mean_delta_U = np.mean(
                delta_U[in_bin]
            )
            variance_delta_U = np.var(
                delta_U[in_bin]
            )

            # Second-order cumulant expansion:
            #
            # (1 / beta) ln <exp(beta ΔU)>
            # ≈ <ΔU> + beta / 2 Var(ΔU)
            correction = (
                mean_delta_U
                + (beta / 2.0)
                * variance_delta_U
            )

            F_star[i, j] = F_star_bin
            F[i, j] = F_star_bin - correction

    # Set the minimum of each PMF to zero.
    if np.any(np.isfinite(F_star)):
        F_star -= np.nanmin(F_star)

    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)

    return F_star, F, bin_centers1, bin_centers2, counts

def calc_pmf_3D(
    coord1: ArrayLike,
    coord2: ArrayLike,
    coord3: ArrayLike,
    boost: ArrayLike,
    cutoff: int,
    bin_size: float,
    temperature: float = 300.0,
    progress: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calculate biased and cumulant-reweighted two-dimensional PMFs.

    Reweighting uses a second-order cumulant expansion of the GaMD
    boost potential within each two-dimensional reaction-coordinate bin.

    Parameters
    ----------
    coord1
        First reaction-coordinate value for each trajectory frame.
    coord2
        Second reaction-coordinate value for each trajectory frame.
    coord3
        Third reaction-coordinate value for each trajectory frame.
    boost
        GaMD boost potential, ΔU, for each corresponding trajectory
        frame, in kcal/mol.
    cutoff
        Minimum number of frames required for a bin to be included.
    bin_size
        Width of bins along the all reaction coordinates.
    temperature
        Simulation temperature in Kelvin.
    progress
        Whether to display a progress bar during PMF calculation.

    Returns
    -------
    F_star
        Biased two-dimensional PMF in kcal/mol.
    F
        Reweighted two-dimensional PMF in kcal/mol.
    bin_centers1
        Centers of bins along the first reaction coordinate.
    bin_centers2
        Centers of bins along the second reaction coordinate.
    bin_centers3
        Centers of bins along the third reaction coordinate.
    counts
        Number of frames in each bin.
    """
    X = np.asarray(
        coord1,
        dtype=float,
    )
    Y = np.asarray(
        coord2,
        dtype=float,
    )
    Z = np.asarray(
        coord3,
        dtype=float,
    )

    delta_U = np.asarray(
        boost,
        dtype=float,
    )

    # Validate input arrays.
    if X.ndim != 1 or Y.ndim != 1 or Z.ndim != 1 or delta_U.ndim != 1:
        raise ValueError(
            "Coordinates and boost potential must be one-dimensional."
        )

    if not (
        len(X) == len(Y) == len(Z) == len(delta_U)
    ):
        raise ValueError(
            "Coordinates and boost potential must contain the same "
            f"number of frames. Got {len(X)}, {len(Y)}, {len(Z)} and "
            f"{len(delta_U)}."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Reaction coordinate 1 contains NaN or infinite values."
        )

    if not np.all(np.isfinite(Y)):
        raise ValueError(
            "Reaction coordinate 2 contains NaN or infinite values."
        )

    if not np.all(np.isfinite(Z)):
        raise ValueError(
            "Reaction coordinate 3 contains NaN or infinite values."
        )

    if not np.all(np.isfinite(delta_U)):
        raise ValueError(
            "Boost potential contains NaN or infinite values."
        )

    if cutoff < 1:
        raise ValueError(
            "Cutoff must be at least 1."
        )

    if bin_size <= 0:
        raise ValueError(
            "Bin size must be greater than zero."
        )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    beta = 1.0 / (
        BOLTZMANN_KCAL * temperature
    )
    total_frames = len(delta_U)

    # Align bin edges with multiples of the requested bin sizes.
    lower_edge1 = (
        np.floor(X.min() / bin_size)
        * bin_size
    )
    upper_edge1 = (
        np.ceil(X.max() / bin_size)
        * bin_size
    )

    lower_edge2 = (
        np.floor(Y.min() / bin_size)
        * bin_size
    )
    upper_edge2 = (
        np.ceil(Y.max() / bin_size)
        * bin_size
    )

    lower_edge3 = (
        np.floor(Z.min() / bin_size)
        * bin_size
    )

    upper_edge3 = (
        np.ceil(Z.max() / bin_size)
        * bin_size
    )

    if np.isclose(lower_edge1, upper_edge1):
        upper_edge1 = lower_edge1 + bin_size

    if np.isclose(lower_edge2, upper_edge2):
        upper_edge2 = lower_edge2 + bin_size

    if np.isclose(lower_edge3, upper_edge3):
        upper_edge3 = lower_edge3 + bin_size

    edges1 = np.arange(
        lower_edge1,
        upper_edge1 + bin_size,
        bin_size,
    )
    edges2 = np.arange(
        lower_edge2,
        upper_edge2 + bin_size,
        bin_size,
    )

    edges3 = np.arange(
        lower_edge3,
        upper_edge3 + bin_size,
        bin_size,
    )

    bin_centers1 = 0.5 * (
        edges1[:-1] + edges1[1:]
    )
    bin_centers2 = 0.5 * (
        edges2[:-1] + edges2[1:]
    )
    bin_centers3 = 0.5 * (
        edges3[:-1] + edges3[1:]
    )

    F_star = np.full(
        (len(bin_centers1), len(bin_centers2), len(bin_centers3)),
        np.nan,
    )
    F = np.full(
        (len(bin_centers1), len(bin_centers2), len(bin_centers3)),
        np.nan,
    )

    counts = np.zeros(
        (len(bin_centers1), len(bin_centers2), len(bin_centers3)),
        dtype=int,
    )

    for i in tqdm(range(len(bin_centers1)), desc="Reweighting") if progress else range(len(bin_centers1)):
        for j in range(len(bin_centers2)):
            for k in range(len(bin_centers3)):
                # Include the upper edges in the final bins so that
                # maximum coordinate values are not discarded.
                if i == len(bin_centers1) - 1:
                    in_bin1 = (
                        (X >= edges1[i])
                        & (X <= edges1[i + 1])
                    )
                else:
                    in_bin1 = (
                        (X >= edges1[i])
                        & (X < edges1[i + 1])
                    )

                if j == len(bin_centers2) - 1:
                    in_bin2 = (
                        (Y >= edges2[j])
                        & (Y <= edges2[j + 1])
                    )
                else:
                    in_bin2 = (
                        (Y >= edges2[j])
                        & (Y < edges2[j + 1])
                    )

                if k == len(bin_centers3) - 1:
                    in_bin3 = (
                        (Z >= edges3[k])
                        & (Z <= edges3[k + 1])
                    )
                else:
                    in_bin3 = (
                        (Z >= edges3[k])
                        & (Z < edges3[k + 1])
                    )

                in_bin = in_bin1 & in_bin2 & in_bin3
                n_frames = np.sum(in_bin)
                counts[i, j, k] = n_frames

                if n_frames < cutoff:
                    continue

                p_star = n_frames / total_frames

                F_star_bin = (
                    -BOLTZMANN_KCAL
                    * temperature
                    * np.log(p_star)
                )

                mean_delta_U = np.mean(
                    delta_U[in_bin]
                )
                variance_delta_U = np.var(
                    delta_U[in_bin]
                )

                # Second-order cumulant expansion:
                #
                # (1 / beta) ln <exp(beta ΔU)>
                # ≈ <ΔU> + beta / 2 Var(ΔU)
                correction = (
                    mean_delta_U
                    + (beta / 2.0)
                    * variance_delta_U
                )

                F_star[i, j, k] = F_star_bin
                F[i, j, k] = F_star_bin - correction

    # Set the minimum of each PMF to zero.
    if np.any(np.isfinite(F_star)):
        F_star -= np.nanmin(F_star)

    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)

    return F_star, F, bin_centers1, bin_centers2, bin_centers3, counts
