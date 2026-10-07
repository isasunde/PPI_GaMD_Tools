"""Statistical metrics for evaluating GaMD boost potentials."""

import numpy as np
from numpy.typing import ArrayLike

from .constants import BOLTZMANN_KCAL

def calc_anharmonicity(
    boost: ArrayLike,
    temperature: float,
    bins: int = 100,
) -> float:
    """Calculate the anharmonicity of a boost-potential distribution.
       Based on equation 16 from Miao & McCammon 2017"""
    delta_U = np.asarray(
        boost/(BOLTZMANN_KCAL*temperature),
        dtype=float,
    )

    if delta_U.ndim != 1:
        raise ValueError(
            "Boost potential must be a one-dimensional array."
        )

    if len(delta_U) < 2:
        raise ValueError(
            "At least two boost-potential values are required."
        )

    if not np.all(np.isfinite(delta_U)):
        raise ValueError(
            "Boost potential contains NaN or infinite values."
        )

    sigma = np.std(delta_U)

    if sigma == 0:
        raise ValueError(
            "Anharmonicity cannot be calculated for a distribution "
            "with zero variance."
        )

    p_delta_U, edges = np.histogram(
        delta_U,
        bins=bins,
        density=True,
    )

    d_delta_U = np.diff(edges)
    nonzero = p_delta_U > 0

    # S_deltaU = sum(p(DeltaU)*ln(p(DeltaU))*dDeltaU)
    S_delta_U = -np.sum(
        p_delta_U[nonzero]
        * np.log(p_delta_U[nonzero])
        * d_delta_U[nonzero]
    )

    # S_max = 1/2*ln(2*pi*e*sigma(DeltaU)^2)
    S_max = 0.5 * np.log(
        2 * np.pi * np.e * sigma**2
    )

    gamma = S_max - S_delta_U

    return gamma
        
