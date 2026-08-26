"""Statistical metrics for evaluating GaMD boost potentials."""

import numpy as np
from numpy.typing import ArrayLike


def calc_anharmonicity(
    boost: ArrayLike,
    bins: int = 100,
) -> float:
    """Calculate the anharmonicity of a boost-potential distribution."""
    delta_U = np.asarray(
        boost,
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

    S_delta_U = -np.sum(
        p_delta_U[nonzero]
        * np.log(p_delta_U[nonzero])
        * d_delta_U[nonzero]
    )

    S_max = 0.5 * np.log(
        2 * np.pi * np.e * sigma**2
    )

    gamma = S_max - S_delta_U

    return gamma
        
