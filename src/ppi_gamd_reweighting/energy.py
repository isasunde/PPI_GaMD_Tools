"""Calculate binding energy from GaMD results."""

import numpy as np
from numpy.typing import ArrayLike

from .constants import BOLTZMANN_KCAL, V0

def binding_free_energy(
    pmf: ArrayLike,
    rb: float,
    ru: float,
    x_bins: ArrayLike,
    y_bins: ArrayLike,
    z_bins: ArrayLike,
    bin_size: ArrayLike,
    V0: float = V0,
    temperature: float = 300.0,
) -> tuple[
    float | None,
    float,
    float,
    float | None,
    float,
    float,
]:
    """Calculate binding free energy from a three-dimensional Cartesian PMF.

    The binding free energy is calculated by separating the PMF into
    bound and unbound regions based on the radial distance from the
    origin.

    The bound region is defined as:

        r <= rb

    and the unbound region as:

        r >= ru

    where:

        r = sqrt(x^2 + y^2 + z^2)

    The intermediate region, rb < r < ru, is excluded from the
    calculation.

    Parameters
    ----------
    pmf
        Three-dimensional PMF in kcal/mol.
    rb
        Upper radial boundary of the bound region.
    ru
        Lower radial boundary of the unbound region.
    x_bins
        Bin centers along the x coordinate.
    y_bins
        Bin centers along the y coordinate.
    z_bins
        Bin centers along the z coordinate.
    bin_size
        Width of the bins along the x, y, and z coordinates.
    V0
        Standard-state reference volume in the same volume units as
        the PMF bin volume.
    temperature
        Simulation temperature in Kelvin.

    Returns
    -------
    dG
        Binding free energy in kcal/mol. Returns ``None`` if no bound
        or unbound bins are available.
    V_bound
        Configurational volume of the bound region.
    V_bound0
        Geometric volume of the bound region.
    dW
        Work associated with the unbound region in kcal/mol.
    V_unbound
        Configurational volume of the unbound region.
    V_unbound0
        Geometric volume of the unbound region.
    """
    # -----------------------------------------------------------------
    # Convert inputs to NumPy arrays.
    # -----------------------------------------------------------------
    pmf = np.asarray(
        pmf,
        dtype=float,
    )

    x_bins = np.asarray(
        x_bins,
        dtype=float,
    )
    y_bins = np.asarray(
        y_bins,
        dtype=float,
    )
    z_bins = np.asarray(
        z_bins,
        dtype=float,
    )

    # -----------------------------------------------------------------
    # Validate input parameters.
    # -----------------------------------------------------------------
    if pmf.ndim != 3:
        raise ValueError(
            "PMF must be a three-dimensional array."
        )

    expected_shape = (
        len(x_bins),
        len(y_bins),
        len(z_bins),
    )

    if pmf.shape != expected_shape:
        raise ValueError(
            "PMF dimensions do not match the supplied bin centers. "
            f"Expected {expected_shape}, got {pmf.shape}."
        )

    if rb < 0:
        raise ValueError(
            "Bound-state radius must be zero or greater."
        )

    if ru <= rb:
        raise ValueError(
            "Unbound-state radius must be greater than the "
            "bound-state radius."
        )

    if len(bin_size) != 3:
        raise ValueError(
            "Bin size must contain three values."
        )

    if any(size <= 0 for size in bin_size):
        raise ValueError(
            "All bin sizes must be greater than zero."
        )

    if V0 <= 0:
        raise ValueError(
            "Reference volume V0 must be greater than zero."
        )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    # -----------------------------------------------------------------
    # Calculate thermodynamic constants and bin volume.
    # -----------------------------------------------------------------
    kBT = BOLTZMANN_KCAL * temperature
    beta = 1.0 / kBT

    dV_bin = (
        bin_size[0]
        * bin_size[1]
        * bin_size[2]
    )

    # -----------------------------------------------------------------
    # Shift PMF baseline to zero.
    # -----------------------------------------------------------------
    pmf = pmf.copy()

    if not np.any(np.isfinite(pmf)):
        raise ValueError(
            "PMF contains no finite values."
        )

    pmf -= np.nanmin(pmf)

    # -----------------------------------------------------------------
    # Initialize configurational volumes and bin counts.
    # -----------------------------------------------------------------
    V_bound = 0.0
    V_unbound = 0.0

    n_bound = 0
    n_unbound = 0

    # -----------------------------------------------------------------
    # Integrate configurational volumes over bound and unbound regions.
    # -----------------------------------------------------------------
    for ix in range(pmf.shape[0]):
        for iy in range(pmf.shape[1]):
            for iz in range(pmf.shape[2]):

                if not np.isfinite(pmf[ix, iy, iz]):
                    continue

                x = x_bins[ix]
                y = y_bins[iy]
                z = z_bins[iz]

                r = np.sqrt(
                    x**2
                    + y**2
                    + z**2
                )

                boltzmann_weight = np.exp(
                    -beta * pmf[ix, iy, iz]
                )

                if r <= rb:
                    n_bound += 1
                    V_bound += boltzmann_weight

                elif r >= ru:
                    n_unbound += 1
                    V_unbound += boltzmann_weight

    # -----------------------------------------------------------------
    # Convert bin counts and Boltzmann-weighted sums to volumes.
    # -----------------------------------------------------------------
    V_bound0 = n_bound * dV_bin
    V_unbound0 = n_unbound * dV_bin

    V_bound *= dV_bin
    V_unbound *= dV_bin

    # -----------------------------------------------------------------
    # Check that both regions contain sampled bins.
    # -----------------------------------------------------------------
    if n_bound == 0:
        print("No bound bins observed.")

        return (
            None,
            n_bound,
            n_unbound,
            V_bound,
            V_bound0,
            None,
            V_unbound,
            V_unbound0,
        )

    if n_unbound == 0:
        print("No unbound bins observed.")

        return (
            None,
            n_bound,
            n_unbound,
            V_bound,
            V_bound0,
            None,
            V_unbound,
            V_unbound0,
        )

    # -----------------------------------------------------------------
    # Calculate unbound-state work and binding free energy.
    # -----------------------------------------------------------------
    dW = -kBT * np.log(
        V_unbound / V_unbound0
    )

    dG = (
        -kBT * np.log(
            V_bound / V0
        )
        - dW
    )

    return (
        dG,
        n_bound,
        n_unbound,
        V_bound,
        V_bound0,
        dW,
        V_unbound,
        V_unbound0,
    )