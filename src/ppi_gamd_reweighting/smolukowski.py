"""Smolukowskii solver for kinetic reweighting of GaMD simulations"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Literal, get_args
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from .constants import BOLTZMANN_KCAL

BoundaryType = Literal[
    "Reflective",
    "Absorbing",
]

def find_wells(
        F: ArrayLike,
        bins: ArrayLike,
        rb_cutoff: float,
        ru_cutoff: float,
        
    ):
    """
        Find the bound & unbound well of a PMF.
        Note the function assumes bound responds to lowest coordinate values 
        As such need to be inversed for i.e. contacts as coordinate
    
        Parameters
        ----------
        F : array-like
            1D PMF in energy units, usually kcal/mol.
    
        bins : array-like
            Bin centers for the reaction coordinate.
    
        rb_cutoff : int
            Max value of reaction coordinate corresponding to bound state.
    
        ru_cutoff : int
            Min value of reaction coordinate corresponding to unbound state.
    
        Returns
        -------
        bound : int
            Bin index corresponding to bound state minimum
        
        unbound : int
            Bin index corresponding to unbound state minimum
        
        barrier : int
            Bin index corresponding to barrier between bound and unbound minima

        """
    # Find local minima
    local_min = F == minimum_filter1d(
        F,
        size=3,
        mode="nearest"
    )

    well_indices = np.where(local_min & np.isfinite(F))[0]

    # Remove boundary minima caused by edge effects
    well_indices = well_indices[
        (well_indices > 0) &
        (well_indices < len(F) - 1)
    ]

    if len(well_indices) < 2:
        print(f"Fewer than two minima found. Skipping kinetic analysis.")
        return None, None, None

    # Convert minima indices to coordinate values
    well_x = bins[well_indices]
    well_F = F[well_indices]

    # ------------------------------------------------------------
    # Option A: Choice minima based on specified cutoffs
    # ------------------------------------------------------------
    # Bound candidates: Low-distance minima
    bound_candidates = well_indices[well_x <= rb_cutoff]

    # Unbound candidates: High-distance minima
    unbound_candidates = well_indices[well_x >= ru_cutoff]

    if len(bound_candidates) > 0 and len(unbound_candidates) > 0:

        # Choose the lowest-energy minimum within each physical region
        bound = int(bound_candidates[np.argmin(F[bound_candidates])])
        unbound = int(unbound_candidates[np.argmin(F[unbound_candidates])])

    else:
        print(
            f"Could not find minima in both bound and unbound regions. "
            "Using fallback based on separated minima."
        )

        # ------------------------------------------------------------
        # Option B: Fallback based on selection of sufficiently separated minima
        # ------------------------------------------------------------
        # Sort minima by energy
        sorted_wells = well_indices[np.argsort(F[well_indices])]

        # First minimum is the deepest one
        first = int(sorted_wells[0])
        first_x = bins[first]

        # Require second minimum to be separated in coordinate space
        min_separation = ru_cutoff - rb_cutoff

        separated = [
            idx for idx in sorted_wells[1:]
            if abs(bins[idx] - first_x) >= min_separation
        ]

        if len(separated) == 0:
            print(
                f"No second minimum sufficiently separated from the deepest minimum. "
                "Skipping kinetic analysis."
                "Potentially try to decrease distance between rb_cutoff & ru_cutoff"
            )
            return None, None, None

        second = int(separated[0])

        # For distance coordinate: lower-distance minimum = bound
        bound = int(min(first, second))
        unbound = int(max(first, second))

    if bound == unbound:
        print(f"Bound and unbound minima identical. Skipping.")
        return None, None, None

    # ------------------------------------------------------------
    # Find barrier between bound and unbound minima
    # ------------------------------------------------------------
    barrier = int(np.argmax(F[bound:unbound + 1]) + bound)

    return bound, unbound, barrier

def find_curvature(
        F: ArrayLike,
        bins: ArrayLike,
        idx: int,
        fit_window: int=3 
    ):
    """
    Find the curvature of a point on a PMF.
    Based on equation 2 of Miao 2019 with update to use np.polyfit in place of the quadratic function plainly
    
    Parameters
    ----------
    F : array-like
        1D PMF in energy units, usually kcal/mol.
    
    bins : array-like
        Bin centers for the reaction coordinate.

    idx : int
        Bin index to find curvature for

    fit_window : int
        No. of bins on either side of idx to include to find curvature
        
    Returns
    -------
    ddF
        Curvature at selected bin calculated over the given window

    w
        Curvature in frequency at the selected bin

    """
    if fit_window < 1:
        raise ValueError('At least 3 bins is required to find curvature using quadratic formula')

    # Find potential bins for fitting
    i0 = max(0, idx - fit_window)
    i1 = min(len(F), idx + fit_window + 1)

    x_fit = bins[i0:i1]
    F_fit = F[i0:i1]

    valid_fit = np.isfinite(x_fit) & np.isfinite(F_fit)

    if np.sum(valid_fit) < 3:
        print("Not enough bins within window to get a valid fit")
        ddF = np.nan

    else:
        # Find curvature at a minima, A_m, from quadratic function 
        # F(A) = a*A^2 + b*A + c => F''(A) = 2*a
        coeff = np.polyfit(x_fit[valid_fit], F_fit[valid_fit], 2)
        ddF = 2.0 * coeff[0]

    # Convert curvatures to frequencies
    w = np.sqrt(abs(ddF)/(2*np.pi))

    return ddF, w

def find_residence_times(
        boost: ArrayLike,
        x: ArrayLike,
        rb_cutoff: float,
        ru_cutoff: float,
        min_event_duration: float,
        frame_dt: float,
        include_censored_event: bool=False,
        
):
    """"
    Calculate residence time in bound & unbound state

    Parameters
    -----------
    boost : array-lile
        Boost throughout the trajectory

    x : array-like
        Reaction coordinate throughout trajectory.
    
    min_event_duration : float
        Minimum duration of one event in ns

    frame_dt : float
        Time step between each frame in ns

    """
    if len(x) != len(boost):
        raise ValueError("The same number of frames must be included for the reaction coordinate and the boost")

    bound_times = []
    unbound_times = []
    
    min_event_frames = int(np.ceil(min_event_duration / frame_dt))
    
    print(
        f"Only including bound/unbound events lasting at least "
        f"{min_event_duration:.1f} ns, corresponding to ({min_event_frames} frames)"
    )
    
    # State labels:
    # -1 = intermediate / ambiguous
    #  0 = bound
    #  1 = unbound
    state = np.full(len(x), -1, dtype=int)
    
    bound_mask = (x <= rb_cutoff)
    unbound_mask = (x >= ru_cutoff)
    
    state[bound_mask] = 0
    state[unbound_mask] = 1
    
    current_state = None
    counter = 0
    
    for s in state:
        # Transition region: count it as part of the current event,
        # but do not start an event in the transition region.
        if s == -1:
            if current_state is not None:
                counter += 1
                continue
    
        if current_state is None:
            current_state = s
            counter = 1
            continue
    
        if s == current_state:
            counter += 1
        else:
            # Only append completed events that last at least 1 ns
            if counter >= min_event_frames:
                if current_state == 0:
                    bound_times.append(counter)
                else:
                    unbound_times.append(counter)
    
            current_state = s
            counter = 1
    
        # The final event is right-censored because no transition out is observed.
        # Usually exclude it from residence-time averages.
        if include_censored_event and current_state is not None:
            if counter >= min_event_frames:
                if current_state == 0:
                    bound_times.append(counter)
                else:
                    unbound_times.append(counter)
    
    # Convert frame counts to seconds
    bound_times = np.asarray(bound_times, dtype=float) * frame_dt * 1e-9
    unbound_times = np.asarray(unbound_times, dtype=float) * frame_dt * 1e-9
    
    if len(bound_times) == 0 or len(unbound_times) == 0:
        print(f"No complete bound/unbound transitions longer than {min_event_duration:.1f} ns observed. Skipping.")
        return None, None
    
    print(f"{len(bound_times)} binding & {len(unbound_times)} unbinding events observed")
    
    tau_b = np.mean(bound_times)
    tau_u = np.mean(unbound_times)
    
    print(
        f"Residence time in bound: {tau_b:.2e} s "
        f"& unbound: {tau_u:.2e} s"
    )
    return tau_b, tau_u

def safe_exp(x: float,
             clip: float=50.0
             ):
    """
    Exponential with clipping to avoid numerical overflow.

    The clipping does not replace proper timestep stability,
    but it prevents immediate floating-point overflow.
    """
    return np.exp(np.clip(x, -clip, clip))


def solve_smoluchowski(
    F: ArrayLike,
    x: ArrayLike,
    well_start: int,
    barrier: int,
    temperature: float,
    D: float=1.0,
    dt: float|None=None,
    nsteps: int=200000,
    output_stride: int=100,
    exp_clip: float=50.0,
    left_boundary: BoundaryType="Reflective",
    right_boundary: BoundaryType="Absorbing",
    stability_safety: float=0.05,
    fit_lnS_range: list=(-4.0, -0.05),
    stop_S: float=1e-8,
    diagnostic: bool=True,
):
    """
    Solve the 1D Smoluchowski equation on a PMF using an explicit
    finite-difference/master-equation style propagator.

    Parameters
    ----------
    F : array-like
        1D PMF in energy units, usually kcal/mol.

    x : array-like
        Bin centers for the reaction coordinate.

    well_start : int
        Left boundary index of the region whose survival probability is followed.

    barrier : int
        Right boundary index of the region whose survival probability is followed.
        For dissociation, this is typically the barrier between bound and unbound.

    temperature : float
        Thermal energy in same units as F. At 300 K, kBT ≈ 0.596 kcal/mol.

    D : float
        Diffusion coefficient in coordinate^2 / Smoluchowski-time.
        Usually set to 1.0 when extracting a model rate.

    dt : float or None
        Numerical integration timestep. If None, a stable value is estimated
        from the maximum local transition rate.

    nsteps : int
        Maximum number of integration steps.

    output_stride : int
        Record S(t), lnS(t), and mass every this many steps.

    exp_clip : float
        Clip exponent arguments to [-exp_clip, exp_clip].

    left_boundary : {"Reflective", "Absorbing"}
        Boundary condition at well_start.

    right_boundary : {"Reflective", "Absorbing"}
        Boundary condition at barrier.

    stability_safety : float
        Safety factor used for automatic dt estimation.
        Smaller values are more stable but slower.

    fit_lnS_range : tuple
        Range of lnS values used for linear fitting, e.g. (-4, -0.05).

    stop_S : float
        Stop integration when S(t) falls below this value.

    diagnostic : bool
        If True, produce diagnostic plots.

    Returns
    -------
    dict
        Contains times, lnS, mass, k_model, final_probability, dt, dx,
        fit coefficients, fit mask, boundaries, and status message.
    """

    kBT = BOLTZMANN_KCAL*temperature

    # -----------------------------
    # Validate boundary options
    # -----------------------------
    if left_boundary not in get_args(BoundaryType):
        raise ValueError(
            f"Left_boundary must be one of {BoundaryType}, got {left_boundary!r}"
        )
    
    if right_boundary not in get_args(BoundaryType):
        raise ValueError(
            f"Right_boundary must be one of {BoundaryType}, got {right_boundary!r}"
        )

    # -----------------------------
    # Convert input arrays
    # -----------------------------
    F = np.asarray(F, dtype=float)
    x = np.asarray(x, dtype=float)

    if F.ndim != 1 or x.ndim != 1:
        raise ValueError("F and x must both be 1D arrays.")

    if len(F) != len(x):
        raise ValueError("F and x must have the same length.")

    if temperature <= 0:
        raise ValueError("Temperature in Kelvin must be positive.")

    if D <= 0:
        raise ValueError("D must be positive.")

    # -----------------------------
    # Handle NaNs safely
    # -----------------------------
    finite = np.isfinite(F) & np.isfinite(x)

    if not np.any(finite):
        raise ValueError("No finite PMF points found.")

    finite_idx = np.where(finite)[0]
    first = finite_idx[0]
    last = finite_idx[-1]

    # Require no internal NaN gaps
    if not np.all(finite[first:last + 1]):
        raise ValueError(
            "F contains internal NaN/non-finite gaps. "
            "For this solver, use a contiguous sampled PMF region."
        )

    F = F[first:last + 1]
    x = x[first:last + 1]

    # Remap user-supplied original indices to trimmed array indices
    well_start = int(well_start) - first
    barrier = int(barrier) - first

    if well_start < 0 or barrier >= len(F):
        raise ValueError(
            "well_start/barrier fall outside the finite PMF region after trimming."
        )

    if well_start >= barrier:
        raise ValueError("well_start must be smaller than barrier.")

    # -----------------------------
    # Check x spacing
    # -----------------------------
    dxs = np.diff(x)

    if np.any(dxs <= 0):
        raise ValueError("x must be strictly increasing.")

    if not np.allclose(dxs, dxs[0], rtol=1e-6, atol=1e-10):
        raise(
            "Warning: x spacing is not perfectly uniform. "
        )

    dx = dxs[0]

    # -----------------------------
    # Boundary sanity checks
    # -----------------------------
    # Absorbing boundary needs one PMF bin outside the interval
    # because the outgoing rate depends on the neighboring potential.
    if left_boundary == "Absorbing" and well_start == 0:
        raise ValueError(
            "Left absorbing boundary requires well_start > 0 "
            "so the solver can evaluate the outgoing flux to the left."
        )

    if right_boundary == "Absorbing" and barrier == len(F) - 1:
        raise ValueError(
            "Right absorbing boundary requires barrier < len(F)-1 "
            "so the solver can evaluate the outgoing flux to the right."
        )

    # -----------------------------
    # Reduced PMF
    # -----------------------------
    F = F - np.nanmin(F)
    u = F / kBT

    # Get indexes & index slices for well
    interval = slice(well_start, barrier)
    interval_indices = np.arange(well_start, barrier)

    # Minimum inside the selected interval
    xmin_local = well_start + np.nanargmin(u[interval])

    # -----------------------------
    # Initial Boltzmann distribution inside the well
    # -----------------------------
    p = np.zeros_like(u)

    # Set the minima to zero within the well
    u_shift = u[interval] - u[xmin_local]

    # Construct initial probability in well
    # p_i(0) = e^(-u_i-u_{min}) 
    p[interval] = safe_exp(-u_shift, clip=exp_clip)

    # Normalize probability
    # p_i(0) = e^(-u_i-u_{min}) /(sum(e^{u-u_{min}})
    norm = np.sum(p[interval])

    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Initial probability normalization failed.")

    p /= norm
    S0 = np.sum(p[interval])

    # -----------------------------
    # Estimate stable dt if not given
    # -----------------------------
    # Delta t_{stable} = s*(dx^2)/(D*max(r_{out}))
    outgoing_rates = []

    for i in interval_indices:

        out_rate = 0.0

        # Left outgoing flux
        if i == well_start and left_boundary == "Reflective":
            left_out = 0.0
        else:
            # k_(i->i+1) = D/(Delta x^2)*exp(-(u_(i+1) - u_i)/2)
            left_out = safe_exp(-(u[i - 1] - u[i]) / 2.0, clip=exp_clip)
        out_rate += left_out

        # Right outgoing flux
        if i == barrier and right_boundary == "Reflective":
            right_out = 0.0
        else:
            # k_(i+i -> i) = D/(Delta x^2)*exp(-(u_i - u_(i+1))/2)
            right_out = safe_exp(-(u[i + 1] - u[i]) / 2.0, clip=exp_clip)
        out_rate += right_out

        outgoing_rates.append(out_rate)

    max_out_rate = np.max(outgoing_rates)

    if max_out_rate <= 0 or not np.isfinite(max_out_rate):
        raise ValueError("Invalid outgoing transition rates.")

    # Delta t_{stable} = s*(dx^2)/(D*max(r_{out}))
    dt_stable = stability_safety * dx**2 / (D * max_out_rate)

    if dt is None:
        dt = dt_stable
    else:
        if dt > dt_stable:
            print(
                f"Warning: dt={dt:.3e} may be unstable. "
                f"Recommended dt <= {dt_stable:.3e}. "
                f"Consider decreasing dt or stability_safety."
            )

    # -----------------------------
    # Storage
    # -----------------------------
    times = []
    lnS = []
    mass = []
    snapshots = []

    status = "completed"

    # -----------------------------
    # Time propagation
    # -----------------------------
    for step in tqdm(range(nsteps), desc = 'Calculating survival function: '):

        p_old = p.copy()
        p_new = np.zeros_like(p_old)

        for i in interval_indices:

            # -----------------------------------
            # Left side flux
            # -----------------------------------
            if i == well_start:

                if left_boundary == "Reflective":
                    left_in = 0.0
                    left_out = 0.0

                elif left_boundary == "Absorbing":
                    # No incoming flux from outside, but probability can leave.
                    left_in = 0.0
                    left_out = safe_exp(
                        -(u[i - 1] - u[i]) / 2.0,
                        clip=exp_clip
                    )

            else:
                left_in = p_old[i - 1] * safe_exp(
                    -(u[i] - u[i - 1]) / 2.0,
                    clip=exp_clip
                )

                left_out = safe_exp(
                    -(u[i - 1] - u[i]) / 2.0,
                    clip=exp_clip
                )

            # -----------------------------------
            # Right side flux
            # -----------------------------------
            if i == barrier - 1:

                if right_boundary == "Reflective":
                    right_in = 0.0
                    right_out = 0.0

                elif right_boundary == "Absorbing":
                    # No incoming flux from outside, but probability can leave.
                    right_in = 0.0
                    right_out = safe_exp(
                        -(u[barrier] - u[i]) / 2.0,
                        clip=exp_clip
                    )

            else:
                right_in = p_old[i + 1] * safe_exp(
                    -(u[i] - u[i + 1]) / 2.0,
                    clip=exp_clip
                )

                right_out = safe_exp(
                    -(u[i + 1] - u[i]) / 2.0,
                    clip=exp_clip
                )

            # -----------------------------------
            # Master-equation / finite-difference update
            # -----------------------------------
            incoming = left_in + right_in
            outgoing = p_old[i] * (left_out + right_out)

            p_new[i] = (
                p_old[i] + D * (incoming - outgoing) * dt / dx**2
            )

        # Avoid tiny negative values from explicit Euler roundoff
        if np.any(p_new < -1e-14):
            status = "Unstable: negative probability at step {step}"
            print(f'Solver became unstable at step {step}')
            break
        p_new[p_new < 0] = 0.0

        if not np.all(np.isfinite(p_new)):
            status = f"unstable: non-finite probability at step {step}"
            print(status)
            break

        S = np.sum(p_new[interval]) / S0
        total_mass = np.sum(p_new)

        if not np.isfinite(S) or S <= 0:
            status = f"invalid survival probability at step {step}: S={S}"
            print(status)
            break

        # With absorbing boundaries, mass should not increase.
        # Small numerical fluctuations are tolerated.
        if total_mass > S0*1.01:
            status = (
                f"unstable: probability mass increased to "
                f"{total_mass:.4f} at step {step}"
            )
            print(status)
            break

        if step % output_stride == 0:
            times.append((step + 1) * dt)
            lnS.append(np.log(S))
            mass.append(total_mass)
            snapshots.append(p_new.copy())

        p = p_new

        if S < stop_S:
            status = f"stopped: S < {stop_S}"
            break

    # -----------------------------
    # Convert outputs
    # -----------------------------
    times = np.asarray(times)
    lnS = np.asarray(lnS)
    mass = np.asarray(mass)

    # -----------------------------
    # Fit ln S(t)
    # -----------------------------
    fit_low, fit_high = fit_lnS_range

    fit_mask = (
        np.isfinite(times)
        & np.isfinite(lnS)
        & (lnS > fit_low)
        & (lnS < fit_high)
    )

    if np.sum(fit_mask) >= 3:
        coeff = np.polyfit(times[fit_mask], lnS[fit_mask], 1)
        k_model = -coeff[0]
    else:
        coeff = None
        k_model = np.nan
        print("Not enough valid points for linear fit.")
        print("Consider decreasing output stride")

    # -----------------------------
    # Diagnostics
    # -----------------------------
    if diagnostic:

        fig, axes = plt.subplots(2, 2, figsize=(8, 6))

        # PMF and boundaries
        axes[0, 0].plot(x, F, lw=2)
        axes[0, 0].axvline(
            x[well_start],
            linestyle="--",
            label=f"Left boundary: {left_boundary}"
        )
        axes[0, 0].axvline(
            x[barrier],
            linestyle="--",
            label=f"Right boundary: {right_boundary}"
        )
        axes[0, 0].scatter(
            x[xmin_local],
            F[xmin_local],
            s=60,
            label="Minimum in interval",
            zorder=5
        )
        axes[0, 0].set_title("PMF and selected interval")
        axes[0, 0].set_xlabel("Reaction coordinate")
        axes[0, 0].set_ylabel("PMF")
        axes[0, 0].legend()

        # Final probability distribution
        axes[0, 1].plot(x, p, lw=2)
        axes[0, 1].axvline(x[well_start], linestyle="--")
        axes[0, 1].axvline(x[barrier], linestyle="--")
        axes[0, 1].set_title("Final probability distribution")
        axes[0, 1].set_xlabel("Reaction coordinate")
        axes[0, 1].set_ylabel("Probability")

        # Survival decay
        if len(times) > 0:
            axes[1, 0].scatter(times, lnS, s=10, label="ln S(t)")

            if coeff is not None:
                axes[1, 0].plot(
                    times[fit_mask],
                    np.polyval(coeff, times[fit_mask]),
                    lw=2,
                    label=f"Fit: k = {k_model:.3e}"
                )

        axes[1, 0].set_title("Survival decay")
        axes[1, 0].set_xlabel("Smoluchowski time")
        axes[1, 0].set_ylabel("ln S(t)")
        axes[1, 0].legend()

        # Probability mass
        if len(times) > 0:
            axes[1, 1].plot(times, mass, lw=2)

        axes[1, 1].set_title("Total probability mass")
        axes[1, 1].set_xlabel("Smoluchowski time")
        axes[1, 1].set_ylabel("Mass")

        for ax in axes.flat:
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        "times": times,
        "lnS": lnS,
        "mass": mass,
        "k_model": k_model,
        "coeff": coeff,
        "fit_mask": fit_mask,
        "final_probability": p,
        "snapshots": snapshots,
        "dt": dt,
        "dt_stable": dt_stable,
        "dx": dx,
        "D": D,
        "kBT": kBT,
        "well_start": well_start + first,
        "well_end": barrier + first,
        "xmin_local": xmin_local + first,
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "status": status,
    }

def calc_kinetic_params(
    F: ArrayLike,
    barrier: int,
    bound: int,
    unbound: int,
    w_b: float,
    w_u: float,
    w_br: float,
    k_off_star: float,
    k_on1_star: float,
    k_off_model: float,
    k_on1_model: float,
    temperature: float,
    conc: float,
):
    """
    Calculate apparent diffusion coefficients and Kramers-corrected rates.

    The apparent diffusion coefficients are obtained by comparing the
    reference/model rates from the Smoluchowski solver with the target
    rates. The resulting diffusion coefficients are then used in a
    Kramers expression to calculate corrected dissociation and association
    rates from the PMF.

    Parameters
    ----------
    F : array-like
        1D PMF in energy units, usually kcal/mol.

    barrier : int
        Index of the PMF barrier separating the bound and unbound states.

    bound : int
        Index of the bound-state PMF minimum.

    unbound : int
        Index of the unbound-state PMF minimum.

    w_b : float
        Frequency of the PMF at the bound-state minimum.

    w_u : float
        Frequency of the PMF at the unbound-state minimum.

    w_br : float
        Frequency of the PMF at the barrier.

    k_off_star : float
        Reference dissociation rate used to determine the apparent
        diffusion coefficient for dissociation.

    k_on1_star : float
        Reference first-order association rate used to determine the
        apparent diffusion coefficient for association.

    k_off_model : float
        Dissociation rate obtained from the Smoluchowski model.

    k_on1_model : float
        First-order association rate obtained from the Smoluchowski model.

    temperature : float
        Temperature in Kelvin.

    conc : float
        Concentration used to convert the first-order association rate
        to a second-order association rate.

    Returns
    -------
    dict
        Dictionary containing:

        D_off : float
            Apparent diffusion coefficient for dissociation.

        D_on : float
            Apparent diffusion coefficient for association.

        k_off : float
            Kramers-corrected dissociation rate.

        k_on1 : float
            Kramers-corrected first-order association rate.

        k_on : float
            Kramers-corrected second-order association rate.

        Kd_M : float
            Dissociation constant calculated as k_off / k_on.

        Ka_M_inv : float
            Association constant calculated as k_on / k_off.

        deltaF_off : float
            Free-energy barrier for dissociation.

        deltaF_on : float
            Free-energy barrier for association.

    """

    F = np.asarray(F, dtype=float)

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    if conc <= 0:
        raise ValueError("Concentration must be positive.")

    if k_off_model <= 0:
        raise ValueError("k_off_model must be positive.")

    if k_on1_model <= 0:
        raise ValueError("k_on1_model must be positive.")

    kBT = BOLTZMANN_KCAL * temperature

    # ------------------------------------------------------------
    # Apparent diffusion coefficients
    # ------------------------------------------------------------
    D_off = k_off_star / k_off_model
    D_on = k_on1_star / k_on1_model

    # ------------------------------------------------------------
    # Free-energy barriers
    # ------------------------------------------------------------
    deltaF_off = F[barrier] - F[bound]
    deltaF_on = F[barrier] - F[unbound]

    # ------------------------------------------------------------
    # Kramers-corrected rates
    # ------------------------------------------------------------
    # xi = kBT/D
    xi_off = kBT/D_off
    xi_on = kBT/D_on

    # k_off = (2*pi*w_m*w_b)/xi * e^(-DeltaF/kBT)
    k_off = (
        (2.0 * np.pi * w_b * w_br) / xi_off
        * np.exp(-deltaF_off / kBT)
    )

    k_on1 = (
        (2.0 * np.pi * w_u * w_br) / xi_on
        * np.exp(-deltaF_on / kBT)
    )

    # Convert first-order association rate to second-order association
    k_on = k_on1 / conc

    # ------------------------------------------------------------
    # Equilibrium constants
    # ------------------------------------------------------------
    Kd_M = k_off / k_on
    Ka_M_inv = k_on / k_off

    return {
        "D_off": D_off,
        "D_on": D_on,
        "k_off": k_off,
        "k_on1": k_on1,
        "k_on": k_on,
        "Kd_M": Kd_M,
        "Ka_M_inv": Ka_M_inv,
        "deltaF_off": deltaF_off,
        "deltaF_on": deltaF_on
    }