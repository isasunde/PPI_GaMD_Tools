"""Input functions for GaMD and cpptraj data."""

from pathlib import Path

import pandas as pd

from .coordinates import (
    COORD_COLUMNS,
    ReactionCoordType,
    construct_reaction_coord,
)

def load_gamd_log(
    log_file: str | Path,
) -> pd.DataFrame:
    """Load a GaMD log file into a pandas DataFrame."""
    column_names = (
        "ntx",
        "step",
        "Vp",
        "Vd",
        "fw_tot",
        "fw_d",
        "deltaVp",
        "deltaVd",
    )

    gamd_df = pd.read_csv(
        log_file,
        sep=r"\s+",
        #comment="#",
        header=2,
        names=column_names,
    )

    if gamd_df.empty:
        raise ValueError(
            f"No GaMD data found in '{log_file}'."
        )

    if gamd_df.isna().any().any():
        raise ValueError(
            f"Missing or malformed data detected in '{log_file}'."
        )
    
    return gamd_df


def load_reaction_coord(
    coord_file: str | Path,
    coord_type: ReactionCoordType,
) -> pd.Series:
    """Load a reaction coordinate from cpptraj output."""
    try:
        columns = COORD_COLUMNS[coord_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown reaction-coordinate type: {coord_type}"
        ) from exc

    coord_df = pd.read_csv(
        coord_file,
        sep=r"\s+",
        comment="#",
        header=None,
        names=columns,
    )

    if coord_df.empty:
        raise ValueError(
            f"No reaction-coordinate data found in '{coord_file}'."
        )

    if coord_df.isna().any().any():
        raise ValueError(
            f"Missing or malformed data detected in '{coord_file}'."
        )

    return construct_reaction_coord(
        coord_df,
        coord_type,
    )