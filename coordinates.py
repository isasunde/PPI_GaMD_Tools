"""Definitions and utilities for supported reaction coordinates."""

from typing import Literal

import pandas as pd


ReactionCoordType = Literal[
    "min_distance",
    "distance",
    "contacts",
    "rmsd",
    "x-dist",
    "y-dist",
    "z-dist",
]


COORD_COLUMNS = {
    "min_distance": ("Frame", "dist_A", "dist_B"),
    "distance": ("Frame", "dist"),
    "contacts": (
        "Frame",
        "Native",
        "Non_native",
        "min_dist",
        "max_dist",
    ),
    "rmsd": ("Frame", "rmsd"),
    "x-dist": ("Frame", "Vx", "Vy", "Vz", "Ox", "Oy", "Oz"),
    "y-dist": ("Frame", "Vx", "Vy", "Vz", "Ox", "Oy", "Oz"),
    "z-dist": ("Frame", "Vx", "Vy", "Vz", "Ox", "Oy", "Oz"),
}


COORD_LABELS = {
    "min_distance": "Minimum distance (Å)",
    "distance": "Distance (Å)",
    "contacts": "Number of contacts",
    "rmsd": "RMSD (Å)",
    "x-dist": "X distance (Å)",
    "y-dist": "Y distance (Å)",
    "z-dist": "Z distance (Å)",
}

def construct_reaction_coord(
    data: pd.DataFrame,
    coord_type: ReactionCoordType,
) -> pd.Series:
    """Construct a reaction coordinate from cpptraj data."""

    if coord_type == "min_distance":
        coord = data[["dist_A", "dist_B"]].min(axis=1)

    elif coord_type == "distance":
        coord = data["dist"]

    elif coord_type == "contacts":
        coord = data["Native"] + data["Non_native"]

    elif coord_type == "rmsd":
        coord = data["rmsd"]
    elif coord_type == "x-dist":
        coord = data["Vx"]
    elif coord_type == "y-dist":
        coord = data["Vy"]
    elif coord_type == "z-dist":
        coord = data["Vz"]

    else:
        raise ValueError(
            f"Unknown reaction-coordinate type: {coord_type}"
        )

    return coord