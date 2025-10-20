"""Utility functions for the Soccer-Project.

This module provides small helpers to extract columns from DataFrames and
lists of statistics, and to build concatenated matrices used for clustering.

Improvements made:
- Follow PEP8 (naming, spacing, imports).
- Use idiomatic numpy / pandas operations instead of repeated np.append.
- Add type hints and docstrings.
- Keep original behaviour (return numpy arrays) but more efficient.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


# Notes about seasons indexing used elsewhere in the project:
# 0-20   -> serie A 2020
# 20-40  -> serie A 2019
# 40-60  -> serie A 2018
# 60-80  -> serie A 2017
# 80-100 -> serie A 2016
# 100-120-> serie A 2015
# 120-140-> serie A 2014


def column_as_array(df: pd.DataFrame, column: int) -> np.ndarray:
    """Return a DataFrame column as a 1-D numpy array.

    Args:
        df: pandas DataFrame with at least `column+1` columns.
        column: integer index of the column to extract.

    Returns:
        1-D numpy array containing the column values.
    """
    return df.iloc[:, column].to_numpy()


def get_G(df: pd.DataFrame) -> np.ndarray:
    """Goals column (originally column 5).

    Kept for backward compatibility with previous API (function `G`).
    """
    return column_as_array(df, 5)


def get_GA(df: pd.DataFrame) -> np.ndarray:
    """Goals against column (originally column 6)."""
    return column_as_array(df, 6)


def get_PTS(df: pd.DataFrame) -> np.ndarray:
    """Points column (originally column 7)."""
    return column_as_array(df, 7)


def get_xG(df: pd.DataFrame) -> np.ndarray:
    """Expected goals column (originally column 8)."""
    return column_as_array(df, 8)


def get_NPxG(df: pd.DataFrame) -> np.ndarray:
    """Non-penalty expected goals column (originally column 9)."""
    return column_as_array(df, 9)


def get_xGA(df: pd.DataFrame) -> np.ndarray:
    """Expected goals against (originally column 10)."""
    return column_as_array(df, 10)


def get_NPxGA(df: pd.DataFrame) -> np.ndarray:
    """Non-penalty expected goals against (originally column 11)."""
    return column_as_array(df, 11)


def get_NPxGD(df: pd.DataFrame) -> np.ndarray:
    """Non-penalty expected goal difference (originally column 12)."""
    return column_as_array(df, 12)


def get_PPDA(df: pd.DataFrame) -> np.ndarray:
    """PPDA metric (originally column 13)."""
    return column_as_array(df, 13)


def get_OPPDA(df: pd.DataFrame) -> np.ndarray:
    """Opponent PPDA (originally column 14)."""
    return column_as_array(df, 14)


def get_DC(df: pd.DataFrame) -> np.ndarray:
    """DC metric (originally column 15)."""
    return column_as_array(df, 15)


def get_ODC(df: pd.DataFrame) -> np.ndarray:
    """Opponent DC metric (originally column 16)."""
    return column_as_array(df, 16)


def champions_team(lista_df: Sequence[pd.DataFrame]) -> np.ndarray:
    """Return the top 4 teams for each DataFrame in lista_df.

    The original implementation took df.iloc[0:4,1:] for each df. This
    returns a concatenated numpy array with those slices stacked along axis 0.
    """
    slices: List[np.ndarray] = []
    for df in lista_df:
        sel = df.iloc[0:4, 1:]
        slices.append(sel.to_numpy())
    if slices:
        return np.vstack(slices)
    return np.empty((0, 0))


def retrocesse(lista_df: Sequence[pd.DataFrame]) -> np.ndarray:
    """Return the relegated teams slices for each DataFrame.

    Original code used rows starting at index 17 and taking 3 rows (17:20).
    We follow the same behaviour and concatenate results.
    """
    slices: List[np.ndarray] = []
    for df in lista_df:
        sel = df.iloc[17:20, 1:]
        slices.append(sel.to_numpy())
    if slices:
        return np.vstack(slices)
    return np.empty((0, 0))


def get_stats(lista_stats: Sequence[dict], campo1: str, campo2: str, campo3: str) -> np.ndarray:
    """Extract a nested stat field from a list of stats dicts.

    Expects lista_stats to be an iterable of mappings where
    lista_stats[i][campo1][campo2][campo3] exists for each i.
    The original code looped over range(140); we iterate over the provided
    list length to be more robust.
    """
    return np.array([s[campo1][campo2][campo3] for s in lista_stats])


def get_stats_against(lista_stats: Sequence[dict], campo1: str, campo2: str, campo3: str) -> np.ndarray:
    """Extract an "against" nested stat field from a list of stats dicts."""
    return np.array([s[campo1][campo2]["against"][campo3] for s in lista_stats])


def RealClusterCreator(X: np.ndarray, cl1: int, cr1: int) -> np.ndarray:
    """Create a concatenated matrix composed of repeating season slices.

    The function concatenates six slices of X, each offset by 20 rows,
    starting from cl1:cr1. This mirrors the original implementation.

    Args:
        X: 2-D numpy array with at least (cr1 + 120) rows.
        cl1: start index of slice for the first season.
        cr1: end index of slice for the first season.

    Returns:
        Concatenated 2-D numpy array.
    """
    slices = [X[cl1:cr1, :], X[cl1 + 20:cr1 + 20, :], X[cl1 + 40:cr1 + 40, :],
              X[cl1 + 60:cr1 + 60, :], X[cl1 + 80:cr1 + 80, :], X[cl1 + 100:cr1 + 100, :],
              X[cl1 + 120:cr1 + 120, :]]
    return np.concatenate(slices, axis=0)
