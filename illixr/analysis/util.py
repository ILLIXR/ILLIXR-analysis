"""Functions which are **independent** of ILLIXR.

It is suitable for importing into another project.

Ideally, every ILLIXR-specific function calls ILLIXR-independent
functions with ILLIXR-specific information."""

import itertools
import sqlite3
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, TypeVar

import pandas as pd
from pandas.api.types import union_categoricals


def normalize_cats(
    dfs: List[pd.DataFrame],
    columns: Optional[List[str]] = None,
    include_indices: bool = True,
) -> List[pd.DataFrame]:
    """Normalize the categorical columns to the same dtype.

    dfs should be a list of dataframes with the same columns.

    Primarily useful before pd.concat(...)

    """
    if not dfs:
        return dfs

    if columns is None:
        if include_indices:
            # peel of indices into columns so that I can mutate them
            indices = [df.index for df in dfs]
            dfs = [df.reset_index() for df in dfs]
        columns2 = [
            column
            for column, dtype in dfs[0].dtypes.iteritems()
            if isinstance(dtype, pd.CategoricalDtype)
        ]
    else:
        columns2 = columns

    columns_union_cat = {
        column: union_categoricals(
            [df[column] for df in dfs], ignore_order=True, sort_categories=True
        ).as_ordered()
        for column in columns2
    }

    dfs = [
        df.assign(
            **{
                column: pd.Categorical(df[column], categories=union_cat.categories)
                for column, union_cat in columns_union_cat.items()
            }
        )
        for df in dfs
    ]

    if columns is None and include_indices:
        dfs = [
            df.reset_index(drop=True).set_index(index)
            for df, index in zip(dfs, indices)
        ]

    return dfs


def pd_read_sqlite_table(
    database: Path,
    table: str,
    index_cols: List[str],
    verify: bool = False,
) -> pd.DataFrame:
    """Reads a whole table from a sqlite3 database."""
    conn = sqlite3.connect(str(database))
    return (
        pd.read_sql_query(f"SELECT * FROM {table};", conn)
        .sort_values(index_cols, inplace=False)
        .set_index(index_cols, verify_integrity=verify)
        .sort_index(inplace=False)
    )


def command_exists(command: str) -> bool:
    """Test if `command` is found on the path."""
    return (
        subprocess.run(["which", command], check=False, capture_output=True).returncode
        == 0
    )


T = TypeVar("T")


def flatten(its: Iterable[Iterable[T]]) -> Iterable[T]:
    """Flatten an iterable of iterable of T to just iterable of T"""
    return itertools.chain.from_iterable(its)
