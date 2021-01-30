import sqlite3
from pathlib import Path
from typing import Iterable, List

import pandas as pd  # type: ignore

from .call_forest import CallForest
from .types import Trial, Trials
from .util import normalize_cats


def read_sqlite(
    database: Path,
    table: str,
    index_cols: List[str],
    verify: bool = False,
) -> pd.DataFrame:
    conn = sqlite3.connect(str(database))
    return (
        pd.read_sql_query(f"SELECT * FROM {table};", conn)
        .sort_values(index_cols)
        .set_index(index_cols, verify_integrity=verify)
        .sort_index()
    )


def read_frames(database: Path, verify: bool = False) -> pd.DataFrame:
    strings = read_sqlite(database, "strings", ["address"], verify)
    frames = read_sqlite(database, "finished", ["tid", "id"], verify)

    return (
        frames.join(strings, on="function_name", how="left")
        .drop(columns=["function_name"])
        .assign(**{"function_name": lambda df: pd.Categorical(df["string"])})
        .drop(columns=["string"])
        .join(strings, on="file_name", how="left")
        .drop(columns=["file_name"])
        .assign(**{"file_name": lambda df: pd.Categorical(df["string"])})
        .drop(columns=["string"])
    )


def read_trial(path: Path, verify: bool = False) -> Trial:
    return Trial(
        call_forest=CallForest(
            pd.concat(
                normalize_cats(
                    [read_frames(database) for database in (path / "frames").iterdir()],
                    include_indices=False,
                ),
                verify_integrity=verify,
            )
        )
    )


def read_trials(paths: Iterable[Path]) -> Trials:
    trials = [read_trial(path) for path in paths]
    return Trials(trials=trials)
