"""Functions which load data from ILLIXR dumps.


They should **not** depend on ILLIXR's DAG or configuration. If that
is necessary to read, take the DAG-specific information as an
argument.

"""

from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd  # type: ignore

from .call_forest import CallForest
from .types import Trial, Trials
from .util import normalize_cats, pd_read_sqlite_table


def read_frames(database: Path, verify: bool = False) -> pd.DataFrame:
    """Reads the cpu_timer data."""
    strings = pd_read_sqlite_table(database, "strings", ["address"], verify)
    frames = pd_read_sqlite_table(database, "finished", ["tid", "id"], verify)

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


def read_trial(
    path: Path,
    info_readers: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
    verify: bool = False,
) -> Trial:
    """Reads all data from the dump of a single ILLIXR trial."""
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


def read_trials(paths: Iterable[Path], verify: bool = False) -> Trials:
    """Reads all data from the a set of ILLIXR trials."""
    trials = [read_trial(path, {}, verify) for path in paths]
    return Trials(each=trials)
