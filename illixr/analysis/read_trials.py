"""Functions which load data from ILLIXR dumps.


They should **not** depend on ILLIXR's DAG or configuration. If that
is necessary to read, take the DAG-specific information as an
argument.

"""

from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

from .call_forest import CallForest
from .types import Trial, Trials


def read_trial(
    metrics: Path,
    info_cols: Mapping[str, object],
    info_readers: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
    verify: bool = False,
) -> Trial:
    """Reads all data from the dump of a single ILLIXR trial.

    This should only contain high-level function-calls.

    """

    return Trial(
        call_forest=CallForest.from_dir(metrics, info_cols, info_readers, verify),
    )


def read_trials(paths: Iterable[Path], verify: bool = False) -> Trials:
    """Reads all data from the a set of ILLIXR trials.

    This should only contain high-level function-calls.

    """
    trials = [read_trial(path, {}, {}, verify) for path in paths]
    return Trials(each=trials)
