"""Functions which load data from ILLIXR dumps.


They should **not** depend on ILLIXR's DAG or configuration. If that
is necessary to read, take the DAG-specific information as an
argument.

"""

from pathlib import Path
from typing import Iterable

import charmonium.cache as ch_cache

from .call_tree import CallTree
from .types import Trial, Trials

CACHE_PATH = Path(".cache")


@ch_cache.decor(ch_cache.FileStore.create(CACHE_PATH))
def read_trial(
    metrics: Path,
    verify: bool = False,
) -> Trial:
    """Reads all data from the dump of a single ILLIXR trial.

    This should only contain high-level function-calls.

    """

    return Trial(
        call_trees=CallTree.from_metrics_dir(metrics, verify),
        output_dir=metrics,
    )


def read_trials(paths: Iterable[Path], verify: bool = False) -> Trials:
    """Reads all data from the a set of ILLIXR trials.

    This should only contain high-level function-calls.

    """
    trials = [read_trial(path, verify) for path in paths]
    return Trials(each=trials)
