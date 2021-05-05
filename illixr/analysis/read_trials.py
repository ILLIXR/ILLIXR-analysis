"""Functions which load data from ILLIXR dumps.


They should **not** depend on ILLIXR's DAG or configuration. If that
is necessary to read, take the DAG-specific information as an
argument.

"""

import functools
import itertools
from pathlib import Path
from typing import Iterable, Mapping
import multiprocessing

import yaml
import dask.bag

from .call_tree import CallTree
from .types import Trial, Trials
from charmonium.cache import memoize, MemoizedGroup
import charmonium.time_block as ch_time_block

group = MemoizedGroup(size="20GiB")

@memoize(group=group)
def read_trial(
    metrics: Path,
    verify: bool,
) -> Trial:
    """Reads all data from the dump of a single ILLIXR trial.

    This should only contain high-level function-calls.

    """
    config = yaml.load((metrics / "config.yaml").read_text(), Loader=yaml.SafeLoader)
    return Trial(
        call_trees=CallTree.from_metrics_dir(metrics, verify),
        output_dir=metrics,
        config=config,
    )


@ch_time_block.decor()
def read_trials(metrics_dirs: Iterable[Path], output_dir: Path, verify: bool = False) -> Trials:
    """Reads all data from the a set of ILLIXR trials.

    This should only contain high-level function-calls.

    """

    # trials = [read_trial(path, verify) for path in metrics_dirs]
    # trials = list(multiprocessing.Pool().map(functools.partial(read_trial, verify=False), metrics_dirs))
    trials = (
        dask.bag.from_sequence(metrics_dirs)
        .map(lambda path: (print(path), read_trial(path, verify))[1])
        .compute()
    )
    read_trial.group._index_write()
    return Trials(each=trials, output_dir=output_dir)
