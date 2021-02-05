"""Types that will be needed in multiple modules."""

from pathlib import Path
from typing import List, Mapping

import attr

from .call_tree import CallTree


@attr.frozen
class Trial:
    """A single trial of ILLIXR."""

    call_trees: Mapping[int, CallTree] = attr.ib()
    output_dir: Path


@attr.frozen
class Trials:
    """A collection of trials of ILLIXR, to be analyzed together."""

    each: List[Trial] = attr.ib()
