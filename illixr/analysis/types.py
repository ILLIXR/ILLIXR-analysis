"""Types that will be needed in multiple modules."""

from typing import List

import attr

from .call_forest import CallForest


@attr.frozen
class Trial:
    """A single trial of ILLIXR."""

    call_forest: CallForest = attr.ib()


@attr.frozen
class Trials:
    """A collection of trials of ILLIXR, to be analyzed together."""

    each: List[Trial] = attr.ib()
