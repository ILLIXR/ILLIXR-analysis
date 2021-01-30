from typing import List

import attr

from .call_forest import CallForest


@attr.s
class Trial:
    call_forest: CallForest = attr.ib()


@attr.s
class Trials:
    trials: List[Trial] = attr.ib()
