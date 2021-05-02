"""Types that will be needed in multiple modules."""

from pathlib import Path
from typing import Dict, List, Mapping, Any

import attr

from .call_tree import CallTree


def conditions2label(conditions: Mapping[str, Any]) -> str:
    label = f"{conditions['scheduler']} {conditions['cpus']}x{conditions['cpu_freq']:.1f}GHz"
    return label

@attr.frozen
class Trial:
    """A single trial of ILLIXR."""

    call_trees: Mapping[int, CallTree]
    output_dir: Path
    config: Mapping[str, Any]
    output: Dict[str, Any] = {}

    def __cache_key__(self) -> Any:
        return None

    def __cache_val__(self) -> Any:
        return self.output_dir

@attr.frozen
class Trials:
    """A collection of trials of ILLIXR, to be analyzed together."""

    each: List[Trial]
    output_dir: Path
    output: Dict[str, Any] = {}

    def __cache_key__(self) -> Any:
        return None

    def __cache_val__(self) -> Any:
        return self.output_dir
