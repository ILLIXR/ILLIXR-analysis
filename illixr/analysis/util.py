"""Functions which are **independent** of ILLIXR.

It is suitable for importing into another project.

Ideally, every ILLIXR-specific function calls ILLIXR-independent
functions with ILLIXR-specific information."""

import abc
import itertools
import subprocess
from typing import Any, Callable, Iterable, List, Optional, TypeVar
import collections
import random
import contextlib
from enum import Enum
from pathlib import Path
import shutil
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Any,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import warnings
import itertools
import io
import multiprocessing

import anytree  # type: ignore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import numpy as np
import pygraphviz  # type: ignore
import pandas as pd  # type: ignore
from pandas.api.types import union_categoricals  # type: ignore
from typing_extensions import Protocol


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


def command_exists(command: str) -> bool:
    """Test if `command` is found on the path."""
    return (
        subprocess.run(["which", command], check=False, capture_output=True).returncode
        == 0
    )


_T = TypeVar("_T")


def flatten(its: Iterable[Iterable[_T]]) -> Iterable[_T]:
    """Flatten an iterable of iterables to just iterable"""
    return itertools.chain.from_iterable(its)


def compose_all(fns: Iterable[Callable[[_T], _T]]) -> Callable[[_T], _T]:
    """Compose all functions; the output of one is input to next."""

    def ret(elem: _T) -> _T:
        for fn in fns:
            elem = fn(elem)
        return elem

    return ret


_C = TypeVar("_C", bound="Comparable")


class Comparable(Protocol):
    """Analog of java.lang.Comparable"""

    # pylint: disable=missing-function-docstring

    @abc.abstractmethod
    def __eq__(self, other: Any) -> bool:
        ...

    @abc.abstractmethod
    def __lt__(self: _C, other: _C) -> bool:
        ...

    @abc.abstractmethod
    def __gt__(self: _C, other: _C) -> bool:
        ...

    @abc.abstractmethod
    def __le__(self: _C, other: _C) -> bool:
        ...

    @abc.abstractmethod
    def __ge__(self: _C, other: _C) -> bool:
        ...


def clip(elem: _C, lower: _C, upper: _C) -> _C:
    """Returns an element between lower and upper.

    Analog of numpy.clip for scalars.

    """
    if elem <= lower:
        return lower
    elif elem >= upper:
        return upper
    else:
        return elem


def sort_and_set_index(
    df: pd.DataFrame, columns: List[str], verify_integrity: bool = False
) -> pd.DataFrame:
    """Sets columns as a uniuqe, sorted index.

    Can use with df.pipe

    """
    return (
        df.sort_values(columns, inplace=False)
        .set_index(columns, verify_integrity=verify_integrity)
        .sort_index(inplace=False)
    )


def to_categories(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Converts columns of df to categories.

    Can use with df.pipe.

    """
    return df.assign(**{column: df[column].astype("category") for column in columns})


def set_index_with_uniquifier(
    df: pd.DataFrame, columns: List[str], uniquifier_column: str
) -> pd.DataFrame:
    """Genreates uniqueifier_column and sets columns + [uniquifier_column] as a unique, sorted index.

    Can use with df.pipe.

    """
    return (
        df.sort_values(columns, kind="mergesort")
        .set_index(columns)
        .assign(**{uniquifier_column: 0})
        .assign(
            **{
                uniquifier_column: lambda df: (
                    df[uniquifier_column]
                    .groupby(by=range(len(columns)))
                    .transform(lambda series: range(len(series)))
                )
            }
        )
        .set_index(uniquifier_column, append=True)
        .sort_index(inplace=False)
    )

def summary_stats(data: np.array, digits: int = 1, percentiles: List[float] = [25, 75, 90, 95]) -> str:
    percentiles_str = " " + " ".join(
        f"[{percentile}%]={np.percentile(data, percentile):,.{digits}f}"
        for percentile in percentiles
    )
    with np.errstate(invalid="ignore"):
        return f"{data.mean():,.{digits}f} +/- {data.std():,.{digits}f} ({data.std() / data.mean() * 100:.0f}%) med={np.median(data):,.{digits}f} count={len(data)}{percentiles_str}"

def right_pad(text: str, length: int) -> str:
    return text + " " * max(0, length - len(text))

Key = TypeVar("Key")
Val = TypeVar("Val")
def dict_concat(dicts: Iterable[Mapping[Key, Val]]) -> Mapping[Key, Val]:
    return {
        key: val
        for dict in dicts
        for key, val in dict.items()
    }

def write_dir(content_map: Mapping[Path, Any]) -> None:
    for path, content in content_map.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.rmtree(path)
        if isinstance(content, str):
            path.write_text(content)
        elif isinstance(content, bytes):
            path.write_bytes(content)
        elif isinstance(content, dict):
            write_dir({path / key: subcontent for key, subcontent in content.items()})
        else:
            raise TypeError(type(content))

def histogram(
        ys: np.array,
        xlabel: str,
        title: str,
        bins: int = 50,
        cloud: bool = True,
        logy: bool = True,
        grid: bool = False,
) -> bytes:
    fake_file = io.BytesIO()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(ys, bins=bins, align='mid')
    if cloud:
        ax.plot(ys, np.random.randn(*ys.shape) * (ax.get_ylim()[1] * 0.2) + (ax.get_ylim()[1] * 0.5) * np.ones(ys.shape), linestyle='', marker='.', ms=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Occurrences (count)")
    if grid:
        ax.grid(True, which="major", axis="both")
    if logy:
        ax.set_yscale("log")
    fig.savefig(fake_file)
    plt.close(fig)
    return fake_file.getvalue()

def timeseries(
        ts: np.array,
        ys: np.array,
        title: str,
        ylabel: str,
        series_label: Optional[str] = None,
        grid: bool = False,
) -> bytes:
    fake_file = io.BytesIO()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.set_xlabel("Time since start (sec)")
    if grid:
        ax.grid(True, which="major", axis="both")
    if len(ts) == len(ys) + 1:
        ts = ts[:-1]
    if len(ts) == len(ys) - 1:
        ys = ys[:-1]
    ax.plot((ts - ts[0]) / 1e9, ys, label=series_label)
    fig.savefig(fake_file)
    plt.close(fig)
    return fake_file.getvalue()

# TODO: replace with toolz
A = TypeVar("A")
B = TypeVar("B")
def second(pair: Tuple[A, B]) -> B:
    return pair[1]

T = TypeVar("T")


def chunker(it: Iterable[T], size: int) -> Iterable[List[T]]:
    """chunk input into size or less chunks
shamelessly swiped from Lib/multiprocessing.py:Pool._get_task"""
    it = iter(it)
    while True:
        x = list(itertools.islice(it, size))
        if not x:
            return
        yield x

# import tracemalloc
# tracemalloc.start(25)

biggest_offenders = []

def track_memory_usage():
    def decorator(function):
        def inner_function(*args, **kwargs):
            snapshot1 = tracemalloc.take_snapshot()
            ret = function(*args, **kwargs)
            snapshot2 = tracemalloc.take_snapshot()
            diffs = snapshot2.compare_to(snapshot1, 'lineno')
            print(str(function))
            for stat in diffs[:10]:
                print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
                for line in stat.traceback.format():
                    print(line)
            return ret
        return inner_function
    return decorator

def omit(dct: Mapping[Key, Val], keys: Set[Key]) -> Mapping[Key, Val]:
    return {key: val for key, val in dct.items() if key not in keys}

def undefault_dict(dct):
    if isinstance(dct, (dict, collections.defaultdict)):
        return dict((key, undefault_dict(val)) for key, val in dct.items())
    else:
        return dct
