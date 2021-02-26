"""Functions which are **independent** of ILLIXR.

It is suitable for importing into another project.

Ideally, every ILLIXR-specific function calls ILLIXR-independent
functions with ILLIXR-specific information."""

import abc
import itertools
import subprocess
from typing import Any, Callable, Iterable, List, Optional, TypeVar

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
