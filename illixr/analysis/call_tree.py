"""See CallForest."""

from __future__ import annotations

import collections
import contextlib
import sqlite3
import dask.bag
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import anytree  # type: ignore
import attr
import pandas  # type: ignore
from tqdm import tqdm  # type: ignore

from .util import sort_and_set_index, to_categories


class DataIntegrityWarning(Warning):
    """Non-fatal warning that the data might be invalid."""


T = TypeVar("T")


def falsy_to_none(elem: T) -> Optional[T]:
    if elem:
        return elem
    else:
        return None


class StaticFrame(anytree.NodeMixin):  # type: ignore
    """A dynamic stack frame."""

    function_name: str
    file_name: str
    line: int
    _plugin: Optional[str]
    _topic_name: Optional[str]

    def __hash__(self) -> int:
        return hash((
            self.function_name,
            self.file_name,
            self.line,
            self.plugin,
            self.topic_name,
        )) ^ (0 if self.parent is not None else hash(self.parent))

    def __determ_hash__(self) -> Any:
        return (self.function_name, self.file_name, self.line, self.plugin, self.topic_name, self.parent)

    def __eq__(self, other: object) -> bool:
        attrs = ["function_name", "file_name", "line", "plugin", "topic_name"]
        return all(getattr(self, attr) == getattr(other, attr) for attr in attrs) and self.parent == other.parent

    def __neq__(self, other: DynamicFrame) -> bool:
        return not self == other

    def plugin_function(self, sep: str = "\n") -> str:
        plugin_str = self.plugin + sep if self.plugin else ""
        return f"{plugin_str}{self.function_name}"

    def plugin_function_topic(self, sep: str="\n") -> str:
        plugin_str = self.plugin + sep if self.plugin else ""
        topic_str = sep + self.topic_name if self.topic_name else ""
        return f"{plugin_str}{self.function_name}{topic_str}"

    def file_function_line(self, sep: str=":") -> str:
        return f"{self.file_name}{sep}{self.line}{sep}{self.function_name}"

    def __str__(self) -> str:
        return self.file_function_line()

    def __init__(
        self,
        row: pandas.Series[Any],
        parent: Optional[StaticFrame] = None,
        children: Optional[Iterable[StaticFrame]] = None,
    ) -> None:
        """Constructs a StaticFrame. See anytree.NodeMixin for parent and children."""
        self.function_name = cast(str, row["function_name"])
        self._file_name = cast(str, row["file_name"])
        self.line = cast(int, row["line"])
        self._plugin = falsy_to_none(cast(str, row["plugin"]))
        self._topic_name = falsy_to_none(cast(str, row["topic_name"]))
        self.parent = parent
        if self.function_name in {"get", "put"}:
            assert self.topic_name is not None
        if children:
            self.children = children

    @property
    def plugin(self) -> Optional[str]:
        """Returns the name of the plugin responsible for calling this static frame.."""
        if self._plugin is None:
            if self.parent is not None:
                self._plugin = self.parent.plugin
            else:
                self._plugin = None
        return self._plugin

    @property
    def topic_name(self) -> str:
        """Returns the topic of the plugin responsible for calling this static frame."""
        if self._topic_name is None:
            if self.function_name == "callback":
                self._topic_name = self.parent.parent.topic_name
            else:
                self._topic_name = None
        return self._topic_name

    @property
    def file_name(self) -> str:
        return "/".join(self._file_name.split('/')[-2:])

class DynamicFrame(anytree.NodeMixin):  # type: ignore
    """A dynamic stack frame."""

    thread_id: int
    _frame_id: int
    cpu_time: int
    wall_time: int
    wall_start: int
    wall_stop: int
    static_frame: StaticFrame
    serial_no: Optional[int]

    def __hash__(self) -> int:
        return hash((self.thread_id, self._frame_id))

    def __determ_hash__(self) -> Any:
        return (self.thread_id, self._frame_id)

    def __eq__(self, other: DynamicFrame) -> bool:
        return self.thread_id == other.thread_id and self._frame_id == other._frame_id

    def __neq__(self, other: DynamicFrame) -> bool:
        return not self == other

    def __str__(self) -> str:
        """Human-readable string representation"""
        return f"{self.thread_id} {self._frame_id}"

    def __repr__(self) -> str:
        return f"{self.thread_id=} {self._frame_id=}"

    def __init__(
        self,
        index: Tuple[int, int],
        row: pandas.Series[Any],
        static_frame: StaticFrame,
        parent: Optional[DynamicFrame] = None,
        children: Optional[Iterable[DynamicFrame]] = None,
    ) -> None:
        """Constructs a DynamicFrame. See anytree.NodeMixin for parent and children."""

        is_custom_time = row["custom_time"] != 0
        self.thread_id = index[0]
        self._frame_id = index[1]
        self.cpu_time = row["cpu_stop"] - row["cpu_start"]
        self.wall_time = row["wall_stop"] - row["wall_start"]
        self.wall_start = row["wall_start"] if not is_custom_time else row["custom_time"]
        self.wall_stop = row["wall_stop"] if not is_custom_time else row["custom_time"]
        self.static_frame = static_frame
        self.serial_no = row["serial_no"]
        self.parent = parent
        if children:
            self.children = children


_Class = TypeVar("_Class", bound="CallTree")


@attr.frozen
class CallTree:
    """Deals with the callgraph generated by cpu_timer for one thread.

    Other analysis should never read the raw dataframe of cpu_timer
    frames; Instead, they should delegate to this module. That way, I
    can easily change how cpu_timer works.

    This will be based on implementation details of
    ILLIXR/common/cpu_timer, ILLIXR/runtime/frame_logger.hpp, and the
    various implementations of FrameInfo but NOT on any other part of
    ILLIXR.

    """

    thread_id: int
    root: DynamicFrame
    calls: int
    static_to_dynamic: Mapping[StaticFrame, List[DynamicFrame]]

    @classmethod
    def from_database(
        cls: Type[_Class],
        database_url: str,
        verify: bool = False,
    ) -> Optional[_Class]:
        """Reads a CallForest from the database

        This is the "opposite" of ILLIXR/runtime/frame_logger2.hpp.

        """

        with contextlib.closing(sqlite3.connect(database_url)) as conn:
            frames = pandas.read_sql_query("SELECT * FROM finished;", conn)
            index = frames[["thread_id", "frame"]]
            dups = index.duplicated()
            assert not dups.any()
            frames2 = sort_and_set_index(frames, ["thread_id", "frame"], verify_integrity=True)
            frames3 = to_categories(frames2, ["function_name", "topic_name"])
            frames = frames3

        calls = sum(frames["cpu_start"] != 0) + sum(frames["cpu_stop"] != 0)

        if verify and not (frames["epoch"] == 0).all():
            raise RuntimeError(
                "Frames come from different epochs;" "They need to be merged."
            )

        if len(frames) == 0:
            return None

        if verify and frames.index.levels[0].nunique() != 1:
            raise RuntimeError("Frames come from different threads")
        else:
            thread_id = frames.index.levels[0][0]

        # Don't create duplicates of the dynamic and static frame
        index_to_frame: Dict[Tuple[int, int], DynamicFrame] = {}
        frame_to_static_children: Dict[
            Union[DynamicFrame, None], Dict[Tuple[str, str, str], StaticFrame]
        ] = collections.defaultdict(dict)
        static_to_dynamic: Dict[
            StaticFrame, List[DynamicFrame]
        ] = collections.defaultdict(list)
        for index, row in tqdm(
            frames.iterrows(),
            total=len(frames),
            desc=f"Reconstrucing stack {thread_id}",
            unit="frame",
        ):
            # Get parent as DynamicFrame or None
            assert (not verify) or row["caller"] == 0 or row["caller"] < index[1]
            parent = index_to_frame.get((index[0], row["caller"]), None)

            # Get StaticFrame, reusing if already exists.
            # However, it must already exist _at the same point in the stack._
            # static_children is all of the StaticFrames that exist at this point in the stack
            static_children = frame_to_static_children[
                parent.static_frame if parent is not None else None
            ]
            static_info = cast(
                Tuple[str, str, str],
                tuple(row[["function_name", "plugin", "topic_name"]]),
            )
            if static_info not in static_children:
                # Not exists; create
                static_children[static_info] = StaticFrame(
                    row, parent=parent.static_frame if parent else None
                )
            static_frame = static_children[static_info]

            frame = DynamicFrame(index, row, static_frame, parent=parent)

            # Update the index_to_frame so its children can find it.
            index_to_frame[index] = frame
            static_to_dynamic[frame.static_frame].append(frame)

        return cls(
            thread_id=thread_id,
            root=index_to_frame[(thread_id, 0)],
            static_to_dynamic=dict(static_to_dynamic),
            calls=calls,
        )

    @classmethod
    def from_metrics_dir(
        cls: Type[_Class],
        metrics: Path,
        verify: bool = False,
    ) -> Mapping[int, _Class]:
        """Returns a forest constructed from each database in the dir."""
        return dict(
            dask.bag.from_sequence((metrics / "frames").iterdir())
            .map(lambda path: cls.from_database(str(path), verify))
            .filter(lambda tree: tree is not None)
            .map(lambda tree: (tree.thread_id, tree))
            .compute()
        )
