"""See CallForest."""

from __future__ import annotations

import collections
import contextlib
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple, Type, TypeVar, Union, cast

import anytree  # type: ignore
import attr
import pandas  # type: ignore
from tqdm import tqdm  # type: ignore

from .util import sort_and_set_index, to_categories


class DataIntegrityWarning(Warning):
    """Non-fatal warning that the data might be invalid."""


class StaticFrame(anytree.NodeMixin):  # type: ignore
    """A dynamic stack frame."""

    _function_name: str
    _plugin_id: int
    _topic_name: Optional[str]

    def __str__(self) -> str:
        """Human-readable string representation"""
        ret = f"{self.function_name}"
        if self.plugin_id:
            ret += f", plugin {self.plugin_id}"
        if self.topic_name:
            ret += f", topic {self.topic_name}"
        return ret

    def __init__(
        self,
        function_name: str,
        plugin_id: int,
        topic_name: str,
        parent: Optional[StaticFrame] = None,
        children: Optional[Iterable[StaticFrame]] = None,
    ) -> None:
        """Constructs a StaticFrame. See anytree.NodeMixin for parent and children."""
        self._function_name = function_name
        self._plugin_id = plugin_id
        self._topic_name = topic_name
        self.parent = parent
        if children:
            self.children = children


class DynamicFrame(anytree.NodeMixin):  # type: ignore
    """A dynamic stack frame."""

    thread_id: int
    _frame_id: int
    static_frame: StaticFrame

    def __str__(self) -> str:
        """Human-readable string representation"""
        return f"{self.thread_id} {self._frame_id}"

    def __init__(
        self,
        thread_id: int,
        frame_id: int,
        static_frame: StaticFrame,
        parent: Optional[DynamicFrame] = None,
        children: Optional[Iterable[DynamicFrame]] = None,
    ) -> None:
        """Constructs a DynamicFrame. See anytree.NodeMixin for parent and children."""
        self.thread_id = thread_id
        self._frame_id = frame_id
        self.static_frame = static_frame
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

    thread_id: int = attr.ib()
    root: DynamicFrame = attr.ib()

    @classmethod
    def from_database(
        cls: Type[_Class],
        database_url: str,
        verify: bool = False,
    ) -> _Class:
        """Reads a CallForest from the database

        This is the "opposite" of ILLIXR/runtime/frame_logger2.hpp.

        """

        with contextlib.closing(sqlite3.connect(database_url)) as conn:
            frames = (
                pandas.read_sql_query("SELECT * FROM finished;", conn).pipe(
                    sort_and_set_index, ["thread_id", "frame"], verify_integrity=True
                )
                # sorting by values first helps the multiindex group data into levels
                .pipe(to_categories, ["function_name", "topic_name"])
                # Omitting "file_name"
            )

        if not (frames["epoch"] == 0).all():
            raise RuntimeError(
                "Frames come from different epochs;" "They need to be merged."
            )

        if frames.index.levels[0].nunique() != 1:
            raise RuntimeError("Frames come from different threads")
        else:
            thread_id = frames.index.levels[0][0]

        # Don't create duplicates of the dynamic and static frame
        frame_to_parent: Dict[Tuple[int, int], DynamicFrame] = {}
        frame_to_static_children: Dict[
            Union[DynamicFrame, None], Dict[Tuple[str, int, str], StaticFrame]
        ] = collections.defaultdict(dict)
        for (thread_id, frame_id), row in tqdm(
            frames.iterrows(),
            total=len(frames),
            desc=f"Reconstrucing stack {thread_id}",
            unit="frame",
        ):
            # Get parent as DynamicFrame or None
            parent = frame_to_parent.get((thread_id, frame_id), None)

            # Get StaticFrame, reusing if already exists.
            # However, it must already exist _at the same point in the stack._
            # static_children is all of the StaticFrames that exist at this point in the stack
            static_children = frame_to_static_children[parent]
            static_info = cast(
                Tuple[str, int, str],
                tuple(row[["function_name", "plugin_id", "topic_name"]]),
            )
            if static_info not in static_children:
                # Not exists; create
                static_children[static_info] = StaticFrame(
                    *static_info, parent=parent.static_frame if parent else None
                )
            static_frame = static_children[static_info]

            frame = DynamicFrame(thread_id, frame_id, static_frame, parent=parent)

            # Update the frame_to_parent so its children can find it.
            frame_to_parent[(thread_id, frame_id)] = frame

        return cls(
            thread_id=thread_id,
            root=frame_to_parent[(thread_id, 0)],
        )

    @classmethod
    def from_metrics_dir(
        cls: Type[_Class],
        metrics: Path,
        verify: bool = False,
    ) -> Mapping[int, _Class]:
        """Returns a forest constructed from each database in the dir."""
        database_paths = list((metrics / "frames").iterdir())
        trees = (
            cls.from_database(str(database_path), verify)
            for database_path in tqdm(
                database_paths,
                total=len(database_paths),
                desc=f"Loading frames database {metrics!s}",
                unit="thread",
            )
        )
        return {tree.thread_id: tree for tree in trees}
