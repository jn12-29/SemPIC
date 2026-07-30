from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SelectionAction = Literal["refresh", "customize", "select_all", "clear"]


@dataclass(frozen=True, slots=True)
class SelectionState:
    selected: tuple[str, ...]
    follow_all: bool


def reconcile_multiselect(
    options: Sequence[str],
    previous: SelectionState | None,
    action: SelectionAction = "refresh",
    user_selection: Sequence[str] = (),
) -> SelectionState:
    """Reconcile options while preserving explicit follow-all intent."""
    available = tuple(dict.fromkeys(options))
    available_set = set(available)
    if action == "select_all":
        return SelectionState(selected=available, follow_all=True)
    if action == "clear":
        return SelectionState(selected=(), follow_all=False)
    if action == "customize":
        requested = set(user_selection)
        return SelectionState(
            selected=tuple(value for value in available if value in requested),
            follow_all=False,
        )
    if action != "refresh":
        raise ValueError(f"Unsupported multiselect action: {action}")
    if previous is None:
        return SelectionState(selected=available, follow_all=True)
    if previous.follow_all:
        return SelectionState(selected=available, follow_all=True)
    return SelectionState(
        selected=tuple(value for value in previous.selected if value in available_set),
        follow_all=False,
    )


@dataclass(frozen=True, slots=True)
class DirectorySelection:
    selected: tuple[str, ...]
    follow_all: bool


def transition_directory_selection(
    discovered: Sequence[str | Path],
    previous: DirectorySelection | None,
    action: SelectionAction = "refresh",
    user_selection: Sequence[str | Path] = (),
) -> DirectorySelection:
    """Apply one directory-selection event without reading or mutating UI state."""
    available = tuple(dict.fromkeys(str(Path(path).resolve()) for path in discovered))
    normalized_previous = (
        None
        if previous is None
        else SelectionState(previous.selected, previous.follow_all)
    )
    normalized_users = tuple(str(Path(path).resolve()) for path in user_selection)
    result = reconcile_multiselect(
        available,
        normalized_previous,
        action=action,
        user_selection=normalized_users,
    )
    return DirectorySelection(result.selected, result.follow_all)
