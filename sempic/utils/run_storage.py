import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


_RUN_SUFFIX_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def validate_run_suffix(suffix: str | None) -> None:
    if suffix is not None and (
        not isinstance(suffix, str) or _RUN_SUFFIX_PATTERN.fullmatch(suffix) is None
    ):
        raise ValueError(
            "run_suffix must start with an alphanumeric character and contain only "
            "letters, digits, dots, underscores, and hyphens."
        )


def _safe_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "run"


def compose_run_suffix(
    *system_parts: str | None,
    user_suffix: str | None = None,
) -> str | None:
    validate_run_suffix(user_suffix)
    parts = [_safe_segment(part) for part in system_parts if part]
    if user_suffix is not None:
        parts.append(user_suffix)
    return "_".join(parts) or None


def allocate_run_dir(
    root: str | Path,
    suffix: str | None,
    now: datetime | None = None,
    collision: str = "increment",
) -> Path:
    validate_run_suffix(suffix)
    if collision != "increment":
        raise ValueError(f"Unsupported run directory collision policy: {collision}")

    run_root = Path(root)
    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_name = timestamp if suffix is None else f"{timestamp}_{suffix}"
    index = 0
    while True:
        name = base_name if index == 0 else f"{base_name}_{index}"
        run_dir = run_root / name
        try:
            run_dir.mkdir()
        except FileExistsError:
            index += 1
            continue
        return run_dir


def select_existing_run_dir(
    root: str | Path,
    run_dir: str | Path,
) -> Path:
    run_root = Path(root)
    resolved_root = run_root.resolve()
    selected = Path(run_dir)
    if selected.is_symlink():
        raise ValueError(f"Run directory must not be a symlink: {selected}")
    if not selected.is_dir():
        raise ValueError(f"Run directory does not exist: {selected}")
    if os.path.abspath(selected.parent) != os.path.abspath(run_root):
        raise ValueError(f"Run directory must be a direct child of {resolved_root}")
    resolved_run_dir = selected.resolve()
    if resolved_run_dir.parent != resolved_root:
        raise ValueError(f"Run directory must be a direct child of {resolved_root}")
    return resolved_run_dir


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(temporary, "w") as f:
            json.dump(value, f, indent=2)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
