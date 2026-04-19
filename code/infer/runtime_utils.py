from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def progress_bar(
    iterable: Iterable,
    *,
    total: Optional[int] = None,
    desc: str = "Progress",
    unit: str = "item",
):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
    )


def print_kv_block(title: str, values: Mapping[str, object]) -> None:
    print(f"\n=== {title} ===")
    for key, value in values.items():
        print(f"{key:<18}: {value}")


def print_output_block(title: str, outputs: Mapping[str, object]) -> None:
    print(f"\n=== {title} ===")
    for key, value in outputs.items():
        if value in (None, ""):
            continue
        print(f"{key:<18}: {value}")


def ensure_parent(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
