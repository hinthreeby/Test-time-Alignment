"""Import-path bootstrap for relocated PARM-TARO and Router V2 sources.

This module changes launch/runtime plumbing only.  It neither copies nor
modifies either source tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping


def required_python_paths(project_root: Path) -> tuple[Path, Path, Path]:
    """Return the frozen source search order required by this experiment."""
    root = project_root.resolve()
    paths = (
        root / "Method/Router_Apdative",
        root / "Method/PARM_TARO",
        root,
    )
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise RuntimeError(f"Required source parent path(s) missing: {missing}")
    router = paths[0] / "router_v2/__init__.py"
    if not router.is_file():
        raise RuntimeError(f"Canonical router_v2 package missing: {router}")
    return paths


def bootstrap_import_paths(project_root: Path) -> tuple[str, ...]:
    """Prepend canonical source parents to this interpreter's ``sys.path``."""
    ordered = tuple(str(path) for path in required_python_paths(project_root))
    retained = [entry for entry in sys.path if entry not in ordered]
    sys.path[:] = [*ordered, *retained]
    return ordered


def subprocess_environment(
    project_root: Path,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an environment whose children inherit the same source order."""
    environment = dict(os.environ if source is None else source)
    ordered = [str(path) for path in required_python_paths(project_root)]
    existing = environment.get("PYTHONPATH", "")
    for entry in existing.split(os.pathsep):
        if entry and entry not in ordered:
            ordered.append(entry)
    environment["PYTHONPATH"] = os.pathsep.join(ordered)
    return environment
