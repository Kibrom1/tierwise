"""Turn a diff into TaskSignals.

Signals were the gap a new user hit first: the router asks for a file count, a
line count and three judgement calls, and nothing produced any of them. Two of
those numbers are sitting in ``git diff`` already, and two more of the signals
can be *read* rather than guessed -- whether the change creates new files or
edits existing ones is exactly what "greenfield" and "needs existing context"
mean.

What this module will not do is invent the rest. ``ambiguity`` and
``dependency_depth`` are not in a diff at any resolution, so they stay explicit
parameters that default to "no signal". A confident-looking number derived from
nothing would be worse than an honest zero, because the router would weight it.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .models import TaskSignals, Tier

#: git's rename notation in --numstat paths: "{old => new}/file" or "old => new".
_RENAME_BRACED = re.compile(r"\{[^{}]*=>\s*([^{}]*)\}")


@dataclass
class DiffStat:
    """What a diff actually says, before any interpretation."""

    files: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    paths: list[str] = field(default_factory=list)
    #: git status letters, when available: A added, M modified, D deleted,
    #: R renamed, C copied.
    statuses: list[str] = field(default_factory=list)
    binary_files: int = 0

    @property
    def lines_changed(self) -> int:
        return self.lines_added + self.lines_removed

    @property
    def is_all_new(self) -> bool:
        """Every file is newly added -- nothing existing to understand."""
        return bool(self.statuses) and all(s == "A" for s in self.statuses)

    @property
    def touches_existing(self) -> bool:
        """Some file that already existed is modified, deleted or renamed."""
        return any(s in {"M", "D", "R", "C"} for s in self.statuses)


def _clean_path(raw: str) -> str:
    """Undo git's rename notation so a path is a path."""
    if "{" in raw and "=>" in raw:
        return re.sub(_RENAME_BRACED, r"\1", raw).replace("//", "/")
    if " => " in raw:
        return raw.split(" => ", 1)[1]
    return raw


def parse_numstat(text: str) -> DiffStat:
    """Parse ``git diff --numstat`` output.

    Binary files appear as ``-\\t-\\tpath``: they count as a touched file and
    contribute no lines, which is the truthful reading -- a 4 MB PNG is not
    400,000 lines of complexity.
    """
    stat = DiffStat()
    for row in text.splitlines():
        parts = row.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        added, removed, raw_path = parts[0], parts[1], parts[-1]
        stat.files += 1
        stat.paths.append(_clean_path(raw_path))
        if added == "-" or removed == "-":
            stat.binary_files += 1
            continue
        if added.isdigit():
            stat.lines_added += int(added)
        if removed.isdigit():
            stat.lines_removed += int(removed)
    return stat


def parse_name_status(text: str) -> list[str]:
    """Parse ``git diff --name-status`` into bare status letters."""
    statuses = []
    for row in text.splitlines():
        parts = row.split("\t")
        if not parts or not parts[0]:
            continue
        statuses.append(parts[0][0].upper())   # R100 -> R, M -> M
    return statuses


def run_git_diff(
    ref: str = "HEAD~1",
    cwd: str | Path | None = None,
    paths: Optional[Sequence[str]] = None,
    staged: bool = False,
) -> DiffStat:
    """Read a diff from a real repository.

    ``staged=True`` reads the index instead of a ref, which is the useful form
    in a pre-commit hook: route the work you are about to ask for, not the work
    you last finished.
    """
    base = ["git", "diff"] + (["--cached"] if staged else ([ref] if ref else []))
    tail = ["--"] + list(paths) if paths else []

    numstat = _git(base + ["--numstat"] + tail, cwd)
    stat = parse_numstat(numstat)
    stat.statuses = parse_name_status(_git(base + ["--name-status"] + tail, cwd))
    return stat


def _git(args: list[str], cwd: str | Path | None) -> str:
    result = subprocess.run(
        args, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def signals_from_stat(
    stat: DiffStat,
    description: str = "",
    category: Optional[str] = None,
    ambiguity: float = 0.0,
    dependency_depth: int = 0,
    requires_context: Optional[bool] = None,
    is_greenfield: Optional[bool] = None,
    tier_hint: Optional[Tier] = None,
    min_tier: Optional[Tier] = None,
) -> TaskSignals:
    """Build TaskSignals from a parsed diff plus what only you can supply.

    ``requires_context`` and ``is_greenfield`` are read from the file statuses
    when git provided them, and left alone when you pass them explicitly.
    ``ambiguity`` and ``dependency_depth`` have no diff-side source at all: they
    default to zero, which means "no signal", not "none".
    """
    if requires_context is None:
        requires_context = stat.touches_existing
    if is_greenfield is None:
        is_greenfield = stat.is_all_new

    return TaskSignals(
        description=description,
        category=category,
        file_count=max(stat.files, 1),
        lines_changed=stat.lines_changed,
        dependency_depth=dependency_depth,
        requires_context=requires_context,
        ambiguity=ambiguity,
        is_greenfield=is_greenfield,
        tier_hint=tier_hint,
        min_tier=min_tier,
        metadata={"paths": list(stat.paths), "binary_files": stat.binary_files},
    )


def signals_from_diff(
    ref: str = "HEAD~1",
    cwd: str | Path | None = None,
    paths: Optional[Sequence[str]] = None,
    staged: bool = False,
    **overrides,
) -> TaskSignals:
    """Read a repository's diff and build TaskSignals from it.

    >>> signals = signals_from_diff("HEAD~1", category="refactor", ambiguity=0.3)
    """
    return signals_from_stat(run_git_diff(ref, cwd, paths, staged), **overrides)


def signals_from_numstat(numstat: str, name_status: str = "", **overrides) -> TaskSignals:
    """Build TaskSignals from diff output you already have in hand."""
    stat = parse_numstat(numstat)
    if name_status:
        stat.statuses = parse_name_status(name_status)
    return signals_from_stat(stat, **overrides)
