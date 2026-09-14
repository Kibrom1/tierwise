"""Project config: where the model names live.

Thresholds are learned, so they persist per machine. Model names are the
opposite -- a project decision that should be committed and identical for every
developer, for CI, and for whatever runs the tuner. Three environment variables
set in three places drift, and when they do, routing differs by environment and
the tuner learns from decisions made against models nobody else is using.

So the names go in a file next to the code, found by walking up from the working
directory:

    # tierwise.toml
    [models]
    low = "claude-haiku-4-5"
    medium = "claude-sonnet-4-5"
    high = "claude-opus-4-5"

``tierwise.json`` with the same shape works too, and needs no TOML parser --
which matters on Python 3.10, where ``tomllib`` does not exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

ENV_CONFIG = "TIERWISE_CONFIG"
CONFIG_NAMES = ("tierwise.toml", "tierwise.json")


class ConfigError(ValueError):
    """A config file exists but cannot be used.

    Missing is fine -- defaults apply. Present and broken is not: silently
    ignoring it would route against models nobody chose.
    """


def find_config(start: str | Path | None = None) -> Optional[Path]:
    """Walk up from `start` (default: cwd) looking for a config file."""
    import os

    override = os.environ.get(ENV_CONFIG)
    if override:
        path = Path(override)
        return path if path.exists() else None

    current = Path(start) if start is not None else Path.cwd()
    current = current.resolve()
    for directory in (current, *current.parents):
        for name in CONFIG_NAMES:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def _parse_toml(text: str, path: Path) -> dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise ConfigError(
                f"{path} is TOML, but this Python has no TOML parser. "
                "Use Python 3.11+, `pip install tomli`, or rename it to "
                "tierwise.json with the same shape."
            ) from exc
    try:
        return tomllib.loads(text)
    except Exception as exc:  # noqa: BLE001 - parser type varies
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the config file, or return {} when there is none."""
    target = Path(path) if path is not None else find_config()
    if target is None or not target.exists():
        return {}

    text = target.read_text(encoding="utf-8")
    if target.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{target} is not valid JSON: {exc}") from exc
    else:
        data = _parse_toml(text, target)

    if not isinstance(data, dict):
        raise ConfigError(f"{target} must contain a table at the top level")
    data["__path__"] = str(target)
    return data
