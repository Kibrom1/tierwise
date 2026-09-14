"""Signal extraction from a diff -- what it reads, and what it refuses to guess."""

import shutil
import subprocess

import pytest

from tierwise import (
    DiffStat,
    Router,
    Tier,
    parse_name_status,
    parse_numstat,
    signals_from_diff,
    signals_from_numstat,
    signals_from_stat,
)

NUMSTAT = """\
12\t3\tsrc/app/auth.py
40\t0\tsrc/app/session.py
0\t18\ttests/test_auth.py
"""

NAME_STATUS = """\
M\tsrc/app/auth.py
A\tsrc/app/session.py
M\ttests/test_auth.py
"""

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


# -- parsing -----------------------------------------------------------------

def test_numstat_counts_files_and_lines():
    stat = parse_numstat(NUMSTAT)
    assert stat.files == 3
    assert stat.lines_added == 52
    assert stat.lines_removed == 21
    assert stat.lines_changed == 73
    assert stat.paths == ["src/app/auth.py", "src/app/session.py", "tests/test_auth.py"]


def test_binary_files_count_as_files_not_as_lines():
    """A 4 MB PNG is one touched file, not 400,000 lines of complexity."""
    stat = parse_numstat("-\t-\tdocs/logo.png\n10\t2\tREADME.md\n")
    assert stat.files == 2
    assert stat.binary_files == 1
    assert stat.lines_changed == 12


def test_renames_resolve_to_the_new_path():
    stat = parse_numstat("3\t3\tsrc/{old => new}/thing.py\n1\t1\ta.py => b.py\n")
    assert stat.paths == ["src/new/thing.py", "b.py"]


def test_blank_and_malformed_rows_are_skipped():
    assert parse_numstat("\n\ngarbage\n5\t5\tok.py\n").files == 1


def test_empty_diff_is_empty():
    stat = parse_numstat("")
    assert stat.files == 0 and stat.lines_changed == 0
    assert stat.is_all_new is False and stat.touches_existing is False


def test_name_status_reduces_to_letters():
    assert parse_name_status("M\ta.py\nR100\told.py\tnew.py\nA\tb.py\n") == ["M", "R", "A"]


# -- what it reads rather than guesses ---------------------------------------

def test_editing_existing_files_means_context_is_needed():
    signals = signals_from_numstat(NUMSTAT, NAME_STATUS)
    assert signals.file_count == 3
    assert signals.lines_changed == 73
    assert signals.requires_context is True
    assert signals.is_greenfield is False


def test_an_all_new_diff_is_greenfield():
    signals = signals_from_numstat("40\t0\tsrc/new.py\n", "A\tsrc/new.py\n")
    assert signals.is_greenfield is True
    assert signals.requires_context is False


def test_deletions_count_as_touching_existing_code():
    signals = signals_from_numstat("0\t30\tsrc/gone.py\n", "D\tsrc/gone.py\n")
    assert signals.requires_context is True
    assert signals.is_greenfield is False


def test_without_statuses_it_assumes_nothing():
    """No name-status available: neither flag may be invented."""
    signals = signals_from_numstat(NUMSTAT)
    assert signals.requires_context is False
    assert signals.is_greenfield is False


def test_explicit_values_win_over_what_was_read():
    signals = signals_from_numstat(NUMSTAT, NAME_STATUS, requires_context=False,
                                   is_greenfield=True)
    assert signals.requires_context is False
    assert signals.is_greenfield is True


def test_judgement_signals_default_to_no_signal():
    """A diff cannot know how underspecified a request was."""
    signals = signals_from_numstat(NUMSTAT, NAME_STATUS)
    assert signals.ambiguity == 0.0
    assert signals.dependency_depth == 0


def test_the_caller_supplies_what_the_diff_cannot():
    signals = signals_from_numstat(
        NUMSTAT, NAME_STATUS, description="rework auth", category="refactor",
        ambiguity=0.4, dependency_depth=3, tier_hint=Tier.HIGH,
    )
    assert signals.category == "refactor"
    assert signals.ambiguity == pytest.approx(0.4)
    assert signals.dependency_depth == 3
    assert signals.tier_hint is Tier.HIGH


def test_paths_are_kept_in_metadata():
    signals = signals_from_numstat(NUMSTAT, NAME_STATUS)
    assert signals.metadata["paths"][0] == "src/app/auth.py"
    assert signals.metadata["binary_files"] == 0


def test_an_empty_diff_still_yields_routable_signals():
    signals = signals_from_stat(DiffStat())
    assert signals.file_count == 1        # never zero; TaskSignals expects a task
    assert Router().route(signals).tier is Tier.LOW


def test_extracted_signals_route():
    """The point of the module: diff in, tier out."""
    big = signals_from_numstat(
        "\n".join(f"{60}\t{40}\tsrc/mod{i}.py" for i in range(14)) + "\n",
        "\n".join(f"M\tsrc/mod{i}.py" for i in range(14)) + "\n",
        category="refactor",
    )
    assert Router().route(big).tier is not Tier.LOW


# -- against a real repository -----------------------------------------------

@needs_git
def test_reads_a_real_repository(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True,
                       capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "kept.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-qm", "first")

    (tmp_path / "kept.py").write_text("x = 1\ny = 2\nz = 3\n")
    (tmp_path / "added.py").write_text("a = 1\nb = 2\n")
    git("add", "-A")
    git("commit", "-qm", "second")

    signals = signals_from_diff("HEAD~1", cwd=tmp_path, category="feature")
    assert signals.file_count == 2
    assert signals.lines_changed == 4          # 2 added in kept.py, 2 in added.py
    assert signals.requires_context is True    # kept.py was modified
    assert signals.is_greenfield is False
    assert sorted(signals.metadata["paths"]) == ["added.py", "kept.py"]


@needs_git
def test_reads_the_staged_index(tmp_path):
    """The pre-commit case: route the work you are about to ask for."""
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True,
                       capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-qm", "first")

    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    git("add", "-A")

    signals = signals_from_diff(cwd=tmp_path, staged=True)
    assert signals.file_count == 1
    assert signals.lines_changed == 1
    assert signals.requires_context is True
