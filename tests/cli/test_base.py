"""Unit tests for meshing_utils.cli.base."""

import argparse
import logging
from pathlib import Path

import pytest

from meshing_utils.cli.base import (
    add_common_args,
    backup_if_needed,
    configure_logging,
    resolve_bmd_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> argparse.Namespace:
    """Return a Namespace with default common-arg values, overridable via kwargs."""
    defaults = {
        "case_dir": None,
        "log_level": "INFO",
        "no_backup": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# add_common_args
# ---------------------------------------------------------------------------

class TestAddCommonArgs:

    def test_adds_case_dir_argument(self):
        """--caseDir must be accepted and default to None."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args([])
        assert args.case_dir is None

    def test_case_dir_parses_to_path(self, tmp_path):
        """--caseDir must be parsed as a Path object."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args(["--caseDir", str(tmp_path)])
        assert isinstance(args.case_dir, Path)
        assert args.case_dir == tmp_path

    def test_adds_log_level_argument(self):
        """--logLevel must be accepted with default INFO."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args([])
        assert args.log_level == "INFO"

    def test_log_level_choices(self):
        """--logLevel must accept all standard log levels."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            args = parser.parse_args(["--logLevel", level])
            assert args.log_level == level

    def test_log_level_invalid_raises(self):
        """An invalid --logLevel value must cause a parse error."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--logLevel", "VERBOSE"])

    def test_adds_no_backup_flag(self):
        """--noBackup must be accepted and default to False."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args([])
        assert args.no_backup is False

    def test_no_backup_sets_true(self):
        """--noBackup flag must set no_backup to True."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args(["--noBackup"])
        assert args.no_backup is True


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

class TestConfigureLogging:

    def test_configure_accepts_debug_level(self):
        """configure_logging must not raise for DEBUG level."""
        args = _make_args(log_level="DEBUG")
        # basicConfig is a no-op when handlers are already set up (e.g. by pytest).
        # We just verify it does not raise and that the level string is valid.
        configure_logging(args)  # must not raise
        assert getattr(logging, args.log_level) == logging.DEBUG

    def test_configure_accepts_warning_level(self):
        """configure_logging must not raise for WARNING level."""
        args = _make_args(log_level="WARNING")
        configure_logging(args)  # must not raise
        assert getattr(logging, args.log_level) == logging.WARNING

    def test_configure_accepts_all_levels(self):
        """configure_logging must accept all five standard log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            args = _make_args(log_level=level)
            configure_logging(args)  # must not raise


# ---------------------------------------------------------------------------
# resolve_bmd_path
# ---------------------------------------------------------------------------

class TestResolveBmdPath:

    def test_returns_correct_paths_when_bmd_exists(self, tmp_path):
        """When blockMeshDict exists, resolve_bmd_path must return correct paths."""
        system_dir = tmp_path / "system"
        system_dir.mkdir()
        bmd_path = system_dir / "blockMeshDict"
        bmd_path.write_text("FoamFile {}")

        args = _make_args(case_dir=tmp_path)
        case_dir, result_bmd = resolve_bmd_path(args)

        assert case_dir == tmp_path
        assert result_bmd == bmd_path

    def test_uses_cwd_when_case_dir_is_none(self, tmp_path, monkeypatch):
        """When case_dir is None, resolve_bmd_path must use the current directory."""
        system_dir = tmp_path / "system"
        system_dir.mkdir()
        bmd_path = system_dir / "blockMeshDict"
        bmd_path.write_text("FoamFile {}")

        monkeypatch.chdir(tmp_path)
        args = _make_args(case_dir=None)
        case_dir, result_bmd = resolve_bmd_path(args)

        assert case_dir == tmp_path
        assert result_bmd == bmd_path

    def test_exits_when_bmd_missing(self, tmp_path):
        """When blockMeshDict does not exist, resolve_bmd_path must call sys.exit(1)."""
        args = _make_args(case_dir=tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            resolve_bmd_path(args)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# backup_if_needed
# ---------------------------------------------------------------------------

class TestBackupIfNeeded:

    def test_creates_bak_file(self, tmp_path):
        """backup_if_needed must create a .bak file when no_backup is False."""
        bmd = tmp_path / "blockMeshDict"
        bmd.write_text("FoamFile {}")

        backup_if_needed(bmd, no_backup=False)

        bak = bmd.with_suffix(".bak")
        assert bak.exists()
        assert bak.read_text() == "FoamFile {}"

    def test_no_backup_skips_bak_creation(self, tmp_path):
        """backup_if_needed must not create a .bak file when no_backup is True."""
        bmd = tmp_path / "blockMeshDict"
        bmd.write_text("FoamFile {}")

        backup_if_needed(bmd, no_backup=True)

        bak = bmd.with_suffix(".bak")
        assert not bak.exists()

    def test_missing_source_file_does_not_raise(self, tmp_path):
        """backup_if_needed must silently skip if the source file does not exist."""
        bmd = tmp_path / "blockMeshDict"
        # File does not exist
        backup_if_needed(bmd, no_backup=False)
        bak = bmd.with_suffix(".bak")
        assert not bak.exists()

    def test_bak_file_overwritten_on_second_call(self, tmp_path):
        """Calling backup_if_needed twice must overwrite the previous .bak."""
        bmd = tmp_path / "blockMeshDict"
        bmd.write_text("version 1")
        backup_if_needed(bmd, no_backup=False)

        bmd.write_text("version 2")
        backup_if_needed(bmd, no_backup=False)

        bak = bmd.with_suffix(".bak")
        assert bak.read_text() == "version 2"

    def test_symlink_bak_is_not_overwritten(self, tmp_path):
        """backup_if_needed must skip the copy when .bak is already a symlink."""
        bmd = tmp_path / "blockMeshDict"
        bmd.write_text("original content")

        # Create a real target file that the symlink will point to
        real_target = tmp_path / "real_target.txt"
        real_target.write_text("real target content")

        bak = bmd.with_suffix(".bak")
        try:
            bak.symlink_to(real_target)
        except OSError:
            pytest.skip("Cannot create symlink on this platform/configuration")

        # backup_if_needed must NOT follow the symlink and overwrite real_target
        backup_if_needed(bmd, no_backup=False)

        # The real target must be unchanged
        assert real_target.read_text() == "real target content"
