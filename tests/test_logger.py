"""The shared logger keeps its output format, level control and stdout target."""

import re

import pytest

from relay_d.utils import coloring_logger as cl

LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} - (INFO|WARNING|ERROR|DEBUG)\s* - (.*)$")


@pytest.fixture(autouse=True)
def _restore_level():
    before = cl.logger.level
    yield
    cl.logger.setLevel(before)


def test_line_format_and_stdout(capsys):
    cl.set_level("INFO")
    cl.logger.info("hello 100% sure")  # a % in the message must not be interpolated
    cl.logger.warning("careful")
    out = capsys.readouterr().out.splitlines()
    assert [LINE.match(line).group(1) for line in out] == ["INFO", "WARNING"]
    assert LINE.match(out[0]).group(2) == "hello 100% sure"


def test_level_filtering_and_validation(capsys):
    cl.set_level("ERROR")
    cl.logger.info("hidden")
    cl.logger.error("shown")
    assert capsys.readouterr().out.count("\n") == 1
    cl.set_level("debug")  # case-insensitive
    cl.logger.debug("now visible")
    assert "now visible" in capsys.readouterr().out
    with pytest.raises(ValueError):
        cl.set_level("LOUD")


def test_unencodable_character_does_not_lose_the_line(capsys, monkeypatch):
    class Ascii:
        encoding = "ascii"

        def isatty(self):
            return False

        def write(self, s):
            s.encode("ascii")  # would raise if the formatter had left a non-ascii character
            return len(s)

        def flush(self):
            pass

    monkeypatch.setattr("sys.stdout", Ascii())
    cl.logger.info("café → ok")  # no exception, line formatted with replacements


def test_single_handler_after_reimport():
    import importlib

    importlib.reload(cl)
    assert sum(getattr(h, "relayd", False) for h in cl.logger.handlers) == 1
