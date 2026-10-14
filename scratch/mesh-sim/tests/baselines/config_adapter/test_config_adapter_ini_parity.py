"""INI structure rules of config.read_ini/load_baseline and parity with the C++ loader.

Expectations come from scripts/baselines/README.md ("The [baseline] section": strict
rejection of duplicates, ':' assignments and continuation lines), scripts/CLAUDE.md
(strip_inline_comment matches the C++ loader), and src/util/ini-parser.{h,cc}: comments
start at the first '#' or ';' anywhere on a line and are stripped *before* the line is
classified as a header or key, so a value can never contain '#' or ';'.
"""

import pytest

from scripts.baselines import config


def _ini(tmp_path, text, name="run.ini"):
    path = tmp_path / name
    path.write_text(text)
    return path


ACTIVE = "algorithm = geometric\nobjective = coverage\nmovable_nodes = uav-a\n"


@pytest.mark.parametrize("raw,expected", [
    ("geometric#x", "geometric"),
    ("geometric;x", "geometric"),
    ("geometric ; a # b", "geometric"),
    ("\tgeometric\t# tab separated", "geometric"),
    ("geometric # x = optimization", "geometric"),
])
def test_inline_comment_markers_match_cpp(tmp_path, raw, expected):
    # ini-parser.cc truncates at the first '#' then the first ';', then trims " \t\r\n".
    cfg = config.load_baseline(_ini(tmp_path, f"[baseline]\nalgorithm = {raw}\n"))
    assert cfg.algorithm == expected


def test_full_line_and_indented_comments_are_ignored(tmp_path):
    text = ("[baseline]\n# algorithm = optimization\n; objective = resilience\n"
            "algorithm = geometric\n    # an indented comment is not a continuation\n"
            "objective = coverage\n\t; tab-indented comment\nmovable_nodes = uav-a\n")
    cfg = config.load_baseline(_ini(tmp_path, text))
    assert (cfg.algorithm, cfg.objective, cfg.movable_nodes) == (
        "geometric", "coverage", ("uav-a",))


def test_commented_out_duplicate_is_not_a_duplicate(tmp_path):
    text = "[baseline]\nobjective = coverage\n#objective = balanced\n"
    assert config.load_baseline(_ini(tmp_path, text)).objective == "coverage"


@pytest.mark.parametrize("header", ["[baseline] # note", "[baseline] ; note",
                                    "[baseline]# note"])
def test_header_with_trailing_comment_is_the_baseline_section(tmp_path, header):
    cfg = config.load_baseline(_ini(tmp_path, f"{header}\n{ACTIVE}"))
    assert cfg.algorithm == "geometric"
