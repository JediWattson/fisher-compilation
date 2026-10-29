"""Terminal TUI helpers: ANSI styling, hrule layout, markdown rendering,
and stream-repaint bookkeeping. No MLX required."""

import contextlib
import io
import re

from fisher_graph.mlx_qwen3_pipeline import (
    _Ansi,
    _RULE_W,
    _hrule,
    _inline_md,
    _render_markdown,
    _TermRepaint,
)

_STRIP = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(s):
    return _STRIP.sub("", s)


def test_ansi_disabled_is_passthrough():
    a = _Ansi(False)
    assert a.bold("x") == "x" and a.cyan("y") == "y" and a.dim("z") == "z"


def test_hrule_width_and_label():
    a = _Ansi(True)
    line = _hrule(a, "model", style=a.b_cyan)
    assert len(_plain(line)) == _RULE_W
    assert " model " in line
    bare = _hrule(a)
    assert len(_plain(bare)) == _RULE_W and set(_plain(bare)) == {"─"}
    tailed = _hrule(a, tail="16 tok")
    assert len(_plain(tailed)) == _RULE_W and _plain(tailed).endswith(" 16 tok ")


def test_inline_spans():
    a = _Ansi(True)
    out = _inline_md(a, "**b** `c` *i* ~~s~~ [t](http://u)")
    p = _plain(out)
    assert "**" not in p and "``" not in p and "~~" not in p
    assert "b" in p and "c" in p and "t (http://u)" in p
    assert "\x1b[1m" in out and "\x1b[33m" in out and "\x1b[4m" in out


def test_render_markdown_blocks():
    a = _Ansi(True)
    md = "## Head\n- one\n- two\n```py\ncode()\n```\n> note\n---\nend"
    out = _render_markdown(md, a)
    p = _plain(out)
    assert "╭─ py" in p and "╰──" in p
    assert "│ code()" in p
    assert "•" in p and "▎" in p
    assert "##" not in p and "```" not in p
    # disabled: passthrough
    assert _render_markdown(md, _Ansi(False)) == md


def test_render_markdown_unclosed_fence():
    a = _Ansi(True)
    out = _render_markdown("```python\nx = 1", a)
    p = _plain(out)
    assert "╭─ python" in p and "│ x = 1" in p and "╰" not in p


def test_repaint_counts_rows():
    a = _Ansi(True)
    s = _TermRepaint(a)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s.feed("ab")
        s.feed("cd\nef")
        s.repaint("RENDERED")
    raw = buf.getvalue()
    assert "abcd\nef" in raw
    assert raw.endswith("RENDERED\n")
    # two physical rows streamed -> cursor up one row, then clear-to-end
    assert "\x1b[1A\x1b[J" in raw


def test_repaint_disabled_writes_raw_only():
    s = _TermRepaint(_Ansi(False))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s.feed("abc\n")
        s.repaint("SHOULD_NOT_APPEAR")
    assert buf.getvalue() == "abc\n"
