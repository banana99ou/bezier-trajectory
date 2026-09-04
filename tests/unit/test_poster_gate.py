"""The poster's build gate, and the promise that no result on it was typed by hand.

`tools/render_poster.py` refuses to build when the sidecar it quotes stops being
figure-grade, or stops showing what the poster claims. A gate that cannot fail is
not evidence, so these tests state the failures explicitly: each `bad_*` case
mutates one field of the real sidecar and asserts the build is refused.

The last tests are the "no hand-typed number" rule made mechanical --
`numbers.tex` must be exactly what the sidecar generates, `poster.tex` must
not contain a result value as a literal, and the numbers the poster's own
figures print must move when the sidecar moves.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
POSTER = ROOT / "paper" / "ksas_2026_fall" / "poster"
SIDECAR = ROOT / "figures" / "paper1" / "occlusion_figure.json"


def _render_poster():
    spec = importlib.util.spec_from_file_location("render_poster", ROOT / "tools" / "render_poster.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rp():
    return _render_poster()


@pytest.fixture(scope="module")
def sidecar():
    return json.loads(SIDECAR.read_text())


def test_the_real_sidecar_passes_the_gate(rp, sidecar):
    """The gate is not vacuously strict: today's measured run gets through."""
    rp.check(sidecar)


@pytest.mark.parametrize(
    "section,field,value,why",
    [
        ("constrained", "graft_min_los_margin", +0.5,
         "the constrained path on the baseline schedule KEEPS the link, so the "
         "poster's central claim -- that retiming is what saved the run -- is false"),
        ("baseline", "min_los_margin", +1.0,
         "the baseline never loses the link, so there is nothing to demonstrate"),
        ("constrained", "min_los_margin", -0.5,
         "the constrained run loses the link, so the method did not work"),
        ("constrained", "unsound_clips", 3.0,
         "some wall covers only a clipped piece, so the certificate does not "
         "speak for the whole keep-out zone"),
        ("constrained", "occlusion_certificate", 1e-3, "the run does not certify"),
        ("constrained", "koz_certificate", 1e-3, "the run does not certify"),
        ("constrained", "total_slack", 1e-2, "the run is standing on slack"),
        ("constrained", "min_clearance", -1.0, "the returned trajectory penetrates"),
    ],
)
def test_the_gate_refuses_when_the_demo_stops_demonstrating(rp, sidecar, section, field, value, why):
    bad = copy.deepcopy(sidecar)
    bad[section][field] = value
    with pytest.raises(SystemExit) as excinfo:
        rp.check(bad)
    assert field in str(excinfo.value), why


def test_numbers_tex_is_exactly_what_the_sidecar_generates(rp, sidecar):
    """A hand-edited numbers.tex is caught here, not on the printed poster."""
    committed = (POSTER / "numbers.tex").read_text(encoding="utf-8")
    assert committed == rp.macros(sidecar)


def test_poster_tex_quotes_no_result_value_as_a_literal(rp, sidecar):
    """Every result on the poster comes through a \\NUM macro, never typed in.

    Table 1's scene parameters (6 m, 30 m, 200 m, ...) are the problem definition
    and are literals by design; only the run's *results* are checked here.
    """
    results = {
        k: v for k, v in
        ((k, v) for k, v in _macro_pairs(rp.macros(sidecar)))
        if k.startswith(("NUMbase", "NUMcon", "NUMgraft", "NUMaxis", "NUMarrival", "NUMsnapshot", "NUMdelay"))
    }
    body = _without_layout_numbers((POSTER / "poster.tex").read_text(encoding="utf-8"))
    offenders = sorted(
        f"{value!r} (should be \\{name})"
        for name, value in results.items()
        if re.search(rf"(?<![\d.]){re.escape(value.lstrip('+'))}(?![\d.])", body)
    )
    assert not offenders, "hand-typed result value(s) in poster.tex: " + ", ".join(offenders)


def _without_layout_numbers(text: str) -> str:
    """Drop the numbers that set geometry, not results -- column widths, scales,
    lengths. Leaving them in makes the check fire on ``p{0.46\\linewidth}``."""
    text = re.sub(r"[\d.]+\s*\\linewidth", " ", text)
    text = re.sub(r"(?:scale|linewidth|roundedcorners)\s*=\s*[\d.]+", " ", text)
    text = re.sub(r"\\vspace\{[\d.]+mm\}", " ", text)
    return re.sub(r"[\d.]+(?:mm|pt|cm|em)\b", " ", text)


def _macro_pairs(text: str):
    for match in re.finditer(r"\\newcommand\{\\(NUM\w+)\}\{(.*)\}", text):
        yield match.group(1), match.group(2)


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext (poppler) not on PATH")
def test_the_poster_figures_print_what_the_sidecar_says(sidecar, tmp_path):
    """The arrival times and the link-loss interval drawn on the scene figure are
    read from the sidecar, never typed. Falsified by mutating the sidecar: a
    figure that kept printing 46.6 s after the sidecar said 99.9 s fails here.
    """
    mutated = copy.deepcopy(sidecar)
    mutated["constrained"]["arrival_time"] = 99.9
    mutated["baseline"]["arrival_time"] = 88.8
    mutated["baseline"]["los_loss_interval"] = [21.1, 22.2]
    side = tmp_path / "sidecar.json"
    side.write_text(json.dumps(mutated))
    out = tmp_path / "figs"
    proc = subprocess.run(
        [sys.executable, str(POSTER / "make_poster_figures.py"), "--sidecar", str(side),
         "--out-dir", str(out), "--only", "fig_scene"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    text = subprocess.run(["pdftotext", str(out / "fig_scene.pdf"), "-"],
                          capture_output=True, text=True).stdout
    for expected in ("99.9 s", "88.8 s", "21.1", "22.2"):
        assert expected in text, f"{expected!r} not drawn: the figure does not follow the sidecar"
    assert "46.6" not in text and "40.0" not in text, "a stale value survived the mutation"
