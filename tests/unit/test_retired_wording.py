"""Retired wording must not reach an artifact.

Several sentences were measured false and retired by name in
``paper/journal_1/SPEC.md`` section 1.2. Retiring them in a document is a rule a
session can forget; this test is the guard that makes forgetting visible.

**Scope: ARTIFACT files only** -- the text that becomes a paper, a poster or a
manuscript. Documentation is deliberately exempt, because a document that
forbids a sentence has to quote it; scanning documentation would make the guard
fire on the very lines that exist to prevent the mistake.

**The invariant is an EQUALITY, not an absence.** The set of (artifact, phrase)
hits must equal ``KNOWN`` exactly, so the test fails in both directions:

* a NEW hit in any artifact -- retired wording reached the paper;
* a KNOWN hit that disappears -- the defect was fixed and this file is now
  lying about the state of the repository, so the entry must be deleted.

The second half is what stops ``KNOWN`` decaying into a list of excuses nobody
rechecks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Files whose text becomes an artifact. A file that does not exist yet is
# skipped, so listing the journal manuscript before it is written is correct.
ARTIFACTS = [
    "paper/ksas_2026_fall/manuscript.md",
    "paper/ksas_2026_fall/poster/poster.tex",
    "paper/journal_1/main.tex",
]

# Each entry: name -> compiled pattern. Names are stable; patterns may be
# widened. Every one of these is retired in `paper/journal_1/SPEC.md` 1.2 or in
# the CLAUDE.md Traps section, with the measurement that retired it.
RETIRED = {
    # "the decomposition binds the passing time" -- true only of a SPACE-ONLY
    # decomposition. A space-time decomposition never commits to timing, and
    # Tang et al.'s ST-GCS optimises arrival time inside one.
    "timing_bound_by_decomposition_ko": re.compile(r"통과\s*시각.{0,24}매인다"),
    "timing_bound_by_decomposition_en": re.compile(
        r"decomposition[^.]{0,80}(commits to|binds)[^.]{0,40}tim", re.IGNORECASE
    ),
    # "a decomposition cannot express visibility" -- false as stated; a
    # space-time decomposition can carve any closed set.
    "cannot_express_ko": re.compile(r"분해[^.\n]{0,30}표현할\s*수\s*없"),
    "cannot_express_en": re.compile(
        r"decomposition[^.]{0,60}cannot express", re.IGNORECASE
    ),
    # The lift was published in August 2025; opening the novelty claim with it
    # claims someone else's result.
    "time_as_a_coordinate": re.compile(r"time as a coordinate", re.IGNORECASE),
    # The objective is a parameter-domain smoothness regulariser and is blind to
    # timing. The phrase is permanently forbidden, not merely deferred.
    "minimises_spatial_acceleration": re.compile(
        r"minimi[sz]es spatial acceleration", re.IGNORECASE
    ),
}

# The exact set of hits that exist on purpose, each with the reason it is
# tolerated and what removes it. Nothing may be added here without a reason of
# the same kind.
KNOWN = {
    # Submitted 2026-09-03 and therefore immutable. Recorded so the sentence is
    # never copied forward into the journal.
    ("paper/ksas_2026_fall/manuscript.md", "timing_bound_by_decomposition_ko"),
    # Draft. Must be corrected before the conference (2026-11-10); when it is,
    # delete this entry and the test goes green again.
    ("paper/ksas_2026_fall/poster/poster.tex", "timing_bound_by_decomposition_ko"),
}


def _hits() -> set[tuple[str, str]]:
    found = set()
    for rel in ARTIFACTS:
        path = REPO / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in RETIRED.items():
            if pattern.search(text):
                found.add((rel, name))
    return found


def test_no_new_retired_wording_reaches_an_artifact():
    """A retired sentence in artifact text is a defect, not a style choice."""
    new = _hits() - KNOWN
    assert not new, (
        "retired wording reached an artifact: "
        + ", ".join(f"{f} -> {n}" for f, n in sorted(new))
        + ". These sentences were measured false; see paper/journal_1/SPEC.md section 1.2."
    )


def test_known_violations_still_exist_or_the_list_is_stale():
    """A fixed defect must leave this file, or KNOWN becomes a list of excuses."""
    gone = KNOWN - _hits()
    assert not gone, (
        "these entries no longer occur and must be deleted from KNOWN: "
        + ", ".join(f"{f} -> {n}" for f, n in sorted(gone))
    )


@pytest.mark.parametrize("name,pattern", sorted(RETIRED.items()))
def test_every_pattern_matches_the_sentence_it_was_written_for(name, pattern):
    """A pattern that matches nothing cannot fail, and would be decoration.

    Each retired sentence is reproduced here in the form that motivated it, so a
    pattern edited into uselessness is caught by this test rather than by a
    reviewer reading the paper.
    """
    specimens = {
        "timing_bound_by_decomposition_ko": "통과 시각도 그 셀 배치에 매인다",
        "timing_bound_by_decomposition_en": (
            "a free-space decomposition must be fixed first, so it commits to the timing"
        ),
        "cannot_express_ko": "볼록 분해로는 표현할 수 없는 제약",
        "cannot_express_en": "a decomposition cannot express visibility",
        "time_as_a_coordinate": "our contribution is time as a coordinate",
        "minimises_spatial_acceleration": "the objective minimizes spatial acceleration",
    }
    assert pattern.search(specimens[name]), f"{name} no longer matches its own specimen"
