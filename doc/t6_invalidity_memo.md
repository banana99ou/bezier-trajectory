# T6 Invalidity Memo

## Verdict

The current `T6` implementation is invalid as a paper-facing evidence artifact.

It may still be retained as an engineering diagnostic, but it should not be treated as defended evidence for `C9`.

## Primary Reason

The downstream comparison is built on a bespoke direct-collocation implementation in:

- `orbital_docking/downstream_collocation.py`

That choice invalidates the scientific intent of `T6`.

Why:

- `T6` was supposed to measure warm-start usefulness for downstream direct collocation
- instead, the result is now entangled with the behavior of a repo-local custom DCM
- once the downstream solver itself is untrusted, the experiment no longer isolates the effect of the initial guess

In short:

- the custom downstream implementation became part of the hypothesis under test
- that is exactly what `T6` needed to avoid

## Supporting Evidence

### 1. The old export bug was real, but it is no longer the main blocker

From `artifacts/paper_artifacts/t6_rust_promotion_gate.json`:

- export consistency gate: `PASS`
- maximum endpoint velocity mismatch after the Rust-backed rerun: about `7.22e-15 km/s`
- overwrite defect growth factor: essentially `1.0`

Interpretation:

- the earlier endpoint-velocity/export inconsistency was a real bug
- but after correcting that issue, the comparison still did not become credible

Therefore the remaining failure cannot honestly be framed as just a warm-start export problem.

### 2. The current downstream stack remains the bottleneck

From `artifacts/paper_artifacts/t6_stage3_credible_rerun.json`:

- both hardened runs failed
- both hit `maxiter = 200`
- naive max equality violation: `3.90e-02`
- warm max equality violation: `1.03e+00`

Interpretation:

- once the custom DCM is asked to carry stronger credibility requirements, it does not deliver a clean solved comparison
- this makes the downstream implementation itself the main uncertainty source

### 3. The experiment stopped answering the intended question

The intended question was:

- does the Bézier warm start help downstream direct collocation?

The actual question answered by the current setup is closer to:

- how does this repo’s custom direct-collocation prototype behave under different initial guesses and patches?

Those are not the same question.

## Consequence For `C9` / `T6`

Unsafe reading:

- that the current `T6` demonstrates warm-start usefulness
- that the present downstream result supports `C9`
- that the comparison is only about initialization quality

Safe reading:

- the repo now contains a useful diagnostic prototype
- the prototype exposed:
  - an export inconsistency bug
  - an upstream/downstream mismatch
  - a deeper downstream-implementation problem

That is useful engineering information.

It is not valid paper evidence.

## Required Reset

The correct response is not to keep patching the bespoke DCM for `T6`.

The correct response is:

1. retire the bespoke downstream DCM from the paper-facing `T6` path
2. replace it with an off-the-shelf downstream optimal-control stack
3. rerun the matched naive-vs-warm-start comparison there

Recommended replacement note:

- `doc/t6_off_the_shelf_dcm_replacement.md`

## Final Decision

Current status:

- `T6` is invalid as defended evidence
- `C9` must remain unsupported by the present downstream comparison
- the current artifacts should be treated as diagnostic only, not promotable evidence
