# Paper Claim, Scope, and Non-Claims

This document fixes the paper's boundaries before further drafting. Its purpose is to prevent the paper from drifting into claims that are broader than the current method, code, or evidence.

## Core claim

The paper's main claim should be:

**This work presents a Bézier-curve-based framework for generating smooth, continuously KOZ-safe trajectories. The method operates entirely in control-point space, uses De Casteljau subdivision to impose conservative supporting half-space constraints on Bézier sub-arcs, and solves a sequence of convex quadratic subproblems within a successive-convexification loop. These trajectories are also intended to serve as structured warm starts for downstream solvers, but that downstream role should support the paper's usefulness case rather than define its core identity.**

That claim is strong enough to be interesting and narrow enough to be defensible.

## Actual contribution

The actual contribution is not "solving constrained trajectory optimization in general." The contribution is the combination of the following elements into one coherent framework:

- a control-point-space formulation based on Bézier curves;
- structured derivative handling using the `D` and `E` operators;
- quadratic objective assembly using precomputable matrix structure, including Gram-matrix reuse;
- conservative continuous KOZ handling through De Casteljau subdivision plus segment-wise supporting half-spaces;
- an SCP implementation that repeatedly relinearizes the nonconvex parts while keeping each subproblem convex;
- a reusable framework for constructing smooth continuously safe trajectories under the paper's stated assumptions;
- a demonstration on a simplified orbital-transfer example.

The paper is therefore a **framework-and-method paper with a demonstration case and a major downstream-usefulness component**, not a domain-maximal application paper and not a final optimal-control paper.

## Scope

The paper should explicitly stay within the following scope:

- It studies **trajectory initialization / warm-start generation**, not final guidance or closed-loop control.
- It focuses on **continuous avoidance of a spherical KOZ** through conservative geometric constraints on Bézier sub-arcs.
- It uses a **control-point-space formulation** rather than a pointwise state/control discretization.
- It allows smooth position, velocity, and acceleration representations through structured Bézier derivative mappings.
- It uses **successive convexification**, so the method is a sequence of convex subproblems, not an exact one-shot convex reformulation of the original nonconvex problem.
- It demonstrates the framework on a **simplified orbital-transfer case**, while keeping the conceptual framing application-agnostic.
- It may discuss broader portability as a property of the formulation, but not as an experimentally established fact unless additional demonstrations are added.

## Contribution-order rule

When the paper introduces itself, the framing order should be:

1. framework identity
2. control-point-space and continuous-safety construction
3. demonstration evidence
4. downstream warm-start usefulness

That last item is important, but it should not eclipse the framework-level contribution.

## What the paper is trying to prove

The paper is trying to prove four things:

1. A Bézier control-point-space formulation can turn continuous spherical-KOZ avoidance into a conservative and computationally manageable constrained optimization procedure.
2. The structured `D`/`E`/Gram-matrix formulation gives a coherent way to represent derivatives and objectives without rebuilding everything from pointwise Bernstein evaluation inside the optimization loop.
3. The framework can produce smooth, feasible, safety-respecting trajectories for a nontrivial demonstration problem.
4. These trajectories are useful as the **initialization stage of a downstream direct-collocation pipeline**, which is tested by comparing the full two-pass DCM pipeline (Pass 1 Hermite-Simpson → peak detect → Pass 2 multi-phase LGL) against a variant in which the Bézier upstream replaces Pass 1 while Pass 2 remains identical.

Item 4 is a major part of the paper's usefulness case, not an optional side branch. However, it should still remain downstream to the framework identity in the paper's rhetoric. The comparison is scoped to circular transfers in which both pipelines converge — a set bounded on one side by eccentricity (every elliptic case in the tested dataset produced an infeasible Bézier upstream) and on the other by downstream Pass 2 convergence at long transfer times. Any wording that extends item 4 beyond that regime, or that implies method-class superiority over direct collocation, is a framing violation.

## Explicit non-claims

The paper should explicitly not claim any of the following:

- It does **not** exactly convert a nonlinear nonconvex trajectory problem into a single linear or convex problem.
- It does **not** prove global optimality.
- It does **not** prove superiority to direct collocation, direct transcription, or other planner classes.
- It does **not** establish cross-domain performance for drones, cars, or other systems from orbital evidence alone.
- It does **not** provide a full high-fidelity rendezvous guidance formulation.
- It does **not** solve general nonconvex obstacle geometry in full generality; the safety argument in the present paper is tied to the spherical-KOZ supporting-half-space construction.
- It does **not** directly optimize true mission delta-v; the objectives used are surrogates.
- It does **not** show real-time capability unless that is measured and defended.

## What the paper is and is not trying to prove

The paper **is** trying to establish that the proposed formulation is mathematically coherent, computationally implementable, useful for generating smooth KOZ-safe trajectories under explicit assumptions, and valuable as a downstream initialization method when evaluated under a fair comparison.

The paper is **not** trying to prove that Bézier methods are universally better than collocation, that the demonstration case is operationally realistic enough for deployment, or that the current surrogate objectives recover true optimal control solutions.

That distinction matters. Without it, the paper becomes easy to attack.

## Framing warnings

The following phrasings should be treated as red flags unless backed by new evidence:

- "converts the nonlinear nonconvex problem into a convex problem"
- "superior to direct collocation"
- "general method for orbital, drone, and car planning" when only one domain is demonstrated
- "physically meaningful delta-v objective" as a claimed advantage over the L2 surrogate
- "continuous safety guarantee" without explicitly stating the assumptions: spherical KOZ, supporting half-space construction, and control-point satisfaction on each subdivided sub-arc

## Recommended discipline rule

Every important sentence in the paper should survive the following test:

**Is this sentence supported by the actual method, the actual demonstration, and the actual evidence map?**

If not, weaken it or delete it.
