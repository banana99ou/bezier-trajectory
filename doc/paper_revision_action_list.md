# Paper revision action list (from professor feedback)

Source: `doc/paper_draft_korean_PF.md` — 83 inline `[[…]]` annotations + 4 `{…}` annotations in the abstract = **87 items total**.

## Legend

- **Status**: 🟢 TODO · 🟡 PARTIAL (action remains, interpretation defused by cache evidence) · ✓ DEFUSED (no action)
- **Imp**: C(ritical) / H(igh) / M(edium) / L(ow)
- **Ease**: T(rivial) / E(asy) / M(edium) / H(ard)

## Phase ordering rationale

The phases are ordered so high-blast-radius changes happen first. Don't polish words in §2 if §2 will be folded into §1. Don't fight the tolerance defense paragraph if you're going to delete it after rerunning at sane tolerance. Phase 0 collapses structure; Phase 1 regenerates numbers and framing sections; Phase 2 rewrites bodies; Phase 3 does line-level polish across the now-stable text.

## Defused-by-cache-evidence summary

The April 2026 paper-run cache (10000-iter at `tol=1e-12`) compared against 100000-iter reruns shows the iterates are essentially settled (cost identical to 4 sig figs for `n_seg ≥ 8`, `dnorm` already at 0.01–0.03 km on a ~6500 km trajectory). What looked like "not converged" is a tolerance-choice byproduct, not a science failure. Three items are affected: #9 fully defused (defense paragraph just gets deleted), #7 and #8 partially defused (interpretation is clear; rerun + col-drop actions remain).

The prof's verbal "선형화 한 번 하고 수렴할 때까지 상수 유지" critique is **separate from** the tolerance issue and is **not** defused by the cache. It's listed below as a Phase-0 decision (frozen-linearization probe) because its outcome dictates how the §4 method is framed.

---

## PHASE 0 — Structural decisions (do first; they invalidate large prose blocks)

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 1 | 40 | "section2 전체 세부절 구분하지 않고, section 1과 합칠 것" | Fold all of §2 into §1, no subsections | C | M | 🟢 |
| 2 | 46 | (same) | Covered by #1 | – | – | 🟢 |
| 3 | 52 | (same) | Covered by #1 | – | – | 🟢 |
| 4 | 433 | "한계점 1-2문장으로 간략히 서술. 결론 섹션에 통합" | Delete §7 as standalone; 1–2 sentence limitations into §8 | C | E | 🟢 |
| 5 | 28 | "2,3번 하나로 합쳐야 할 듯" | Merge contributions 2 and 3 | H | T | 🟢 |
| 6 | 30 | "줄글 형식으로 수정" (contributions list) | Convert (1)…(5) → prose paragraph | H | E | 🟢 |
| – | – | **Decision: frozen-linearization probe (NOT in PF, from prof's verbal feedback)** | Run 1-line code change (freeze gravity Jacobian at iter-0 P_ref). Decide: keep SCP framing OR reframe §4 as "one-shot linearization + KOZ fixed point". Outcome dictates §4 wording, citation set [6][7], and §6.x runtime numbers. | C | M | 🟢 |

---

## PHASE 1 — Numbers regen + abstract/conclusion (after Phase 0; before in-section polish)

Driver: rerun all paper configs at `tol=1e-6, max_iter=1000` (or whatever the Phase-0 probe lands on). Cache evidence shows table cost numbers won't move; runtime column shrinks ~30×; convergence labels become honest.

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 7 | 358 | "tolerance 명시하고 적정 수준으로 조정하여 수렴하도록" | Rerun all paper configs; report `tol=1e-6` | C | E | 🟡 (interp defused; rerun still needed) |
| 8 | 354 | "Setting column 삭제. 최대 반복…수렴 못한 것?" | (a) drop Setting col from T2 ✓; (b) rerun → iter < cap | C | T+E | 🟡 (a) done; (b) pending Phase 1 rerun |
| 9 | 356 | "?" on §6.1 defense paragraph | ~~Delete the entire 5000/10000-iter sensitivity defense paragraph~~ | – | T | ✓ DEFUSED — paragraph being removed |
| 10 | 393 | "Setting column 삭제" (T4) | Drop Setting col from T4 | M | T | ✓ done |
| 11 | 6 (브) | "조금 더 추상적으로 작성, 알고리즘/방법론 과상세" | Rewrite abstract to drop algorithmic detail | C | M | 🟢 |
| 12 | 12 (브) | "초록 정량값 지양, 정성적 서술" | Strip all numbers from abstract | C | E | 🟢 |
| 13 | 8 (브) | "예) 곡선분할 기법을 통해 구형 장애물에 대하 비볼록 부등식 제약 조건을 선형화 하였다." | Use prof's exemplar sentence as the abstract's method description | C | T | 🟢 |
| 14 | 10 (브) | "궤도전이" (orbital transfer 한국어 치환) | Word swap: "orbital transfer" → "궤도전이" in abstract and through paper | L | T | ✓ done (line 346 caption left for #54 rewrite) |
| 15 | 451 | "결론 정량값 지양, 정성적 서술" | Strip all numbers from §8 | C | E | 🟢 |
| 16 | 303 | "단순 궤도전이 문제인데 target/chaser 가 왜 나오는지?" | Either drop target/chaser language or justify upfront | H | E | 🟢 |
| 17 | 305 | "세팅 → 시나리오" | Word swap | L | T | ✓ done |
| 18 | 315 | §5.1 metrics: bullet → 줄글 | Convert metric list to prose | H | E | ✓ done |

After Phase 1: T2/T3/T4 cost numbers stable; runtime column realistic; abstract and conclusion honest. The §6.1 defense paragraph disappears. Then it's safe to rewrite sections.

---

## PHASE 2 — In-section rewrites (depend on Phase 0/1)

### §3 (problem setup)

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 19 | 63 | "'우주비행체의 3차원 궤도전이 문제에서 우주비행체의 궤적'" | Make trajectory definition specific to spacecraft 3D transfer | M | T | ✓ done |
| 20 | 137 | "각 내용의 연결이 매끄럽지 못함, 흐름 따라가기 어려움" | Add connective intro sentences in §3.3 | H | M | 🟢 |
| 21 | 139 | "Gram matrix가 무엇이고, 왜 갑자기 등장하는지 설명" | Define Gram matrix; motivate why it appears here | H | E | 🟢 |
| 22 | 145 | "가속도 연산자 설명 필요" | Define $L_{2,N}$ before use | H | T | 🟢 |
| 23 | 157 | "이차형식, 왜 이차형식 정리가 의미 있는지 최적화 정식화 관점에서 설명" | Add 1–2 sentence motivation; rename "quadratic form" → "이차형식" | H | E | 🟢 |

### §4 (method)

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 24 | 171 | "현재 실험 → 본 연구" | Word swap | L | T | 🟢 |
| 25 | 171 | "원점 중심의 경우 — 무슨 의미?" | Clarify Earth-centric KOZ assumption inline | M | T | 🟢 |
| 26 | 173 | "제어다각형 → 제어점" | Term unification | L | T | 🟢 |
| 27 | 173 | "기존 vs 분할된 곡선 제어점 시각 그림 추가" | New diagram for §4.1 | H | M | 🟢 |
| 28 | 179 | "제어다각형 → 제어점" | Covered by #26 | – | – | 🟢 |
| 29 | 223 | "그림 1 캡션 상세 부족" | Expand F1 caption to describe geometry | M | E | 🟢 |
| 30 | 227 | "논문 → 연구" | Word swap | L | T | 🟢 |
| 31 | 227 | "affine 선형화 중력장 — 이게 뭔지 설명 부터" | Define linearization step before §4.2 formula | H | E | 🟢 |
| 32 | 227 | "u(t) 삽입" | Notation insertion | M | T | 🟢 |
| 33 | 233 | "affine linearization — 이게 뭔지 설명 부터" | Covered by #31 | – | – | 🟢 |
| 34 | 241 | "i, A, B, C 무엇인지 먼저 설명, 왜 이렇게 정의되는지, 물리적 의미 보강" | Define notation block; explain physical meaning of each term | H | M | 🟢 |
| 35 | 243 | "IRLS 무슨 과정인지 설명 보강 + 참고문헌 추가" | Add 2–3 sentence IRLS explanation + cite (e.g., Holland & Welsch 1977 / Daubechies et al.) | H | E | 🟢 |
| 36 | 249 | (empty — context: "안전" undefined again) | Same fix as #71/#72 in Phase 3 | M | T | 🟢 |
| 37 | 262 | "ℓ 볼드체 일관, PDF 렌더 깨짐" | Fix Markdown→PDF bold rendering for `\ell` symbol | M | T | 🟢 |
| 38 | 267 | "필요한 경우 — 본 연구는 포함했나? 명확히" | State definitively: prox term is on, weight value | M | T | 🟢 |
| 39 | 289 | "Algorithm 형식으로 정리" | Convert §4.3 procedure to numbered Algorithm pseudocode | H | M | 🟢 |
| 40 | 293 | "제어다각형 → 제어점" (F2 caption) | Covered by #26 | – | – | 🟢 |

### §5.2 — vague experiment description (full rewrite)

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 41 | 319 | "안전 근사 — 무슨 의미?" | Resolved when "안전" is defined upfront (#71/#72) | – | – | 🟢 |
| 42 | 319 | "어떤 변화? — 구체화" | Rewrite §5.2 with explicit predicted outcomes | H | E | 🟢 |
| 43 | 319 | "이 효과 — 무슨 효과?" | Same rewrite | – | – | 🟢 |
| 44 | 321 | "차수별 추세 — 무엇의 추세?" | Specify metric (cost vs degree) | M | T | 🟢 |
| 45 | 321 | "표현 모호, 명확성 떨어짐, 전체 재작성" | **Rewrite §5.2 paragraph from scratch** | C | M | 🟢 |

### §5.3 — pipeline intro (full rewrite)

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 46 | 323 | "Downstream Pass-1-replacement 난해, 한국어 풀어쓰기" | **Rewrite §5.3 in Korean academic prose, define every pipeline term** | C | H | 🟢 |
| 47 | 325 | "downstream 활용 가능성 — 풀어쓰기" | Covered by #46 | – | – | 🟢 |
| 48 | 325 | "matched pipeline-variant 비교 — 풀어쓰기" | Covered by #46 | – | – | 🟢 |
| 49 | 332 | "학술 논문 아니라 codebase 설명서, 전반적 재작성" | Covered by #46 (this is the worst paragraph in the section) | – | – | 🟢 |

### §6 (results)

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 50 | 336 | "수치 삽입" | "결과" → "수치 결과" | L | T | ✓ done |
| 51 | 338 | "질문을 다룬다 어색" | Rewrite §6 opening as "확인한 결과" framing | M | E | 🟢 |
| 52 | 338 | "어떤 → 구체적 명확하게" | Spell out "subdivision count vs cost & runtime" | M | T | 🟢 |
| 53 | 338 | "어떤 → 구체적" (dup) | Same | – | – | 🟢 |
| 54 | 346 | "F3 캡션: 차수별 궤적, 범례 불명, delta-v proxy 무엇" | Rewrite F3 caption + define delta-v proxy | H | E | 🟢 |
| 55 | 381 | "F4: n_seg=2 infeasible 별도 표시" | Annotate marker on F4 | M | T | 🟢 |

### §6.4 — most heavily attacked section

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 56 | 405 | "약어 풀어쓰기, 각 pipeline 그림으로 설명" | Add new pipeline-diagram figure; spell out DCM/H-S/LGL | C | M | 🟢 |
| 57 | 405 | "질문 형식으로 서술하지 말 것" | Rewrite "본 비교가 다루는 질문은…" → declarative | M | T | 🟢 |
| 58 | 405 | "각 pipeline 적절한 소개 없이 매우 부적절" | Covered by #56 | – | – | 🟢 |
| 59 | 407 | "circular 사례 무엇?" | Define case-selection criteria upfront | C | E | 🟢 |
| 60 | 417 | "case들 어디서 갑자기 나왔는지 소개 먼저" | Covered by #59 | – | – | 🟢 |
| 61 | 419 | "사례 20, 114?" | Define what makes them notable when introducing cases | M | T | 🟢 |
| 62 | 419 | "drop-in 대체?" | Drop the term; rewrite in plain Korean | M | T | 🟢 |
| 63 | 419 | "phase 구조 일치 여부에 조건부?" | Rewrite in plain Korean | M | E | 🟢 |
| 64 | 421 | "End-to-end speedup?" | Define on first use; pair with Korean | L | T | 🟢 |
| 65 | 425 | "F6 캡션 1–2줄, 본문에서 설명, 왜 보여주는지" | Tighten F6 caption to 2 lines; move content to body | H | E | 🟢 |
| 66 | 429 | "명사화 표현 과다, 논문 전반 문체 개설" | **Paper-wide nominalization sweep** (touches every section) | C | H | 🟢 |
| 67 | 429 | "관찰 사실 나열일 뿐, 해석 보강" | Add interpretive sentences after each result paragraph in §6.4 | H | M | 🟢 |

---

## PHASE 3 — Line-level polish (do last, after paragraphs are stable)

### Term substitutions (Korean preferred)

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 68 | 28 | "논컨벡스 → 비볼록" | Word swap | L | T | ✓ done |
| 69 | 36 | "section 2 영어 용어 과다 → 한국어" | After §2 folded into §1, do Korean swap pass on merged text | M | E | 🟢 |
| 70 | 38 | "컨벡스화 → 볼록화" | Word swap (paper-wide) | L | T | ✓ done (title + body; line 46 left struck through pending Phase 0 #1) |
| 71 | 48 | "논컨벡스 → 비볼록" | Same as #68 | – | – | ✓ done |

### "안전" definition + forward-reference cleanup

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 72 | 18 | "안전이 무슨 의미인지 명확하지 않음" | Define "연속 안전" precisely on first use in §1 | C | E | 🟢 |
| 73 | 18 | (same, dup) | Covered by #72 | – | – | 🟢 |
| 74 | 30 | "Pass 1 단계 — 설명 없이 등장" | Resolved naturally when §6.4 (#56) introduces pipelines | – | – | 🟢 |
| 75 | 30 | "정의된 작동 영역 안에서의 초기화 대체 가능성 — 명확한 표현" | Rephrase contribution item | M | T | 🟢 |
| 76 | 12 | "downstream 용어 남용" | Resolved when abstract is rewritten (#11) and §6.4 reintroduced | – | – | 🟢 |
| 77 | 12 | "Pass 2 solver·dynamics·tolerance" | Same | – | – | 🟢 |
| 78 | 12 | "two-pass direct collocation pipeline" | Same | – | – | 🟢 |
| 79 | 12 | "end-to-end runtime" | Same | – | – | 🟢 |
| 80 | 12 | "Pass 2" | Same | – | – | 🟢 |

### Citations

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 81 | 18 | "참고문헌 달기" (direct transcription/collocation) | Cite Betts 1998 / Hargraves & Paris (already in refs) at point of claim | H | T | 🟢 |
| 82 | 20 | "참고문헌 달기" (initialization sensitivity) | Add citation (e.g., Betts; or warm-start literature) | H | E | 🟢 |
| 83 | 54 | "참고문헌 달기" (직선 보간, 형상화, DB 초기화) | Add citations for each initialization style mentioned | M | E | 🟢 |

### Vague phrasing cleanups (post-rewrite check)

| # | L | Annotation gist | Action | Imp | Ease | Status |
|---|---|---|---|---|---|---|
| 84 | 54 | "초기화는 연속 안전 — 명확하지 못한 표현" | Rephrase after "안전" defined (#72) | M | T | 🟢 |
| 85 | 56 | "기하학적인 안전 제약 — 명확하지 못한 표현" | Same | M | T | 🟢 |

---

## Resolution summary

- ✓ Fully defused (cache evidence): **1 item** (#9 — §6.1 defense paragraph deletion)
- ✓ Done in this session: **9 items** (#10, #14, #17, #18, #19, #50, #68, #70, #71); #8(a) partially done
- 🟡 Partially defused (interpretation clear, action remains): **2 items** (#7, #8(b))
- 🟢 Active TODO: **75 items**

## Effective unique work units

The 87 annotations collapse to **~37 distinct work units** because of duplicates (#1–3, #56/58/60, etc.) and natural roll-ups (#76–80 all resolve when the abstract and §6.4 are rewritten with proper terminology).

## Critical-path order for actually doing this

1. **Decide frozen-linearization probe** (gate for everything in Phase 1/2 method-side wording).
2. **Phase 0** structural collapses (§2 fold, §7 fold, contributions merge).
3. **Phase 1** rerun + numbers regen + abstract/conclusion rewrite.
4. **Phase 2** in-section rewrites in this order: §3 (foundation) → §4 (method) → §5.1 → §5.2 → §5.3 → §6 → §6.4. §6.4 is biggest and goes last because it depends on §5.3's pipeline definitions.
5. **Phase 3** line-level (term subs, citations, vague-phrase cleanups, paper-wide nominalization sweep #66).

Item #66 (paper-wide nominalization sweep) is the only Phase-3 item that's both *high importance* and *high effort*. It's the one polish-level item worth budgeting real time for, because it touches every section and takes the paper from "edited" to "publishable."

## Items NOT in the PF but raised by prof's verbal feedback (Slack)

These four are not in the 87 but are part of the same review round:

- **V1.** Frozen-linearization probe (listed as Phase-0 decision above)
- **V2.** Two-pass discoverability — `tools/dcm_downstream_experiment.py` and `dcm_baseline/` are not visible from the `orbital_docking/` directory the prof browsed. Add a top-level pointer in README and `Project_Spec.md`. Imp H, Ease T.
- **V3.** "수십초 걸리는게 이상해요" — addressed automatically by Phase 1 (sane tolerance → ~30× runtime drop). No separate action needed.
- **V4.** "5회 이내 수렴해야" — addressed by V1 outcome. If frozen linearization works, true. If not, reframe expectation in §6.1.

## nseg=4 wart (newly visible after Phase 1)

The cache shows nseg=4 has `dnorm = 152` even at 100000 iter — genuinely oscillating, not converging. Once Phase 1 rerun lands and other cases honestly say "converged at iter N < cap," nseg=4 will be the only legitimate non-convergence in T3. Decide before the rerun: drop the row, mark it non-convergent with a footnote, or stabilize it (more proximal damping / smaller trust radius). Don't let it surface unannounced.
