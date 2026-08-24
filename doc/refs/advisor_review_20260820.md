# Advisor review, 2026-08-20 — gist

Sidecar to [`advisor_review_20260820.pdf`](advisor_review_20260820.pdf), 4 pages, by 이수원
(Future Mobility Control Laboratory, Kookmin University). Received over Slack in reply to the two
notes sent 2026-08-19.

**This document outranks any agent's assertion about paper scope, venue, or the objective
function.** Where this file and a solver-side claim disagree, this file wins and the disagreement
is recorded here.

**Numbering gotcha.** The review cites 연구노트 004 and 009. Those are the numbers the PDFs carried
when they were sent. In this repository's `doc/notes/` they are now **002** (probabilistic KOZ) and
**003** (paper-1 report to advisor). Same documents, renumbered after the notes folder was
reorganised. Do not create new 004/009 directories chasing the review's numbers.

---

## 1. The item on a clock — venue

The review's central judgement: **two pages is too small for this content.** 항공우주학회's 2-page
format is effectively an extended abstract, so it buys discussion at the venue rather than a
finished argument, and the theoretical core — 지지 반공간 구성, 매 반복 재구성, 볼록성 논의 — cannot
be developed in that space.

Proposal: submit to **대한기계학회 추계학술대회 (max 6 pages)** as well. It runs 동역학·제어·로봇
sessions where 궤적최적화 / 이동 장애물 회피 / 자율이동체 경로계획 appear regularly, so the scope
fits.

| | 한국항공우주학회 추계 | 대한기계학회 추계 |
|---|---|---|
| 지면 | 2 pages | **6 pages** |
| 발표신청 마감 | — | **2026-08-26 (수)** |
| 논문제출 마감 | 2026-09-04 (금) | 2026-09-02 (수) |

The review's own words: 기계학회로 가려면 **이번 주 안에 결정하고 신청부터 해야 한다.**

The 종합 section frames it as both venues — 항공우주 kept light and aimed at discussion, the
theory carried by the 6-page 기계학회 manuscript.

## 2. What the review approved

**The objective function — our plan stands.** 매개변수 영역 매끄러움 정규화 + 도착시각 페널티,
물리량은 제약으로. The review supplies the sentence that connects it to the standard construction
and says one sentence is enough:

> 시각이 매개변수에 대해 선형인 경우 이 항은 가속도 에너지와 일치한다.

It also confirms 도착시각 페널티와 속도 제한을 반드시 함께 도입한다 is sound.

**One instruction we did not have.** 장애물 페널티 항은 목적함수에서 **뺀다** — avoidance is a hard
constraint through the 지지 반공간, so it does not belong in the cost. SCP's slack variables and
their penalty may still appear as a numerical device, and the write-up must say plainly that this
is not the avoidance mechanism.

**The clipping idea is sound**, and the review connects it to the advisor's own in-progress work on
spatial branch-and-bound for global-optimality guarantees (연구노트 005, temporal-profile-shaping,
2026-06). See §5 below — that note has not been read.

## 3. The structure the review supplied

We had reached "잘라내기는 신뢰 구간과 같은 성격의 조정 방식을 따로 둔다" and could not fill in what
the update indicator would be. The review fills it, by defining each mechanism as three elements:

| | 신뢰 구간 | 잘라내기 |
|---|---|---|
| 조정 변수 | 반경 Δ | 시공간 창의 길이 W |
| 유효성 조건 | 선형화 근사의 성립 | 도달 가능성 — W ≥ Δ + 장애물 크기 |
| 갱신 지표 | 예측 감소 대비 실제 감소의 비 (ratio test) | 잘린 조각의 볼록껍질과 실제 영역 사이의 간격, 또는 그 조각의 곡률 |

The point of writing it this way, in the review's words: it makes explicit that **W's lower bound
is dependent on Δ, while the margin above that lower bound is an independent tuning target.** That
is the precise answer to the question we left open — neither "tie them together" nor "keep them
fully separate."

Recording each indicator per iteration gives the basis for designing an adaptive rule and for
verifying that it works. The review notes this has the same form as choosing a branch variable and
contracting a box in the branch-and-bound work.

**The 갱신 지표 is already measured.** The 2026-08-19 convexity sweep recorded exactly this
quantity — worst gap between a clipped piece's convex hull and the true region, against window
length. It lives in the shadow-convexity measurement, not in a prose file. Reuse it rather than
re-deriving it.

## 4. What must be added or changed

**가시선 차폐 제약의 임무 관점 동기 (§2.2).** The constraint is currently presented only as a
methodological differentiator — something convex-decomposition methods cannot express — with no
statement of *어떤 임무, 어떤 동기로* it is needed in aerospace practice. The reviewer wants to know
what mission is in mind. Required in both the manuscript and the talk.

**제목 — decide after the venue (§2.4).**

- 항공우주학회 → mission-situation first. The review endorses our current 우선안: *지상국 가시선을
  유지하며 이동 장애물을 회피하는 3차원 궤적 최적화*.
- 기계학회 → method first. The reviewer's own candidate: *신뢰구간 기반 지지 반공간 재구성을 이용한
  공간-시간 이동 장애물 회피 궤적최적화*, on the grounds that 신뢰 구간 carries both the validity of
  the linearization and the justification of the clipping, making it the formulation's central
  device.

branch-and-bound 전역최적성 is the reviewer's own direction and is **not** in this manuscript's
scope — keep it out of the title, mention it only as future work.

## 5. On 연구노트 002 (probabilistic KOZ) — future work, three gaps

The review agrees this stays out of the conference manuscript. Three items to strengthen before it
becomes paper 2:

1. **정규화 제약.** A probability density must integrate to 1 over space. That requires the obstacle
   to exist *somewhere* at all times, and it means the model describes the obstacle's **position**,
   so its **volume** is not directly represented. A high-probability region may stand in for volume,
   but whether that correspondence is principled needs checking.
2. **점유(occupancy) 관점.** For collision avoidance the question is not "where is this object" but
   "is this space-time position dangerous." Modelling each position's risk may fit better than
   modelling each object's position probability. The note's own `ρ = Σ pⱼ` already does not
   integrate to 1, so it is not a density — it is closer to an occupancy quantity. **Survey the
   위험밀도장 / occupancy literature.**
3. **로브 반지름 식 (3)의 유도 근거.** The step from eq. (2) to eq. (3) is omitted; supply the
   derivation or a reference.

## 6. Not yet read — the branch-and-bound note

Reference [3] of the review:

> S. Lee, "단일 볼록화를 넘어: spatial branch-and-bound에 의한 전역최적성 보증," Research Note 005,
> temporal-profile-shaping, 2026-06.
> <https://fmcl-nextcloud.duckdns.org/f/37414> — 연구실 Nextcloud 계정 필요.

**Fetching it returns HTTP 403** and no local copy or Nextcloud client exists on this machine. The
review says it is also registered in the FMCLResearch app. Until it is read, the §3 claim that the
clipping construction "has the same form as" the branch-and-bound box contraction is the
reviewer's assertion, taken on trust and not verified here.
