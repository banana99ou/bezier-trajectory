---
name: reusv-2026-plan 프로젝트 겹침 분석
description: bezier-orbit-transfer-scp와 reusv-2026-plan 간 공유 가능한 코드/이론 5개 영역 식별 (2026-03-28)
type: project
---

bezier-orbit-transfer-scp와 /Users/suwon/research/reusv-2026-plan 간 겹침 분석 결과.

**Why:** 두 프로젝트가 Bernstein/Bézier 기반 궤적 최적화라는 동일한 수학 프레임워크를 공유하며, 코드 재사용 가능성이 높다.

**How to apply:** reusv 재진입 구현 시 bezier-orbit의 기존 모듈을 우선 활용. 공통 패키지 분리 검토.

## 겹치는 5개 영역

1. **Bernstein 기저/평가**: bezier-orbit `basis.py` → reusv에서 독립 구현한 `bvec()`, `bez()` 대체 가능
2. **구간별 선형화 + O(1/K²) 수렴**: bezier-orbit 보고서 009 (중력) ↔ reusv 보고서 009 (양력/밀도) — 동일한 수학 프레임워크
3. **Bernstein 대수 (곱/합성/차수축소)**: bezier-orbit `algebra.py`의 `gravity_composition_pipeline` 패턴 → reusv에서 `ρ(h)` 합성에 응용 가능
4. **SCP 구조 (CVXPY, trust region)**: bezier-orbit `inner_loop.py` → reusv 재진입 SCP에 구조 재사용
5. **자기일관적 비선형항 보정**: bezier-orbit `_corrected_position_algebraic()` → 재진입 밀도/항력 보정에 일반화 가능
