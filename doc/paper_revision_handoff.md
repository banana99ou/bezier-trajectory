# 논문 개정 + SCvx 마무리 — session handoff

_Curated context for the next session. Supersedes the completed portions of `scvx_fix_handoff.md` (그 파일의 검증·수정 작업은 전부 완료·커밋됨)._

## TL;DR 상태 (전부 커밋·푸시됨, 예외는 명시)

- **SCvx 솔버**: dv 모드 완전 삭제 + energy-mode ρ-test 진짜로 만듦 (`1581e54`), 3-tier feasibility 정의 (`863dd64`). 5-pillar 검증 전부 PASS (ρ=1.001 실측, cold-start NLP가 동일 최적해 독립 도달). 골든·Rust 테스트 green.
- **논문 §3.3**: proximal 루프 서술 → 신뢰 구간 SCvx로 재작성 + 전면 문체 정비 94건 (`4589fe1`). 교수 주석 중 언어 관련은 대부분 해소.
- **한국어 문체 시스템**: `doc/korean_writing_case_collection.md`에 Case 33–34(조어 금지), Rules 19–20, **§6 SCP/SCvx 용어 판정표**(코퍼스 조사 기반, 인쇄 인용 포함) 추가 (`1434d0a`). `.claude/skills/korean-prose/` skill 동기화 완료 — **단, `.claude/`가 gitignore라 skill은 로컬에만 존재** (버전 관리 여부 미결).

## 확정된 결정 (되묻지 말 것)

- **용어**: 사례집 §6 판정표가 최종 권위. 핵심: 신뢰 구간(trust region)·볼록(컨벡스 금지, 교수 지시)·실현 가능(실행 가능 금지)·하위 문제·여유 변수·페널티 항/계수·"새로운 기준점으로 간주"(수락 금지)·예측/실제 감소량·임계값·수렴 조건. **전문 용어 사전식 조어 절대 금지** — 인쇄 선례 우선, 없으면 영어 유지.
- **자기 지칭**: 제안 기법(프레임워크 금지), 파이프라인(한글), 후속(downstream 금지), 분할구간(세그먼트 금지).
- **알고리즘**: canonical SCvx **freeze-off**가 논문 서술·실험 기준 (`disable_scvx_freeze=True`). energy가 유일 목적함수 (dv는 코드에서 삭제됨).
- **수정 방식**: D1 — 지적 지점만 국소 수정, 전면 재작성 금지, 편집 전 `.bak-YYYYMMDD` 백업 (최신: `.bak-20260805`).

## 남은 작업 (우선순위)

1. **STALE 3곳 + 표 재생성** — §4.1 솔버 파라미터 문단(구식: 10000회·prox 1e-6·clipping → 신규: 신뢰 구간 초기 크기 2000 km·×2/×0.5 조절·η=0.1·w_s=1e4·tol), §5.1 10000회 합리화 문단(삭제/재작성), §5.3 "10000회 도달" 문장+수치. **freeze-off 재실험으로 T2/T3/T4 숫자부터 재생성해야 문장이 나옴.** T6(§5.4)는 dv 삭제로 downstream 파이프라인 기본값이 깨져 있음(`dymos_t6.py`·`downstream_collocation.py`가 objective_mode="dv" 전달 → 에러) — energy 재실험 시 함께 수정.
2. **`doc/tmp.md`의 개념 질문 답변** — 원래 4단계 계획의 3단계, 아직 미이행. (SCP/SCvx/QP 개념 질문들; ρ-test는 이 세션에서 실물로 배웠으니 답이 훨씬 쉬워짐.)
3. **미해결 내용 주석 12줄** (rev2, `grep -n '\[\['`) — 데이터베이스 설명 요구(§4.3), chaser/target 의문(§4.1), §4.2 전면 재작성 지시, §2.3 흐름 지적(부분 해소), 그림 TODO 등. 언어가 아니라 내용 결정 필요.
4. **소소한 보류**: `T_normed` 기호 정의(표와 연동), 표 머리글 언어 통일, "경험적 형상화" 원어 확인, §1 로드맵("문제 설정과 표기법")↔§2 실제 제목 불일치.
5. **skill 버전 관리**: `.gitignore`에 `!.claude/skills/` 예외 추가 + 커밋할지 로컬 유지할지.
6. (선택) **컨벡스/볼록 divergence를 교수에게 보고할지** — 2025 춘계 KSAS 논문집 컨벡스 17:0. 현재는 교수 지시대로 볼록.

## 운영 gotchas (전 세션에서 비용 치르고 배운 것)

1. **Rust 재빌드**: `.so`는 자동 갱신 안 됨. `cd rust_optimizer/pybind && ../../.venv/bin/maturin build --release -i ../../.venv/bin/python3.11 && cd ../.. && .venv/bin/pip install --force-reinstall --no-deps rust_optimizer/target/wheels/bezier_opt-0.1.0-cp311-cp311-macosx_11_0_arm64.whl` (maturin develop은 실패함).
2. **캐시는 바이너리를 해시하지 않음** — 재빌드 후 `CACHE_VERSION` bump (현재 `8.0-scvx-true-rho`) 또는 `use_cache=False`. 검증 하니스는 항상 cache-off.
3. `.venv/bin/python`(3.11) 사용, 맨 `python3` 금지.
4. **pytest 전체 실행 시** `--ignore=tests/unit/test_dymos_t6.py` (dymos 미설치로 collection 자체가 죽음). `test_constraints_spec.py::test_koz_origin...`은 pre-existing red (warning 미발생).
5. **한국어 산출물은 반드시 `/korean-prose` skill 로드 후 작성** — 이 세션에서 초안 2회 기각의 근본 원인이 조어·번역체였음. 새 어휘가 필요하면 발명하지 말고 KoreaScience/KCI에서 인쇄 선례부터 검색.

## 핵심 파일 포인터

- 논문: `doc/paper_draft_korean_rev2.md` (§3.3가 최신 모범; STALE 3곳 표시는 이 문서 위 참조)
- 문체: `doc/korean_writing_case_collection.md` (§6 판정표 = 용어 최종 권위), `.claude/skills/korean-prose/SKILL.md` (로컬)
- 솔버: `rust_optimizer/core/src/optimizer.rs` (eval_residual_terms = 진짜 ρ; 시그니처에서 objective_mode 삭제됨)
- 검증: `tools/verify/` 5-pillar + `artifacts/verify/VERDICT.md` (OVERALL PASS)
- 이전 handoff: `doc/scvx_fix_handoff.md` (검증 방법론 상세 — 완료된 작업)
