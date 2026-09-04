# 포스터 — KSAS 2026 추계학술대회

**원고:** [`../README.md`](../README.md) — 제출 완료 2026-09-03, venue rules, manuscript state.
**Idea:** [`idea/spacetime.md`](../../../idea/spacetime.md) — 주장, 정식화, 기하, novelty.
**숫자:** 전부 `figures/paper1/occlusion_figure.json`에서 나온다. 이 디렉터리의 어떤 파일에도
결과 숫자를 손으로 적지 않는다.

## 빌드

```bash
python3 tools/render_poster.py        # numbers.tex 생성 → 그림 5장 생성 → xelatex 2회 → poster.pdf
python3 tools/render_poster.py --numbers-only
python3 paper/ksas_2026_fall/poster/make_poster_figures.py   # 그림만 다시 그릴 때
```

필요한 것은 TeX Live(`xetexko`, `a0poster`, `geometry`), **NanumBarunGothic**, matplotlib이고, 이
기계에는 이미 있다. `poster.pdf`와 `fig_*.pdf`는 렌더이므로 `.gitignore`(`paper/**/*.pdf`)에
걸려 있다. `make_poster_figures.py --png`가 만드는 미리보기 PNG는 걸려 있지 않으니 커밋하지 말
것. 소스는 `poster.tex`, `make_poster_figures.py`, 그리고 생성물인 `numbers.tex`.

| 파일 | 무엇 |
|---|---|
| `poster.tex` | 본문. A0 세로, `a0poster` 클래스, 머리글 + 상단 띠(리프팅 그림 + 임무 배경) + 2단 + 바닥글 |
| `make_poster_figures.py` | 포스터의 그림 5장을 그리는 스크립트. 아래 표 |
| `numbers.tex` | **생성물.** 손으로 고치지 말 것 |

| 그림 | 포스터 위치 | 출처 |
|---|---|---|
| `fig_lift.pdf` (Fig. 1) | 상단 띠, 폭 58 % | (a) schematic, (b) `scenario_loiter()` + 사이드카의 두 반환 곡선 |
| `concept_figure.pdf` (Fig. 2) | 상단 띠 오른쪽 | `figures/paper1/`, 원고의 Fig. 1을 벡터 그대로 |
| `fig_clip.pdf` (Fig. 3) | 왼쪽 단, 3절 | schematic. $r=\max(d,E+\delta)$, 클리핑된 볼륨 $L$, 그 위의 벽 |
| `fig_halfspace.pdf` (Fig. 4) | 왼쪽 단, 4절 | schematic. 거리 극대점에서 자른 두 근접 구간, 벽 두 개 |
| `occlusion_figure.pdf` (Fig. 5) | 오른쪽 단, 8절 | `figures/paper1/`, 원고의 Fig. 2를 벡터 그대로 |
| `fig_iterate.pdf`, `fig_runs.pdf` | **포스터에 없음.** 그려 두었으나 지면이 없어 뺐다 | schematic / 사이드카 |

그림 양식은 저자의 `clipwall` 그림(2026-08-26, 세션 스크래치패드에서 복구)을 따른다: 흰 바탕,
등축, 옅은 격자, KOZ는 회색 관에 얇은 slate 중심선, 클리핑 구는 빨간 점선, 클리핑된 볼륨은 벽마다
한 색(주황, 보라), 벽은 그 색의 굵은 선 하나, 허용 영역은 민트, 벽이 제외하지만 KOZ가 아닌 곳은
흰색으로 남겨 보수성이 보이게. 그림은 인쇄 폭의 절반으로 그려서(단 폭 약 14.8 in, 그림 7.4 in)
12 pt 글자가 24 pt로 찍힌다. 측정 실행에서 나온 그림(`fig_runs`, Fig. 1(b)의 두 곡선)은 원고
그림의 팔레트(파랑 제안, 검정 기준)를 그대로 쓴다.

## 왜 이 포스터가 있는가

2페이지 원고가 잘라낸 논증을 되살리는 자리다. 되살린 것은 세 가지 —
**지지 반공간 구성**(4절), **매 반복 재구성**(4절 끝), **볼록성 논의**(6절). 저널판이 같은 세 가지를
6페이지로 전개한다.

그리고 발표 자리이므로 **임무 동기를 말로 설명해야 한다**. 1절이 그 자리이고, 지도교수
검토의견이 원고와 발표 양쪽에 요구한 항목이다.

## 숫자가 손으로 들어올 수 없는 이유

`tools/render_poster.py`가 빌드 전에 사이드카를 검사하고, 통과하지 못하면 **빌드를 거부한다.**
거부 조건은 각각 "이 데모가 데모이기를 그만두는 방식" 하나씩이다:

- certificate > 1e-6, slack > 1e-6, `unsound_clips` ≠ 0 → 인증하지 않는 실행이다
- clearance ≤ 0 → 반환된 궤적이 뚫고 지나간다
- **기준 실행이 링크를 잃지 않으면** → 보여줄 것이 없다
- **제약을 부과한 실행이 링크를 지키지 못하면** → 기법이 동작하지 않은 것이다
- **기준 시각 배분으로 날린 경로가 링크를 지키면** → 재타이밍이 살린 것이 아니고, 포스터의
  중심 주장이 거짓이다

이 게이트가 실제로 실패할 수 있다는 것은 [`tests/unit/test_poster_gate.py`](../../../tests/unit/test_poster_gate.py)가
증명한다 — 여덟 개의 변조 사이드카가 각각 거부되는 것을 확인하고, `numbers.tex`가 사이드카가
생성하는 것과 글자 단위로 같은지, `poster.tex`에 결과 숫자가 리터럴로 박혀 있지 않은지, 그리고
**그림에 찍히는 도착 시각과 링크 상실 구간이 사이드카를 바꾸면 따라 바뀌는지**(변조 사이드카로
`fig_lift.pdf`를 다시 그려 `pdftotext`로 읽는다)도 함께 본다.

```bash
python3 -m pytest tests/unit/test_poster_gate.py
```

## 인쇄 규격 — A0 세로, 확정

학회 포스터 발표 안내(2025 추계, 2026 춘계 동일 문구): 보드판 95 cm(가로) × 238 cm(세로), 논문
내용은 A4 8장 이내 또는 비슷한 크기, **포스터로 제작할 경우 A0 부착 가능.** 논문번호는 대회본부가
보드 상단에 미리 붙이므로 포스터에 넣지 않는다. 학회 템플릿은 없다.

그래서 **A0 세로(841 × 1189 mm)** 이고 바꿀 계획이 없다. 그래도 바꿔야 한다면 `poster.tex`의
`\documentclass` 첫 인자(`a0`)와 `geometry`의 종이 크기 두 줄이다. 글자 크기는 `a0poster`가
종이에 맞춰 정하고(A0 본문 24.9 pt), 폭은 전부 `\textwidth`의 비율이다. 빌드 로그의
`POSTER band/left/right height` 줄이 상단 띠와 두 단의 높이(pt)이니, 바꾼 뒤에는 그 합이
`\textheight`를 넘지 않는지와 `pdfinfo`의 `Pages: 1`을 확인할 것.
