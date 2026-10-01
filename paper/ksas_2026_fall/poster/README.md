# 포스터 — KSAS 2026 추계학술대회

원고는 [`../README.md`](../README.md), 주장과 정식화는
[`idea/spacetime.md`](../../../idea/spacetime.md).

## 빌드

```bash
python3 tools/render_poster.py          # 게이트 → numbers.tex → 그림 6장 → xelatex 2회
python3 tools/render_poster.py --numbers-only
python3 paper/ksas_2026_fall/poster/make_poster_figures.py [--png] [--only fig_clip ...]
python3 paper/ksas_2026_fall/poster/lift_viewer.py   # 3차원 그림의 시점을 고를 때만
python3 -m pytest tests/unit/test_poster_gate.py
```

TeX Live(`xetexko`, `a0poster`, `geometry`), **NanumBarunGothic**, matplotlib이 필요하고 이 기계에는
있다. `poster.pdf`, `fig_*.pdf`, `lift_viewer.html`은 렌더이므로 `.gitignore`에 걸려 있다.

| 파일 | 무엇 |
|---|---|
| `poster.tex` | 본문. A0 세로, `a0poster`, 머리글 + 2단 + 바닥글 |
| `make_poster_figures.py` | Fig. 2--5와 아직 쓰지 않는 `fig_iterate`, `fig_runs` |
| `lift_viewer.py` | 3차원 장면을 브라우저에서 돌려 `view_init(elev, azim)`을 읽는 도구 |
| `numbers.tex` | **생성물.** 손으로 고치지 말 것 |

Fig. 1(`concept_figure.pdf`)과 Fig. 6(`occlusion_figure.pdf`)은 원고 그림을 그대로 쓴다.

## 두 가지 규칙

**결과 숫자는 손으로 적지 않는다.** 전부 `figures/paper1/occlusion_figure.json`에서
`tools/render_poster.py`가 뽑아 `numbers.tex`에 넣는다. 그 사이드카가 인증하지 않는 실행이 되거나
포스터가 주장하는 바를 더는 보이지 못하면 **빌드를 거부한다.** 거부 조건이 실제로 발동한다는 것은
[`tests/unit/test_poster_gate.py`](../../../tests/unit/test_poster_gate.py)가 변조 사이드카 여덟 개로
확인한다.

**높이는 빌드 로그가 말한다.** `POSTER left/right height` 두 줄이 각 단의 높이(pt)다. 어느 쪽이든
약 2890 pt를 넘으면 두 쪽이 되고, `pdfinfo`의 `Pages: 1`이 그 확인이다.

## 인쇄 규격 — A0 세로, 확정

보드판 95 × 238 cm 세로, A0 부착 가능, 학회 템플릿 없음, 논문번호는 대회본부가 붙인다
(근거와 원문은 [`../template/PROVENANCE.md`](../template/PROVENANCE.md) §포스터 규격).
바꾸려면 `poster.tex`의 `\documentclass` 첫 인자와 `geometry`의 종이 크기 두 줄이다.
