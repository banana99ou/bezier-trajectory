# KSAS 2026 추계학술대회 template — provenance

Fetched **2026-08-20** from the conference's 발표논문 제출 페이지
(`http://ksas.or.kr/Conference/ConferencePaper.asp?AC=0&CODE=CC20260701&CpPage=P`).

| file | source URL |
|---|---|
| `ksas_2026_fall_template.hwp` | `http://ksas.or.kr/UploadData/Editor/Conference/202607/CB4AE170E3C041AAA9065E054E9DB8A1.hwp` |
| `ksas_2026_fall_template.docx` | `http://ksas.or.kr/UploadData/Editor/Conference/202607/FAE005E7F02E41E1B49A5E6E638CC620.doc` (served as `.doc`, is a real docx) |
| `ksas_2026_fall_template.pdf` | rendered locally by LibreOffice from the docx — **derived**, not from the society |

**HTTPS on ksas.or.kr is broken** (TLS handshake fails; `https://` returns nothing). Plain HTTP
over IPv4 works. Any tool that force-upgrades to HTTPS will fail to reach this site.

## Rules read off the template itself

- A4, **two-column body** (single-column title block), margins top 3.5 cm / bottom 2.5 cm /
  left-right 2.0 cm, column gap 1.0 cm. Header is preprinted: "한국항공우주학회 2026 추계학술대회 논문집".
- Title block: 국문 제목, 저자 (**발표자에 `*`**), 소속, 영문 제목, 영문 저자 — same author order as
  the web submission form.
- **Key Words**: 2줄 이내, 국영문 병기 형식 (`Domain/Boundary Decomposition(영역/경계 분할)`).
- Body font: 한글·영문 모두 **굴림체**; use the template's named styles (본문내용, 소제목,
  세부절 제목, 그림표캡션, 수식).
- Section headings fixed: 서 론 / 본 론 / 결 론 / 후 기 / 참고문헌. Blank line between sections.
- Citations are **superscript numbers in parentheses**: `홍길동 등(1)`, `설계방안(2,3)`, `이론이 제안되었다(2~4)`.
- **참고문헌 5개 이내**, all in English, KSAS journal citation style. Table captions above,
  figure captions below, **both in English**.
- 수식 numbered sequentially with parentheses.
- All guide sentences must be deleted before submission.

## Rules read off the submission page

- **논문(초록 및 최종본) 제출 마감: 2026년 9월 4일(금)**, online. The 400자 이내 초록 is typed into
  the web form, separately from the manuscript.
- Presenting requires **both**: society membership with the annual fee paid, **and** 사전등록
  신청 + 결제 완료. 사전등록 마감 is 10/30, but the submission page lists paid registration as a
  precondition for submitting — treat it as due before 9/4.
- Oral vs poster: poster-requested papers and **late oral requests** go to poster. Oral talks start
  11/11(수) morning. Submitting early is what buys an oral slot.
- All co-authors must be entered on the website in the same order as the manuscript file.

## Not verified — do not state as a rule

The manuscript **page limit**. The template file is exactly 2 pages of A4, and the repo has been
assuming "2 pages" since 2026-08-19, but neither the 논문모집안내 page nor the 제출 page states a
limit for the regular conference. (The "A4 4쪽 이내" on that page belongs to the separate
항공우주산업·정책 논문 경진대회, which is a different submission path.) The 2-page figure is the
template's own length, not a quoted rule. If the difference would change what we cut, ask the
society office (ksas@ksas.or.kr) rather than inferring.
