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

## 포스터 규격 — fetched 2026-09-03

**The 2026 추계 poster guidance is not published as of 2026-09-03.** The conference page has no
발표 안내 tab, the 모집 안내문 PDF names 포스터 only as the overflow rule (구두 신청 선착순 200편
이후 접수분), and both notice boards carry nothing. KSAS publishes 포스터 발표 안내 as a page of the
프로그램 안내 PDF two to three weeks before the event; expect it in mid-to-late October 2026.

The two most recent conferences state the rule in identical words, so it is being treated as
stable. **Labelled as a prior conference's spec, not this one's:**

| file | conference | source URL |
|---|---|---|
| `ksas_2026_spring_poster_guide_p16.pdf` | 2026 춘계 프로그램 v8, p.16 | `http://ksas.or.kr/UploadData/Editor/Conference/202603/4AB4FADC257C43CF9EF56C67750BD96F.pdf` (linked from `ConferenceView.asp?AC=0&CODE=CC20251202&CpPage=1323`) |
| `ksas_2025_fall_poster_guide_p16.pdf` | 2025 추계 프로그램, p.16 | `http://ksas.or.kr/UploadData/Editor/Conference/202511/0CEB…pdf` (linked from `ConferenceView.asp?AC=0&CODE=CC20250701&CpPage=1308`) |

Verbatim 포스터 발표 준비 요령 (2026 춘계; 2025 추계 is word-for-word the same):

> 1) 보드판 크기 : 95cm(가로)×238cm(세로)
> 2) 내용은 간결하고 분명할 것
> 3) 논문 내용은 A4용지 8장 이내로 하거나 이와 비슷한 크기로 제한함 (포스터로 제작할 경우 A0 사이즈 부착가능)
> 4) 포스터 발표 Panel 견본 참조
>
> ☞ 논문번호는 프로그램에 주어진 번호로서 대회본부에서 부착합니다.
> ☞ 보드판의 논문번호 외에는 본인이 직접 만들어서 발표시작 20분 전까지 주어진 번호의 보드판에 부착하여야 합니다.
> ☞ 우수논문 선정을 위하여 포스터 세션 좌장께서 채점을 진행할 예정입니다. 저자 중 1명은 필히 발표시간 동안 포스터 앞에서 질문에 답변하여야 합니다.

What that fixes for the poster: **A0 portrait (841 × 1189 mm), printed** — the society offers no
digital option and ships **no poster template** (only the .hwp/.doc paper templates above). The
board is 95 cm wide, so A0 sits with about 5 cm to each side; the 논문번호 label is affixed by
대회본부 at the top and must not be printed on the poster. Put up by 20 minutes before the session;
one author stands at the board for the whole session because a 좌장 scores it for 우수논문.
Session date/time/venue for 2026 추계 are unknown until the 프로그램 안내 appears.
