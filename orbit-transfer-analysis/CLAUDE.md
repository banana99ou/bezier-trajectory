# 논문 작성 프로젝트

## 프로젝트 개요

- 논문 제목: Classification of Optimal Continuous Thrust Profiles for Low Earth Orbit Transfer Missions: A Parametric Database Study
- 대상 저널: Acta Astronautica (Elsevier)
- 주요 키워드: Low-thrust trajectory optimization, Continuous thrust profile, Low Earth orbit transfer, Direct collocation, Parametric study

## 연구 개요

LEO-to-LEO 연속 추력 궤도전이 임무에서 최적 추력 프로파일을 체계적으로 분류하는 연구.
- Direct collocation 기반 최적 궤적 데이터베이스 구축
- 추력 프로파일 유형 분류 (unimodal, bimodal, multimodal)
- 임펄스 기동과 연속 추력 해의 정량적 관계 규명

## 디렉토리 구조

```
.
├── manuscript/          # 논문 원고 (Elsevier CAS double-column)
│   ├── manuscript.tex   # 메인 원고
│   ├── cas-refs.bib     # BibTeX 참고문헌
│   ├── cas-dc.cls       # Elsevier CAS 클래스 파일
│   └── manuscript.pdf   # 컴파일된 PDF
├── docs/                # 참고 논문 PDF 및 명세서
├── els-cas-templates/   # Elsevier CAS 템플릿 원본
├── sessions/            # 세션 기록
└── 글감/                # 연구 목적 및 참고 자료
```

- 데이터 경로는 상대 경로를 사용하라.

## 추적관리 문서 (traceability.md)

프로젝트 루트에 `traceability.md`를 유지하라. 이 문서는 논문 내 모든 그림, 표, 본문 수치가 어디서 생성되었는지 추적하기 위한 문서이다.

### 관리 대상

- **그림(Figure)**: 그림 파일 경로, 생성 스크립트 경로, 사용된 데이터 소스
- **표(Table)**: 표에 포함된 수치의 데이터 소스, 생성 스크립트 (있는 경우)
- **본문 수치**: 본문에 등장하는 정량적 수치와 해당 데이터 소스

### 문서 형식

```markdown
# Traceability

## Figures

| Figure | 파일 경로 | 생성 스크립트 | 데이터 소스 | 비고 |
|---|---|---|---|---|
| Fig. 1 | latex/figures/fig1.pdf | scripts/plot/fig1.py | data/processed/case01.db | |

## Tables

| Table | 위치 (섹션) | 생성 스크립트 | 데이터 소스 | 비고 |
|---|---|---|---|---|

## In-text Values

| 수치 | 위치 (섹션/문장) | 데이터 소스 | 산출 방법 | 비고 |
|---|---|---|---|---|
```

### 갱신 규칙

- 그림, 표, 본문 수치를 생성하거나 수정할 때 반드시 `traceability.md`를 함께 업데이트하라.
- 기존 항목을 삭제할 때도 문서에서 제거하라.
- 스크립트가 없이 수동으로 생성된 항목은 비고에 "수동 생성"이라고 기재하라.

## 작성 규칙

### 영어 표현

- 한국어 화자가 읽기에 자연스러운 간결한 학술 영어를 사용하라.
- 짧은 문장을 선호하라. 한 문장에 절이 3개 이상 이어지면 분리하라.
- 불필요한 수동태, 형식적 표현(it is worth noting that 등)을 피하라.

### LaTeX 규칙

- 벡터: \mathbf{x} (볼드 소문자)
- 행렬: \mathbf{A} (볼드 대문자)
- 스칼라: 이탤릭 (기본)
- 집합: \mathcal{S} (캘리그래피)
- 수식 번호는 참조되는 수식에만 부여하라.
- \label, \ref, \cite 키는 일관된 네이밍을 유지하라.
  - 수식: eq:이름 (예: eq:dynamics)
  - 그림: fig:이름 (예: fig:trajectory)
  - 표: tab:이름 (예: tab:results)
  - 섹션: sec:이름 (예: sec:method)

### 참고문헌

- BibTeX cite key: 저자연도 형식 (예: Kim2023)
- 모든 참고문헌은 cas-refs.bib에서 관리하라.
- 본문에 없는 논문을 인용하지 마라.

## 주의사항

- 본문에 없는 내용을 추가하지 마라.
- 정량적 수치는 데이터/원고의 값을 정확히 인용하라.
- 파일 수정 전 변경 사항을 사용자에게 보고하라.

## 답변 방식

- 모든 답변은 한국어로 답변하도록 한다.

## Git Commit Guidelines

커밋 메시지 작성 규칙:
- **한국어 개조식**으로 간결하게 작성
- "Claude Code로 작성" 등 AI 도구 관련 문구 **포함하지 않음**
- Co-Authored-By 태그 **사용하지 않음**

예시:
```
프로젝트 초기 구성 및 분석 환경 구축

- DuckDB 기반 데이터베이스 구조 설계
- 논문 A/B 분석 스크립트 작성
- 출판 계획 및 분석 문서화
```

## Python 환경 설정

- 모든 Python 스크립트는 conda 가상환경에서 실행
- 가상환경 디렉토리: `./env` 또는 `./venv`
- 의존성 관리: `requirements.txt` 또는 `environment.yml` 생성

## 대화 간 세션 내용 보존

- 대화가 길어지거나 내용을 요약해야 할 때마다, `sessions/` 디렉토리에 현재까지의 대화 내용을 요약하여 별도의 마크다운 파일로 저장한다. 파일 제목은 날짜 및 시간 기준으로 작성하여 겹치지 않도록 한다.
- 저장된 세션 내용 요약 마크다운 파일들을 읽어서 다른 대화에서도 내용을 기억할 수 있도록 한다.
