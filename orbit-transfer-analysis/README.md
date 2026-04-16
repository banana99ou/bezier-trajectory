# Orbit Transfer Analysis

LEO-to-LEO 연속 추력 궤도전이의 최적 추력 프로파일 분류 연구

## 개요

본 프로젝트는 저궤도(LEO) 간 전이 임무에서 최적 연속 추력 프로파일의 형태를 체계적으로 분류하고, 궤도 파라미터와 추력 프로파일 유형 간의 관계를 규명하는 연구입니다.

## 연구 목표

1. **파라메트릭 데이터베이스 구축**: 다양한 궤도 구성에 대해 직접 배치법(Direct Collocation)으로 최적 궤적 생성
2. **추력 프로파일 분류**: 피크 특성에 따라 단봉형(unimodal), 쌍봉형(bimodal), 다봉형(multimodal)으로 분류
3. **지배 파라미터 규명**: 추력 프로파일 형태를 결정하는 주요 궤도 파라미터 식별
4. **연속-이산 추력 관계 분석**: 연속 추력 프로파일과 고전적 임펄스 기동 간의 대응 관계 정량화

## 방법론

### 2단계 직접 배치법 (Two-Pass Direct Collocation)
- **Pass 1**: Hermite-Simpson 배치 (균일 노드) - 초기 해 및 피크 검출
- **Pass 2**: Multi-Phase LGL 배치 (적응적 노드) - 고정밀 해

### 적응적 샘플링 전략
- 라틴 하이퍼큐브 샘플링 (LHS)으로 초기 커버리지 확보
- 가우시안 프로세스 분류 + 엔트로피 기반 능동 학습으로 결정 경계 집중 샘플링

## 디렉토리 구조

```
orbit-transfer-analysis/
├── manuscript/          # 논문 원고 (LaTeX)
│   ├── manuscript.tex   # 메인 문서
│   └── cas-refs.bib     # 참고문헌
├── docs/                # 기술 문서
│   ├── database_sampling_spec.md           # 적응적 샘플링 명세서
│   ├── LEO-to-LEO 궤적최적화 구현 명세서.md  # 구현 명세서
│   └── Computational Methods for.../       # Bézier 궤적 설계 방법론
├── 글감/                 # 연구 노트 및 분석 자료
└── src/                 # 소스 코드 (예정)
```

## 구성 파라미터

| 파라미터 | 기호 | 범위 | 단위 |
|---------|------|------|------|
| 장반경 변화 | Δa | [-500, +2000] | km |
| 경사각 변화 | Δi | [0, 15] | deg |
| 전이 시간 | T/T₀ | [0.5, 5.0] | - |
| 초기 고도 | h₀ | {400, 600, 800, 1000} | km |

## 참고문헌

주요 자기인용:
- Lee, S. (2023). A Generative Verification Framework on Statistical Stability for Data-Driven Controllers. *IEEE Access*, 11, 5267-5280.
- Lee, S. & Kim, Y. (2021). Optimal Output Trajectory Shaping Using Bézier Curves. *JGCD*, 44(5), 1027-1035.
- Lee, S. (2025). A Shape-based Approach Suited for Short-Duration Orbit Transfer Trajectory Design. *EUCASS 2025* (to appear).

## 사사

본 연구는 국방기술진흥연구소(KRIT)의 지원을 받아 수행되었습니다 (No. KRIT-CT-22-030, ReUSV-41, 2025).

## 저자

이수원 (Suwon Lee)
국민대학교 미래모빌리티학과
suwon.lee@kookmin.ac.kr
