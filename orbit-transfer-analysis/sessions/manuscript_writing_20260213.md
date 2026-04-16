# 논문 작성 세션 - 2026-02-13

## 진행 사항

### 완료된 작업
1. **Phase 1.1**: 단일 케이스 검증 (h0=400, da=200, di=5, T=2.0 → bimodal, 0.72s)
2. **Phase 3.0**: Sec 3.1 수정 — fmincon → CasADi/IPOPT
3. **Phase 4.2**: 참고문헌 37건 검증 완료 (33 확인, 3 수정, 1 미확인-자기인용)
4. **Phase 2.0**: generate_figures.py (11개 그림), generate_tables.py (4개 표) 작성
5. **Phase 3.3a**: Sec 6 Discussion 초안 완성 (물리적 해석, 임펄스 대응, 실용적 시사점, 한계점)
6. **Phase 3.4a**: Sec 7 Conclusions 뼈대 완성
7. **Sec 4/5 구조 작성**: 본문 텍스트, 그림/표 참조 구조 완성
8. **DB 빌드 완료**: 4개 고도 슬라이스 × 500 = 2000 총 샘플, 1780 수렴 (89.0%)
9. **그림/표 생성**: 11개 그림 + 5개 표 모두 생성 완료
10. **수치 채우기**: 모든 placeholder를 실제 DB 수치로 교체
    - Table 2: 수렴 기준 클래스 분포 (Uni=520, Bi=598, Multi=662)
    - Table 3: 클래스별 비용 통계
    - Table 4: ARD 길이 척도 (T/T₀=0.039~0.248, Δa=0.285~1.856, Δi=0.374~0.702)
    - Sec 7: N_TOTAL=1780, CONV_RATE=89.0%, UNI=29.2%, BI=33.6%, MULTI=37.2%
11. **그림 includegraphics**: 11개 모두 주석 해제
12. **English polish**: Sec 1~7 학술 영어 다듬기 완료
13. **traceability.md 최종 업데이트**: 실제 수치로 갱신
14. **최종 PDF 컴파일**: 19페이지, 에러 없음

### DB 빌드 결과
| h0 | Total | Converged | Rate | Uni | Bi | Multi |
|----|-------|-----------|------|-----|-----|-------|
| 400 | 500 | 431 | 86.2% | 126 | 139 | 166 |
| 600 | 500 | 436 | 87.2% | 116 | 166 | 154 |
| 800 | 500 | 460 | 92.0% | 141 | 149 | 170 |
| 1000 | 500 | 453 | 90.6% | 137 | 144 | 172 |
| **Total** | **2000** | **1780** | **89.0%** | **520** | **598** | **662** |

### 주요 버그 수정
1. **GP CompoundKernel**: sklearn OvR GPC가 CompoundKernel 생성 → `kernel.kernels` 리스트로 접근하도록 수정
2. **normalize_params 열 순서**: sorted key 순서 `[T_normed, delta_a, delta_i]` 일치하도록 수정
3. **Table 2 클래스 분포**: 전체(2000) → 수렴 기준(1780)으로 수정

## 원고 최종 구조
- Sec 1 (Introduction): 완료
- Sec 2 (Problem Formulation): 완료
- Sec 3 (Methodology): 완료 (CasADi/IPOPT, less than 3s)
- Sec 4 (Results: Database): 완료 — Table 2-4, Fig 1-8
- Sec 5 (Results: Geometric): 완료 — Fig 9-10
- Sec 6 (Discussion): 완료 — Hohmann Δv 수식, ARD 해석, Fig 11
- Sec 7 (Conclusions): 완료 — 정량적 수치 모두 반영
- Appendix A (GP Sampling): 완료
- 전체 19페이지, 11개 그림, 5개 표, 37개 참고문헌
