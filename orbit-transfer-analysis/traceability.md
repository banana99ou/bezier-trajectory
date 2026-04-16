# Traceability

## Figures

| Figure | 파일 경로 | 생성 스크립트 | 데이터 소스 | 비고 |
|---|---|---|---|---|
| Fig. 1 (fig:representative_profiles) | manuscript/figures/fig1_representative_profiles.pdf | scripts/generate_figures.py fig1 | data/trajectories.duckdb | 5D DB 기반 재생성 필요 |
| Fig. 2 (fig:representative_trajectories) | manuscript/figures/fig2_representative_trajectories.pdf | scripts/generate_figures.py fig2 | data/trajectories.duckdb | 5D DB 기반 재생성 필요 |
| Fig. 3 (fig:classification_da_di) | manuscript/figures/fig3_classification_da_di.pdf | scripts/generate_figures.py fig3 | data/trajectories.duckdb | 5D→2D 프로젝션 (Δa vs Δi) |
| Fig. 4 (fig:classification_da_T) | manuscript/figures/fig4_classification_da_T.pdf | scripts/generate_figures.py fig4 | data/trajectories.duckdb | 5D→2D 프로젝션 (Δa vs T_max/T₀) |
| Fig. 5 (fig:classification_di_T) | manuscript/figures/fig5_classification_di_T.pdf | scripts/generate_figures.py fig5 | data/trajectories.duckdb | 5D→2D 프로젝션 (Δi vs T_max/T₀) |
| Fig. 6 (fig:classification_3d) | manuscript/figures/fig6_classification_3d.pdf | scripts/generate_figures.py fig6 | data/trajectories.duckdb | 대표 h₀ 3D 분류 지도 |
| Fig. 7 (fig:sampling_progress) | manuscript/figures/fig7_sampling_progress.pdf | scripts/generate_figures.py fig7 | data/trajectories.duckdb | 적응적 샘플링 이력 |
| Fig. 8 (fig:ard_lengthscales) | manuscript/figures/fig8_ard_lengthscales.pdf | scripts/generate_figures.py fig8 | data/trajectories.duckdb | 5D ARD 길이척도 (5개 바) |
| Fig. 9 (fig:cost_vs_params) | manuscript/figures/fig9_cost_vs_params.pdf | scripts/generate_figures.py fig9 | data/trajectories.duckdb | 비용 vs 5D 파라미터 |
| Fig. 10 (fig:peak_timing) | manuscript/figures/fig10_peak_timing.pdf | scripts/generate_figures.py fig10 | data/trajectories.duckdb | 피크 타이밍 분포 |
| Fig. 11 (fig:hohmann_correspondence) | manuscript/figures/fig11_hohmann_correspondence.pdf | scripts/generate_figures.py fig11 | data/trajectories.duckdb | 호만전이 Δv vs L2 비용 |

## Tables

| Table | 위치 (섹션) | 생성 스크립트 | 데이터 소스 | 비고 |
|---|---|---|---|---|
| Table 1 (tab:param_ranges) | Sec 3.2 | — | src/orbit_transfer/config.py | 수동 생성. 5D 매개변수: Δa∈[-500,2000]km, Δi∈[0,15]°, e₀∈[0,0.1], e_f∈[0,0.1], T_max/T₀∈[0.1,1.2] |
| Table 2 (tab:db_summary) | Sec 4.1 | scripts/generate_tables.py table2 | data/trajectories.duckdb | DB 구축 후 갱신 |
| Table 3 (tab:class_cost) | Sec 4.2 | scripts/generate_tables.py table3 | data/trajectories.duckdb | DB 구축 후 갱신 |
| Table 4 (tab:ard_lengthscales) | Sec 4.3 | scripts/generate_tables.py table4 | data/trajectories.duckdb | 5개 ARD 길이척도 |
| Table 5 (tab:solver_performance) | Sec 4.4 | scripts/generate_tables.py table5 | data/trajectories.duckdb | 솔버 성능 통계 |

## In-text Values

| 수치 | 위치 (섹션/문장) | 데이터 소스 | 산출 방법 | 비고 |
|---|---|---|---|---|
| Δa: [-500, 2000] km | Sec 3.2 / Table 1 | config.py | 설계 파라미터 | 확정 |
| Δi: [0, 15] deg | Sec 3.2 / Table 1 | config.py | 설계 파라미터 | 확정 |
| e₀: [0, 0.1] | Sec 3.2 / Table 1 | config.py | 설계 파라미터 | 확정 |
| e_f: [0, 0.1] | Sec 3.2 / Table 1 | config.py | 설계 파라미터 | 확정 |
| T_max/T₀: [0.1, 1.2] | Sec 3.2 / Table 1 | config.py | 설계 파라미터 | 확정. sub-1.2-revolution |
| T_f/T₀: [0.1, T_max/T₀] | Sec 3.2 / Table 1 | config.py | 솔버 결정변수 | T_MIN_FACTOR=0.1 |
| h₀: {400, 600, 800, 1000} km | Sec 3.2 / Table 1 | config.py | 설계 파라미터 | 확정 |
| u_max = 0.01 km/s² | Sec 2.1 | types.py | 기본값 | 필요시 상향 가능 |
| N_init = 150 (LHS 초기 샘플) | Sec 3.2 | config.py | 설계 파라미터 | 5D용 증가 |
| N_max = 800 (최대 샘플) | Sec 3.2 | config.py | 설계 파라미터 | 5D용 증가 |
| GP_BATCH_SIZE = 15 | Sec 3.2 | config.py | 설계 파라미터 | 5D용 증가 |
| 수렴 엔트로피 임계값 < 0.3 | Appendix A.3 | config.py | 설계 파라미터 | 유지 |
| Hermite-Simpson 30 세그먼트 (Pass 1) | Sec 3.1 | config.py | 설계 파라미터 | 유지 |
| 수렴율 54.4% (전체), h400=40.9%, h600=52.2%, h800=35.7%, h1000=88.9% | Sec 4.1, Sec 7 | data/trajectories_all.duckdb | 5D DB 구축 결과 | 3,240 총 샘플 |
| 총 수렴 건수 1,763 | Sec 7 | data/trajectories_all.duckdb | 5D DB 구축 결과 | |
| 클래스별: Class 0=1,144, Class 1=521, Class 2=98 | Sec 4.2, Sec 7 | data/trajectories_all.duckdb | 5D DB 구축 결과 | |
| ARD 길이척도 [TBD] | Sec 6.1 | data/trajectories_all.duckdb | DB 구축 후 조회 | 5개 차원 |
