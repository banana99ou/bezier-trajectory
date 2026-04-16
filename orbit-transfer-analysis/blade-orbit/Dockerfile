FROM python:3.12-slim
WORKDIR /workspace

# 의존성 설치 (캐시 활용)
COPY pyproject.toml .
RUN pip install --no-cache-dir numpy scipy cvxpy duckdb scikit-learn pyyaml

# blade-orbit 소스
COPY src/ ./src/

# orbit-transfer-analysis 분류 모듈 (최소 의존성만)
COPY vendor/orbit_transfer/ ./vendor/orbit_transfer/

# collocation DB (Phase 1용, Phase 2/3에서도 --resume 시 필요)
COPY data/trajectories_all.duckdb ./data/trajectories_all.duckdb

# 배치 스크립트
COPY scripts/batch_blade_vs_collocation.py ./scripts/batch_blade_vs_collocation.py
COPY scripts/batch_parametric_grid.py ./scripts/batch_parametric_grid.py

# 격자 설정
COPY configs/ ./configs/

# 출력 디렉토리
RUN mkdir -p outputs scripts/results
