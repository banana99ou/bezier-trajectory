#!/bin/bash
# gpu-farm 제출 전 vendor/ 및 data/ 준비
# 사용법: bash scripts/prepare_gpu_farm.sh

set -e
cd "$(dirname "$0")/.."

echo "=== gpu-farm 제출 준비 ==="

# 1. vendor/ — orbit_transfer 분류 모듈 (최소 파일만)
echo "  vendor/orbit_transfer 준비..."
rm -rf vendor/orbit_transfer
mkdir -p vendor/orbit_transfer/classification

OTA=~/gitlab/orbit-transfer-analysis/src/orbit_transfer

cp "$OTA/__init__.py"    vendor/orbit_transfer/
cp "$OTA/config.py"      vendor/orbit_transfer/
cp "$OTA/constants.py"   vendor/orbit_transfer/
cp "$OTA/classification/__init__.py"       vendor/orbit_transfer/classification/
cp "$OTA/classification/peak_detection.py" vendor/orbit_transfer/classification/
cp "$OTA/classification/classifier.py"     vendor/orbit_transfer/classification/

echo "  vendor/orbit_transfer 완료 ($(du -sh vendor/orbit_transfer | cut -f1))"

# 2. data/ — collocation DB
echo "  data/trajectories_all.duckdb 복사..."
mkdir -p data
cp ~/gitlab/orbit-transfer-analysis/data/trajectories_all.duckdb data/

echo "  data/ 완료 ($(du -sh data/trajectories_all.duckdb | cut -f1))"

# 3. 확인
echo ""
echo "=== 준비 완료 ==="
echo "  vendor/orbit_transfer/  — 분류 모듈"
echo "  data/trajectories_all.duckdb  — collocation DB (3,240 cases)"
echo ""
echo "제출:"
echo "  ~/gitlab/gpu-farm/client/gpu-farm submit"
