"""어떤 파라미터가 profile_class를 결정하는지 정량 분석.

분석 내용:
  1. Decision Tree / Random Forest feature importance
  2. 각 파라미터별 단변량 분포 (violin/box plot)
  3. 2D 파라미터 슬라이스별 class 경계 시각화
  4. 결과를 CSV + PNG로 저장

Usage:
    python scripts/analyze_dominant_factors.py
    python scripts/analyze_dominant_factors.py --db data/trajectories_circular.duckdb
    python scripts/analyze_dominant_factors.py --max_depth 4 --n_estimators 200
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# scikit-learn (optional – graceful fallback)
try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.inspection import permutation_importance
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[경고] scikit-learn 없음 – 트리 분석 생략. pip install scikit-learn")


RESULTS_DIR = Path("results")
FIG_DIR = Path("fig/dominant_factors")
CLASS_LABELS = {0: "unimodal", 1: "bimodal", 2: "multimodal"}
CLASS_COLORS = {0: "#2196F3", 1: "#FF9800", 2: "#4CAF50"}


def parse_args():
    p = argparse.ArgumentParser(description="dominant factor 분석")
    p.add_argument("--db", default="data/trajectories.duckdb")
    p.add_argument("--max_depth", type=int, default=5, help="Decision Tree 최대 깊이")
    p.add_argument("--n_estimators", type=int, default=300, help="RF 트리 수")
    p.add_argument("--min_samples", type=int, default=5,
                   help="각 클래스별 최소 샘플 수")
    return p.parse_args()


def load_data(db_path: str) -> pd.DataFrame:
    """수렴 케이스 로드. 필요 컬럼만 선택."""
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute(
        "SELECT h0, delta_a, delta_i, T_normed, e0, ef, profile_class, n_peaks "
        "FROM trajectories WHERE converged = TRUE AND profile_class IS NOT NULL "
        "ORDER BY id"
    ).df()
    conn.close()
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """피처 행렬 X, 레이블 y, 피처 이름 반환."""
    feature_cols = ["h0", "delta_a", "delta_i", "T_normed"]

    # e0, ef가 constant가 아닌 경우만 추가
    for col in ["e0", "ef"]:
        if col in df.columns and df[col].nunique() > 1:
            feature_cols.append(col)

    X = df[feature_cols].values.astype(float)
    y = df["profile_class"].values.astype(int)
    return X, y, feature_cols


# ── 1. Decision Tree 분석 ────────────────────────────────────────────────────

def run_decision_tree(X, y, feature_names, max_depth=5) -> dict:
    if not SKLEARN_OK:
        return {}

    dt = DecisionTreeClassifier(
        max_depth=max_depth, random_state=42,
        class_weight="balanced",
        min_samples_leaf=5,
    )
    dt.fit(X, y)

    cv_scores = cross_val_score(dt, X, y, cv=5, scoring="accuracy")

    result = {
        "model": dt,
        "importance": dict(zip(feature_names, dt.feature_importances_)),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "tree_text": export_text(dt, feature_names=feature_names),
    }

    print("\n[Decision Tree]")
    print(f"  5-fold CV 정확도: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("  Feature importances (DT):")
    for feat, imp in sorted(result["importance"].items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {feat:>10}: {imp:.4f}  {bar}")

    return result


# ── 2. Random Forest 분석 ────────────────────────────────────────────────────

def run_random_forest(X, y, feature_names, n_estimators=300) -> dict:
    if not SKLEARN_OK:
        return {}

    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )
    rf.fit(X, y)

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")

    # permutation importance (더 신뢰성 높음)
    perm = permutation_importance(rf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    perm_imp = dict(zip(feature_names, perm.importances_mean))

    result = {
        "model": rf,
        "importance_gini": dict(zip(feature_names, rf.feature_importances_)),
        "importance_perm": perm_imp,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }

    print("\n[Random Forest]")
    print(f"  5-fold CV 정확도: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("  Feature importances (Gini / Permutation):")
    for feat in feature_names:
        g = result["importance_gini"][feat]
        p = perm_imp[feat]
        bar_g = "█" * int(g * 40)
        print(f"    {feat:>10}: Gini={g:.4f}  Perm={p:.4f}  {bar_g}")

    return result


# ── 3. 단변량 분포 플롯 ──────────────────────────────────────────────────────

def plot_univariate(df: pd.DataFrame, feature_names: list[str]):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n = len(feature_names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    classes = sorted(df["profile_class"].dropna().unique().astype(int))

    for ax, feat in zip(axes, feature_names):
        data_by_class = [df[df["profile_class"] == c][feat].dropna().values for c in classes]
        parts = ax.violinplot(data_by_class, positions=classes, showmedians=True)

        for i, (pc, cls) in enumerate(zip(parts["bodies"], classes)):
            pc.set_facecolor(CLASS_COLORS.get(cls, "gray"))
            pc.set_alpha(0.7)

        ax.set_xticks(classes)
        ax.set_xticklabels([f"class {c}\n({CLASS_LABELS.get(c, '')})" for c in classes],
                            fontsize=8)
        ax.set_title(feat, fontsize=10)
        ax.set_ylabel(feat)
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("파라미터별 profile_class 분포 (violin)", fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / "univariate_violin.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  violin 플롯 저장: {out}")


# ── 4. Feature importance 바 차트 ────────────────────────────────────────────

def plot_feature_importance(dt_result: dict, rf_result: dict, feature_names: list[str]):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    has_dt = bool(dt_result.get("importance"))
    has_rf = bool(rf_result.get("importance_gini"))

    if not has_dt and not has_rf:
        return

    n_methods = has_dt + has_rf * 2  # DT=1, RF gini+perm=2
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    ax_idx = 0
    feat_arr = np.array(feature_names)

    if has_dt:
        imp = np.array([dt_result["importance"][f] for f in feature_names])
        order = np.argsort(imp)
        axes[ax_idx].barh(feat_arr[order], imp[order], color="#2196F3", alpha=0.8)
        axes[ax_idx].set_title(
            f"Decision Tree (Gini)\nCV acc={dt_result['cv_mean']:.3f}±{dt_result['cv_std']:.3f}",
            fontsize=9,
        )
        axes[ax_idx].set_xlabel("Feature Importance")
        ax_idx += 1

    if has_rf:
        for key, label, color in [
            ("importance_gini", "Random Forest (Gini)", "#FF9800"),
            ("importance_perm", "Random Forest (Permutation)", "#4CAF50"),
        ]:
            imp = np.array([rf_result[key][f] for f in feature_names])
            order = np.argsort(imp)
            axes[ax_idx].barh(feat_arr[order], imp[order], color=color, alpha=0.8)
            title_suffix = f"\nCV acc={rf_result['cv_mean']:.3f}±{rf_result['cv_std']:.3f}" \
                           if "gini" in key else ""
            axes[ax_idx].set_title(label + title_suffix, fontsize=9)
            axes[ax_idx].set_xlabel("Feature Importance")
            ax_idx += 1

    fig.suptitle("profile_class 결정 요인 – Feature Importance", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "feature_importance.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  feature importance 플롯 저장: {out}")


# ── 5. 2D 슬라이스 class 분포 ────────────────────────────────────────────────

def plot_2d_slices(df: pd.DataFrame):
    """T_normed 고정 슬라이스에서 (delta_a, delta_i) 산점도."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    t_vals = sorted(df["T_normed"].unique())
    # 대표값 4개 선택
    if len(t_vals) > 4:
        indices = np.linspace(0, len(t_vals) - 1, 4, dtype=int)
        t_vals = [t_vals[i] for i in indices]

    n = len(t_vals)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    classes = sorted(df["profile_class"].dropna().unique().astype(int))

    for ax, tv in zip(axes, t_vals):
        sub = df[np.isclose(df["T_normed"], tv, atol=1e-6)]
        if len(sub) == 0:
            sub = df[(df["T_normed"] - tv).abs() < 0.1]

        for cls in classes:
            csub = sub[sub["profile_class"] == cls]
            ax.scatter(
                csub["delta_a"], csub["delta_i"],
                c=CLASS_COLORS.get(cls, "gray"),
                s=20, alpha=0.7, label=f"class {cls} ({CLASS_LABELS.get(cls, '')})",
            )

        ax.set_title(f"T_normed ≈ {tv:.2f}", fontsize=10)
        ax.set_xlabel("Δa [km]")
        if ax is axes[0]:
            ax.set_ylabel("Δi [deg]")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle("(Δa, Δi) 슬라이스별 profile_class 분포", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "2d_class_slices.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  2D 슬라이스 플롯 저장: {out}")


# ── 6. Decision Tree boundary (2D projection) ───────────────────────────────

def plot_dt_boundary(dt_result: dict, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    """가장 중요한 2개 피처 공간에서 DT 결정 경계 시각화."""
    if not dt_result.get("model") or not dt_result.get("importance"):
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    imp = dt_result["importance"]
    top2 = sorted(imp, key=lambda k: -imp[k])[:2]
    idx0 = feature_names.index(top2[0])
    idx1 = feature_names.index(top2[1])

    X2 = X[:, [idx0, idx1]]

    # 2D 재학습
    dt2 = DecisionTreeClassifier(
        max_depth=dt_result["model"].max_depth,
        random_state=42, class_weight="balanced", min_samples_leaf=5,
    )
    dt2.fit(X2, y)

    x_min, x_max = X2[:, 0].min() * 0.95, X2[:, 0].max() * 1.05
    y_min, y_max = X2[:, 1].min() * 0.95, X2[:, 1].max() * 1.05
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    Z = dt2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("Set1", 3)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=[-0.5, 0.5, 1.5, 2.5])

    classes = sorted(np.unique(y))
    for cls in classes:
        mask = y == cls
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=CLASS_COLORS.get(cls, "gray"), s=15, alpha=0.6,
                   label=f"class {cls} ({CLASS_LABELS.get(cls, '')})")

    ax.set_xlabel(top2[0])
    ax.set_ylabel(top2[1])
    ax.set_title(f"Decision Tree 결정 경계\n(top-2 features: {top2[0]} vs {top2[1]})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "dt_boundary_top2.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  DT 경계 플롯 저장: {out}")


# ── 7. 결과 CSV 저장 ─────────────────────────────────────────────────────────

def save_importance_csv(dt_result: dict, rf_result: dict, feature_names: list[str]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for feat in feature_names:
        row = {"feature": feat}
        if dt_result.get("importance"):
            row["DT_gini"] = dt_result["importance"].get(feat, np.nan)
        if rf_result.get("importance_gini"):
            row["RF_gini"] = rf_result["importance_gini"].get(feat, np.nan)
        if rf_result.get("importance_perm"):
            row["RF_perm"] = rf_result["importance_perm"].get(feat, np.nan)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    # 평균으로 랭킹
    numeric_cols = [c for c in df_out.columns if c != "feature"]
    df_out["rank_mean"] = df_out[numeric_cols].mean(axis=1).rank(ascending=False).astype(int)
    df_out = df_out.sort_values("rank_mean")
    out = RESULTS_DIR / "dominant_factors.csv"
    df_out.to_csv(out, index=False)
    print(f"\n  feature importance CSV 저장: {out}")
    print(df_out.to_string(index=False))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"DB 로드: {args.db}")
    df = load_data(args.db)
    print(f"수렴 케이스: {len(df)}")

    if len(df) == 0:
        print("데이터 없음 – 종료")
        return

    # 클래스별 샘플 수 확인
    print("\n[클래스 분포]")
    for cls in sorted(df["profile_class"].dropna().unique().astype(int)):
        n = (df["profile_class"] == cls).sum()
        print(f"  class {cls} ({CLASS_LABELS.get(cls, '?')}): {n}")

    # 클래스별 최소 샘플 수 검사
    min_count = df["profile_class"].value_counts().min()
    if min_count < args.min_samples:
        print(f"\n[경고] 일부 클래스 샘플이 {min_count}개로 적음. 분석 신뢰도 낮을 수 있음.")

    X, y, feature_names = build_feature_matrix(df)
    print(f"\n피처: {feature_names}")
    print(f"샘플 수: {len(X)}")

    # ─── 트리 분석 ───────────────────────────────────────────────────────────
    dt_result = run_decision_tree(X, y, feature_names, max_depth=args.max_depth)
    rf_result = run_random_forest(X, y, feature_names, n_estimators=args.n_estimators)

    # ─── 플롯 ────────────────────────────────────────────────────────────────
    plot_univariate(df, feature_names)
    plot_feature_importance(dt_result, rf_result, feature_names)
    plot_2d_slices(df)
    if SKLEARN_OK and dt_result:
        plot_dt_boundary(dt_result, X, y, feature_names)

    # ─── CSV 저장 ─────────────────────────────────────────────────────────────
    save_importance_csv(dt_result, rf_result, feature_names)

    # ─── Decision Tree 구조 출력 ─────────────────────────────────────────────
    if dt_result.get("tree_text"):
        print("\n[Decision Tree 규칙 (요약)]")
        lines = dt_result["tree_text"].splitlines()
        # 너무 길면 앞부분만
        for line in lines[:40]:
            print(" ", line)
        if len(lines) > 40:
            print(f"  ... (총 {len(lines)}줄)")

    print("\n분석 완료.")


if __name__ == "__main__":
    main()
