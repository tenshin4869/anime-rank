"""
分析・可視化スクリプト（Phase 0）

1. 特徴量重要度（RF MDI + SHAP for XGBoost）
2. 残差分析（誤差の大きい作品群の特定）
3. 失敗パターン分類（仮説 A / B / C の検証）
4. 予測時点による精度比較

出力先: results/figures/*.png
"""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import japanize_matplotlib          # noqa: F401  日本語フォント
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import shap
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, LABELS_DIR, RESULTS_DIR, FIGURES_DIR,
    MODELS_DIR, ANIME_CATALOG_PATH, setup_logging,
)

logger = setup_logging("analyze_results")

TARGETS = ["Y_1m_gt", "Y_2m_gt", "Y_3m_gt", "Y_4m_gt", "Y_5m_gt"]
TARGET_LABELS = {
    "Y_1m_gt": "Y_1m\n（放送開始1ヶ月後）",
    "Y_2m_gt": "Y_2m\n（放送開始2ヶ月後）",
    "Y_3m_gt": "Y_3m\n（放送開始3ヶ月後）",
    "Y_4m_gt": "Y_4m\n（放送開始4ヶ月後）",
    "Y_5m_gt": "Y_5m\n（放送開始5ヶ月後）",
}

FEATURE_COLS = [
    "pv_w1",     "pv_w2",     "pv_w3",
    "gt_w1",     "gt_w2",     "gt_w3",
    "edit_w1",   "edit_w2",   "edit_w3",
    "editor_w1", "editor_w2", "editor_w3",
]

FEATURE_LABELS = {
    "pv_w1":     "PV 第1週",     "pv_w2":     "PV 第2週",
    "pv_w3":     "PV 第3週",
    "gt_w1":     "GT 第1週",     "gt_w2":     "GT 第2週",
    "gt_w3":     "GT 第3週",
    "edit_w1":   "編集数 第1週", "edit_w2":   "編集数 第2週",
    "edit_w3":   "編集数 第3週",
    "editor_w1": "編集者 第1週", "editor_w2": "編集者 第2週",
    "editor_w3": "編集者 第3週",
}


def preprocess(X: pd.DataFrame) -> np.ndarray:
    X = X.copy().astype(float)
    pv_cols = [c for c in X.columns if c.startswith("pv_")]
    X[pv_cols] = np.log1p(X[pv_cols])
    return X.values


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "master.csv")
    with open(ANIME_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    title_meta = {a["anime_title"]: a for a in catalog}
    df["popularity_tier"] = df["anime_title"].map(
        lambda t: title_meta.get(t, {}).get("popularity_tier", "unknown"))
    df["anime_title_ja"] = df["anime_title"].map(
        lambda t: title_meta.get(t, {}).get("anime_title_ja", t))
    df["year"] = df["anime_title"].map(
        lambda t: int(title_meta.get(t, {}).get("air_start", "2020")[:4]))
    return df


# ─── 1. 精度サマリ表（棒グラフ） ────────────────────────────────────────────

def plot_metric_summary():
    cv_path = RESULTS_DIR / "cv_results.csv"
    if not cv_path.exists():
        logger.warning("cv_results.csv が見つかりません")
        return

    df = pd.read_csv(cv_path)
    metrics = ["RMSE", "Spearman_rho", "R2"]
    model_order = ["Ridge", "RandomForest", "XGBoost"]
    target_order = ["Y_1m_gt", "Y_2m_gt", "Y_3m_gt", "Y_4m_gt", "Y_5m_gt"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("5-fold CV 精度サマリ（Phase 1 ベースライン）", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        pivot = (df[df["model"].isin(model_order)]
                 .pivot_table(values=metric, index="target", columns="model", aggfunc="mean")
                 .reindex(target_order)[model_order])
        pivot.index = [TARGET_LABELS[t] for t in pivot.index]
        pivot.plot(kind="bar", ax=ax, width=0.7)
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="モデル", fontsize=8)
        if metric in ("Spearman_rho", "R2"):
            ax.axhline(y=0.6, color="red", linestyle="--", linewidth=0.8,
                       label="ρ=0.6 基準線")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "metric_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("精度サマリ保存: %s", path)


# ─── 2. 特徴量重要度（RF MDI） ──────────────────────────────────────────────

def plot_feature_importance(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle("Random Forest 特徴量重要度（MDI）", fontsize=14, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        sub = df[FEATURE_COLS + [target]].dropna()
        if len(sub) < 5:
            ax.set_title(TARGET_LABELS[target])
            ax.text(0.5, 0.5, "データ不足", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        X = preprocess(sub[FEATURE_COLS])
        y = sub[target].values

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_s, y)
        importances = rf.feature_importances_

        feat_labels = [FEATURE_LABELS[c] for c in FEATURE_COLS]
        sorted_idx = np.argsort(importances)[::-1][:10]

        colors = []
        for c in [FEATURE_COLS[i] for i in sorted_idx]:
            if c.startswith("pv_"):
                colors.append("#4C72B0")
            elif c.startswith("gt_"):
                colors.append("#DD8452")
            elif c.startswith("edit_"):
                colors.append("#55A868")
            else:
                colors.append("#C44E52")

        ax.barh(range(len(sorted_idx)),
                importances[sorted_idx][::-1],
                color=colors[::-1])
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feat_labels[i] for i in sorted_idx][::-1], fontsize=9)
        ax.set_xlabel("重要度（MDI）")
        ax.set_title(TARGET_LABELS[target], fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color="#4C72B0", label="Wikipedia PV"),
        Patch(color="#DD8452", label="Google Trends"),
        Patch(color="#55A868", label="編集回数"),
        Patch(color="#C44E52", label="編集者数"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = FIGURES_DIR / "feature_importance_rf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("RF特徴量重要度保存: %s", path)


# ─── 3. SHAP 分析（XGBoost） ────────────────────────────────────────────────

def plot_shap(df: pd.DataFrame):
    for target in TARGETS:
        sub = df[FEATURE_COLS + [target]].dropna()
        if len(sub) < 5:
            continue

        X = preprocess(sub[FEATURE_COLS])
        y = sub[target].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                  max_depth=4, random_state=42, verbosity=0)
        model.fit(X_s, y)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_s)

        feature_names_ja = [FEATURE_LABELS[c] for c in FEATURE_COLS]

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_s,
                          feature_names=feature_names_ja,
                          show=False, max_display=12)
        plt.title(f"SHAP Beeswarm — XGBoost / {TARGET_LABELS[target]}",
                  fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        path = FIGURES_DIR / f"shap_{target}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP保存: %s", path)


# ─── 4. 残差分析（上位・下位 5 作品） ───────────────────────────────────────

def plot_residuals(df: pd.DataFrame):
    model_path = MODELS_DIR / "xgboost_Y_1m_gt.joblib"
    if not model_path.exists():
        logger.warning("XGBoost モデルが見つかりません（先に train_models.py を実行）")
        return

    artifact = joblib.load(model_path)
    model   = artifact["model"]
    scaler  = artifact["scaler"]

    target = "Y_1m_gt"
    sub = df[FEATURE_COLS + [target, "anime_title_ja", "popularity_tier"]].dropna()
    if sub.empty:
        return

    X = preprocess(sub[FEATURE_COLS])
    X_s = scaler.transform(X)
    y_true = sub[target].values
    y_pred = model.predict(X_s)
    residuals = y_true - y_pred

    sub = sub.copy()
    sub["y_pred"]   = y_pred
    sub["residual"] = residuals

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"残差分析（XGBoost / {TARGET_LABELS[target]}）",
                 fontsize=13, fontweight="bold")

    # 散布図（実測 vs 予測）
    tier_colors = {"high": "#D62728", "medium": "#1F77B4", "low": "#2CA02C"}
    for tier, grp in sub.groupby("popularity_tier"):
        axes[0].scatter(grp["y_pred"], grp[target],
                        c=tier_colors.get(tier, "gray"),
                        label=tier, alpha=0.75, s=60, edgecolors="white", linewidths=0.5)
    lim = max(sub[target].max(), sub["y_pred"].max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "k--", linewidth=1)
    axes[0].set_xlabel("予測値（GT スコア）")
    axes[0].set_ylabel("実測値（GT スコア）")
    axes[0].set_title("予測 vs 実測")
    axes[0].legend(title="人気階層")
    axes[0].grid(alpha=0.3)

    # 残差上位/下位 5 作品
    top5_over  = sub.nlargest(5, "residual")   # 過小予測
    top5_under = sub.nsmallest(5, "residual")  # 過大予測
    highlight = pd.concat([top5_over, top5_under])

    axes[1].axhline(0, color="k", linewidth=0.8)
    axes[1].scatter(sub["y_pred"], sub["residual"],
                    c="lightgray", s=40, zorder=1)
    axes[1].scatter(highlight["y_pred"], highlight["residual"],
                    c="tomato", s=80, zorder=2)

    for _, row in highlight.iterrows():
        axes[1].annotate(row["anime_title_ja"],
                         (row["y_pred"], row["residual"]),
                         fontsize=8, xytext=(5, 3),
                         textcoords="offset points")
    axes[1].set_xlabel("予測値（GT スコア）")
    axes[1].set_ylabel("残差（実測 − 予測）")
    axes[1].set_title("残差プロット（赤点：上下位5作品）")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "residual_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("残差分析保存: %s", path)

    # CSV にも出力
    out_path = RESULTS_DIR / "high_residual_titles.csv"
    sub.sort_values("residual", ascending=False)[
        ["anime_title_ja", "popularity_tier", target, "y_pred", "residual"]
    ].to_csv(out_path, index=False)
    logger.info("残差 CSV 保存: %s", out_path)


# ─── 5. 予測時点別精度比較 ───────────────────────────────────────────────────

def plot_accuracy_by_horizon():
    cv_path = RESULTS_DIR / "cv_results.csv"
    if not cv_path.exists():
        return

    df = pd.read_csv(cv_path)
    target_short = {"Y_1m_gt": "Y_1m (+30)", "Y_2m_gt": "Y_2m (+60)", 
                    "Y_3m_gt": "Y_3m (+90)", "Y_4m_gt": "Y_4m (+120)", 
                    "Y_5m_gt": "Y_5m (+150)"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("予測時点別精度（5-fold CV）", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, ["Spearman_rho", "RMSE"]):
        pivot = (df.pivot_table(values=metric, index="target", columns="model")
                 .reindex(TARGETS))
        pivot.index = [target_short[t] for t in pivot.index]
        pivot.plot(marker="o", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("予測時点（放送開始からの距離 →）")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        if metric == "Spearman_rho":
            ax.axhline(y=0.6, color="red", linestyle="--", linewidth=0.8,
                       label="ρ=0.6 基準線")
        ax.legend(title="モデル", fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "accuracy_by_horizon.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("予測時点別精度保存: %s", path)


# ─── 6. データ概要（相関ヒートマップ） ──────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    sub = df[FEATURE_COLS + TARGETS].dropna()
    if len(sub) < 5:
        return

    X = preprocess(sub[FEATURE_COLS])
    X_df = pd.DataFrame(X, columns=[FEATURE_LABELS[c] for c in FEATURE_COLS])
    for i, t in enumerate(TARGETS):
        X_df[TARGET_LABELS[t]] = sub[t].values

    corr = X_df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("特徴量間 Spearman 相関ヒートマップ", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("相関ヒートマップ保存: %s", path)


# ─── 7. GT スコア分布（人気階層別） ────────────────────────────────────────

def plot_gt_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle("GT スコア分布（人気階層別）", fontsize=13, fontweight="bold")
    palette = {"high": "#D62728", "medium": "#1F77B4", "low": "#2CA02C"}

    for ax, target in zip(axes, TARGETS):
        sub = df[[target, "popularity_tier"]].dropna()
        for tier, grp in sub.groupby("popularity_tier"):
            ax.hist(grp[target], bins=12, alpha=0.65,
                    color=palette.get(tier, "gray"),
                    label=tier, edgecolor="white")
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("GT スコア（0–100）")
        ax.set_ylabel("作品数")
        ax.legend(title="人気階層")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "gt_score_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("GT分布保存: %s", path)


# ─── エントリポイント ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 0 分析・可視化")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    logger.info("データ: %d 作品 / 特徴量有効: %d",
                len(df), df[FEATURE_COLS].dropna().shape[0])

    plot_metric_summary()
    plot_feature_importance(df)
    plot_shap(df)
    plot_residuals(df)
    plot_accuracy_by_horizon()
    plot_correlation_heatmap(df)
    plot_gt_distribution(df)

    logger.info("=== 分析完了 ===")


if __name__ == "__main__":
    main()
