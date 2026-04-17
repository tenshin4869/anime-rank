"""
モデル学習・評価スクリプト（Phase 0 ベースライン）

- 入力: processed/master.csv
- 使用モデル: Ridge, Random Forest, XGBoost
- 評価: RMSE, Spearman ρ, R²
- 検証: 5-fold CV + 時系列 Split（2018-2021 訓練 / 2022-2023 テスト）
- 出力: results/cv_results.csv, results/timesplit_results.csv
"""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, LABELS_DIR, RESULTS_DIR, MODELS_DIR,
    ANIME_CATALOG_PATH, setup_logging,
)

logger = setup_logging("train_models")

TARGETS     = ["Y_1m_gt", "Y_3m_gt"]
TARGET_NAMES = {"Y_1m_gt": "Y_1m（放送開始1ヶ月後）",
                "Y_3m_gt": "Y_3m（放送開始3ヶ月後）"}

FEATURE_COLS = [
    "pv_w1",     "pv_w2",     "pv_w3",
    "gt_w1",     "gt_w2",     "gt_w3",
    "edit_w1",   "edit_w2",   "edit_w3",
    "editor_w1", "editor_w2", "editor_w3",
]

MODELS = {
    "Ridge":         Ridge(alpha=1.0),
    "RandomForest":  RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost":       xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, random_state=42,
                                       verbosity=0),
}


# ─── 前処理 ────────────────────────────────────────────────────────────────

def preprocess(X: pd.DataFrame) -> np.ndarray:
    """PV列に log1p 変換を適用して ndarray を返す。"""
    X = X.copy().astype(float)
    pv_cols = [c for c in X.columns if c.startswith("pv_")]
    X[pv_cols] = np.log1p(X[pv_cols])
    return X.values


def load_data() -> pd.DataFrame:
    master_path = PROCESSED_DIR / "master.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"master.csv が見つかりません: {master_path}")

    df = pd.read_csv(master_path)

    # air_start 年を付与（時系列 Split 用）
    with open(ANIME_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    title_year = {a["anime_title"]: int(a["air_start"][:4]) for a in catalog}
    df["year"] = df["anime_title"].map(title_year)

    return df


# ─── 評価指標 ───────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = root_mean_squared_error(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "Spearman_rho": rho, "R2": r2}


# ─── 5-fold 交差検証 ────────────────────────────────────────────────────────

def run_cv(df: pd.DataFrame, target: str) -> pd.DataFrame:
    sub = df[FEATURE_COLS + [target, "anime_title"]].dropna()
    if len(sub) < 5:
        logger.warning("  %s: サンプル数不足（%d）", target, len(sub))
        return pd.DataFrame()

    X = preprocess(sub[FEATURE_COLS])
    y = sub[target].values

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in MODELS.items():
        oof_pred = np.zeros(len(y))
        for tr, va in kf.split(X):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xva = scaler.transform(X[va])
            m = type(model)(**model.get_params()) if hasattr(model, "get_params") else model
            m.fit(Xtr, y[tr])
            oof_pred[va] = m.predict(Xva)

        metrics = evaluate(y, oof_pred)
        results.append({
            "target": target,
            "model":  model_name,
            "n_samples": len(sub),
            **metrics,
        })
        logger.info("  [CV] %s | %s → RMSE=%.2f, ρ=%.3f, R²=%.3f",
                    target, model_name,
                    metrics["RMSE"], metrics["Spearman_rho"], metrics["R2"])

    return pd.DataFrame(results)


# ─── 時系列 Split ───────────────────────────────────────────────────────────

def run_timesplit(df: pd.DataFrame, target: str) -> pd.DataFrame:
    sub = df[FEATURE_COLS + [target, "anime_title", "year"]].dropna(
        subset=FEATURE_COLS + [target])
    train = sub[sub["year"] <= 2021]
    test  = sub[sub["year"] >= 2022]

    if len(train) < 5 or len(test) < 2:
        logger.warning("  [TimeSplit] %s: 学習(%d) / テスト(%d) サンプル不足",
                       target, len(train), len(test))
        return pd.DataFrame()

    X_tr = preprocess(train[FEATURE_COLS])
    y_tr = train[target].values
    X_te = preprocess(test[FEATURE_COLS])
    y_te = test[target].values

    results = []
    for model_name, model in MODELS.items():
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_tr)
        Xte_s = scaler.transform(X_te)
        m = type(model)(**model.get_params()) if hasattr(model, "get_params") else model
        m.fit(Xtr_s, y_tr)
        y_pred = m.predict(Xte_s)

        metrics = evaluate(y_te, y_pred)
        results.append({
            "target":    target,
            "model":     model_name,
            "n_train":   len(train),
            "n_test":    len(test),
            **metrics,
        })
        logger.info("  [TimeSplit] %s | %s → RMSE=%.2f, ρ=%.3f, R²=%.3f",
                    target, model_name,
                    metrics["RMSE"], metrics["Spearman_rho"], metrics["R2"])

    return pd.DataFrame(results)


# ─── 最終モデル保存（全データで学習） ────────────────────────────────────────

def save_final_models(df: pd.DataFrame):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        sub = df[FEATURE_COLS + [target, "anime_title"]].dropna()
        if sub.empty:
            continue
        X = preprocess(sub[FEATURE_COLS])
        y = sub[target].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        for model_name, model in MODELS.items():
            m = type(model)(**model.get_params()) if hasattr(model, "get_params") else model
            m.fit(X_s, y)
            out_path = MODELS_DIR / f"{model_name.lower()}_{target}.joblib"
            joblib.dump({"model": m, "scaler": scaler, "feature_cols": FEATURE_COLS},
                        out_path)
    logger.info("最終モデル保存: %s", MODELS_DIR)


# ─── エントリポイント ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="モデル学習・評価")
    parser.add_argument("--skip-save", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    logger.info("データ読み込み: %d 行, ターゲット有効数 → %s",
                len(df),
                {t: df[t].notna().sum() for t in TARGETS})

    cv_all, ts_all = [], []
    for target in TARGETS:
        logger.info("── %s ──────────────────────", target)
        cv_df = run_cv(df, target)
        ts_df = run_timesplit(df, target)
        if not cv_df.empty:
            cv_all.append(cv_df)
        if not ts_df.empty:
            ts_all.append(ts_df)

    if cv_all:
        cv_result = pd.concat(cv_all, ignore_index=True)
        cv_path = RESULTS_DIR / "cv_results.csv"
        cv_result.to_csv(cv_path, index=False)
        logger.info("5-fold CV 結果保存: %s", cv_path)

    if ts_all:
        ts_result = pd.concat(ts_all, ignore_index=True)
        ts_path = RESULTS_DIR / "timesplit_results.csv"
        ts_result.to_csv(ts_path, index=False)
        logger.info("時系列 Split 結果保存: %s", ts_path)

    if not args.skip_save:
        save_final_models(df)


if __name__ == "__main__":
    main()
