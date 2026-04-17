"""
特徴量構築スクリプト

raw/ の Wikipedia + Google Trends データを読み込み、以下を生成する:
  - processed/daily/{anime}_daily.csv    : 日次データ（phase ラベル付き）
  - processed/weekly/{anime}_weekly.csv  : 週次データ（7日間総和）
  - processed/master.csv                 : 全作品フラット（モデル学習用）
  - labels/targets.csv                   : 正解ラベル（Y_1m, Y_3m）
"""
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_DIR, PROCESSED_DAILY_DIR, PROCESSED_WEEKLY_DIR,
    PROCESSED_DIR, LABELS_DIR, ANIME_CATALOG_PATH,
    INPUT_WINDOW_DAYS,
    setup_logging,
)

logger = setup_logging("build_features")


# ─── phase ラベルを付ける ────────────────────────────────────────────────────

def assign_phase(date_str: str, air_start: str, y_1m_date: str, y_2m_date: str,
                 y_3m_date: str, y_4m_date: str, y_5m_date: str, input_end: str) -> Optional[str]:
    """
    各日付を phase に対応させる。
    input 範囲外 & 予測時点以外は None（ファイルに含めない）。
    """
    if air_start <= date_str <= input_end:
        return "input"
    if date_str == y_1m_date:
        return "Y_1m"
    if date_str == y_2m_date:
        return "Y_2m"
    if date_str == y_3m_date:
        return "Y_3m"
    if date_str == y_4m_date:
        return "Y_4m"
    if date_str == y_5m_date:
        return "Y_5m"
    return None


# ─── GT 日付の最近傍マッチング ───────────────────────────────────────────────

def find_nearest_gt(gt_df: pd.DataFrame, target_date: str,
                    max_offset_days: int = 7) -> Optional[float]:
    """
    target_date の GT スコアを返す。
    - 週次データ（GT が 7 日刻み）の場合、±max_offset_days 以内で最近傍を使う
    - 見つからなければ None
    """
    if gt_df is None or gt_df.empty:
        return None
    target = pd.Timestamp(target_date)
    gt_df["_dt"] = pd.to_datetime(gt_df["date"])
    diffs = (gt_df["_dt"] - target).abs()
    idx = diffs.idxmin()
    if diffs[idx].days > max_offset_days:
        return None
    return float(gt_df.loc[idx, "gt_score"])


# ─── 1 作品の処理 ───────────────────────────────────────────────────────────

def build_one(anime_title: str, force: bool = False) -> Optional[dict]:
    """1 作品を処理してマスター行 dict を返す（None = スキップ）。"""
    raw_dir   = RAW_DIR / anime_title
    meta_file = raw_dir / "metadata.json"
    wp_file   = raw_dir / "wikipedia_daily.csv"
    gt_file   = raw_dir / "google_trends_daily.csv"

    # --- メタデータ読み込み ---
    if not meta_file.exists():
        logger.warning("[SKIP] %s — metadata.json なし", anime_title)
        return None
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    air_start   = meta["air_start"]
    y_1m_date   = meta["y_1m_date"]
    y_2m_date   = meta["y_2m_date"]
    y_3m_date   = meta["y_3m_date"]
    y_4m_date   = meta["y_4m_date"]
    y_5m_date   = meta["y_5m_date"]
    air_end     = meta["air_end"]

    input_end = (datetime.strptime(air_start, "%Y-%m-%d") +
                 timedelta(days=INPUT_WINDOW_DAYS - 1)).strftime("%Y-%m-%d")

    # --- Wikipedia データ読み込み ---
    if not wp_file.exists():
        logger.warning("[SKIP] %s — wikipedia_daily.csv なし", anime_title)
        return None
    wp_df = pd.read_csv(wp_file)
    wp_df["date"] = wp_df["date"].astype(str)

    # PV が全 NaN の場合は除外（ページ未存在）
    if wp_df["pv"].isna().all():
        logger.warning("[SKIP] %s — PV が全 NaN（Wikipedia ページ未存在）", anime_title)
        return None

    # --- Google Trends データ読み込み ---
    gt_df = None
    if gt_file.exists():
        gt_df = pd.read_csv(gt_file)
        gt_df["date"] = gt_df["date"].astype(str)
    else:
        logger.warning("[WARN] %s — google_trends_daily.csv なし（GT=NaN 扱い）", anime_title)

    # --- 日次データ構築 ---
    # 入力期間 + 予測時点のみ抽出
    target_dates = {y_1m_date, y_2m_date, y_3m_date, y_4m_date, y_5m_date}
    air_start_dt = datetime.strptime(air_start, "%Y-%m-%d")
    input_dates = set(
        (air_start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(INPUT_WINDOW_DAYS)
    )
    keep_dates = input_dates | target_dates

    rows = []
    for _, wp_row in wp_df[wp_df["date"].isin(keep_dates)].iterrows():
        date_str  = wp_row["date"]
        day_index = (datetime.strptime(date_str, "%Y-%m-%d") -
                     air_start_dt).days + 1
        week_index = (day_index - 1) // 7 + 1

        phase = assign_phase(date_str, air_start, y_1m_date, y_2m_date,
                             y_3m_date, y_4m_date, y_5m_date, input_end)
        if phase is None:
            continue

        # GT スコア（日次があれば直接、なければ最近傍）
        gt_score = np.nan
        if gt_df is not None:
            day_gt = gt_df[gt_df["date"] == date_str]
            if not day_gt.empty:
                gt_score = float(day_gt.iloc[0]["gt_score"])
            elif phase in ("Y_1m", "Y_2m", "Y_3m", "Y_4m", "Y_5m"):
                # ターゲット日は最近傍（GT は週次の場合がある）
                gt_score = find_nearest_gt(gt_df, date_str)

        rows.append({
            "date":         date_str,
            "pv":           wp_row.get("pv", np.nan),
            "gt_score":     gt_score,
            "edit_count":   int(wp_row.get("edit_count", 0)),
            "editor_count": int(wp_row.get("editor_count", 0)),
            "day_index":    day_index,
            "week_index":   week_index,
            "phase":        phase,
        })

    if not rows:
        logger.warning("[SKIP] %s — 出力行ゼロ", anime_title)
        return None

    daily_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # 出力先確認
    PROCESSED_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    daily_out = PROCESSED_DAILY_DIR / f"{anime_title}_daily.csv"
    if not daily_out.exists() or force:
        daily_df.to_csv(daily_out, index=False)

    # --- 週次データ構築（入力期間のみ） ---
    input_df = daily_df[daily_df["phase"] == "input"].copy()

    weekly_rows = []
    for wk in range(1, 4):
        wk_df = input_df[input_df["week_index"] == wk]
        if wk_df.empty:
            continue
        weekly_rows.append({
            "week_start":  wk_df["date"].min(),
            "week_end":    wk_df["date"].max(),
            "week_index":  wk,
            "pv_sum":      wk_df["pv"].sum(),
            "gt_sum":      wk_df["gt_score"].sum(),
            "edit_sum":    wk_df["edit_count"].sum(),
            "editor_sum":  wk_df["editor_count"].sum(),
            "phase":       "input",
        })

    weekly_df = pd.DataFrame(weekly_rows)
    weekly_out = PROCESSED_WEEKLY_DIR / f"{anime_title}_weekly.csv"
    if not weekly_out.exists() or force:
        weekly_df.to_csv(weekly_out, index=False)

    # --- マスター行（1 行 1 作品） ---
    # GT の週次集計（入力期間全体の GT は gt_sum に基づく週次分割）
    def _gt_w(wk: int) -> float:
        if gt_df is None:
            return np.nan
        wk_df_wp = input_df[input_df["week_index"] == wk]
        if wk_df_wp.empty:
            return np.nan
        week_start = wk_df_wp["date"].min()
        week_end   = wk_df_wp["date"].max()
        wk_gt = gt_df[(gt_df["date"] >= week_start) & (gt_df["date"] <= week_end)]
        if wk_gt.empty:
            # 週次 GT：該当週の最近傍を使用
            mid = (datetime.strptime(week_start, "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")
            v = find_nearest_gt(gt_df, mid, max_offset_days=7)
            return v if v is not None else np.nan
        return float(wk_gt["gt_score"].sum())

    master_row = {"anime_title": anime_title}
    for wk in range(1, 4):
        wk_df = input_df[input_df["week_index"] == wk]
        master_row[f"pv_w{wk}"]     = wk_df["pv"].sum() if not wk_df.empty else np.nan
        master_row[f"edit_w{wk}"]   = wk_df["edit_count"].sum() if not wk_df.empty else 0
        master_row[f"editor_w{wk}"] = wk_df["editor_count"].sum() if not wk_df.empty else 0
        master_row[f"gt_w{wk}"]     = _gt_w(wk)

    # 正解ラベル
    def _get_gt_target(phase_label: str) -> float:
        target_row = daily_df[daily_df["phase"] == phase_label]
        if not target_row.empty and not pd.isna(target_row.iloc[0]["gt_score"]):
            return float(target_row.iloc[0]["gt_score"])
        # 日次ファイルに直接ない場合は最近傍
        date_map = {"Y_1m": y_1m_date, "Y_2m": y_2m_date, "Y_3m": y_3m_date,
                    "Y_4m": y_4m_date, "Y_5m": y_5m_date}
        return find_nearest_gt(gt_df, date_map[phase_label]) or np.nan

    master_row["Y_1m_gt"] = _get_gt_target("Y_1m")
    master_row["Y_2m_gt"] = _get_gt_target("Y_2m")
    master_row["Y_3m_gt"] = _get_gt_target("Y_3m")
    master_row["Y_4m_gt"] = _get_gt_target("Y_4m")
    master_row["Y_5m_gt"] = _get_gt_target("Y_5m")

    # targets.csv 用の追加情報
    targets_row = {
        "anime_title": anime_title,
        "air_start":   air_start,
        "air_end":     air_end,
        "Y_1m_date":   y_1m_date,
        "Y_1m_gt":     master_row["Y_1m_gt"],
        "Y_2m_date":   y_2m_date,
        "Y_2m_gt":     master_row["Y_2m_gt"],
        "Y_3m_date":   y_3m_date,
        "Y_3m_gt":     master_row["Y_3m_gt"],
        "Y_4m_date":   y_4m_date,
        "Y_4m_gt":     master_row["Y_4m_gt"],
        "Y_5m_date":   y_5m_date,
        "Y_5m_gt":     master_row["Y_5m_gt"],
    }

    logger.info("✓ %s（rows=%d, Y_1m=%.1f, Y_5m=%.1f）",
                anime_title,
                len(daily_df),
                master_row["Y_1m_gt"] or 0,
                master_row["Y_5m_gt"] or 0)

    return {"master": master_row, "targets": targets_row}


# ─── エントリポイント ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="特徴量構築")
    parser.add_argument("--force", action="store_true", help="既存ファイルを上書き")
    args = parser.parse_args()

    with open(ANIME_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    master_rows  = []
    targets_rows = []

    for anime in catalog:
        title = anime["anime_title"]
        result = build_one(title, force=args.force)
        if result is None:
            continue
        master_rows.append(result["master"])
        targets_rows.append(result["targets"])

    if not master_rows:
        logger.error("処理できた作品がゼロ。raw/ データが存在するか確認してください。")
        return

    # master.csv
    master_df = pd.DataFrame(master_rows)
    cols = ["anime_title",
            "pv_w1",     "pv_w2",     "pv_w3",
            "gt_w1",     "gt_w2",     "gt_w3",
            "edit_w1",   "edit_w2",   "edit_w3",
            "editor_w1", "editor_w2", "editor_w3",
            "Y_1m_gt", "Y_2m_gt", "Y_3m_gt", "Y_4m_gt", "Y_5m_gt"]
    master_df = master_df.reindex(columns=cols)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master_path = PROCESSED_DIR / "master.csv"
    master_df.to_csv(master_path, index=False)
    logger.info("master.csv 保存: %s（%d 作品）", master_path, len(master_df))

    # targets.csv
    targets_df = pd.DataFrame(targets_rows)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    targets_path = LABELS_DIR / "targets.csv"
    targets_df.to_csv(targets_path, index=False)
    logger.info("targets.csv 保存: %s", targets_path)

    # 有効行の確認
    for col in ["Y_1m_gt", "Y_2m_gt", "Y_3m_gt", "Y_4m_gt", "Y_5m_gt"]:
        valid = master_df[col].notna().sum()
        logger.info("  %s: 有効=%d / 全=%d", col, valid, len(master_df))


if __name__ == "__main__":
    main()
