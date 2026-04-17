"""
Google Trends データ収集スクリプト

- pytrends を使って各作品の Google Trends スコアを取得
- 1作品 = 1リクエスト（自己完結型：X と Y が同一スケール）
- データは data/raw/{anime_title}/google_trends_daily.csv に保存する

NOTE:
  リクエスト期間が 90 日超の場合、pytrends は週次データを返す。
  その場合でも data/raw/ には週次データを保存し、
  後段の build_features.py で週次特徴量として扱う。
"""
import json
import time
import argparse
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_DIR, ANIME_CATALOG_PATH,
    GT_REQUEST_DELAY_SECONDS, RETRY_COUNT,
    setup_logging,
)

logger = setup_logging("collect_google_trends")


# ─── 取得ロジック ────────────────────────────────────────────────────────────

def fetch_trends(gt_query: str, start_date: str, end_date: str,
                 pytrends: TrendReq) -> pd.DataFrame:
    """
    指定クエリ・期間の Google Trends を取得して DataFrame を返す。
    columns: date (str YYYY-MM-DD), gt_score (int)
    """
    timeframe = f"{start_date} {end_date}"
    pytrends.build_payload([gt_query], cat=0, timeframe=timeframe, geo="JP")
    df = pytrends.interest_over_time()

    if df.empty:
        logger.warning("  データなし（クエリ: %s）", gt_query)
        return pd.DataFrame(columns=["date", "gt_score"])

    df = df.reset_index()[["date", gt_query]].rename(columns={gt_query: "gt_score"})
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["gt_score"] = df["gt_score"].astype(int)
    return df


def collect_one(anime: dict, pytrends: TrendReq, force: bool = False) -> bool:
    """1作品分の Google Trends データを収集して raw/ に保存する。"""
    title      = anime["anime_title"]
    gt_query   = anime["gt_query"]
    air_start  = anime["air_start"]

    out_dir    = RAW_DIR / title
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_file    = out_dir / "google_trends_daily.csv"
    meta_file  = out_dir / "metadata.json"

    if gt_file.exists() and not force:
        logger.info("[SKIP] %s — GT データ既存。--force で上書き可能", title)
        return True

    # metadata.json から y_3m_date を読む（Wikipedia収集済み前提）
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        y_3m_date = meta.get("y_3m_date")
        if not y_3m_date:
            logger.warning("  metadata.json に y_3m_date がない: %s", title)
            return False
    else:
        # metadata がなければ air_start + 90 日で推定
        from datetime import datetime, timedelta
        y_3m_date = (datetime.strptime(air_start, "%Y-%m-%d") +
                     timedelta(days=90)).strftime("%Y-%m-%d")
        logger.warning("  metadata.json なし。y_3m_date を推定: %s", y_3m_date)

    logger.info("▶ %s（クエリ: %s, 期間: %s〜%s）", title, gt_query, air_start, y_3m_date)

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            df = fetch_trends(gt_query, air_start, y_3m_date, pytrends)
            if df.empty:
                logger.warning("  [SKIP] %s — GT スコアが空", title)
                return False
            df.to_csv(gt_file, index=False)
            logger.info("  保存: %s（%d行）", gt_file, len(df))

            # metadata の gt_request_period を更新
            if meta_file.exists():
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                meta["gt_request_period"] = f"{air_start}_to_{y_3m_date}"
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            return True

        except Exception as e:
            logger.warning("  GT取得失敗（%d/%d） [%s]: %s", attempt, RETRY_COUNT, title, e)
            if attempt < RETRY_COUNT:
                wait = GT_REQUEST_DELAY_SECONDS * attempt
                logger.info("  %d 秒待機して再試行...", wait)
                time.sleep(wait)

    logger.error("  最大リトライ超過: %s", title)
    return False


# ─── エントリポイント ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Google Trends データ収集")
    parser.add_argument("--force",  action="store_true", help="既存データを上書き")
    parser.add_argument("--titles", nargs="*", help="特定の anime_title のみ収集（省略時は全件）")
    args = parser.parse_args()

    with open(ANIME_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    if args.titles:
        catalog = [a for a in catalog if a["anime_title"] in args.titles]
        logger.info("対象を %d 作品に絞り込み", len(catalog))

    pytrends = TrendReq(hl="ja-JP", tz=540, timeout=(10, 25))

    success, failed = [], []
    for i, anime in enumerate(catalog):
        try:
            ok = collect_one(anime, pytrends, force=args.force)
            (success if ok else failed).append(anime["anime_title"])
        except Exception as e:
            logger.error("予期しないエラー [%s]: %s", anime["anime_title"], e)
            failed.append(anime["anime_title"])

        # 最後の作品以外は待機
        if i < len(catalog) - 1:
            logger.info("  次のリクエストまで %d 秒待機...", GT_REQUEST_DELAY_SECONDS)
            time.sleep(GT_REQUEST_DELAY_SECONDS)

    logger.info("=== 完了 === 成功: %d / 失敗: %d", len(success), len(failed))
    if failed:
        logger.warning("失敗した作品: %s", failed)


if __name__ == "__main__":
    main()
