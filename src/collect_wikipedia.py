"""
Wikipedia データ収集スクリプト

- Wikipedia Pageviews REST API でページビュー数（日次）を取得
- Wikipedia MediaWiki API でリビジョン履歴を取得し、日次の編集回数・ユニーク編集者数を算出
- データは data/raw/{anime_title}/ 以下に保存する
"""
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import requests
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_DIR, ANIME_CATALOG_PATH, INPUT_WINDOW_DAYS,
    Y_1M_OFFSET_DAYS, Y_3M_OFFSET_DAYS,
    RETRY_COUNT, RETRY_WAIT_SECONDS,
    WIKIPEDIA_PV_API, WIKIPEDIA_REVISIONS_API,
    setup_logging,
)

logger = setup_logging("collect_wikipedia")


# ─── ユーティリティ ─────────────────────────────────────────────────────────

def _request_with_retry(url: str, params: dict | None = None,
                        session: requests.Session | None = None) -> requests.Response:
    s = session or requests.Session()
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = s.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = RETRY_WAIT_SECONDS * attempt
                logger.warning("429 Too Many Requests. %d秒待機して再試行...", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logger.warning("リクエスト失敗（%d/%d）: %s", attempt, RETRY_COUNT, e)
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_WAIT_SECONDS)
    raise RuntimeError(f"最大リトライ回数超過: {url}")


# ─── Pageviews API ──────────────────────────────────────────────────────────

def fetch_pageviews(page_title: str, start_date: str, end_date: str,
                    session: requests.Session) -> pd.DataFrame:
    """
    日次ページビューを取得して DataFrame を返す。
    columns: date (str YYYY-MM-DD), pv (int)
    """
    # Wikipedia Pageviews API: 日本語文字はエンコード、スペースは _ 、括弧はそのまま
    # アンダースコア区切りのタイトルをスペースに戻してから再エンコード
    clean_title = page_title.replace("_", " ")
    encoded_title = quote(clean_title.replace(" ", "_"), safe="()~")
    start_fmt = start_date.replace("-", "") + "00"
    end_fmt   = end_date.replace("-", "") + "00"
    url = f"{WIKIPEDIA_PV_API}/{encoded_title}/daily/{start_fmt}/{end_fmt}"

    resp = _request_with_retry(url, session=session)
    items = resp.json().get("items", [])

    records = []
    for item in items:
        ts = item["timestamp"]                     # "2022100100"
        date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        records.append({"date": date_str, "pv": item["views"]})

    return pd.DataFrame(records)


# ─── Revisions API（編集回数・ユニーク編集者） ─────────────────────────────────

def fetch_revisions_for_date_range(page_title: str, start_date: str,
                                   end_date: str,
                                   session: requests.Session) -> pd.DataFrame:
    """
    MediaWiki API でリビジョン一覧を取得し、日次の edit_count / editor_count を返す。
    columns: date (str YYYY-MM-DD), edit_count (int), editor_count (int)
    """
    daily: dict[str, set] = {}   # date -> set of user names

    params = {
        "action":   "query",
        "prop":     "revisions",
        "titles":   page_title,
        "rvprop":   "timestamp|user",
        "rvlimit":  "max",
        "rvdir":    "newer",
        "rvstart":  f"{start_date}T00:00:00Z",
        "rvend":    f"{end_date}T23:59:59Z",
        "format":   "json",
    }

    while True:
        resp = _request_with_retry(WIKIPEDIA_REVISIONS_API,
                                   params=params, session=session)
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for page_data in pages.values():
            for rev in page_data.get("revisions", []):
                ts   = rev["timestamp"][:10]   # "YYYY-MM-DD"
                user = rev.get("user", "anonymous")
                if ts not in daily:
                    daily[ts] = set()
                daily[ts].add(user)

        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
        time.sleep(0.5)

    records = [
        {"date": d, "edit_count": len(users), "editor_count": len(users)}
        for d, users in daily.items()
    ]
    if not records:
        return pd.DataFrame(columns=["date", "edit_count", "editor_count"])
    return pd.DataFrame(records)


# ─── メイン収集ロジック ──────────────────────────────────────────────────────

def compute_target_dates(air_start: str) -> tuple[str, str]:
    """Y_1m, Y_3m の日付を返す（いずれも air_start 基準）。"""
    start = datetime.strptime(air_start, "%Y-%m-%d")
    y_1m  = start + timedelta(days=Y_1M_OFFSET_DAYS)
    y_3m  = start + timedelta(days=Y_3M_OFFSET_DAYS)
    return y_1m.strftime("%Y-%m-%d"), y_3m.strftime("%Y-%m-%d")


def collect_one(anime: dict, force: bool = False) -> bool:
    """1作品分の Wikipedia データを収集して raw/ に保存する。"""
    title    = anime["anime_title"]
    title_ja = anime["anime_title_ja"]
    wp_title = anime["wikipedia_page_title"]
    air_start = anime["air_start"]
    air_end   = anime["air_end"]

    out_dir = RAW_DIR / title
    out_dir.mkdir(parents=True, exist_ok=True)

    pv_file   = out_dir / "wikipedia_daily.csv"
    meta_file = out_dir / "metadata.json"

    # 既存データをスキップ（--force なし）
    if pv_file.exists() and meta_file.exists() and not force:
        logger.info("[SKIP] %s — 既にデータ存在。--force で上書き可能", title)
        return True

    logger.info("▶ %s（%s）の収集を開始", title, title_ja)

    y_1m_date, y_3m_date = compute_target_dates(air_start)

    # 収集範囲: air_start〜Y_3m（air_start+90日）
    fetch_start = air_start
    fetch_end   = y_3m_date

    session = requests.Session()
    session.headers["User-Agent"] = "anime-rank-research/0.1 (research project)"

    # ── Pageviews 取得 ──────────────────────────────────────────────────────
    try:
        pv_df = fetch_pageviews(wp_title, fetch_start, fetch_end, session)
        logger.info("  PV取得: %d 日分", len(pv_df))
    except Exception as e:
        logger.error("  PV取得失敗 [%s]: %s", title, e)
        return False

    # air_start 時点のページが存在するか確認
    input_pvs = pv_df[
        (pv_df["date"] >= air_start) &
        (pv_df["date"] <= (datetime.strptime(air_start, "%Y-%m-%d") +
                           timedelta(days=INPUT_WINDOW_DAYS - 1)).strftime("%Y-%m-%d"))
    ]
    if input_pvs["pv"].isna().all():
        logger.warning("  [SKIP] %s — 入力期間にPVデータなし（Wikipediaページ未存在の可能性）", title)
        return False

    # ── Revisions 取得 ─────────────────────────────────────────────────────
    try:
        rev_df = fetch_revisions_for_date_range(wp_title, fetch_start, fetch_end, session)
        logger.info("  Revisions取得: %d 日分", len(rev_df))
    except Exception as e:
        logger.error("  Revisions取得失敗 [%s]: %s", title, e)
        rev_df = pd.DataFrame(columns=["date", "edit_count", "editor_count"])

    # ── マージ ────────────────────────────────────────────────────────────
    # 全日付の連続インデックスを作成
    date_range = pd.date_range(fetch_start, fetch_end, freq="D")
    base_df = pd.DataFrame({"date": date_range.strftime("%Y-%m-%d")})

    merged = base_df.merge(pv_df, on="date", how="left")
    merged = merged.merge(rev_df, on="date", how="left")
    merged["edit_count"]   = merged["edit_count"].fillna(0).astype(int)
    merged["editor_count"] = merged["editor_count"].fillna(0).astype(int)

    # PV が NaN の行はページ未存在を意味する（仕様: 0とNaNを区別）
    # pv_df にない日付 = NaN のまま（後段で除外判定に使う）

    # ── 保存 ──────────────────────────────────────────────────────────────
    merged.to_csv(pv_file, index=False)
    logger.info("  保存: %s", pv_file)

    meta = {
        "anime_title":           title,
        "anime_title_ja":        title_ja,
        "wikipedia_page_title":  wp_title,
        "gt_query":              anime.get("gt_query", ""),
        "gt_query_rule":         "Wikipediaページタイトルから括弧表記を除去",
        "gt_request_period":     f"{fetch_start}_to_{fetch_end}",
        "air_start":             air_start,
        "air_end":               air_end,
        "total_episodes":        anime.get("total_episodes", 0),
        "y_1m_date":             y_1m_date,
        "y_3m_date":             y_3m_date,
        "data_collected_at":     datetime.now().strftime("%Y-%m-%d"),
        "wikipedia_page_exists": True,
        "notes":                 "",
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("  メタデータ保存: %s", meta_file)

    return True


# ─── エントリポイント ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Wikipedia データ収集")
    parser.add_argument("--force",  action="store_true", help="既存データを上書き")
    parser.add_argument("--titles", nargs="*", help="特定の anime_title のみ収集（省略時は全件）")
    args = parser.parse_args()

    with open(ANIME_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    if args.titles:
        catalog = [a for a in catalog if a["anime_title"] in args.titles]
        logger.info("対象を %d 作品に絞り込み", len(catalog))

    success, failed = [], []
    for anime in catalog:
        try:
            ok = collect_one(anime, force=args.force)
            (success if ok else failed).append(anime["anime_title"])
        except Exception as e:
            logger.error("予期しないエラー [%s]: %s", anime["anime_title"], e)
            failed.append(anime["anime_title"])
        time.sleep(1)   # Wikipedia API への礼儀

    logger.info("=== 完了 === 成功: %d / 失敗: %d", len(success), len(failed))
    if failed:
        logger.warning("失敗した作品: %s", failed)


if __name__ == "__main__":
    main()
