"""
既存50作品のmetadata.jsonをY_1m/Y_3m定義に更新するスクリプト。

変更内容:
  - y_1m_date を追加（air_start + 30日）
  - y_3m_date を air_start + 90日 に再定義（旧定義は air_end + 90日）
  - y_mid_date / y_end_date を削除
  - gt_request_period も新しい y_3m_date に合わせて更新
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import sys
from config import (
    RAW_DIR, Y_1M_OFFSET_DAYS, Y_2M_OFFSET_DAYS, Y_3M_OFFSET_DAYS,
    Y_4M_OFFSET_DAYS, Y_5M_OFFSET_DAYS, setup_logging
)

logger = setup_logging("update_metadata")

updated = 0
skipped = 0

for anime_dir in sorted(RAW_DIR.iterdir()):
    meta_file = anime_dir / "metadata.json"
    if not meta_file.exists():
        continue

    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    air_start = meta.get("air_start")
    if not air_start:
        logger.warning("[SKIP] %s — air_start なし", anime_dir.name)
        skipped += 1
        continue

    start_dt = datetime.strptime(air_start, "%Y-%m-%d")
    y_1m_date = (start_dt + timedelta(days=Y_1M_OFFSET_DAYS)).strftime("%Y-%m-%d")
    y_2m_date = (start_dt + timedelta(days=Y_2M_OFFSET_DAYS)).strftime("%Y-%m-%d")
    y_3m_date = (start_dt + timedelta(days=Y_3M_OFFSET_DAYS)).strftime("%Y-%m-%d")
    y_4m_date = (start_dt + timedelta(days=Y_4M_OFFSET_DAYS)).strftime("%Y-%m-%d")
    y_5m_date = (start_dt + timedelta(days=Y_5M_OFFSET_DAYS)).strftime("%Y-%m-%d")

    # 旧キーを削除
    meta.pop("y_mid_date", None)
    meta.pop("y_end_date", None)

    # 新キーを設定
    meta["y_1m_date"] = y_1m_date
    meta["y_2m_date"] = y_2m_date
    meta["y_3m_date"] = y_3m_date
    meta["y_4m_date"] = y_4m_date
    meta["y_5m_date"] = y_5m_date
    meta["gt_request_period"] = f"{air_start}_to_{y_5m_date}"

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("✓ %s: y_1m=%s, y_3m=%s", anime_dir.name, y_1m_date, y_3m_date)
    updated += 1

logger.info("=== 完了 === 更新: %d / スキップ: %d", updated, skipped)
