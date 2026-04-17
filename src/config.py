"""
プロジェクト共通設定・定数定義
"""
import logging
from pathlib import Path

# ─── ディレクトリ設定 ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR              = PROJECT_ROOT / "data"
RAW_DIR               = DATA_DIR / "raw"
PROCESSED_DIR         = DATA_DIR / "processed"
PROCESSED_DAILY_DIR   = PROCESSED_DIR / "daily"
PROCESSED_WEEKLY_DIR  = PROCESSED_DIR / "weekly"
LABELS_DIR            = DATA_DIR / "labels"

RESULTS_DIR   = PROJECT_ROOT / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
MODELS_DIR    = RESULTS_DIR / "models"
LOGS_DIR      = PROJECT_ROOT / "logs"

SRC_DIR             = Path(__file__).parent
ANIME_CATALOG_PATH  = SRC_DIR / "anime_catalog.json"

# ─── データ収集パラメータ ────────────────────────────────────────────────────
INPUT_WINDOW_DAYS   = 21   # Day1〜Day21（第1〜3週）
Y_1M_OFFSET_DAYS    = 30   # 放送開始から1ヶ月後（air_start + 30日）
Y_2M_OFFSET_DAYS    = 60   # 放送開始から2ヶ月後（air_start + 60日）
Y_3M_OFFSET_DAYS    = 90   # 放送開始から3ヶ月後（air_start + 90日）
Y_4M_OFFSET_DAYS    = 120  # 放送開始から4ヶ月後（air_start + 120日）
Y_5M_OFFSET_DAYS    = 150  # 放送開始から5ヶ月後（air_start + 150日）

# API
RETRY_COUNT             = 3
RETRY_WAIT_SECONDS      = 5
GT_REQUEST_DELAY_SECONDS = 5   # Google Trends レートリミット対策

# Wikipedia Pageviews REST API
WIKIPEDIA_PV_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    "/ja.wikipedia/all-access/all-agents"
)

# Wikipedia MediaWiki API（編集履歴）
WIKIPEDIA_REVISIONS_API = "https://ja.wikipedia.org/w/api.php"

# ─── ロガー設定 ──────────────────────────────────────────────────────────────
def setup_logging(name: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
