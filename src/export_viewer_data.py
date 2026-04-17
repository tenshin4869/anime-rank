import json
import pandas as pd
import numpy as np
from pathlib import Path
import os
import joblib

# Load paths and config
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, PROCESSED_DIR, MODELS_DIR, ANIME_CATALOG_PATH

def load_catalog_meta():
    with open(ANIME_CATALOG_PATH, encoding="utf-8") as f:
        return {item["anime_title"]: item for item in json.load(f)}

def export_viewer_data():
    master_path = PROCESSED_DIR / "master.csv"
    if not master_path.exists():
        print("master.csv not found")
        return
    
    df = pd.read_csv(master_path)
    model_path = MODELS_DIR / "ridge_Y_3m_gt.joblib"
    if not model_path.exists():
        print("Model ridge_Y_3m_gt.joblib not found")
        return

    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    
    # Feature columns used in the 3-week phase
    feature_cols = [
        "pv_w1",     "pv_w2",     "pv_w3",
        "gt_w1",     "gt_w2",     "gt_w3",
        "edit_w1",   "edit_w2",   "edit_w3",
        "editor_w1", "editor_w2", "editor_w3",
    ]
    target = "Y_3m_gt"
    
    sub = df[feature_cols + [target, "anime_title"]].dropna()
    catalog = load_catalog_meta()
    
    # Predict and calculate residuals
    # preprocessing same as analyze_results.py (log1p for PV)
    X = sub[feature_cols].copy().astype(float)
    pv_cols = [c for c in X.columns if c.startswith("pv_")]
    X[pv_cols] = np.log1p(X[pv_cols])
    
    X_s = scaler.transform(X.values)
    y_true = sub[target].values
    y_pred = model.predict(X_s)
    residuals = y_true - y_pred  # >0: under-predicted (Pattern B), <0: over-predicted (Pattern A)

    sub["y_pred"] = y_pred
    sub["residual"] = residuals
    
    # Categorize patterns
    def categorize(res):
        if res < -15:
            return "Pattern A (失速・過大予測)"
        elif res > 15:
            return "Pattern B (再燃・過小予測)"
        else:
            return "Normal (適正)"
            
    sub["pattern"] = sub["residual"].apply(categorize)
    
    # Load raw daily data for each anime
    viewer_data = []
    
    for _, row in sub.iterrows():
        title = row["anime_title"]
        meta_file = RAW_DIR / title / "metadata.json"
        if not meta_file.exists():
            continue
            
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
            
        air_start = meta["air_start"]
        anime_title_ja = meta.get("anime_title_ja", title)
        y_3m_date = meta.get("y_3m_date")
        
        # Load time-series
        pv_file = RAW_DIR / title / "wikipedia_daily.csv"
        gt_file = RAW_DIR / title / "google_trends_daily.csv"
        
        ts_data = []
        
        pv_df = None
        if pv_file.exists():
            pv_df = pd.read_csv(pv_file)
            pv_df["date"] = pd.to_datetime(pv_df["date"])
            pv_df.set_index("date", inplace=True)
            
        gt_df = None
        if gt_file.exists():
            gt_df = pd.read_csv(gt_file)
            gt_df["date"] = pd.to_datetime(gt_df["date"])
            gt_df.set_index("date", inplace=True)
            
        # Extract 150 days
        start_dt = pd.to_datetime(air_start)
        
        dates = pd.date_range(start=start_dt, periods=151, freq="D")
        
        for d in dates:
            day_str = d.strftime("%Y-%m-%d")
            
            pv_val = 0
            if pv_df is not None and d in pv_df.index:
                val = pv_df.loc[d, "pv"]
                # some dates might have duplicates yielding Series, take first
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if not pd.isna(val):
                    pv_val = int(val)
                
            gt_val = None
            if gt_df is not None and d in gt_df.index:
                # If there are duplicate records, take first
                gt_rec = gt_df.loc[[d]]
                if not gt_rec.empty and not pd.isna(gt_rec.iloc[0]["gt_score"]):
                    gt_val = int(gt_rec.iloc[0]["gt_score"])
            
            ts_data.append({
                "day_index": (d - start_dt).days,
                "date": day_str,
                "pv": pv_val,
                "gt": gt_val
            })
            
        # Also clean up GT values: PyTrends weekly data might leave days as None.
        # Let's forward fill GT score if it's weekly for better charting visuals.
        last_gt = None
        for item in ts_data:
            if item["gt"] is not None:
                last_gt = item["gt"]
            elif last_gt is not None:
                item["gt"] = last_gt
                
        # Fill remaining None with 0
        for item in ts_data:
            if item["gt"] is None:
                item["gt"] = 0
                
        viewer_data.append({
            "id": title,
            "title_ja": anime_title_ja,
            "air_start": air_start,
            "pattern": row["pattern"],
            "y_3m_gt": float(row[target]),
            "y_pred": float(row["y_pred"]),
            "residual": float(row["residual"]),
            "timeseries": ts_data
        })
        
    out_dir = Path(__file__).parent.parent / "tools" / "residual-viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "viewer_data.js"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("window.VIEWER_DATA = ")
        json.dump(viewer_data, f, ensure_ascii=False, indent=2)
        f.write(";")
        
    print(f"Exported {len(viewer_data)} records to {out_path}")

if __name__ == "__main__":
    export_viewer_data()
