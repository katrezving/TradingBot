import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--id-cols", required=True, help="comma-separated: e.g. symbol,timestamp,fwd_ret")
    p.add_argument("--model-out", required=True)
    p.add_argument("--features-out", required=True)
    p.add_argument("--holdout-ratio", type=float, default=0.2)
    p.add_argument("--nan-col-threshold", type=float, default=0.99)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    logger.info(f"Cargando: {args.data}")
    df = pd.read_csv(args.data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    base_exclude = set(id_cols + [args.target])
    cand = [c for c in df.columns if c not in base_exclude]
    num_cols = [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]
    na_ratio = df[num_cols].isna().mean()
    keep_cols = [c for c in num_cols if na_ratio[c] <= args.nan_col_threshold and not (df[c].nunique(dropna=True) <= 1)]
    dropped = sorted(set(num_cols) - set(keep_cols))
    logger.info(f"n_features(candidatas)={len(num_cols)} → usadas={len(keep_cols)} (descartadas={len(dropped)})")
    fg_feats = [c for c in keep_cols if c.startswith("fg_")]
    cp_feats = [c for c in keep_cols if c.startswith("cp_")]
    cg_feats = [c for c in keep_cols if c.startswith("cg_")]
    logger.info(f"fg_feats={fg_feats}")
    logger.info(f"cp_feats={cp_feats}")
    logger.info(f"cg_feats={cg_feats}")
    X = df[keep_cols]
    y = df[args.target]
    n = len(df)
    split_idx = int(n * (1 - args.holdout_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(f"shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        logger.error("Split temporal produjo 0 filas en train o test. Revisa holdout-ratio o cobertura temporal.")
        return
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(random_state=args.random_state))
        ]
    )
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    f1 = f1_score(y_test, pred)
    logger.info(f"F1 holdout: {f1:.4f} (test={len(y_test)})")
    try:
        rep = classification_report(y_test, pred, digits=4)
        logger.info(f"Classification report:\n{rep}")
    except Exception:
        pass
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model_out)
    with open(args.features_out, "w", encoding="utf-8") as f:
        json.dump(keep_cols, f, ensure_ascii=False, indent=2)
    logger.info(f"Guardado modelo: {args.model_out}")
    logger.info(f"Guardado features: {args.features_out}")

if __name__ == "__main__":
    main()
