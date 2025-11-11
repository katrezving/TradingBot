# run_train.py
# -*- coding: utf-8 -*-
"""
Entrena el modelo rápido (train_hgb_quick), evalúa (evaluate_baseline),
y registra qué grupos de features (fg_, cp_, cg_, cmc_) están en el modelo.
Incluye validación rolling window y análisis robusto de feature importance.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.inspection import permutation_importance
import numpy as np
import shap
import matplotlib.pyplot as plt

def _print(msg: str) -> None:
    print(f"[TRAIN] {msg}")

def _latest_mod_time(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "NA"

def _load_features_list(feats_path: str):
    with open(feats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("features", [])

def _group_counts(feats):
    groups = {
        "fg_": 0,
        "cp_": 0,
        "cg_": 0,
        "cmc_": 0,
    }
    for c in feats:
        for p in groups.keys():
            if c.startswith(p):
                groups[p] += 1
    return groups

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="BNBUSDT_1h_features_full.csv")
    p.add_argument("--target", default="y")
    p.add_argument("--id-cols", default="symbol,timestamp,fwd_ret")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--evals-dir", default="evals")
    p.add_argument("--module-train", default="src.ml.train_hgb_quick")
    p.add_argument("--module-eval", default="src.ml.evaluate_baseline")
    p.add_argument("--optimize-threshold", default="f1", choices=["f1", "accuracy", "precision", "recall"])
    return p

def validate_rolling_window(data_path, target, id_cols, evals_dir="evals", n_splits=5):
    """Rolling window CV + feature importance compatible, sin errores de shape"""
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    id_cols_list = [c.strip() for c in id_cols.split(',')]
    features_all = [col for col in df.columns if col not in id_cols_list and col != target]
    X_full = df[features_all].copy()
    y_full = df[target]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    f1_scores = []
    models = []
    val_sets = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

        # Solo columnas sin all-nan antes de imputar
        cols_valid = [col for col in X_tr.columns if not X_tr[col].isna().all()]
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr[cols_valid]), columns=cols_valid)
        X_val_imp = pd.DataFrame(imputer.transform(X_val[cols_valid]), columns=cols_valid)

        model = HistGradientBoostingClassifier(random_state=42)
        model.fit(X_tr_imp, y_tr)
        preds = model.predict(X_val_imp)
        f1_scores.append(f1_score(y_val, preds))

        print(f"\n[Rolling Fold {fold+1}]:")
        print(classification_report(y_val, preds))

        models.append(model)
        val_sets.append((X_val_imp.copy(), y_val.copy(), cols_valid))

    print(f"\n[Rolling F1 Window Avg]: {np.mean(f1_scores):.4f}")

    # --- Feature Importance con permutation_importance (último fold) ---
    X_val_last, y_val_last, used_features_last = val_sets[-1]
    print("\n[IMPORTANCIA DE FEATURES — PERMUTATION IMPORTANCE]")
    result = permutation_importance(models[-1], X_val_last, y_val_last, n_repeats=5, random_state=42)
    fi_df = pd.DataFrame({'feature': used_features_last, 'importance': result.importances_mean})
    print(fi_df.sort_values('importance', ascending=False).head(10))

    # --- SHAP: visualización sobre las features realmente usadas ---
    try:
        explainer = shap.Explainer(models[-1])
        shap_values = explainer(X_val_last)
        os.makedirs(evals_dir, exist_ok=True)
        plt.figure()
        shap.summary_plot(shap_values, X_val_last, plot_type="bar", show=False)
        plt.savefig(f"{evals_dir}/shap_summary_rolling.png")
        print(f"\n[SHAP] Guardado en {evals_dir}/shap_summary_rolling.png")
    except Exception as e:
        print(f"[SHAP] Error al graficar: {e}")

def main():
    args = build_parser().parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.evals_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path = os.path.join(args.models_dir, f"hgb_{stamp}.joblib")
    feats_path = os.path.join(args.models_dir, f"hgb_{stamp}.features.json")

    # --- TRAIN ---
    cmd_train = [
        sys.executable, "-m", args.module_train,
        "--data", args.data,
        "--target", args.target,
        "--id-cols", args.id_cols,
        "--model-out", model_path,
        "--features-out", feats_path,
    ]
    _print(f"Ejecutando: {' '.join(cmd_train)}")
    subprocess.check_call(cmd_train)
    _print(f"Modelo guardado: {model_path} (mod: {_latest_mod_time(model_path)})")
    _print(f"Features guardadas: {feats_path} (mod: {_latest_mod_time(feats_path)})")

    # --- FEATURE GROUP LOG ---
    feats = _load_features_list(feats_path)
    groups = _group_counts(feats)
    _print("Resumen de grupos de features usados por el MODELO:")
    _print(f" fg_: {groups['fg_']}")
    _print(f" cp_: {groups['cp_']}")
    _print(f" cg_: {groups['cg_']}")
    _print(f" cmc_: {groups['cmc_']}")

    # --- VALIDACIÓN ROLLING WINDOW ---
    print("\n----- VALIDACIÓN AVANZADA ROLLING WINDOW -----")
    validate_rolling_window(args.data, args.target, args.id_cols, evals_dir=args.evals_dir)
    print("----- FIN VALIDACIÓN ROLLING -----\n")

    # --- EVAL ---
    eval_dir = os.path.join(args.evals_dir, f"eval_{stamp}")
    os.makedirs(eval_dir, exist_ok=True)

    # Carga lista de features reales desde el JSON
    with open(feats_path, "r", encoding="utf-8") as f:
        features_cli = json.load(f)
    if isinstance(features_cli, dict) and "features" in features_cli:
        features_cli = features_cli["features"]

    cmd_eval = [
        sys.executable, "-m", args.module_eval,
        "--data", args.data,
        "--model", model_path,
        "--features"
    ] + features_cli + [
        "--target", args.target,
        "--out-metrics", os.path.join(eval_dir, "metrics.json"),
        "--out-preds", os.path.join(eval_dir, "predictions.csv")
    ]

    _print(f"\n[EVAL] Ejecutando: {' '.join(cmd_eval)}")
    subprocess.check_call(cmd_eval)

    # Guardar referencias “LATEST”
    latest_model_txt = os.path.join(args.models_dir, "LATEST.txt")
    with open(latest_model_txt, "w", encoding="utf-8") as f:
        f.write(model_path + "\n")
        f.write(feats_path + "\n")

    _print(f"\n[SUMMARY]\n Modelo : {model_path} (mod: {_latest_mod_time(model_path)})\n Feats : {feats_path} (mod: {_latest_mod_time(feats_path)})\n Eval dir: {eval_dir}")
    metrics_path = os.path.join(eval_dir, "metrics.json")
    thr_out = os.path.join(args.evals_dir, "LATEST_THRESHOLD.txt")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        thr = m.get("optimized_threshold", {}).get("best_threshold", None)
        if thr is not None:
            with open(thr_out, "w", encoding="utf-8") as f:
                f.write(str(thr))
            _print(f" Threshold óptimo ({args.optimize_threshold}): {thr} -> {thr_out}")
        else:
            _print(" WARN: No se encontró best_threshold en metrics.json")
    except Exception as e:
        _print(f" WARN: no se pudo leer metrics.json: {e}")

    _print("\n[OK] Archivos útiles:")
    _print(f" - {latest_model_txt}")
    _print(f" - {thr_out}")
    _print(f" - {os.path.join(eval_dir, 'metrics.json')}")
    _print(f" - {os.path.join(eval_dir, 'predictions.csv')}")
    # Puedes añadir otros reportes aquí si lo deseas

if __name__ == "__main__":
    main()
