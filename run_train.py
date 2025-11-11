# run_train.py
# -*- coding: utf-8 -*-
"""
Entrena el modelo rápido (train_hgb_quick), evalúa (evaluate_baseline),
y registra qué grupos de features (fg_, cp_, cg_, cmc_) están en el modelo.

Uso rápido (PowerShell):
  python .\run_train.py
  # o especificando dataset:
  python .\run_train.py --data data/ALLPAIRS_ctx_1h_h6_ts.csv

Genera:
  - models\hgb_<STAMP>Z.joblib
  - models\hgb_<STAMP>Z.features.json
  - models\LATEST.txt
  - evals\eval_<STAMP>\* (incluye metrics.json)
  - evals\LATEST_THRESHOLD.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

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
    p.add_argument("--data", default="data/ALLPAIRS_ctx_1h_h6_ts.csv")
    p.add_argument("--target", default="y")
    p.add_argument("--id-cols", default="symbol,timestamp,fwd_ret")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--evals-dir", default="evals")
    p.add_argument("--module-train", default="src.ml.train_hgb_quick")
    p.add_argument("--module-eval", default="src.ml.evaluate_baseline")
    p.add_argument("--optimize-threshold", default="f1", choices=["f1", "accuracy", "precision", "recall"])
    return p

def main():
    args = build_parser().parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.evals_dir, exist_ok=True)

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path = os.path.join(args.models_dir, f"hgb_{stamp}.joblib")
    feats_path  = os.path.join(args.models_dir, f"hgb_{stamp}.features.json")

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
    _print(f"  fg_:  {groups['fg_']}")
    _print(f"  cp_:  {groups['cp_']}")
    _print(f"  cg_:  {groups['cg_']}")
    _print(f"  cmc_: {groups['cmc_']}")

    # --- EVAL ---
    eval_dir = os.path.join(args.evals_dir, f"eval_{stamp}")
    os.makedirs(eval_dir, exist_ok=True)

    cmd_eval = [
        sys.executable, "-m", args.module_eval,
        "--data", args.data,
        "--model-path", model_path,
        "--features-path", feats_path,
        "--target", args.target,
        "--id-cols", args.id_cols,
        "--outdir", eval_dir,
        "--optimize-threshold", args.optimize_threshold,
    ]

    _print(f"\n[EVAL] Ejecutando: {' '.join(cmd_eval)}")
    subprocess.check_call(cmd_eval)

    # Guardar referencias “LATEST”
    latest_model_txt = os.path.join(args.models_dir, "LATEST.txt")
    with open(latest_model_txt, "w", encoding="utf-8") as f:
        f.write(model_path + "\n")
        f.write(feats_path + "\n")
    _print(f"\n[SUMMARY]\n  Modelo  : {model_path} (mod: {_latest_mod_time(model_path)})\n  Feats   : {feats_path} (mod: {_latest_mod_time(feats_path)})\n  Eval dir: {eval_dir}")

    metrics_path = os.path.join(eval_dir, "metrics.json")
    thr_out = os.path.join(args.evals_dir, "LATEST_THRESHOLD.txt")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        thr = m.get("optimized_threshold", {}).get("best_threshold", None)
        if thr is not None:
            with open(thr_out, "w", encoding="utf-8") as f:
                f.write(str(thr))
            _print(f"  Threshold óptimo ({args.optimize_threshold}): {thr}  -> {thr_out}")
        else:
            _print("  WARN: No se encontró best_threshold en metrics.json")
    except Exception as e:
        _print(f"  WARN: no se pudo leer metrics.json: {e}")

    _print("\n[OK] Archivos útiles:")
    _print(f"  - {latest_model_txt}")
    _print(f"  - {thr_out}")
    _print(f"  - {os.path.join(eval_dir, 'metrics.json')}")
    _print(f"  - {os.path.join(eval_dir, 'predictions.csv')}")
    _print(f"  - {os.path.join(eval_dir, 'classification_report.txt')}")
    _print(f"  - {os.path.join(eval_dir, 'confusion_matrix.png')}")
    _print(f"  - {os.path.join(eval_dir, 'confusion_matrix_normalized.png')}")

if __name__ == "__main__":
    main()

# Entrenar + evaluar (usa rutas por defecto que ya vienes usando): python .\run_train.py
# Luego puedes predecir con un solo tiro usando el wrapper anterior:
