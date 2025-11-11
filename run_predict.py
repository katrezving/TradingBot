# run_predict.py
# -*- coding: utf-8 -*-
"""
Ejecuta predict_latest con el último modelo por defecto y registra
qué grupos de features (fg_, cp_, cg_, cmc_) usa el modelo.

Uso rápido (PowerShell):
  python .\run_predict.py
  # Cambiando side/threshold explícito:
  python .\run_predict.py --side long --threshold 0.25
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
    print(f"[RUN] {msg}")

def _find_latest_model(models_dir: str) -> tuple[str, str]:
    cand = sorted(glob.glob(os.path.join(models_dir, "hgb_*.joblib")))
    if not cand:
        raise FileNotFoundError("No hay modelos en 'models/'. Entrena primero.")
    mp = cand[-1]
    fp = mp.replace(".joblib", ".features.json")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"No existe features.json para el modelo: {fp}")
    return mp, fp

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

def _read_latest_threshold(evals_dir: str) -> float | None:
    path = os.path.join(evals_dir, "LATEST_THRESHOLD.txt")
    if os.path.exists(path):
        try:
            return float(open(path, "r", encoding="utf-8").read().strip())
        except Exception:
            return None
    # fallback: buscar el 'metrics.json' más reciente
    metric_files = sorted(glob.glob(os.path.join(evals_dir, "eval_*", "metrics.json")))
    if not metric_files:
        return None
    latest = metric_files[-1]
    try:
        m = json.load(open(latest, "r", encoding="utf-8"))
        return float(m.get("optimized_threshold", {}).get("best_threshold"))
    except Exception:
        return None

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ALLPAIRS_ctx_1h_h6_ts.csv")
    p.add_argument("--id-cols", default="symbol,timestamp,fwd_ret")
    p.add_argument("--outdir", default="signals/live")
    p.add_argument("--side", default="long", choices=["long", "short", "longshort"])
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--btc-regime-level", default="none", choices=["none", "bull", "bear", "neutral"])
    p.add_argument("--sync-common-timestamp", action="store_true", default=True)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--evals-dir", default="evals")
    p.add_argument("--module-predict", default="src.ml.predict_latest")
    return p

def main():
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model_path, feats_path = _find_latest_model(args.models_dir)
    _print(f"Modelo   : {model_path} (mod: {_latest_mod_time(model_path)})")
    _print(f"Features : {feats_path} (mod: {_latest_mod_time(feats_path)})")
    _print(f"Data     : {args.data} (mod: {_latest_mod_time(args.data)})")
    _print(f"OutDir   : {args.outdir}")
    _print(f"Side     : {args.side}")

    thr = args.threshold
    if thr is None:
        thr = _read_latest_threshold(args.evals_dir)
        _print(f"Thr      : {thr if thr is not None else 'N/A'} (tomado de evals/ si existe)")
    else:
        _print(f"Thr      : {thr} (forzado por CLI)")

    # Log de grupos de features que usa el modelo
    feats = _load_features_list(feats_path)
    groups = _group_counts(feats)
    _print("Feature groups en el MODELO:")
    _print(f"  fg_:  {groups['fg_']}")
    _print(f"  cp_:  {groups['cp_']}")
    _print(f"  cg_:  {groups['cg_']}")
    _print(f"  cmc_: {groups['cmc_']}")

    # --- PREDICT ---
    cmd = [
        sys.executable, "-m", args.module_predict,
        "--data", args.data,
        "--model-prefix", "hgb_",
        "--id-cols", args.id_cols,
        "--outdir", args.outdir,
        "--side", args.side,
        "--btc-regime-level", args.btc_regime_level,
        "--sync-common-timestamp",
    ]
    if thr is not None:
        cmd += ["--threshold", str(thr)]

    _print("\nEjecutando predict_latest...\n")
    subprocess.check_call(cmd)

    # Previsualizar salida generada (si existe)
    csv_out = os.path.join(args.outdir, "signals_latest.csv")
    json_out = os.path.join(args.outdir, "signals_latest.json")
    _print("[OK] Archivos generados:")
    if os.path.exists(csv_out):
        _print(f"  - {csv_out} (mod: {_latest_mod_time(csv_out)})")
    if os.path.exists(json_out):
        _print(f"  - {json_out} (mod: {_latest_mod_time(json_out)})")

    # Pequeña vista previa sin depender de pandas:
    try:
        if os.path.exists(csv_out):
            _print("\n[Preview]")
            with open(csv_out, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    print(line.rstrip())
                    if i >= 6:  # cabecera + 6 filas
                        break
    except Exception:
        pass

if __name__ == "__main__":
    main()

# python run_predict.py
# Forzar un threshold concreto: Forzar un threshold concreto:
# Forzar un threshold concreto: python run_predict.py --no-latest-threshold --default-thr 0.25
# Fijar un modelo específico: python run_predict.py --model-path models\hgb_20251014T225807Z.joblib
# Cambiar side / outdir: Cambiar side / outdir: