# -*- coding: utf-8 -*-
"""
Evaluación de un modelo entrenado:
- Alinea columnas con features del entrenamiento (.features.json).
- Predice y guarda predictions.csv.
- Calcula métricas (accuracy, f1_macro, roc_auc, mcc) y guarda metrics.json.
- Optimiza umbral si se solicita (--optimize-threshold f1|precision|recall|youden).
- Guarda classification_report y (si hay matplotlib) matrices de confusión.

Uso típico:
python -m src.ml.evaluate_baseline ^
  --data data\\ALLPAIRS_ctx_1h_h6_ts.csv ^
  --model-path models\\hgb_YYYYMMDDTHHMMSSZ.joblib ^
  --features-path models\\hgb_YYYYMMDDTHHMMSSZ.features.json ^
  --target y ^
  --id-cols symbol,timestamp,fwd_ret ^
  --outdir evals\\hgb_ts_no_leak ^
  --optimize-threshold f1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

from pandas.api.types import is_numeric_dtype


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained model and export predictions/metrics.")
    p.add_argument("--data", required=True, help="CSV con datos de evaluación (mismo pipeline de entrenamiento).")
    p.add_argument("--model-path", required=True, help="Ruta al .joblib del modelo.")
    p.add_argument("--features-path", required=True, help="Ruta al .features.json con lista de columnas usadas.")
    p.add_argument("--target", default="y", help="Nombre de la columna objetivo (si existe).")
    p.add_argument("--id-cols", default="", help="Columnas ID/aux a excluir (coma), e.g.: symbol,timestamp,fwd_ret")
    p.add_argument("--outdir", default="evals/baseline", help="Directorio de salida.")
    p.add_argument("--optimize-threshold", choices=["f1", "precision", "recall", "youden"], default=None,
                   help="Optimiza umbral sobre prob_1 para maximizar la métrica elegida.")
    p.add_argument("--threshold-grid", default="auto",
                   help='Grid para búsqueda de umbral; "auto" usa np.linspace(0.01,0.99,99) o puedes pasar "0.1,0.2,..."')
    return p.parse_args()

def load_features_list(path):
    """
    Carga la lista de features desde JSON, compatible con versiones antiguas o nuevas.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "features" in data:
        return data["features"]
    else:
        raise ValueError(f"No se pudo interpretar el formato del archivo de features: {path}")

def build_matrix(df: pd.DataFrame, feature_cols: List[str], target: str, id_cols: List[str]) -> (pd.DataFrame, Optional[pd.Series], pd.DataFrame):
    """
    Devuelve:
      X (solo features, alineadas a training),
      y (si target existe, caso contrario None),
      meta (columnas útiles: symbol, timestamp, close si existen)
    """
    d = df.copy()

    # eliminar columnas id explícitas
    drop_cols = [c for c in id_cols if c in d.columns]
    d = d.drop(columns=drop_cols, errors="ignore")

    # mantener target si existe para y; QUITARLO de X
    y = d[target] if target in d.columns else None
    if target in d.columns:
        d = d.drop(columns=[target])

    # seleccionar solo numéricas (evita conflictos tz-aware)
    num_cols = [c for c in d.columns if is_numeric_dtype(d[c])]
    Xall = d[num_cols].copy().replace([np.inf, -np.inf], np.nan)

    # alinear al orden y conjunto de training features
    for c in feature_cols:
        if c not in Xall.columns:
            Xall[c] = 0.0
    X = Xall[feature_cols]

    # meta informativa (si existen)
    meta_cols = [c for c in ["symbol", "timestamp", "close"] if c in df.columns]
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    return X, y, meta


def optimize_threshold(y_true: np.ndarray, prob1: np.ndarray, metric: str, grid: np.ndarray) -> Dict[str, Any]:
    best_thr, best_score = None, -np.inf
    for t in grid:
        y_pred = (prob1 >= t).astype(int)
        if metric == "f1":
            s = f1_score(y_true, y_pred, average="binary", zero_division=0)
        elif metric == "precision":
            s = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            s = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            # Youden J = TPR - FPR
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
            tpr = tp / max(tp + fn, 1e-12)
            fpr = fp / max(fp + tn, 1e-12)
            s = tpr - fpr
        else:
            s = -np.inf
        if s > best_score:
            best_score, best_thr = s, float(t)
    return {"metric": metric, "best_threshold": best_thr, "best_score": best_score}


def save_confusion_fig(cm: np.ndarray, labels: List[str], path: Path, title: str):
    if not _HAVE_MPL:
        return
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    print(f"Cargando modelo: {args.model_path}")
    model = joblib.load(args.model_path)

    print(f"Cargando features: {args.features_path}")
    feature_cols = load_features_list(args.features_path)

    print(f"Cargando datos: {args.data}")
    df = pd.read_csv(args.data)

    # Construir matrices
    X, y, meta = build_matrix(df, feature_cols=feature_cols, target=args.target, id_cols=id_cols)

    # Predicción
    print("Prediciendo...")
    y_prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            y_prob = proba[:, 1]
    y_pred = model.predict(X)

    # Métricas (si hay y)
    metrics: Dict[str, Any] = {}
    if y is not None:
        y_true = y.values.astype(int)
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        # roc_auc solo si hay proba y dos clases presentes
        try:
            if y_prob is not None and len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        try:
            metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            pass

    # Optimización de threshold (opcional)
    optimized = None
    if y is not None and y_prob is not None and args.optimize_threshold:
        if args.threshold_grid == "auto":
            grid = np.linspace(0.01, 0.99, 99)
        else:
            grid = np.array([float(x) for x in args.threshold_grid.split(",") if x.strip()])

        optimized = optimize_threshold(y_true, y_prob, args.optimize_threshold, grid)
        print(f"Umbral optimizado ({optimized['metric']}): {optimized['best_threshold']:.4f}  score={optimized['best_score']:.4f}")

    # Guardar predictions.csv
    out_pred = pd.DataFrame({"pred": y_pred})
    if y_prob is not None:
        out_pred["prob_1"] = y_prob
    if args.target in df.columns:
        out_pred["y_true"] = df[args.target].values

    # Añadir meta si existe
    for c in ["symbol", "timestamp", "close"]:
        if c in meta.columns:
            out_pred[c] = meta[c].values

    pred_path = outdir / "predictions.csv"
    out_pred.to_csv(pred_path, index=False)
    print(f"Predicciones guardadas en: {pred_path}")

    # Guardar metrics.json
    bundle = {
        "metrics": metrics,
        "optimized_threshold": optimized if optimized is not None else None,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "classes_detected": int(len(np.unique(y_pred))),
    }
    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"Métricas guardadas en: {metrics_path}")

    # Classification report
    if y is not None:
        report_txt = classification_report(y_true, y_pred, digits=4, zero_division=0)
        rep_path = outdir / "classification_report.txt"
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(report_txt)
        print(f"Classification report guardado en: {rep_path}")

        # Confusion matrices (si hay matplotlib)
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0,1])
            if _HAVE_MPL:
                save_confusion_fig(cm, labels=["0","1"], path=outdir / "confusion_matrix.png",
                                   title="Confusion Matrix (counts)")
                cm_norm = (cm.T / np.clip(cm.sum(axis=1), 1e-12, None)).T
                save_confusion_fig(cm_norm, labels=["0","1"], path=outdir / "confusion_matrix_normalized.png",
                                   title="Confusion Matrix (normalized)")
                print("Matriz de confusión guardada en: "
                      f"{outdir / 'confusion_matrix.png'} y {outdir / 'confusion_matrix_normalized.png'}")
        except Exception as e:
            print(f"[warn] No se pudo guardar matrices de confusión: {e}")

    print("\n✔ Evaluation done.")


if __name__ == "__main__":
    main()
