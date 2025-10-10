import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# ======================================================
# FUNCIONES AUXILIARES
# ======================================================

def load_dataset(path: str):
    df = pd.read_csv(path)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def train_val_test_split_time(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


def pick_feature_cols(df):
    drop_cols = {"ts","open","high","low","close","fwd_ret","y"}
    seen = set()
    def base(c): return c.split(".")[0]
    cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        b = base(c)
        if b in seen:
            continue
        seen.add(b)
        cols.append(c)
    return cols


def evaluate_cls(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }
    return out


def simple_pnl(back_df, y_prob, fee=0.0007, threshold=0.55):
    y_prob = np.asarray(y_prob)
    signals = (y_prob >= threshold).astype(int)
    gross = back_df["fwd_ret"].fillna(0).to_numpy() * signals
    net = gross - (2 * fee * signals)  # ida y vuelta
    equity = (1 + pd.Series(net)).cumprod()
    return {
        "trades": int(signals.sum()),
        "avg_trade_ret_%": float((gross[signals == 1].mean() * 100) if signals.sum() else 0.0),
        "total_ret_%": float((equity.iloc[-1] - 1) * 100),
        "equity_curve": equity.tolist(),
    }

# ======================================================
# ENTRENAMIENTO PRINCIPAL
# ======================================================

def main():
    ap = argparse.ArgumentParser(description="Entrena modelo (GB o RF) optimizando precisión con tabla de thresholds.")
    ap.add_argument("--data", type=str, default="data/BNBUSDT_4h_features_h6.csv")
    ap.add_argument("--model", type=str, choices=["gb","rf"], default="gb")
    ap.add_argument("--fee", type=float, default=0.0007)
    ap.add_argument("--outdir", type=str, default="models")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ==========================
    # Cargar dataset
    # ==========================
    df = load_dataset(args.data)
    feat_cols = pick_feature_cols(df)

    train, val, test = train_val_test_split_time(df)
    X_train, y_train = train[feat_cols].values, train["y"].values
    X_val,   y_val   = val[feat_cols].values,   val["y"].values
    X_test,  y_test  = test[feat_cols].values,  test["y"].values

    # ==========================
    # Escalado de datos
    # ==========================
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ==========================
    # Modelo
    # ==========================
    if args.model == "gb":
        clf = GradientBoostingClassifier(random_state=42)
        Xtr, Xva, Xte = X_train_s, X_val_s, X_test_s
    else:
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42
        )
        Xtr, Xva, Xte = X_train, X_val, X_test

    # ==========================
    # Entrenamiento
    # ==========================
    clf.fit(Xtr, y_train)
    
    # ==========================
    # Búsqueda de threshold (0.55–0.80) y tabla comparativa
    # ==========================
    val_prob = clf.predict_proba(Xva)[:, 1]

    candidates = np.linspace(0.55, 0.80, 11)  # 0.55, 0.58, ..., 0.80
    rows = []
    for th in candidates:
        sig = (val_prob >= th)
        trades = int(sig.sum())
        tp = int(((sig == 1) & (y_val == 1)).sum())
        prec = (tp / trades) if trades > 0 else 0.0
        rows.append({"threshold": float(th), "precision": float(prec), "trades": trades})

    # imprimir tabla ordenada
    table = sorted(rows, key=lambda r: (r["precision"], r["trades"]), reverse=True)
    print("\n=== Tabla thresholds (VALIDACIÓN) ===")
    print("thr\tprec\ttrades")
    for r in table:
        print(f"{r['threshold']:.2f}\t{r['precision']:.3f}\t{r['trades']}")

    # Objetivo: precisión mínima con un volumen decente de señales
    TARGET_PREC = 0.70
    MIN_TRADES = 150  # sube/baja este número para ajustar frecuencia

    best = None
    for r in table:
        if r["precision"] >= TARGET_PREC and r["trades"] >= MIN_TRADES:
            best = r
            break

    if best is None:
        # fallback: el de mayor precisión que cumpla al menos 80 trades
        for r in table:
            if r["trades"] >= 80:
                best = r
                break

    if best is None:
        # último recurso: el top de la tabla
        best = table[0]

    chosen_th = float(best["threshold"])
    print(f"\n>>> Threshold elegido (por PRECISIÓN objetivo): {chosen_th:.3f} | precision {best['precision']:.3f} | trades {best['trades']}")

    # ==========================
    # Evaluación
    # ==========================
    test_prob = clf.predict_proba(Xte)[:, 1]

    val_metrics  = evaluate_cls(y_val,  val_prob,  threshold=chosen_th)
    test_metrics = evaluate_cls(y_test, test_prob, threshold=chosen_th)
    val_back  = simple_pnl(val.reset_index(drop=True),  val_prob,  fee=args.fee, threshold=chosen_th)
    test_back = simple_pnl(test.reset_index(drop=True), test_prob, fee=args.fee, threshold=chosen_th)

    # ==========================
    # Resultados
    # ==========================
    print("\n=== Features usadas ===")
    print(feat_cols)
    print("\n=== VALIDACIÓN ===")
    print(val_metrics)
    print("\nPnL VALIDACIÓN:", val_back)
    print("\n=== TEST ===")
    print(test_metrics)
    print("\nPnL TEST:", test_back)

    # ==========================
    # Guardar modelo y reporte
    # ==========================
    model_path  = os.path.join(args.outdir, f"baseline_{args.model}.pkl")
    scaler_path = os.path.join(args.outdir, f"scaler_{args.model}.pkl")
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    summary = {
        "feat_cols": feat_cols,
        "val_metrics": val_metrics,
        "val_back": val_back,
        "test_metrics": test_metrics,
        "test_back": test_back,
        "best_threshold": chosen_th,
        "fee": args.fee,
        "rows": len(df),
        "model": args.model,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "data": args.data,
        "threshold_table": table,
        "min_trades_constraint": MIN_TRADES,
    }

    pd.Series(summary).to_json(os.path.join(args.outdir, f"report_{args.model}.json"), indent=2)
    print(f"\nModelo guardado en: {model_path}")
    print(f"Reporte JSON en:    {os.path.join(args.outdir, f'report_{args.model}.json')}")

if __name__ == "__main__":
    main()
