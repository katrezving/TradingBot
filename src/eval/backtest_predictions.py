# -*- coding: utf-8 -*-
"""
Backtest sencillo basado en predictions.csv + dataset original.
Señal long 1-bar cuando pred==1 (o prob_1>=threshold si se indica).
Calcula PnL, win rate, profit factor, max drawdown, Sharpe y resumen por símbolo.

Uso:
python -m src.eval.backtest_predictions --preds evals/hgb_ts_no_leak/predictions.csv --data data/ALLPAIRS_ctx_1h_h6_ts.csv --fee 0.0007 --side long --threshold 0.5794
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--fee", type=float, default=0.0)
    p.add_argument("--side", choices=["long","short","longshort"], default="long")
    p.add_argument("--threshold", type=float, default=None, help="Si se setea y hay prob_1 usa prob>=thr en vez de pred")
    p.add_argument("--out", default="evals/backtest_summary.json")
    return p.parse_args()

def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.clip(peak, 1e-12, None)
    return float(dd.min())

def sharpe(returns: np.ndarray, eps: float=1e-12) -> float:
    mu, sigma = returns.mean(), returns.std(ddof=1)
    return float(mu / (sigma + eps) * np.sqrt(365*24))  # anualización aproximada para 1h

def profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns>0].sum()
    losses = -returns[returns<0].sum()
    return float(gains / (losses if losses>0 else np.nan))

def main():
    args = parse_args()
    preds = pd.read_csv(args.preds)
    data  = pd.read_csv(args.data)

    # Alineación por keys si existen; si no, por posición
    has_keys = all(c in preds.columns for c in ["timestamp","symbol"])
    if has_keys and all(c in data.columns for c in ["timestamp","symbol"]):
        key_cols = ["timestamp","symbol"]
        use_cols = key_cols + ["ret1","close"]
        use_cols = [c for c in use_cols if c in data.columns]
        merged = pd.merge(preds, data[use_cols], on=key_cols, how="left", suffixes=("",""))
    else:
        merged = preds.copy()
        for col in ["ret1","close"]:
            if col in data.columns and len(data)==len(preds):
                merged[col] = data[col].values

    if "ret1" not in merged.columns:
        raise ValueError("No se encontró ret1 para calcular PnL. Usa el dataset con ret1.")

    # Señal
    if args.threshold is not None and "prob_1" in merged.columns:
        base = (merged["prob_1"] >= args.threshold).astype(int)
    else:
        if "pred" not in merged.columns:
            raise ValueError("predictions.csv no tiene 'pred' ni 'prob_1'.")
        base = merged["pred"].astype(int)

    if args.side == "long":
        merged["signal"] = base              # 1 = long, 0 = flat
    elif args.side == "short":
        merged["signal"] = -base             # 1 (prob alta) => short; 0 = flat
    else:  # longshort
        if "prob_1" in merged.columns:
            merged["signal"] = np.where(merged["prob_1"]>=0.5, 1, -1)
        else:
            merged["signal"] = np.where(base==1, 1, -1)

    fee = float(args.fee)
    merged["ret_real"] = merged["ret1"]
    merged["pnl"] = merged["signal"] * (merged["ret_real"] - fee * (merged["signal"]!=0).astype(int))

    trades = (merged["signal"]!=0).sum()
    winrate = float((merged["pnl"]>0).sum()/trades) if trades>0 else np.nan
    eq = (1.0 + merged["pnl"].fillna(0)).cumprod()
    md = max_drawdown(eq.values)
    sp = sharpe(merged["pnl"].fillna(0).values)
    pf = profit_factor(merged["pnl"].fillna(0).values)
    total_ret = float(eq.values[-1]-1.0)

    summary = {
        "rows": int(len(merged)),
        "trades": int(trades),
        "winrate": winrate,
        "profit_factor": pf,
        "sharpe_like": sp,
        "max_drawdown": md,
        "total_return_mult": float(eq.values[-1]),
        "total_return_pct": total_ret,
        "threshold_used": args.threshold,
        "side": args.side,
        "fee": fee,
    }

    if "symbol" in merged.columns:
        per_sym = []
        for sym, g in merged.groupby("symbol"):
            eqs = (1.0 + g["pnl"].fillna(0)).cumprod()
            per_sym.append({
                "symbol": sym,
                "rows": int(len(g)),
                "trades": int((g["signal"]!=0).sum()),
                "winrate": float((g["pnl"]>0).mean()) if len(g)>0 else np.nan,
                "ret_mult": float(eqs.values[-1]),
                "max_dd": max_drawdown(eqs.values),
            })
        summary["per_symbol"] = per_sym

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
