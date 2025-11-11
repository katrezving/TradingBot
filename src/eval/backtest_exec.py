# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--fee", type=float, default=0.0)
    p.add_argument("--side", choices=["long","short","longshort"], default="long")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--entry-at", choices=["open","close"], default="open")
    p.add_argument("--spread-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--stop-loss-bps", type=float, default=None)
    p.add_argument("--take-profit-bps", type=float, default=None)
    p.add_argument("--fill-priority", choices=["adverse_first","tp_first","sl_first"], default="adverse_first")
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--position-usd", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=None)
    p.add_argument("--max-positions", type=int, default=1)
    p.add_argument("--out", default="evals/backtest_exec_summary.json")
    return p.parse_args()

def bps_to_frac(bps: Optional[float]) -> float:
    return 0.0 if bps is None else bps/10000.0

def entry_price(next_open: float, next_close: float, side: int, entry_at: str, spread_frac: float, slip_frac: float) -> float:
    base = next_open if entry_at=="open" else next_close
    half = spread_frac/2.0
    return base*(1+half+slip_frac) if side>0 else base*(1-half-slip_frac)

def exit_price(next_open: float, next_close: float, side: int, at: str, spread_frac: float, slip_frac: float) -> float:
    base = next_open if at=="open" else next_close
    half = spread_frac/2.0
    return base*(1-half-slip_frac) if side>0 else base*(1+half+slip_frac)

def intrabar_hit(entry: float, side: int, hi: float, lo: float, slf: float, tpf: float, prio: str):
    if side>0:
        sl_lvl = entry*(1-slf) if slf>0 else None
        tp_lvl = entry*(1+tpf) if tpf>0 else None
        sl_hit = (sl_lvl is not None) and (lo<=sl_lvl)
        tp_hit = (tp_lvl is not None) and (hi>=tp_lvl)
    else:
        sl_lvl = entry*(1+slf) if slf>0 else None
        tp_lvl = entry*(1-tpf) if tpf>0 else None
        sl_hit = (sl_lvl is not None) and (hi>=sl_lvl)
        tp_hit = (tp_lvl is not None) and (lo<=tp_lvl)
    order = {"adverse_first":["sl","tp"], "sl_first":["sl","tp"], "tp_first":["tp","sl"]}[prio]
    for k in order:
        if k=="sl" and sl_hit: return "sl", sl_lvl
        if k=="tp" and tp_hit: return "tp", tp_lvl
    return None, None

def compute_signal_row(row, thr: Optional[float], side: str) -> int:
    if thr is not None and "prob_1" in row and not pd.isna(row["prob_1"]):
        base = int(row["prob_1"]>=thr)
    else:
        base = int(row["pred"]) if "pred" in row else 0
    if side=="long": return base
    if side=="short": return -base
    if "prob_1" in row and not pd.isna(row["prob_1"]): return 1 if row["prob_1"]>=0.5 else -1
    return 1 if base==1 else -1

def max_drawdown(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq); dd=(eq-peak)/np.clip(peak,1e-12,None); return float(dd.min())

def profit_factor(r: np.ndarray) -> float:
    g=r[r>0].sum(); l=-r[r<0].sum(); return float(g/(l if l>0 else np.nan))

def sharpe_like(r: np.ndarray, ann:int=365*24) -> float:
    mu=r.mean(); sd=r.std(ddof=1); return float(mu/ (sd+1e-12) * np.sqrt(ann))

def backtest_symbol(df: pd.DataFrame, thr: Optional[float], side: str, fee: float, entry_at: str,
                    spread: float, slip: float, slf: float, tpf: float, prio: str,
                    capital0: float, pos_usd: float, risk: Optional[float], maxpos: int):
    df=df.sort_values("timestamp").reset_index(drop=True)
    for c in ["open","high","low","close"]:
        if c not in df.columns: raise ValueError(f"Falta columna '{c}' en data.")
    eq=capital0; pos=0; size=0.0; entry=None; trades=[]; series=[]
    open_pos=0
    for i in range(len(df)-1):
        row=df.iloc[i]; nxt=df.iloc[i+1]
        sig=compute_signal_row(row, thr, side)
        if pos==0 and open_pos<maxpos and sig!=0:
            price_for_size = nxt["open"] if entry_at=="open" else nxt["close"]
            eprice_tmp = entry_price(price_for_size, price_for_size, sig, entry_at, spread, slip)
            if (risk is not None) and (slf>0):
                stop_dist=slf*eprice_tmp; risk_usd=eq*float(risk); size=risk_usd/max(stop_dist,1e-12)
            else:
                size=pos_usd/max(eprice_tmp,1e-12)
            eprice=entry_price(nxt["open"],nxt["close"],sig,entry_at,spread,slip)
            cost=eprice*size*fee; eq-=cost
            pos=sig; entry=eprice; open_pos+=1
            trades.append({"t":nxt["timestamp"],"action":"ENTER","side":"LONG" if sig>0 else "SHORT","price":eprice,"size":size,"fee":cost,"equity":eq})
        if pos!=0:
            hit, px = intrabar_hit(entry,pos,nxt["high"],nxt["low"],slf,tpf,prio)
            reason=None; ex=None
            if hit is not None:
                ex=px; reason=hit.upper()
            else:
                if sig!=pos:
                    ex=exit_price(nxt["open"],nxt["close"],pos,entry_at,spread,slip); reason="SIGNAL"
            if ex is not None:
                cost=ex*size*fee
                pnl=(ex-entry)*size*(1 if pos>0 else -1)
                eq+=pnl-cost
                trades.append({"t":nxt["timestamp"],"action":"EXIT","reason":reason,"side":"LONG" if pos>0 else "SHORT","price":ex,"size":size,"fee":cost,"pnl":pnl,"equity":eq})
                pos=0; size=0.0; entry=None; open_pos=max(0,open_pos-1)
        series.append(eq)
    eq_arr=np.array(series) if series else np.array([capital0])
    step=np.diff(eq_arr,prepend=capital0)/max(capital0,1e-12)
    exits=[t for t in trades if t.get("action")=="EXIT"]; wins=[t for t in exits if t.get("pnl",0)>0]
    return {
        "equity": eq_arr.tolist(),
        "ret_mult": float(eq_arr[-1]/capital0),
        "total_return_pct": float(eq_arr[-1]/capital0-1.0),
        "profit_factor": profit_factor(step),
        "max_drawdown": max_drawdown(eq_arr),
        "sharpe_like": sharpe_like(step),
        "winrate": (len(wins)/len(exits)) if exits else np.nan,
        "trades": trades,
        "n_exits": len(exits),
    }

def main():
    args=parse_args()
    preds=pd.read_csv(args.preds)
    data =pd.read_csv(args.data)

    need=["timestamp","symbol"]
    have_ids_preds=set(need).issubset(preds.columns)
    have_ids_data =set(need).issubset(data.columns)

    if have_ids_preds and have_ids_data:
        cols_needed=["timestamp","symbol","open","high","low","close"]
        miss=[c for c in cols_needed if c not in data.columns]
        if miss: raise ValueError(f"Faltan columnas en data: {miss}")
        merged=pd.merge(preds, data[cols_needed], on=["timestamp","symbol"], how="left")
    else:
        if len(preds)!=len(data):
            raise ValueError("Preds y data no tienen la misma longitud y no hay IDs para alinear.")
        need_ohlc=["open","high","low","close"]
        miss=[c for c in need_ohlc if c not in data.columns]
        if miss: raise ValueError(f"Faltan columnas OHLC en data: {miss}")
        merged=preds.copy()
        for c in need_ohlc: merged[c]=data[c].values
        for c in ["timestamp","symbol"]:
            if c in data.columns: merged[c]=data[c].values

    # coalesce open/high/low/close si hay sufijos _x/_y
    for col in ["open","high","low","close"]:
        if col not in merged.columns:
            x,y=f"{col}_x",f"{col}_y"
            if x in merged.columns or y in merged.columns:
                merged[col]=merged.get(y, pd.Series([np.nan]*len(merged))).combine_first(
                            merged.get(x, pd.Series([np.nan]*len(merged))))
                for dcol in [x,y]:
                    if dcol in merged.columns: merged.drop(columns=[dcol], inplace=True)

    spread=bps_to_frac(args.spread_bps); slip=bps_to_frac(args.slippage_bps)
    slf=bps_to_frac(args.stop_loss_bps); tpf=bps_to_frac(args.take_profit_bps)

    out={"symbols":[]}; all_eq=None; weights=[]
    for sym, g in merged.groupby(merged["symbol"] if "symbol" in merged.columns else (np.zeros(len(merged)))):
        res=backtest_symbol(
            df=g, thr=args.threshold, side=args.side, fee=float(args.fee), entry_at=args.entry_at,
            spread=spread, slip=slip, slf=slf, tpf=tpf, prio=args.fill_priority,
            capital0=float(args.capital), pos_usd=float(args.position_usd), risk=args.risk_per_trade,
            maxpos=int(args.max_positions)
        )
        if isinstance(sym, (int,float)) and not ("symbol" in merged.columns): sym="ALL"
        out["symbols"].append({"symbol": str(sym), "winrate": res["winrate"], "profit_factor": res["profit_factor"],
                               "max_drawdown": res["max_drawdown"], "ret_mult": res["ret_mult"], "n_exits": res["n_exits"]})
        eq=np.array(res["equity"],dtype=float)
        if all_eq is None: all_eq=eq
        else:
            L=min(len(all_eq),len(eq)); all_eq=all_eq[:L]+eq[:L]
        weights.append(1.0)

    if all_eq is None: all_eq=np.array([args.capital],dtype=float)
    else: all_eq=all_eq/max(len(weights),1)
    step=np.diff(all_eq,prepend=all_eq[0])/max(args.capital,1e-12)
    summary={
        "global":{
            "ret_mult": float(all_eq[-1]/args.capital),
            "total_return_pct": float(all_eq[-1]/args.capital-1.0),
            "profit_factor": profit_factor(step),
            "max_drawdown": max_drawdown(all_eq),
            "sharpe_like": sharpe_like(step),
        },
        "params":{
            "threshold_used": args.threshold, "side": args.side, "fee": float(args.fee),
            "entry_at": args.entry_at, "spread_bps": args.spread_bps, "slippage_bps": args.slippage_bps,
            "stop_loss_bps": args.stop_loss_bps, "take_profit_bps": args.take_profit_bps, "fill_priority": args.fill_priority,
            "capital": args.capital, "position_usd": args.position_usd, "risk_per_trade": args.risk_per_trade, "max_positions": args.max_positions
        },
        "per_symbol": out["symbols"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary, indent=2))

if __name__=="__main__":
    main()