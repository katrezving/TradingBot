import argparse
import numpy as np
import pandas as pd
import joblib
import ccxt
from datetime import datetime, timezone

from src.features.indicators import compute_indicators, select_feature_cols

def fetch_ohlcv(symbol="BNB/USDT", timeframe="1h", limit=500):
    ex = ccxt.binance()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    d = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    d["ts"] = pd.to_datetime(d["ts"], unit="ms", utc=True).dt.tz_convert("UTC")
    return d

def decide(prob, row, thr):
    """
    Confirmadores livianos para subir precisión:
    - Tendencia: ema20 > ema50
    - RSI no sobrecomprado: 50 <= rsi <= 68
    - Momentum: rsi_momentum > 0 (si existe)
    - Volatilidad razonable: -1.5 <= volatility_z <= 2.0 (si existe)
    """
    if prob < thr:
        return False, "prob<thr"

    conds = []
    # tendencia
    conds.append(row.get("ema20", np.nan) > row.get("ema50", np.nan))
    # rsi banda media
    rsi = row.get("rsi", np.nan)
    conds.append((rsi >= 50) and (rsi <= 68))
    # momentum rsi
    if "rsi_momentum" in row:
        conds.append(row["rsi_momentum"] > 0)
    # volatilidad
    if "volatility_z" in row:
        vz = row["volatility_z"]
        conds.append((vz >= -1.5) and (vz <= 2.0))

    # todas deben cumplirse
    if all(bool(c) for c in conds):
        return True, "ok"
    return False, "filters"

    # MACD hist > 0 (sesgo alcista)
    if "macd_hist" in row:
        conds.append(row["macd_hist"] > 0)

    # %B Bollinger en mitad superior del canal (evita debilidad)
    if "bb_perc" in row:
        conds.append(row["bb_perc"] >= 0.5)


def main():
    ap = argparse.ArgumentParser(description="Predicción próxima vela con filtros de confirmación (paper-trading).")
    ap.add_argument("--symbol", type=str, default="BNB/USDT")
    ap.add_argument("--timeframe", type=str, default="1h")
    ap.add_argument("--model", type=str, default="models/baseline_gb.pkl")
    ap.add_argument("--scaler", type=str, default="models/scaler_gb.pkl")
    ap.add_argument("--threshold", type=float, default=0.62)  # ajusta según tu tabla
    args = ap.parse_args()

    # 1) Datos recientes
    raw = fetch_ohlcv(args.symbol, args.timeframe, limit=600)
    feats = compute_indicators(raw)
    feats = feats.dropna().reset_index(drop=True)

    # 2) Cargar modelo + scaler y features
    clf = joblib.load(args.model)
    scaler = joblib.load(args.scaler)
    feat_cols = select_feature_cols(feats)

    X = feats[feat_cols].values
    Xs = scaler.transform(X)
    probs = clf.predict_proba(Xs)[:,1]

    # 3) Última fila (señal actual)
    last = feats.iloc[-1]
    last_prob = float(probs[-1])
    go, reason = decide(last_prob, last, args.threshold)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("\n================ RESULTADO ================")
    print(f"Fecha (UTC): {now}")
    print(f"Par / TF:    {args.symbol} / {args.timeframe}")
    print(f"Prob subida: {last_prob:.3f}  (thr={args.threshold:.3f})")
    print(f"RSI: {last.get('rsi', np.nan):.2f} | ema20>ema50: {bool(last.get('ema20',0)>last.get('ema50',0))} | MACD hist: {last.get('macd_hist', np.nan):.6f}")
    print(f"Volatility_z: {last.get('volatility_z', np.nan)} | rsi_momentum: {last.get('rsi_momentum', np.nan)}")
    print("------------------------------------------")
    if go:
        print("✅ Señal: BUY (confirmada por filtros)")
    else:
        print(f"⛔ Sin señal: {reason}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
