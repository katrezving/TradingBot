import pandas as pd
import pandas_ta as ta

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("ts").reset_index(drop=True)

    # Retornos
    d["ret1"] = d["close"].pct_change()
    d["ret3"] = d["close"].pct_change(3)
    d["ret12"] = d["close"].pct_change(12)

    # EMAs
    d["ema20"] = ta.ema(d["close"], length=20)
    d["ema50"] = ta.ema(d["close"], length=50)
    d["ema100"] = ta.ema(d["close"], length=100)
    d["ema200"] = ta.ema(d["close"], length=200)
    d["ema_gap_20_50"] = (d["ema20"] - d["ema50"]) / d["close"]
    d["ema_gap_50_100"] = (d["ema50"] - d["ema100"]) / d["close"]
    d["ema_gap_100_200"] = (d["ema100"] - d["ema200"]) / d["close"]
    d["ema_cross_20_50"] = (d["ema20"] > d["ema50"]).astype(int)

    # RSI y momentum
    d["rsi"] = ta.rsi(d["close"], length=14)
    d["rsi_slope"] = d["rsi"].diff()
    d["rsi_overbought"] = (d["rsi"] > 70).astype(int)
    d["rsi_oversold"] = (d["rsi"] < 30).astype(int)
    d["rsi_momentum"] = (d["rsi"] - d["rsi"].rolling(7).mean()) / 100

    # MACD
    macd = ta.macd(d["close"], fast=12, slow=26, signal=9)
    d["macd"] = macd["MACD_12_26_9"]
    d["macd_sig"] = macd["MACDs_12_26_9"]
    d["macd_hist"] = macd["MACDh_12_26_9"]

    # ATR + volatilidad relativa
    d["atr"] = ta.atr(d["high"], d["low"], d["close"], length=14)
    d["atr_frac"] = d["atr"] / d["close"]
    d["volatility_z"] = (d["atr_frac"] - d["atr_frac"].rolling(20).mean()) / (d["atr_frac"].rolling(20).std() + 1e-9)

    # Bollinger %B (20, 2.0) – cálculo manual robusto
    ma = d["close"].rolling(20).mean()
    sd = d["close"].rolling(20).std()
    lower = ma - 2 * sd
    upper = ma + 2 * sd
    d["bb_perc"] = (d["close"] - lower) / ((upper - lower).abs() + 1e-9)

    # Volumen relativo + aceleración
    d["vol_rel"] = d["volume"] / (d["volume"].rolling(20).mean() + 1e-9)
    d["vol_accel"] = d["vol_rel"].diff()

    return d

def make_labeled_dataset(df_ind: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    d = df_ind.copy()
    d["fwd_ret"] = d["close"].pct_change(horizon).shift(-horizon)
    d["y"] = (d["fwd_ret"] > 0).astype(int)
    d = d.dropna().reset_index(drop=True)
    return d

def select_feature_cols(df: pd.DataFrame):
    cols = [
        "ret1","ret3","ret12","rsi","rsi_slope","rsi_overbought","rsi_oversold","rsi_momentum",
        "ema20","ema50","ema100","ema200","ema_gap_20_50","ema_gap_50_100","ema_gap_100_200","ema_cross_20_50",
        "macd","macd_sig","macd_hist","atr_frac","volatility_z","bb_perc",
        "vol_rel","vol_accel","volume"
    ]
    return [c for c in cols if c in df.columns]
