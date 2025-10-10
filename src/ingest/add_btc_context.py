import os
import pandas as pd
import numpy as np

def _rolling_corr(a, b, win):
    return a.rolling(win).corr(b)

def _rolling_beta(a_ret, b_ret, win):
    cov = a_ret.rolling(win).cov(b_ret)
    var = b_ret.rolling(win).var()
    return cov / (var.replace(0, np.nan))

def add_btc_context(all_file="data/ALLPAIRS_1h_features_h6.csv", out_file="data/ALLPAIRS_ctx_1h_h6.csv"):
    df = pd.read_csv(all_file)
    # asegurar tipos
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["ts", "symbol"]).reset_index(drop=True)

    # separar por símbolo y quedarnos con columnas clave para pivotear
    # asumimos que el archivo ya trae: ['ts','symbol','close','ret1','ret3','ret12','rsi','ema20','ema50','atr_frac', ...]
    # 1) Extraer serie BTC
    btc = df[df["symbol"] == "BTCUSDT"].copy()
    if btc.empty:
        raise ValueError("No se encontró BTCUSDT en el dataset combinado.")

    # Serie BTC: returns adicionales y métricas
    btc = btc.sort_values("ts").reset_index(drop=True)
    btc["btc_ret1"] = btc["close"].pct_change()
    btc["btc_ret3"] = btc["close"].pct_change(3)
    btc["btc_ret6"] = btc["close"].pct_change(6)
    btc["btc_trend"] = (btc["ema20"] > btc["ema50"]).astype(int)
    btc["btc_rsi"] = btc["rsi"]
    # volatilidad BTC: z-score de atr_frac
    btc["btc_vol_z"] = (btc["atr_frac"] - btc["atr_frac"].rolling(48).mean()) / (btc["atr_frac"].rolling(48).std() + 1e-9)
    btc_ctx = btc[["ts","btc_ret1","btc_ret3","btc_ret6","btc_trend","btc_rsi","btc_vol_z","close"]].rename(columns={"close":"btc_close"})

    # 2) Proxy de "alt index": promedio de cierres normalizados (excluye BTC)
    non_btc = df[df["symbol"] != "BTCUSDT"].copy()
    piv = non_btc.pivot(index="ts", columns="symbol", values="close").sort_index()

    if piv.shape[1] < 1:
        raise ValueError("No se encontraron símbolos ALT (distintos de BTCUSDT) para construir el alt_index.")

    # normalizar cada serie por su media móvil larga para evitar escala
    # usamos min_periods para no llenar de NaN al inicio
    norm = piv / (piv.rolling(240, min_periods=100).mean() + 1e-9)

    # promedio transversal -> alt_index
    alt_index = norm.mean(axis=1)
    alt_index.name = "alt_index"          # ← NOMBRE EXPLÍCITO
    alt_df = alt_index.to_frame().reset_index()  # cols: ts, alt_index

    # 3) BTC/ALT ratio y su pendiente
    btc_close_df = btc_ctx[["ts", "btc_close"]]  # ya trae ts alineado y ordenado
    # merge por 'ts' para asegurar columna con nombre correcto
    data = pd.merge(alt_df, btc_close_df, on="ts", how="inner").sort_values("ts")
    data["btc_alt_ratio"] = data["btc_close"] / (data["alt_index"] + 1e-9)
    data["btc_alt_ratio_slope"] = data["btc_alt_ratio"].diff()  # puedes cambiar por slope rolling si quieres

    ratio_df = data[["ts", "btc_alt_ratio", "btc_alt_ratio_slope"]]

    # 4) Unir contexto BTC + ratio a TODO el dataset
    out = df.merge(btc_ctx, on="ts", how="left").merge(ratio_df, on="ts", how="left")

    # 5) Features de fuerza relativa y co-movimiento por símbolo
    out = out.sort_values(["symbol","ts"]).reset_index(drop=True)
    out["rel_vs_btc_3"] = out["ret3"] - out["btc_ret3"]

    # Corr y beta rolling respecto a BTC (por símbolo)
    out["corr_btc_24h"] = np.nan
    out["beta_btc_24h"] = np.nan
    for sym, g in out.groupby("symbol"):
        r_alt = g["ret1"]
        # alinear con serie BTC por ts ya está hecho tras merge; usamos columnas ya en 'out'
        r_btc = g["btc_ret1"]
        corr = _rolling_corr(r_alt, r_btc, 24)
        beta = _rolling_beta(r_alt, r_btc, 24)
        out.loc[g.index, "corr_btc_24h"] = corr.values
        out.loc[g.index, "beta_btc_24h"] = beta.values

    # Guardar
    out.to_csv(out_file, index=False)
    print(f"✅ Guardado con contexto BTC: {out_file}")
    return out

if __name__ == "__main__":
    add_btc_context()
