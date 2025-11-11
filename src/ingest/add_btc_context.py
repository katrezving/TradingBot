import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rolling_corr(df, x, y, window):
    try:
        return df[x].rolling(window).corr(df[y])
    except Exception as e:
        logger.error(f"Error en rolling_corr {x} vs {y}: {e}")
        return pd.Series(np.nan, index=df.index)

def rolling_beta(df, x, y, window):
    try:
        cov = df[x].rolling(window).cov(df[y])
        var = df[y].rolling(window).var()
        beta = cov / var
        return beta
    except Exception as e:
        logger.error(f"Error en rolling_beta {x} vs {y}: {e}")
        return pd.Series(np.nan, index=df.index)

def add_btc_context(df, btc_symbol="BTCUSDT", window_corr=24, window_beta=24, window_rel=24):
    # Validar que btc_symbol y columnas clave existan
    if not any(df['symbol'] == btc_symbol):
        logger.error(f"No se encontró BTC en el dataframe (symbol='{btc_symbol}').")
        return df
    df = df.copy()
    rows_btc = df[df['symbol'] == btc_symbol].sort_values('ts')
    for symbol in df['symbol'].unique():
        if symbol == btc_symbol:
            continue
        rows_alt = df[df['symbol'] == symbol].sort_values('ts')
        base = pd.merge_asof(rows_alt, rows_btc, on='ts', suffixes=('', '_btc'), direction="backward")
        # Añadir contexto
        base[f"{symbol}_corr_btc"] = rolling_corr(base, "close", "close_btc", window_corr)
        base[f"{symbol}_beta_btc"] = rolling_beta(base, "close", "close_btc", window_beta)
        base[f"{symbol}_rel_btc"] = base["close"] / base["close_btc"]
        # Rolling z-score
        base[f"{symbol}_rel_btc_z"] = (base[f"{symbol}_rel_btc"] - base[f"{symbol}_rel_btc"].rolling(window_rel).mean()) / base[f"{symbol}_rel_btc"].rolling(window_rel).std()
        # Update ALT rows in df
        for col in base.columns:
            if col not in rows_alt.columns:
                df.loc[rows_alt.index, col] = base[col].values
        logger.info(f"Contexto BTC añadido para {symbol}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Añade contexto BTC (beta, correlación, fuerza relativa) al dataset de pares.")
    parser.add_argument("--in", required=True, help="CSV de entrada con pares y velas base.")
    parser.add_argument("--out", required=True, help="CSV de salida con contexto BTC añadido.")
    parser.add_argument("--btc-symbol", default="BTCUSDT")
    parser.add_argument("--window-corr", type=int, default=24)
    parser.add_argument("--window-beta", type=int, default=24)
    parser.add_argument("--window-rel", type=int, default=24)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.in)
        if "ts" not in df.columns or "symbol" not in df.columns or "close" not in df.columns:
            logger.error("Columnas 'ts', 'symbol', 'close' obligatorias no encontradas en archivo de entrada.")
            return
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        logger.info(f"Archivo base cargado: {args.in}, filas={len(df)} cols={len(df.columns)}")
    except Exception as e:
        logger.error(f"Error cargando archivo base: {e}")
        return

    df_ctx = add_btc_context(df, btc_symbol=args.btc_symbol, window_corr=args.window_corr, window_beta=args.window_beta, window_rel=args.window_rel)
    try:
        pd.DataFrame(df_ctx).to_csv(args.out, index=False)
        logger.info(f"Archivo guardado con contexto BTC: {args.out}")
    except Exception as e:
        logger.error(f"Error guardando archivo de salida: {e}")

if __name__ == "__main__":
    main()
