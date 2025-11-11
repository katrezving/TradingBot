import time
import logging
from typing import Optional, List
import ccxt
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_TIMEFRAMES = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}

def _init_exchange(exchange_id: str = "binance", enable_rate_limit: bool = True):
    ex_cls = getattr(ccxt, exchange_id)
    ex = ex_cls({"enableRateLimit": enable_rate_limit})
    return ex

def fetch_ohlcv_full(symbol: str = "BNB/USDT", timeframe: str = "1h",
                     since_iso: Optional[str] = "2019-01-01T00:00:00Z",
                     exchange_id: str = "binance", limit: int = 1000,
                     max_candles: Optional[int] = None, pause_sec: float = 0.8,
                     max_retries: int = 5) -> pd.DataFrame:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe}; choose one of {list(SUPPORTED_TIMEFRAMES)}")
    ex = _init_exchange(exchange_id)
    since_ms = ex.parse8601(since_iso) if since_iso else None
    all_rows: List[list] = []
    fetched = 0

    while True:
        retries = 0
        while retries < max_retries:
            try:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
                break
            except Exception as e:
                logger.warning(f"Error al traer datos OHLCV ({retries+1}/{max_retries}): {e}")
                time.sleep(pause_sec * 2)
                retries += 1
        else:
            logger.error("Máximo de reintentos alcanzado, abortando descarga.")
            break

        if not ohlcv:
            logger.error("La API no devolvió datos. ¿Símbolo/timeframe correcto?")
            break
        all_rows += ohlcv
        fetched += len(ohlcv)
        since_ms = ohlcv[-1][0] + 1
        if max_candles and fetched >= max_candles:
            break
        time.sleep(pause_sec)
        if len(ohlcv) < limit:
            break

    if not all_rows:
        raise RuntimeError("No se obtuvieron datos OHLCV. Verifica símbolo, red o API.")
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if df.isnull().sum().sum() > 0:
        logger.warning("¡Atención! Hay valores nulos en el DataFrame final.")
    return df

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = fetch_ohlcv_full("BNB/USDT", "1h", "2019-01-01T00:00:00Z", max_candles=1500)
    logger.info(f"Primeras filas:\n{df.head()}\nUltimas filas:\n{df.tail()}")
    save_csv(df, "data/BNBUSDT_1h.csv")
    logger.info(f"Guardado exitosamente - data/BNBUSDT_1h.csv, filas: {len(df)}")
