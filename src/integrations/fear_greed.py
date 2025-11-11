# -*- coding: utf-8 -*-
"""
Fear & Greed (alternative.me) – integración robusta

Uso CLI:
  python -m src.integrations.fear_greed \
    --cache data/integrations/fear_greed.csv \
    [--refresh] [--save-hourly data/integrations/fear_greed_hourly.csv]

Notas:
- No requiere API key.
- Si no hay red, intenta leer del cache; si tampoco hay, crea CSV vacío con cabeceras.
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Optional, List
import json
import time
import requests
import pandas as pd

API_URL = "https://api.alternative.me/fng/"

def _read_env():
    # Compatibilidad: no exige dotenv, pero si python-dotenv está disponible, lo usa
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

@dataclass
class FGConfig:
    timeout: int = 20
    max_retries: int = 3
    backoff_sec: int = 2

def _fetch_from_api(cfg: FGConfig) -> pd.DataFrame:
    last_err = None
    for i in range(cfg.max_retries):
        try:
            r = requests.get(API_URL, params={"limit": 0, "format": "json"}, timeout=cfg.timeout)
            r.raise_for_status()
            payload = r.json()
            data = payload.get("data", [])
            rows = []
            for it in data:
                # API entrega "timestamp" (segundos unix) y "value"
                ts = pd.to_datetime(int(it["timestamp"]), unit="s", utc=True)
                rows.append({
                    "timestamp": ts,
                    "fg_value": float(it["value"]),
                    "fg_class": it.get("value_classification", None)
                })
            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception as e:
            last_err = e
            time.sleep(cfg.backoff_sec * (i + 1))
    raise RuntimeError(f"No se pudo obtener Fear&Greed desde la API tras {cfg.max_retries} intentos: {last_err}")

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                         "fg_value": pd.Series(dtype="float"),
                         "fg_class": pd.Series(dtype="object")})

def load_fear_greed(cache_path: str, refresh: bool = False, allow_api: bool = True) -> pd.DataFrame:
    """
    Devuelve df con columnas: timestamp (UTC), fg_value, fg_class
    """
    cfg = FGConfig()
    if refresh:
        try:
            df = _fetch_from_api(cfg)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            out = df.copy()
            out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
            out.to_csv(cache_path, index=False)
            print(f"[fear_greed] Descargados {len(df)} registros -> {cache_path}")
            return df
        except Exception as e:
            print(f"[fear_greed] WARN: {e}. Intentando leer cache…")

    if os.path.isfile(cache_path):
        df = pd.read_csv(cache_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        else:
            df = _empty_df()
        print(f"[fear_greed] Registros: {len(df)}  rango: {df['timestamp'].min()} -> {df['timestamp'].max()}")
        return df

    if allow_api:
        df = _fetch_from_api(cfg)
        return df

    # fallback vacío
    print("[fear_greed] WARN: sin datos (ni API ni cache).")
    return _empty_df()

def to_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de diarios → index horario con ffill limitado y lags útiles
    Genera: fg_value, fg_z, fg_lag1, fg_lag6
    """
    if df.empty:
        return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                             "fg_value": [], "fg_z": [], "fg_lag1": [], "fg_lag6": []})

    df = df.copy()
    df = df.sort_values("timestamp").set_index("timestamp")
    # Reindex horario
    full = pd.date_range(df.index.min().floor("h"), df.index.max().ceil("h"), freq="h", tz="UTC")
    out = df.reindex(full)
    # ffill hasta 72h por defecto (criterio utilizado en build_dataset)
    out = out.ffill(limit=72)

    # z-score sobre ventana larga (robusta a escala)
    v = out["fg_value"]
    fg_z = (v - v.rolling(200, min_periods=25).mean()) / (v.rolling(200, min_periods=25).std(ddof=0))
    out = out.assign(
        fg_z=fg_z,
        fg_lag1=v.shift(1),
        fg_lag6=v.shift(6),
    ).reset_index().rename(columns={"index": "timestamp"})
    return out[["timestamp", "fg_value", "fg_z", "fg_lag1", "fg_lag6"]]

def _main():
    _read_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--save-hourly")
    args = ap.parse_args()

    df = load_fear_greed(args.cache, refresh=args.refresh, allow_api=True)

    if args.save_hourly:
        hf = to_hourly_features(df)
        os.makedirs(os.path.dirname(args.save_hourly), exist_ok=True)
        out = hf.copy()
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        out.to_csv(args.save_hourly, index=False)
        print(f"[fear_greed] hourly rows={len(hf)} -> {args.save_hourly}")
    else:
        print(f"[fear_greed] Registros: {len(df)}  rango: {df['timestamp'].min()} -> {df['timestamp'].max()}")

if __name__ == "__main__":
    _main()
