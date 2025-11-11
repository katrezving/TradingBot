# -*- coding: utf-8 -*-
"""
CoinGlass integración (v4 primero, v1 fallback) – Dominance BTC + Open Interest agregado (hourly)

Uso:
  python -m src.integrations.coinglass ^
    --cache data/integrations/coinglass.csv ^
    --refresh ^
    --days 180 ^
    --symbols BTC,ETH,SOL,BNB,LTC,DOT ^
    --save-hourly data/integrations/coinglass_hourly.csv ^
    --ffill-hours 12 ^
    --prefix cg_ ^
    --api-version auto

ENV:
  COINGLASS_API_KEY=...        (obligatorio)
  COINGLASS_BASE_URL=...       (opcional; default https://open-api.coinglass.com)

Salidas:
  - crudo: timestamp, oi_usd, dominance_btc
  - hourly: timestamp, cg_oi_usd, cg_dominance_btc, cg_oi_usd_roll12
"""

from __future__ import annotations
import argparse
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd


# ------------------------- Utilidades -------------------------

def _read_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

def _base_url() -> str:
    b = os.getenv("COINGLASS_BASE_URL", "https://open-api.coinglass.com").rstrip("/")
    return b

def _headers() -> Dict[str, str]:
    key = os.getenv("COINGLASS_API_KEY", "").strip()
    h = {"Accept": "application/json"}
    if key:
        # Enviar ambos por compatibilidad
        h["coinglassSecret"] = key
        h["CG-API-KEY"] = key
    return h

def _utc_now() -> pd.Timestamp:
    now = pd.Timestamp.utcnow()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    return now

def _safe_json_get(url: str, params: Dict[str, Any] | None = None, retries: int = 2, backoff: float = 0.6) -> Optional[Dict[str, Any]]:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=_headers(), params=params or {}, timeout=15)
            if r.status_code in (400, 401, 403, 404):
                # Petición inválida / sin plan / path inexistente -> no insistimos
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(backoff * (i + 1))
    print(f"[coinglass] WARN request falló tras {retries} intentos: {last}")
    return None

def _empty_raw_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "oi_usd", "dominance_btc"])


# ------------------------- Endpoints candidatos -------------------------

def _candidate_paths(api_version: str) -> Dict[str, List[str]]:
    """
    Devuelve listas de paths candidatos para cada recurso según versión elegida.
    'auto' probará v4 y luego v1.
    """
    v4 = {
        "dominance": [
            "/public/v4/market/global_dominance",     # naming v4 típico
            "/public/v4/market/globalDominance",      # variante
        ],
        "oi_history": [
            "/public/v4/futures/open_interest/history",   # naming v4 típico
            "/public/v4/futures/openInterest/history",    # variante
        ],
    }
    v1 = {
        "dominance": [
            "/api/pro/v1/market/globalDominance",
        ],
        "oi_history": [
            "/api/pro/v1/futures/openInterest/history",
        ],
    }
    if api_version == "v4":
        return v4
    if api_version == "v1":
        return v1
    # auto
    return {"dominance": v4["dominance"] + v1["dominance"],
            "oi_history": v4["oi_history"] + v1["oi_history"]}


# ------------------------- Fetchers robustos -------------------------

def _fetch_dominance(paths: List[str]) -> pd.DataFrame:
    base = _base_url()
    for p in paths:
        url = f"{base}{p}"
        js = _safe_json_get(url)
        if not js:
            continue

        # Buscar la clave de datos
        data = js.get("data") or js.get("result") or js.get("list") or js.get("rows") or []
        if not isinstance(data, list):
            # Algunas APIs v4 devuelven {data:{list:[...]}}:
            if isinstance(js.get("data"), dict):
                data = js["data"].get("list") or js["data"].get("rows") or []

        rows = []
        if isinstance(data, list):
            for it in data:
                ts = it.get("time") or it.get("timestamp") or it.get("ts")
                ts = pd.to_datetime(ts, utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                dom = it.get("btcDominance") or it.get("BTC") or it.get("btc") or it.get("dominance")
                try:
                    dom = float(dom)
                except Exception:
                    dom = None
                rows.append({"timestamp": ts, "dominance_btc": dom})

        if rows:
            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            return df

    return pd.DataFrame(columns=["timestamp", "dominance_btc"])

def _fetch_oi_aggregated(paths: List[str], symbols: List[str], days: int) -> pd.DataFrame:
    base = _base_url()
    since = _utc_now().floor("h") - pd.Timedelta(days=days)
    out_rows: List[Dict[str, Any]] = []

    for sym in symbols:
        payload = None
        for p in paths:
            url = f"{base}{p}"
            payload = _safe_json_get(url, params={"symbol": sym, "interval": "60min"})
            if payload:
                break
        if not payload:
            continue

        data = payload.get("data") or payload.get("result") or payload.get("list") or payload.get("rows") or []
        if not isinstance(data, list):
            if isinstance(payload.get("data"), dict):
                data = payload["data"].get("list") or payload["data"].get("rows") or []

        for it in (data or []):
            ts = it.get("time") or it.get("timestamp") or it.get("ts")
            ts = pd.to_datetime(ts, utc=True, errors="coerce")
            if pd.isna(ts) or ts < since:
                continue
            oi = it.get("openInterest") or it.get("oi") or it.get("value") or it.get("oiUsd")
            try:
                oi = float(oi)
            except Exception:
                oi = None
            out_rows.append({"timestamp": ts, "symbol": sym, "oi_usd": oi})
        time.sleep(0.12)  # cortesía

    if not out_rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "oi_usd"])

    df = pd.DataFrame(out_rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


# ------------------------- API pública del módulo -------------------------

def load_coinglass(cache_path: str,
                   refresh: bool = False,
                   days: int = 180,
                   symbols: Optional[List[str]] = None,
                   api_version: str = "auto") -> pd.DataFrame:
    """
    Devuelve df 'raw' con columnas: timestamp, oi_usd, dominance_btc
    """
    symbols = symbols or ["BTC"]

    if refresh:
        cand = _candidate_paths(api_version)
        dom = _fetch_dominance(cand["dominance"])
        oi = _fetch_oi_aggregated(cand["oi_history"], symbols, days)

        if dom.empty and oi.empty:
            print("[coinglass] WARN: endpoints sin datos o no disponibles (se guarda CSV vacío con cabeceras).")
            raw = _empty_raw_df()
        else:
            if oi.empty:
                raw = dom.rename(columns={"dominance_btc": "dominance_btc"})
            elif dom.empty:
                raw = oi[["timestamp", "oi_usd"]]
            else:
                raw = pd.merge(
                    oi[["timestamp", "oi_usd"]],
                    dom[["timestamp", "dominance_btc"]],
                    on="timestamp",
                    how="outer"
                )

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        out = raw.copy()
        if "timestamp" in out.columns and not out.empty:
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")
        out.to_csv(cache_path, index=False)
        print(f"[coinglass] crudo rows={len(raw)} -> {cache_path}")
        return raw

    if os.path.isfile(cache_path):
        df = pd.read_csv(cache_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df

    print("[coinglass] WARN: sin cache y sin refresh; devolviendo vacío.")
    return _empty_raw_df()

def hourly_features(df: pd.DataFrame, ffill_hours: int = 12, prefix: str = "cg_") -> pd.DataFrame:
    """
    Devuelve hourly features:
      - cg_oi_usd
      - cg_dominance_btc
      - cg_oi_usd_roll12 (media móvil 12h)
    """
    # Garantizar columnas
    for c in ["timestamp", "oi_usd", "dominance_btc"]:
        if c not in df.columns:
            df[c] = pd.NA

    if df.empty:
        return pd.DataFrame({
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            f"{prefix}oi_usd": pd.Series(dtype="float"),
            f"{prefix}dominance_btc": pd.Series(dtype="float"),
            f"{prefix}oi_usd_roll12": pd.Series(dtype="float"),
        })

    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    agg = d.groupby(pd.Grouper(freq="h")).agg({
        "oi_usd": "mean",
        "dominance_btc": "mean"
    })

    if not agg.empty:
        full = pd.date_range(agg.index.min().floor("h"), agg.index.max().ceil("h"), freq="h", tz="UTC")
        agg = agg.reindex(full)
    agg = agg.ffill(limit=ffill_hours)

    agg[f"{prefix}oi_usd"] = agg["oi_usd"]
    agg[f"{prefix}dominance_btc"] = agg["dominance_btc"]
    agg[f"{prefix}oi_usd_roll12"] = agg["oi_usd"].rolling(12, min_periods=1).mean()

    out = agg.reset_index().rename(columns={"index": "timestamp"})
    return out[["timestamp", f"{prefix}oi_usd", f"{prefix}dominance_btc", f"{prefix}oi_usd_roll12"]]


# ------------------------- CLI -------------------------

def _main():
    _read_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--symbols", type=str, default="BTC")
    ap.add_argument("--save-hourly", type=str)
    ap.add_argument("--ffill-hours", type=int, default=12)
    ap.add_argument("--prefix", type=str, default="cg_")
    ap.add_argument("--api-version", type=str, default="auto", choices=["auto", "v4", "v1"])
    ap.add_argument("--base-url", type=str, default=None)
    args = ap.parse_args()

    if args.base_url:
        os.environ["COINGLASS_BASE_URL"] = args.base_url

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    raw = load_coinglass(args.cache, refresh=args.refresh, days=args.days, symbols=symbols, api_version=args.api_version)

    if args.save_hourly:
        hf = hourly_features(raw, ffill_hours=args.ffill_hours, prefix=args.prefix)
        os.makedirs(os.path.dirname(args.save_hourly), exist_ok=True)
        out = hf.copy()
        if not out.empty and "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")
        out.to_csv(args.save_hourly, index=False)
        print(f"[coinglass] hourly rows={len(hf)} -> {args.save_hourly}")
    else:
        print(f"[coinglass] raw rows={len(raw)}")

if __name__ == "__main__":
    _main()
