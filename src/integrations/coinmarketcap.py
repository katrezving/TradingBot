"""
CoinMarketCap integration
- Lee la API key desde CMC_API_KEY (env o .env)
- Toma un snapshot "ahora" de:
    * Global metrics: total market cap (USD), BTC/ETH dominance
    * Listings: ranking y %change_24h para calcular top-gainers
- Acumula snapshots en un CSV "cache" (histórico propio)
- Agrega a frecuencia HORARIA con forward-fill y saca features útiles:
    cmc_total_mcap_usd
    cmc_btc_dominance
    cmc_eth_dominance
    cmc_alt_dominance
    cmc_topgainers20_count_tracked (cuántas de tus symbols están en el top20 24h)
    cmc_topgainers20_mean_change   (media % de los top20 24h)
- Opcional: rolling media (roll_hours) y prefijo personalizable

Uso (PowerShell):
  $env:CMC_API_KEY = "TU_KEY"   # o en .env CMC_API_KEY=...
  python -m src.integrations.coinmarketcap `
    --cache data\integrations\cmc.csv `
    --refresh `
    --save-hourly data\integrations\cmc_hourly.csv `
    --ffill-hours 12 `
    --roll-hours 6 `
    --prefix cmc_ `
    --symbols BTC,ETH,SOL,BNB,LTC,DOT

Notas:
- La API free de CMC no da histórico por endpoint; generamos histórico
  acumulando snapshots cada vez que ejecutes este script (cron/Task Scheduler).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# -------- helpers --------

def _load_api_key() -> str:
    # intenta dotenv si existe
    api_key = os.environ.get("CMC_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
            api_key = os.environ.get("CMC_API_KEY")
        except Exception:
            api_key = os.environ.get("CMC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CMC_API_KEY no encontrada. Ponla en el entorno o en .env (CMC_API_KEY=...)."
        )
    return api_key

def _get(url: str, params: Optional[Dict] = None, timeout: int = 20) -> Dict:
    headers = {
        "Accept": "application/json",
        "X-CMC_PRO_API_KEY": _load_api_key(),
    }
    r = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
    # No hacemos backoff aquí para dejar claro si la API falla (p.ej. 5xx)
    r.raise_for_status()
    return r.json()

def _utc_now_floor_hour() -> pd.Timestamp:
    return pd.Timestamp.utcnow().floor("h").tz_convert("UTC")

# -------- fetchers --------

def fetch_global_metrics() -> Dict:
    """
    /v1/global-metrics/quotes/latest
    """
    url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
    payload = _get(url)
    data = payload.get("data", {}) or {}
    quote = (data.get("quote") or {}).get("USD") or {}
    total_mcap = quote.get("total_market_cap")
    btc_dom = data.get("btc_dominance")
    eth_dom = data.get("eth_dominance")
    ts = _utc_now_floor_hour()
    return {
        "timestamp": ts,
        "total_mcap_usd": total_mcap,
        "btc_dominance": btc_dom,
        "eth_dominance": eth_dom,
    }


def fetch_listings(limit: int = 5000, convert: str = "USD") -> pd.DataFrame:
    """
    /v1/cryptocurrency/listings/latest
    Usamos percent_change_24h para "top gainers" a nivel global.
    """
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    payload = _get(url, params={"limit": limit, "convert": convert})
    rows = []
    ts = _utc_now_floor_hour()
    for it in (payload.get("data") or []):
        sym = it.get("symbol")
        quote = (it.get("quote") or {}).get(convert) or {}
        rows.append(
            {
                "timestamp": ts,
                "symbol": sym,
                "price": quote.get("price"),
                "pct_1h": quote.get("percent_change_1h"),
                "pct_24h": quote.get("percent_change_24h"),
                "pct_7d": quote.get("percent_change_7d"),
                "market_cap": quote.get("market_cap"),
                "volume_24h": quote.get("volume_24h"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# -------- cache / hourly aggregation --------

def _append_snapshot(cache_path: str, df_snap: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    if os.path.exists(cache_path):
        base = pd.read_csv(cache_path)
        # parse datetime
        if "timestamp" in base.columns:
            base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    else:
        base = pd.DataFrame(columns=key_cols)

    out = pd.concat([base, df_snap], ignore_index=True)
    out = out.drop_duplicates(subset=key_cols, keep="last").sort_values("timestamp")
    out.to_csv(cache_path, index=False)
    return out


def _compute_hourly_features(
    df_global: pd.DataFrame,
    df_list: pd.DataFrame,
    tracked_symbols: List[str],
    ffill_hours: int = 12,
    roll_hours: int = 0,
    prefix: str = "cmc_",
) -> pd.DataFrame:
    """
    - df_global: cols [timestamp, total_mcap_usd, btc_dominance, eth_dominance]
    - df_list:   cols [timestamp, symbol, pct_24h, ...]
    """
    # Global metrics -> 1 fila por hora
    g = (
        df_global[["timestamp", "total_mcap_usd", "btc_dominance", "eth_dominance"]]
        .dropna(how="all", subset=["total_mcap_usd", "btc_dominance", "eth_dominance"])
        .copy()
    )
    if g.empty:
        g = pd.DataFrame(
            columns=["timestamp", "total_mcap_usd", "btc_dominance", "eth_dominance"]
        )

    g = g.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    g = g.set_index("timestamp").asfreq("h")
    g = g.ffill(limit=ffill_hours)
    g["alt_dominance"] = (100.0 - (g["btc_dominance"].fillna(0) + g["eth_dominance"].fillna(0))).clip(lower=0)

    # Listings -> top gainers por hora
    tl = df_list.copy()
    if not tl.empty:
        tl["timestamp"] = pd.to_datetime(tl["timestamp"], utc=True, errors="coerce")
        tl = tl.dropna(subset=["timestamp"])
        # top 20 por hora
        top20 = (
            tl.sort_values(["timestamp", "pct_24h"], ascending=[True, False])
              .groupby("timestamp")
              .head(20)
        )
        # count tracked symbols en top20
        mask_tracked = top20["symbol"].isin(tracked_symbols)
        top_counts = top20.groupby("timestamp")["symbol"].agg(
            tracked_count=lambda s: int((s.isin(tracked_symbols)).sum()),
            total="count",
        ).rename(columns={"tracked_count": "topg_tracked_count", "total": "topg_total"})

        top_mean = top20.groupby("timestamp")["pct_24h"].mean().rename("topg_mean_change")

        listing_feat = pd.concat([top_counts, top_mean], axis=1)
        listing_feat = listing_feat.asfreq("h").ffill(limit=ffill_hours)
    else:
        listing_feat = pd.DataFrame(index=g.index if not g.empty else pd.DatetimeIndex([], tz="UTC"))
        listing_feat["topg_tracked_count"] = pd.Series(dtype="float")
        listing_feat["topg_total"] = pd.Series(dtype="float")
        listing_feat["topg_mean_change"] = pd.Series(dtype="float")

    # merge
    out = pd.concat([g, listing_feat], axis=1)
    # opcional rolling
    if roll_hours and roll_hours > 1 and not out.empty:
        roll = out.rolling(f"{roll_hours}h", min_periods=max(1, roll_hours // 2))
        out[f"total_mcap_usd_roll{roll_hours}h_mean"] = roll["total_mcap_usd"].mean()
        out[f"btc_dom_roll{roll_hours}h_mean"] = roll["btc_dominance"].mean()
        out[f"topg_mean_change_roll{roll_hours}h_mean"] = roll["topg_mean_change"].mean()

    # rename con prefijo
    rename_map = {
        "total_mcap_usd": f"{prefix}total_mcap_usd",
        "btc_dominance": f"{prefix}btc_dominance",
        "eth_dominance": f"{prefix}eth_dominance",
        "alt_dominance": f"{prefix}alt_dominance",
        "topg_tracked_count": f"{prefix}topgainers20_count_tracked",
        "topg_total": f"{prefix}topgainers20_total",
        "topg_mean_change": f"{prefix}topgainers20_mean_change",
        f"total_mcap_usd_roll{roll_hours}h_mean": f"{prefix}total_mcap_usd_roll{roll_hours}h_mean",
        f"btc_dom_roll{roll_hours}h_mean": f"{prefix}btc_dom_roll{roll_hours}h_mean",
        f"topg_mean_change_roll{roll_hours}h_mean": f"{prefix}topgainers20_mean_change_roll{roll_hours}h_mean",
    }
    out = out.rename(columns=rename_map)
    out = out.reset_index().rename(columns={"index": "timestamp"})
    return out


# -------- public load --------

def load_coinmarketcap(
    cache_path: str,
    refresh: bool,
    tracked_symbols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - df_global_cache (histórico acumulado)
      - df_listings_cache (histórico acumulado de listings top N)
    """
    # GLOBAL SNAPSHOT
    snap_g = pd.DataFrame([fetch_global_metrics()])
    snap_g["timestamp"] = pd.to_datetime(snap_g["timestamp"], utc=True)
    df_global = _append_snapshot(
        cache_path=cache_path.replace(".csv", "_global.csv"),
        df_snap=snap_g,
        key_cols=["timestamp"],
    )

    # LISTINGS SNAPSHOT (guardamos top 500 por tamaño para ahorrar espacio)
    df_listings_now = fetch_listings(limit=500, convert="USD")
    # guardamos todo el listado (para permitir otros cálculos después)
    df_list = _append_snapshot(
        cache_path=cache_path.replace(".csv", "_listings.csv"),
        df_snap=df_listings_now,
        key_cols=["timestamp", "symbol"],
    )
    return df_global, df_list


# -------- CLI --------

def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Ruta base del cache CSV (se crean *_global.csv y *_listings.csv)")
    ap.add_argument("--refresh", action="store_true", help="Forzar nuevo snapshot ahora y anexar al cache")
    ap.add_argument("--save-hourly", default=None, help="Ruta para guardar features horarios")
    ap.add_argument("--ffill-hours", type=int, default=12)
    ap.add_argument("--roll-hours", type=int, default=0, help="0 = sin rolling")
    ap.add_argument("--prefix", default="cmc_")
    ap.add_argument("--symbols", default="BTC,ETH,SOL,BNB,LTC,DOT", help="Símbolos a trackear para 'count_tracked'")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]

    if args.refresh:
        print("[cmc] tomando snapshot y actualizando cache…")

    df_global, df_list = load_coinmarketcap(args.cache, args.refresh, symbols)

    # construir hourly features
    hf = _compute_hourly_features(
        df_global=df_global,
        df_list=df_list,
        tracked_symbols=symbols,
        ffill_hours=args.ffill_hours,
        roll_hours=args.roll_hours,
        prefix=args.prefix,
    )

    # guardar crudos "unificados" por conveniencia
    # (opcional: unificamos en un solo CSV por simplicidad)
    full = df_global.merge(
        df_list.groupby(["timestamp"]).size().rename("listings_count").reset_index(),
        on="timestamp",
        how="left",
    ).sort_values("timestamp")
    full.to_csv(args.cache, index=False)
    print(f"[cmc] crudo rows={len(full)} -> {args.cache}")

    if args.save_hourly:
        # aseguramos formato timestamp de salida consistente
        if not hf.empty:
            hf["timestamp"] = pd.to_datetime(hf["timestamp"], utc=True)
            hf["timestamp"] = hf["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        hf.to_csv(args.save_hourly, index=False)
        print(f"[cmc] hourly rows={len(hf)} -> {args.save_hourly}")


if __name__ == "__main__":
    _main()
