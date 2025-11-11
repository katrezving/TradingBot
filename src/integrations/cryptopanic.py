# -*- coding: utf-8 -*-
"""
CryptoPanic – integración robusta (posts → agregación horaria)

Uso CLI:
  python -m src.integrations.cryptopanic \
    --cache data/integrations/cryptopanic.csv \
    [--refresh] [--days 30] [--currencies BTC,ETH,SOL] \
    [--save-hourly data/integrations/cryptopanic_hourly.csv] \
    [--ffill-hours 6] [--roll-hours 3] [--prefix cp_] [--public]

Salida hourly:
  timestamp, <prefix>n_posts, <prefix>score_simple_mean, <prefix>score_weighted_mean,
  <prefix>important_sum, <prefix>bull_minus_bear

Notas:
- Usa CRYPTOPANIC_API_KEY del entorno si está disponible (para endpoint full).
- Con --public usa el endpoint público (menos campos).
- Siempre emite CSV con cabeceras; si no hay datos, queda vacío pero útil para merge.
"""

from __future__ import annotations
import argparse
import os
import time
from typing import Dict, Any, List, Optional
import requests
import pandas as pd

BASE_URL = "https://cryptopanic.com/api/v1/posts/"

def _read_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

def _auth_params(use_public: bool) -> Dict[str, Any]:
    if use_public:
        return {"public": "true"}
    token = os.getenv("CRYPTOPANIC_API_KEY", "").strip()
    return {"auth_token": token} if token else {"public": "true"}

def _request_page(url: str, params: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _paginate(days: int, currencies: Optional[List[str]], use_public: bool) -> List[Dict[str, Any]]:
    # CryptoPanic pagina con "next" en payload; limit fixed server-side
    params = _auth_params(use_public)
    if currencies:
        params["currencies"] = ",".join(currencies)
    # Usamos "from" para limitar por fecha, si la API lo respeta
    since_iso = (pd.Timestamp.utcnow().floor("h") - pd.Timedelta(days=days)).isoformat()
    params["from"] = since_iso

    url = BASE_URL
    out: List[Dict[str, Any]] = []
    for _ in range(50):  # límite defensivo de páginas
        payload = _request_page(url, params)
        results = payload.get("results", [])
        out.extend(results)
        next_url = payload.get("next")
        if not next_url:
            break
        url = next_url
        params = {}  # después de la primera página, la URL ya lleva query
        time.sleep(0.2)  # cortesía
    return out

def _normalize_rows(items: List[Dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=[
            "timestamp","score","votes_positive","votes_negative","is_important","currencies"
        ])

    rows = []
    for it in items:
        # published_at puede venir como 'published_at' o 'created_at' según endpoint
        ts = it.get("published_at") or it.get("created_at") or it.get("date") or it.get("created")
        ts = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        votes = it.get("votes") or {}
        pos = votes.get("positive") or 0
        neg = votes.get("negative") or 0
        score = (pos - neg)
        cur_list = []
        for c in (it.get("currencies") or []):
            # API retorna objects {code: "BTC", title: "..."}
            code = c.get("code") or c.get("title") or ""
            if code:
                cur_list.append(str(code))
        rows.append({
            "timestamp": ts,
            "score": float(score),
            "votes_positive": float(pos),
            "votes_negative": float(neg),
            "is_important": bool(it.get("important") or it.get("is_important") or False),
            "currencies": ",".join(sorted(set(cur_list))) if cur_list else ""
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df

def load_cryptopanic(cache_path: str,
                     refresh: bool = False,
                     days: int = 30,
                     currencies: Optional[List[str]] = None,
                     use_public: bool = False) -> pd.DataFrame:
    """
    Devuelve df 'raw' con columnas:
      timestamp (UTC), score, votes_positive, votes_negative, is_important, currencies
    """
    if refresh:
        try:
            items = _paginate(days=days, currencies=currencies, use_public=use_public)
            df = _normalize_rows(items)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            out = df.copy()
            if not out.empty:
                out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
            out.to_csv(cache_path, index=False)
            print(f"[cryptopanic] descargado rows={len(df)} -> {cache_path}")
            return df
        except Exception as e:
            print(f"[cryptopanic] WARN: {e}. Intentando leer cache…")

    if os.path.isfile(cache_path):
        df = pd.read_csv(cache_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        else:
            df = pd.DataFrame(columns=["timestamp","score","votes_positive","votes_negative","is_important","currencies"])
        return df

    # fallback vacío
    print("[cryptopanic] WARN: sin datos (ni API ni cache).")
    return pd.DataFrame(columns=["timestamp","score","votes_positive","votes_negative","is_important","currencies"])

def hourly_features(df: pd.DataFrame, ffill_hours: int = 6, roll_hours: int = 3, prefix: str = "cp_") -> pd.DataFrame:
    """
    Agrega a frecuencia horaria y genera features básicas de sentimiento/noticias:
      <prefix>n_posts
      <prefix>score_simple_mean
      <prefix>score_weighted_mean (más peso si importante)
      <prefix>important_sum
      <prefix>bull_minus_bear (= votes_positive - votes_negative) promedio
    """
    cols = ["timestamp","score","votes_positive","votes_negative","is_important"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    if df.empty:
        out = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                            f"{prefix}n_posts": [], f"{prefix}score_simple_mean": [],
                            f"{prefix}score_weighted_mean": [], f"{prefix}important_sum": [],
                            f"{prefix}bull_minus_bear": []})
        return out

    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    d = d.set_index("timestamp")

    # Agregación por hora
    grp = d.groupby(pd.Grouper(freq="h"))
    agg = pd.DataFrame({
        f"{prefix}n_posts": grp["score"].count(),
        f"{prefix}score_simple_mean": grp["score"].mean(),
        f"{prefix}score_weighted_mean": (grp.apply(lambda g: (g["score"] * (1 + g["is_important"].astype(float))).mean())),
        f"{prefix}important_sum": grp["is_important"].sum(),
        f"{prefix}bull_minus_bear": grp.apply(lambda g: (g["votes_positive"] - g["votes_negative"]).mean())
    })

    # Reindex horario y ffill limitado
    if not agg.empty:
        full = pd.date_range(agg.index.min().floor("h"), agg.index.max().ceil("h"), freq="h", tz="UTC")
        agg = agg.reindex(full)
    agg = agg.ffill(limit=ffill_hours)

    # Rolling (suavizado corto)
    for c in list(agg.columns):
        agg[c] = agg[c].rolling(roll_hours, min_periods=1).mean()

    agg = agg.reset_index().rename(columns={"index": "timestamp"})
    return agg

def _main():
    _read_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--currencies", type=str, default="")
    ap.add_argument("--save-hourly", type=str)
    ap.add_argument("--ffill-hours", type=int, default=6)
    ap.add_argument("--roll-hours", type=int, default=3)
    ap.add_argument("--prefix", type=str, default="cp_")
    ap.add_argument("--public", action="store_true", help="Usar endpoint público (sin auth_token)")
    args = ap.parse_args()

    currencies = [c.strip() for c in args.currencies.split(",") if c.strip()] or None
    df = load_cryptopanic(args.cache, refresh=args.refresh, days=args.days, currencies=currencies, use_public=args.public)

    if args.save_hourly:
        hf = hourly_features(df, ffill_hours=args.ffill_hours, roll_hours=args.roll_hours, prefix=args.prefix)
        os.makedirs(os.path.dirname(args.save_hourly), exist_ok=True)
        out = hf.copy()
        if not out.empty:
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")
        out.to_csv(args.save_hourly, index=False)
        print(f"[cryptopanic] hourly rows={len(hf)} -> {args.save_hourly}")
    else:
        print(f"[cryptopanic] rows={len(df)}")

if __name__ == "__main__":
    _main()
