"""
build_dataset.py
================
Une el dataset base (por hora o minuto) con integraciones externas:
- Fear & Greed (diario)
- CryptoPanic (horario)
- CoinGlass / CoinMarketCap (variable)

Usa merge_asof con tolerancia para respetar diferencias temporales (e.g. FG diario vs velas horarias).
"""

import argparse
import pandas as pd
from pathlib import Path


# ==============================
# Utilidades
# ==============================
def _merge_asof_with_tolerance(base_df, add_df, on="timestamp", tolerance_hours=6, prefix=None):
    """Realiza un merge_asof tolerante al tiempo para combinar integraciones de distinta frecuencia."""
    if add_df is None or add_df.empty:
        return base_df

    add_df = add_df.sort_values(on)
    base_df = base_df.sort_values(on)
    merged = pd.merge_asof(
        base_df,
        add_df,
        on=on,
        direction="backward",
        tolerance=pd.Timedelta(hours=tolerance_hours)
    )

    if prefix:
        merged = merged.rename(columns={c: f"{prefix}{c}" for c in add_df.columns if c != on})
    return merged


def _add_lags(df, col, lags=[1, 6]):
    """Crea columnas de rezago (lag) para features de baja frecuencia como FG."""
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


# ==============================
# Carga de integraciones
# ==============================
def load_fear_greed(path, ffill_hours=72):
    fg = pd.read_csv(path)
    fg["timestamp"] = pd.to_datetime(fg["timestamp"], utc=True)
    fg = fg.sort_values("timestamp")

    # Valor numérico principal
    fg["fg_value"] = fg["fg_value"].astype(float)
    fg["fg_z"] = (fg["fg_value"] - fg["fg_value"].mean()) / fg["fg_value"].std()
    fg = _add_lags(fg, "fg_value")
    fg = _add_lags(fg, "fg_z")
    return fg


def load_hourly_csv(path):
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"El archivo {path} no contiene columna 'timestamp'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")


# ==============================
# MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Construye dataset completo uniendo integraciones externas.")
    parser.add_argument("--base", required=True, help="Ruta al CSV base con velas y contexto BTC.")
    parser.add_argument("--out", required=True, help="Ruta de salida final.")
    parser.add_argument("--write-mode", default="backup", choices=["overwrite", "backup"], help="Modo de guardado.")

    # Fear & Greed
    parser.add_argument("--fg-source", choices=["none", "integrations"], default="none")
    parser.add_argument("--fg-cache", default="data/integrations/fear_greed.csv")
    parser.add_argument("--fg-ffill-hours", type=int, default=72)

    # CryptoPanic
    parser.add_argument("--cp-source", choices=["none", "csv"], default="none")
    parser.add_argument("--cp-hourly", default="data/integrations/cryptopanic_hourly.csv")
    parser.add_argument("--cp-ffill-hours", type=int, default=6)
    parser.add_argument("--cp-prefix", default="cp_")

    # CoinGlass
    parser.add_argument("--cg-source", choices=["none", "csv"], default="none")
    parser.add_argument("--cg-hourly", default="data/integrations/coinglass_hourly.csv")
    parser.add_argument("--cg-ffill-hours", type=int, default=12)
    parser.add_argument("--cg-prefix", default="cg_")

    # CoinMarketCap
    parser.add_argument("--cmc-source", choices=["none", "csv"], default="none")
    parser.add_argument("--cmc-hourly", default="data/integrations/cmc_hourly.csv")
    parser.add_argument("--cmc-ffill-hours", type=int, default=12)
    parser.add_argument("--cmc-prefix", default="cmc_")

    args = parser.parse_args()

    # ==============================
    # Cargar base
    # ==============================
    print(f"[build_dataset] Cargando base: {args.base}")
    df = pd.read_csv(args.base)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"[build_dataset] Base filas={len(df)} cols={len(df.columns)}")

    # ==============================
    # Fear & Greed
    # ==============================
    if args.fg_source == "integrations":
        try:
            fg_df = load_fear_greed(args.fg_cache, args.fg_ffill_hours)
            print(f"[fear_greed] Registros: {len(fg_df)} rango: {fg_df['timestamp'].min()} -> {fg_df['timestamp'].max()}")
            df = _merge_asof_with_tolerance(df, fg_df, tolerance_hours=args.fg_ffill_hours)
            print("[build_dataset] FG cols añadidas:",
                  [c for c in df.columns if c.startswith("fg_")])
            print("[build_dataset] FG NA ratio:",
                  {c: round(float(df[c].isna().mean()), 4) for c in df.columns if c.startswith("fg_")})
        except Exception as e:
            print(f"[build_dataset] Error cargando Fear&Greed: {e}")
    else:
        print("[build_dataset] Fear&Greed: desactivado (--fg-source=none).")

    # ==============================
    # CryptoPanic
    # ==============================
    if args.cp_source == "csv":
        try:
            cp_df = load_hourly_csv(args.cp_hourly)
            df = _merge_asof_with_tolerance(df, cp_df, tolerance_hours=args.cp_ffill_hours, prefix=args.cp_prefix)
            print(f"[build_dataset] CP cols añadidas: {[c for c in df.columns if c.startswith(args.cp_prefix)]}")
            print(f"[build_dataset] CP NA ratio: {{c: round(float(df[c].isna().mean()),4) for c in df.columns if c.startswith(args.cp_prefix)}}")
        except Exception as e:
            print(f"[build_dataset] Error cargando CryptoPanic: {e}")
    else:
        print("[build_dataset] CryptoPanic: desactivado.")

    # ==============================
    # CoinGlass
    # ==============================
    if args.cg_source == "csv":
        try:
            cg_df = load_hourly_csv(args.cg_hourly)
            df = _merge_asof_with_tolerance(df, cg_df, tolerance_hours=args.cg_ffill_hours, prefix=args.cg_prefix)
            print(f"[build_dataset] CG cols añadidas: {[c for c in df.columns if c.startswith(args.cg_prefix)]}")
        except Exception as e:
            print(f"[build_dataset] Error cargando CoinGlass: {e}")
    else:
        print("[build_dataset] CoinGlass: desactivado.")

    # ==============================
    # CoinMarketCap
    # ==============================
    if args.cmc_source == "csv":
        try:
            cmc_df = load_hourly_csv(args.cmc_hourly)
            df = _merge_asof_with_tolerance(df, cmc_df, tolerance_hours=args.cmc_ffill_hours, prefix=args.cmc_prefix)
            print(f"[build_dataset] CMC cols añadidas: {[c for c in df.columns if c.startswith(args.cmc_prefix)]}")
        except Exception as e:
            print(f"[build_dataset] Error cargando CoinMarketCap: {e}")
    else:
        print("[build_dataset] CoinMarketCap: desactivado.")

    # ==============================
    # Guardado
    # ==============================
    out_path = Path(args.out)
    if args.write_mode == "backup" and out_path.exists():
        backup_path = out_path.with_suffix(out_path.suffix + ".bak")
        if backup_path.exists():
            backup_path.unlink()  # elimina el anterior
            out_path.rename(backup_path)
        print(f"[build_dataset] Respaldando {out_path.name} -> {backup_path.name}")

    df.to_csv(out_path, index=False)
    print(f"[build_dataset] Guardando: {out_path}")
    print("[build_dataset] Done.")


if __name__ == "__main__":
    main()