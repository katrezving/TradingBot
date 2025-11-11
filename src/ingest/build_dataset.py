import argparse
import pandas as pd
from pathlib import Path
import logging
import os

# ==========================================================
# Logging configurado
# ==========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Utilidades
# ==============================
def _merge_asof_with_tolerance(base_df, add_df, on="timestamp", tolerance_hours=6, prefix=None):
    if add_df is None or add_df.empty:
        logger.warning("El DataFrame adicional está vacío o es None; se omite el merge_asof.")
        return base_df
    try:
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
        logger.info(f"Merge_asof completado; cols añadidas: {list(add_df.columns)}")
        return merged
    except Exception as e:
        logger.error(f"Error en merge_asof: {e}")
        return base_df

def _add_lags(df, col, lags=[1, 6]):
    try:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        logger.info(f"Lags añadidos para columna {col}: {lags}")
    except Exception as e:
        logger.error(f"Error añadiendo lags: {e}")
    return df

def load_fear_greed(path, ffill_hours=72):
    try:
        fg = pd.read_csv(path)
        fg["timestamp"] = pd.to_datetime(fg["timestamp"], utc=True)
        fg = fg.sort_values("timestamp")
        fg["fg_value"] = fg["fg_value"].astype(float)
        fg["fg_z"] = (fg["fg_value"] - fg["fg_value"].mean()) / fg["fg_value"].std()
        fg = _add_lags(fg, "fg_value")
        fg = _add_lags(fg, "fg_z")
        logger.info("Fear&Greed cargado y lags realizados.")
        return fg
    except Exception as e:
        logger.error(f"Error cargando fear_greed: {e}")
        return pd.DataFrame()  # vacío para no romper pipeline

def load_hourly_csv(path):
    try:
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            msg = f"El archivo {path} no contiene columna 'timestamp'"
            logger.error(msg)
            raise ValueError(msg)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        logger.info(f"Archivo horario cargado OK: {path}, filas={len(df)}")
        return df.sort_values("timestamp")
    except Exception as e:
        logger.error(f"Error cargando CSV: {path} — {e}")
        return pd.DataFrame()

def safe_save(df, path, backup=True):
    out_path = Path(path)
    try:
        if backup and out_path.exists():
            backup_path = out_path.with_suffix(out_path.suffix + ".bak")
            if backup_path.exists():
                backup_path.unlink()
            out_path.rename(backup_path)
            logger.info(f"Backup realizado: {out_path.name} -> {backup_path.name}")
        df.to_csv(out_path, index=False)
        logger.info(f"Archivo guardado en {out_path}")
    except Exception as e:
        logger.error(f"Error guardando archivo: {e}")

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
    logger.info(f"Cargando base: {args.base}")
    try:
        df = pd.read_csv(args.base)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        logger.info(f"Base filas={len(df)} cols={len(df.columns)}")
    except Exception as e:
        logger.error(f"Error cargando base: {e}")
        return

    # ==============================
    # Fear & Greed
    # ==============================
    if args.fg_source == "integrations":
        fg_df = load_fear_greed(args.fg_cache, args.fg_ffill_hours)
        df = _merge_asof_with_tolerance(df, fg_df, tolerance_hours=args.fg_ffill_hours)
        fg_cols = [c for c in df.columns if c.startswith("fg_")]
        logger.info(f"FG cols añadidas: {fg_cols}")
        logger.info(f"FG NA ratio: { {c: round(float(df[c].isna().mean()), 4) for c in fg_cols} } ")
    else:
        logger.info("Fear&Greed: desactivado (--fg-source=none).")

    # ==============================
    # CryptoPanic
    # ==============================
    if args.cp_source == "csv":
        cp_df = load_hourly_csv(args.cp_hourly)
        df = _merge_asof_with_tolerance(df, cp_df, tolerance_hours=args.cp_ffill_hours, prefix=args.cp_prefix)
        cp_cols = [c for c in df.columns if c.startswith(args.cp_prefix)]
        logger.info(f"CP cols añadidas: {cp_cols}")
        logger.info(f"CP NA ratio: { {c: round(float(df[c].isna().mean()), 4) for c in cp_cols} } ")
    else:
        logger.info("CryptoPanic: desactivado.")

    # ==============================
    # CoinGlass
    # ==============================
    if args.cg_source == "csv":
        cg_df = load_hourly_csv(args.cg_hourly)
        df = _merge_asof_with_tolerance(df, cg_df, tolerance_hours=args.cg_ffill_hours, prefix=args.cg_prefix)
        cg_cols = [c for c in df.columns if c.startswith(args.cg_prefix)]
        logger.info(f"CG cols añadidas: {cg_cols}")
    else:
        logger.info("CoinGlass: desactivado.")

    # ==============================
    # CoinMarketCap
    # ==============================
    if args.cmc_source == "csv":
        cmc_df = load_hourly_csv(args.cmc_hourly)
        df = _merge_asof_with_tolerance(df, cmc_df, tolerance_hours=args.cmc_ffill_hours, prefix=args.cmc_prefix)
        cmc_cols = [c for c in df.columns if c.startswith(args.cmc_prefix)]
        logger.info(f"CMC cols añadidas: {cmc_cols}")
    else:
        logger.info("CoinMarketCap: desactivado.")

    # ==============================
    # Guardado
    # ==============================
    safe_save(df, args.out, backup=args.write_mode == "backup")
    logger.info("Proceso build_dataset finalizado.")

if __name__ == "__main__":
    main()
