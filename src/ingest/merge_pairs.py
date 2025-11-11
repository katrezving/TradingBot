import argparse
import pandas as pd
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_feature_file(filepath, required_cols=["ts", "symbol", "close"]):
    try:
        df = pd.read_csv(filepath)
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Archivo {filepath} sin columnas requeridas: {missing}")
            return None
        logger.info(f"Archivo {filepath} cargado correctamente ({len(df)} filas).")
        return df
    except Exception as e:
        logger.error(f"Error leyendo archivo {filepath}: {e}")
        return None

def merge_feature_files(files):
    dfs = []
    for file in files:
        df = load_feature_file(file)
        if df is not None:
            dfs.append(df)
    if not dfs:
        logger.error("No se cargó ningún archivo; abortando merge.")
        return pd.DataFrame()
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["ts", "symbol"])
    merged = merged.sort_values(["symbol", "ts"]).reset_index(drop=True)
    logger.info(f"Merge finalizado: {len(merged)} filas, {len(merged.columns)} columnas.")
    return merged

def main():
    parser = argparse.ArgumentParser(description="Merge de features para distintos pares.")
    parser.add_argument("--in-folder", default="data/", help="Carpeta con archivos *_features*.csv")
    parser.add_argument("--out", default="data/all_pairs_features.csv", help="Ruta de salida para el merge final.")
    parser.add_argument("--pattern", default="_features", help="Patrón para archivos de features.")
    args = parser.parse_args()

    path = Path(args.in_folder)
    files = [str(f) for f in path.glob(f"*{args.pattern}*.csv") if f.is_file()]
    logger.info(f"Archivos a unir: {files}")

    merged = merge_feature_files(files)
    if not merged.empty:
        try:
            merged.to_csv(args.out, index=False)
            logger.info(f"Archivo guardado exitosamente en {args.out}")
        except Exception as e:
            logger.error(f"Error guardando archivo final: {e}")

if __name__ == "__main__":
    main()
