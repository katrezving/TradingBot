import argparse
import logging
import pandas as pd
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_features(path, output_prefix="stats_", save_plots=False):
    try:
        df = pd.read_csv(path)
        logger.info(f"{path}: cargado con {len(df)} filas, {len(df.columns)} columnas")
        
        # Balance de clases si columna 'y' existe
        if "y" in df.columns:
            counts = df["y"].value_counts(dropna=False)
            logger.info(f"Balance de clases y = {counts.to_dict()}")
        
        # NaNs por columna
        nan_ratios = df.isnull().mean()
        logger.info(f"NaNs ratio por columna: {nan_ratios.to_dict()}")
        
        # Stats descriptivas
        stats = df.describe(include="all").T
        stats.to_csv(f"{output_prefix}describe.csv")
        logger.info(f"Summary stats exportados en {output_prefix}describe.csv")
        
        # Correlación features numéricos
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr = df[numeric_cols].corr()
        corr.to_csv(f"{output_prefix}corr.csv")
        logger.info(f"Matriz de correlación exportada en {output_prefix}corr.csv")
        
        # Plots automáticos si activado
        if save_plots:
            try:
                import matplotlib.pyplot as plt
                for col in numeric_cols:
                    plt.figure()
                    df[col].hist(bins=40)
                    plt.title(col)
                    plt.savefig(f"{output_prefix}{col}_hist.png")
                    plt.close()
                logger.info(f"Histogramas guardados para cada feature numérico")
            except Exception as e:
                logger.warning(f"Error en generación de histogramas: {e}")
    except Exception as e:
        logger.error(f"Error analizando {path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Análisis descriptivo y de correlación de features")
    parser.add_argument("--paths", nargs="+", required=True, help="Lista de paths CSV de features")
    parser.add_argument("--output-prefix", default="stats_")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    for path in args.paths:
        base = os.path.splitext(os.path.basename(path))[0]
        prefix = f"{args.output_prefix}{base}_"
        analyze_features(path, output_prefix=prefix, save_plots=args.save_plots)

if __name__ == "__main__":
    main()
