import argparse
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_equity(path, equity_col="equity", date_col="ts", out_img="equity_plot.png", title="Equity Curve"):
    try:
        df = pd.read_csv(path)
        if equity_col not in df.columns or date_col not in df.columns:
            logger.error(f"Columnas faltantes en {path}: requerido {equity_col}, {date_col}")
            return
        plt.figure(figsize=(12,5))
        plt.plot(pd.to_datetime(df[date_col], utc=True), df[equity_col])
        plt.title(title)
        plt.xlabel("Fecha")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_img)
        plt.close()
        logger.info(f"Equity plot guardado en {out_img}")
    except Exception as e:
        logger.error(f"Error en plot_equity: {e}")

def main():
    parser = argparse.ArgumentParser(description="Grafica curva de equity o PnL de backtest")
    parser.add_argument("--path", required=True)
    parser.add_argument("--equity-col", default="equity")
    parser.add_argument("--date-col", default="ts")
    parser.add_argument("--out-img", default="equity_plot.png")
    parser.add_argument("--title", default="Equity Curve")
    args = parser.parse_args()
    plot_equity(args.path, args.equity_col, args.date_col, args.out_img, args.title)

if __name__ == "__main__":
    main()
