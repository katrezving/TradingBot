import pandas as pd
from glob import glob
import os

def merge_pairs(input_dir="data", output_file="data/ALLPAIRS_1h_features_h6.csv"):
    """
    Une todos los datasets *_1h_features_h6.csv en un único archivo.
    Añade una columna 'symbol' para identificar el par.
    """

    files = glob(os.path.join(input_dir, "*_1h_features_h6.csv"))
    if not files:
        print("❌ No se encontraron archivos *_1h_features_h6.csv en la carpeta data/")
        return

    dfs = []
    for f in files:
        sym = os.path.basename(f).split("_")[0]  # ejemplo: BTCUSDT_1h_features_h6.csv -> BTCUSDT
        d = pd.read_csv(f)
        d["symbol"] = sym
        dfs.append(d)
        print(f"✅ Añadido: {sym} ({len(d)} filas)")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["ts", "symbol"]).reset_index(drop=True)
    df_all = df_all.sort_values(["symbol", "ts"])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_all.to_csv(output_file, index=False)
    print(f"\n📦 Dataset unificado guardado en: {output_file}")
    print(f"Total de filas combinadas: {len(df_all)}")
    print(f"Símbolos incluidos: {df_all['symbol'].unique().tolist()}")

if __name__ == "__main__":
    merge_pairs()
