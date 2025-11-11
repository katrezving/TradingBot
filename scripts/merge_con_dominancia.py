import pandas as pd

# Ajusta el nombre/ruta aquí
ALTCOIN_FILE = 'altcoin_data.csv' # <-- tu dataset: debe tener columna timestamp (o Date, ver más abajo)
DOMINANCE_FILE = 'dominancia_features.csv' # Ya está en la raíz
OUTFILE = 'altcoin_with_dominance_features.csv'

# Lee ambos archivos
df_main = pd.read_csv(ALTCOIN_FILE)
dom = pd.read_csv(DOMINANCE_FILE)

dom['Date'] = pd.to_datetime(dom['Date'])

# Ajusta aquí según cómo se llame tu columna de tiempo:
if 'timestamp' in df_main.columns:
    df_main['Date'] = pd.to_datetime(df_main['timestamp']).dt.date
    dom['Date'] = dom['Date'].dt.date
elif 'Date' in df_main.columns:
    df_main['Date'] = pd.to_datetime(df_main['Date']).dt.date
    dom['Date'] = dom['Date'].dt.date
else:
    raise ValueError('Tu dataset principal debe tener una columna llamada timestamp o Date')

merged = pd.merge(df_main, dom, on='Date', how='left')

merged.to_csv(OUTFILE, index=False)
print(f"Archivo enriquecido generado como {OUTFILE}")
