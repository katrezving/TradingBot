import pandas as pd

INFILE = 'src/integrations/Bitcoin_Altcoin__dominance.csv'
OUTFILE = 'dominancia_features.csv'

print('Leyendo', INFILE)
df = pd.read_csv(INFILE)
print('Columnas detectadas en el archivo:', list(df.columns))

df['Date'] = pd.to_datetime(df['Date'])

def guess_column(candidates, columns):
    for cand in candidates:
        for col in columns:
            if cand.lower() in col.lower():
                return col
    return None

btc_col = guess_column(['bitcoin dominance', 'bitcoin'], df.columns)
eth_col = guess_column(['ethereum dominance', 'ethereum'], df.columns)

if not btc_col or not eth_col:
    raise ValueError(f"No se pudieron encontrar las columnas: btc({btc_col}) eth({eth_col}) en tu archivo. Revisa los nombres luego de print(df.columns)")

col_map = {
    btc_col: 'btc_dominance',
    eth_col: 'eth_dominance'
}
df = df.rename(columns=col_map)

cols_final = ['Date', 'btc_dominance', 'eth_dominance']
df = df[cols_final].copy()

for c in ['btc_dominance', 'eth_dominance']:
    df[c + '_ret'] = df[c].pct_change()

print('Guardando features macro en', OUTFILE)
df.to_csv(OUTFILE, index=False)
print('Archivo listo:', OUTFILE)
