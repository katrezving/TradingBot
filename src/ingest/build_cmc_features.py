import pandas as pd
import numpy as np

# Carga el archivo
df = pd.read_csv('src/integrations/coinmarketcap_historical.csv')

# Renombra 'date' a 'ts'
df.rename(columns={'date': 'ts'}, inplace=True)

# FILTRA filas para los mercados de interés
df_btc = df[df['coin_id'] == 'bitcoin'][['ts', 'market_cap', 'volume_24h']]
df_eth = df[df['coin_id'] == 'ethereum'][['ts', 'market_cap', 'volume_24h']]

# Market cap y volumen global (por fecha, sumar TODO)
df_total = df.groupby('ts', as_index=False)[['market_cap', 'volume_24h']].sum()
df_total.rename(columns={
    'market_cap': 'cmc_total_mcap_usd',
    'volume_24h': 'cmc_total_vol_24h_usd'
}, inplace=True)

# Merge con BTC y ETH
df_btc.rename(columns={'market_cap': 'cmc_btc_mcap_usd', 'volume_24h': 'cmc_btc_vol_24h_usd'}, inplace=True)
df_eth.rename(columns={'market_cap': 'cmc_eth_mcap_usd', 'volume_24h': 'cmc_eth_vol_24h_usd'}, inplace=True)

df_merged = df_total.merge(df_btc, on='ts', how='left')
df_merged = df_merged.merge(df_eth, on='ts', how='left')

# Dominancias
df_merged['cmc_btc_dominance'] = df_merged['cmc_btc_mcap_usd'] / df_merged['cmc_total_mcap_usd']
df_merged['cmc_eth_dominance'] = df_merged['cmc_eth_mcap_usd'] / df_merged['cmc_total_mcap_usd']

# Retornos históricos y recientes (log returns para estabilidad numérica)
for col in ['cmc_total_mcap_usd', 'cmc_btc_mcap_usd', 'cmc_eth_mcap_usd']:
    df_merged[f'{col}_ret1'] = np.log(df_merged[col] / df_merged[col].shift(1))
    df_merged[f'{col}_ret7'] = np.log(df_merged[col] / df_merged[col].shift(7))
    df_merged[f'{col}_ret30'] = np.log(df_merged[col] / df_merged[col].shift(30))

# Momento relativo: rolling mean y z-score
df_merged['mc_z_30'] = (df_merged['cmc_total_mcap_usd'] - df_merged['cmc_total_mcap_usd'].rolling(30).mean()) / (df_merged['cmc_total_mcap_usd'].rolling(30).std() + 1e-9)
df_merged['btc_dom_z_30'] = (df_merged['cmc_btc_dominance'] - df_merged['cmc_btc_dominance'].rolling(30).mean()) / (df_merged['cmc_btc_dominance'].rolling(30).std() + 1e-9)

# Combinaciones: marketcap/volumen global y para BTC/ETH
df_merged['mcap_vol_ratio_total'] = df_merged['cmc_total_mcap_usd'] / (df_merged['cmc_total_vol_24h_usd'] + 1e-9)
df_merged['mcap_vol_ratio_btc'] = df_merged['cmc_btc_mcap_usd'] / (df_merged['cmc_btc_vol_24h_usd'] + 1e-9)
df_merged['mcap_vol_ratio_eth'] = df_merged['cmc_eth_mcap_usd'] / (df_merged['cmc_eth_vol_24h_usd'] + 1e-9)

# Guarda solo lo útil (ajusta si quieres agregar otras columnas)
df_merged[['ts',
           'cmc_total_mcap_usd', 'cmc_total_vol_24h_usd',
           'cmc_btc_mcap_usd','cmc_eth_mcap_usd',
           'cmc_btc_dominance','cmc_eth_dominance',
           'cmc_total_mcap_usd_ret1','cmc_total_mcap_usd_ret7','cmc_total_mcap_usd_ret30',
           'cmc_btc_mcap_usd_ret1','cmc_btc_mcap_usd_ret7','cmc_btc_mcap_usd_ret30',
           'cmc_eth_mcap_usd_ret1','cmc_eth_mcap_usd_ret7','cmc_eth_mcap_usd_ret30',
           'mc_z_30','btc_dom_z_30',
           'mcap_vol_ratio_total','mcap_vol_ratio_btc','mcap_vol_ratio_eth'
           ]].to_csv('src/integrations/coinmarketcap_historico_prep.csv', index=False)

print("Archivo listo: src/integrations/coinmarketcap_historico_prep.csv")