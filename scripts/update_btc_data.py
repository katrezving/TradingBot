import requests
import pandas as pd
import os
from dotenv import load_dotenv

def get_btc_history_coingecko():
    print("Intentando obtener datos de CoinGecko...")
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365'
    r = requests.get(url, timeout=15)
    btc_data = r.json()
    print("CoinGecko response keys:", list(btc_data.keys()))
    if 'prices' not in btc_data:
        print("Error CoinGecko: no se encontró 'prices'. Respuesta:", btc_data)
        return None, None
    prices = pd.DataFrame(btc_data['prices'], columns=['timestamp', 'btc_close'])
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')

    # CoinGecko NO ofrece histórico de dominancia, sólo el valor actual
    try:
        global_url = 'https://api.coingecko.com/api/v3/global'
        r2 = requests.get(global_url, timeout=15)
        data = r2.json()
        dominance = data['data']['market_cap_percentage']['btc']
        print(f"Dominancia BTC actual (CoinGecko): {dominance:.2f}%")
        # Columna para todos igual al actual
        prices['btc_dominance'] = dominance
    except Exception as e:
        print("Error obteniendo dominancia CoinGecko:", e)
        prices['btc_dominance'] = None

    return prices, 'COINGECKO'

def get_btc_dominance_coinmarketcap():
    print("Intentando obtener dominancia de CoinMarketCap...")
    load_dotenv()
    API_KEY = os.getenv("CMC_API_KEY")
    if not API_KEY:
        raise ValueError("No se encontró CMC_API_KEY en tu .env o variables de entorno")
    headers = {'X-CMC_PRO_API_KEY': API_KEY}
    url = 'https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest'
    r = requests.get(url, headers=headers, timeout=15)
    data = r.json()
    try:
        dominance = data['data']['btc_dominance']
    except Exception as e:
        print("Error leyendo dominancia CMC:", data)
        return None, None
    print(f"Dominancia BTC actual (CMC): {dominance:.2f}%")
    # Usamos precio histórico desde CoinGecko, pero solo dominancia actual desde CMC
    df_prices, _ = get_btc_history_coingecko()
    if df_prices is None:
        print("No fue posible obtener histórico de precios BTC.")
        return None, None
    df_prices['btc_dominance'] = dominance
    return df_prices, 'COINMARKETCAP'

def get_btc_data():
    # Intenta CoinGecko
    prices, source = get_btc_history_coingecko()
    if prices is not None:
        print("Datos obtenidos de CoinGecko.")
        return prices, source
    # Si falla, intenta CoinMarketCap
    prices, source = get_btc_dominance_coinmarketcap()
    if prices is not None:
        print("Datos obtenidos de CoinMarketCap.")
        return prices, source
    print("No fue posible obtener datos de BTC.")
    return None, None

if __name__ == "__main__":
    df_btc, source = get_btc_data()
    if df_btc is not None:
        df_btc.to_csv('btc_data.csv', index=False)
        print(f"Archivo btc_data.csv generado de {source}. Columnas: {df_btc.columns}")
    else:
        print("No se pudo crear btc_data.csv: todas las fuentes fallaron.")
