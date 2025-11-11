import pandas as pd
import matplotlib.pyplot as plt

# Carga el archivo con las columnas: Date, Bitcoin, altcoins_dominance
# Si solo tienes la dominancia de cada coin, calcula altcoins_dominance como suma de Ethereum, BCH, LTC, etc.
df = pd.read_csv('btc_vs_altcoins_dominance.csv')
df['Date'] = pd.to_datetime(df['Date'])

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Bitcoin'], label='Dominancia Bitcoin', color='orange')
plt.plot(df['Date'], df['altcoins_dominance'], label='Dominancia Altcoins', color='purple')
plt.title('Dominancia Bitcoin vs Altcoins: Ciclo Histórico')
plt.xlabel('Fecha')
plt.ylabel('Dominancia (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
