import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('trades_historial.csv')

# Mejores y peores trades
print("TOP 10 mejores trades:\n", df.sort_values("pnl", ascending=False).head(10)[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl']])
print("\nTOP 10 peores trades:\n", df.sort_values("pnl").head(10)[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl']])

# PnL por año y mes
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['year'] = df['entry_time'].dt.year
df['month'] = df['entry_time'].dt.month
print("\nGanancia promedio por año:")
print(df.groupby('year')['pnl'].mean())
print("\nGanancia promedio por mes:")
print(df.groupby(['year','month'])['pnl'].mean())

# Histograma de PnL por trade
plt.hist(df['pnl'], bins=50)
plt.title("Distribución PnL por trade")
plt.xlabel("PnL (%)")
plt.ylabel("Frecuencia")
plt.show()

# PnL acumulado
df['cum_pnl'] = df['pnl'].cumsum()
plt.plot(df['entry_time'], df['cum_pnl'])
plt.title("PnL acumulado del bot ML")
plt.xlabel("Fecha")
plt.ylabel("Cumulative PnL")
plt.show()

# Duración promedio
df['exit_time'] = pd.to_datetime(df['exit_time'])
df['duration_horas'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
print("\nDuración promedio de los trades (horas):", df['duration_horas'].mean())