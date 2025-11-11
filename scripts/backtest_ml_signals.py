import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carga el csv de predicciones generado por tu pipeline ML (ajusta la ruta)
df_bt = pd.read_csv('evals/eval_20251015T163112Z/predictions.csv', parse_dates=['timestamp'])

# Señal ML: solo operar cuando pred == 1
df_bt['signal'] = df_bt['pred']
# Retorno de la siguiente barra (shift -1)
df_bt['future_return'] = df_bt['close'].pct_change().shift(-1)

# Rendimiento del bot: solo gana el retorno si la señal era 1
df_bt['bot_ret'] = df_bt['signal'] * df_bt['future_return']
# Acumulado
initial_cash = 1.0
df_bt['equity_curve'] = (1 + df_bt['bot_ret'].fillna(0)).cumprod() * initial_cash

# Baseline buy & hold
bh_curve = (1 + df_bt['future_return'].fillna(0)).cumprod() * initial_cash

# Métricas básicas
max_drawdown = (df_bt['equity_curve'].cummax() - df_bt['equity_curve']).max() / df_bt['equity_curve'].cummax().max()
bot_final = df_bt['equity_curve'].iloc[-1]
bh_final = bh_curve.iloc[-1]
sharpe_bot = (df_bt['bot_ret'].mean() / df_bt['bot_ret'].std()) * np.sqrt(24*365)  # anualizado para barras 1h
num_trades = df_bt['signal'].sum()

print(f"\nResultados Backtest:")
print(f"Equity final ML: {bot_final:.4f}   Buy&Hold: {bh_final:.4f}")
print(f"Max drawdown: {max_drawdown:.3%}   Sharpe ML: {sharpe_bot:.2f}   Trades: {num_trades}")

# Plot
plt.figure(figsize=(12,6))
plt.plot(df_bt['timestamp'], df_bt['equity_curve'], label='Bot ML')
plt.plot(df_bt['timestamp'], bh_curve, label='Buy & Hold')
plt.legend()
plt.title('Curva de equity vs Buy & Hold')
plt.grid(True)
plt.tight_layout()
plt.show()