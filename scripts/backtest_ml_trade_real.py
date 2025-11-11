import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def run_backtest(args):
    # Cargar archivo con fechas parseadas
    df = pd.read_csv('evals/eval_20251015T163112Z/predictions.csv', parse_dates=['timestamp'])
    if args.symbol is not None and 'symbol' in df.columns:
        df = df[df['symbol'] == args.symbol]
    # Filtrado por ventana temporal si deseas
    if args.start is not None:
        df = df[df['timestamp'] >= args.start]
    if args.end is not None:
        df = df[df['timestamp'] <= args.end]

    # Filtrado por probabilidad mínima para señal ML
    if args.prob_threshold is not None and 'prob_1' in df.columns:
        df['signal'] = ((df['pred'] == 1) & (df['prob_1'] > args.prob_threshold)).astype(int)
    else:
        df['signal'] = df['pred']

    COMMISSION = args.commission
    INITIAL_CASH = args.initial_cash

    cash = INITIAL_CASH
    position = 0
    entry_price = 0
    trades = []
    equity_curve = []

    for i in range(len(df)-1):
        signal = df.iloc[i]['signal']
        price = df.iloc[i]['close']
        next_price = df.iloc[i+1]['close']
        time = df.iloc[i]['timestamp']

        # Entrada solo cuando pasa de 0 a 1
        if position == 0 and signal == 1:
            position = 1
            entry_price = price * (1 + COMMISSION)
            trade = {'type': 'buy', 'entry_time': time, 'entry_price': entry_price}
        # Salida por señal, stop loss, take profit
        elif position == 1 and (
                signal == 0 or
                ((next_price-entry_price)/entry_price < -args.stop_loss) or
                ((next_price-entry_price)/entry_price > args.take_profit)):
            position = 0
            exit_price = price * (1 - COMMISSION)
            profit = (exit_price - entry_price) / entry_price
            cash *= (1 + profit)
            trade['exit_time'] = time
            trade['exit_price'] = exit_price
            trade['pnl'] = profit
            trades.append(trade)
        equity_curve.append(cash if position == 0 else cash * (next_price / entry_price))

    # Liquidar posición final si quedaste dentro
    df_last = df.iloc[-1]
    if position == 1:
        exit_price = df_last['close'] * (1 - COMMISSION)
        profit = (exit_price - entry_price) / entry_price
        cash *= (1 + profit)
        trade['exit_time'] = df_last['timestamp']
        trade['exit_price'] = exit_price
        trade['pnl'] = profit
        trades.append(trade)
        equity_curve[-1] = cash

    # Exporta historial trades a CSV
    pd.DataFrame(trades).to_csv("trades_historial.csv", index=False)
    print("Historial de trades exportado a trades_historial.csv")

    equity_curve = np.array(equity_curve)
    max_drawdown = (np.maximum.accumulate(equity_curve) - equity_curve).max() / np.maximum.accumulate(equity_curve).max()
    final_equity = equity_curve[-1]
    num_trades = len(trades)
    pnls = [t['pnl'] for t in trades if 'pnl' in t]
    sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(len(pnls)) if len(pnls) > 1 else 0

    print(f"\nBacktest trading ajustable:")
    print(f"Equity final: {final_equity:.4f}")
    print(f"Drawdown máximo: {max_drawdown:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Trades ejecutados: {num_trades}")
    print(f"PNL promedio por trade: {np.mean(pnls) if pnls else 0:.5f}")

    # Graficar Bot ML y Buy & Hold juntos
    bh_curve = (df['close'] / df['close'].iloc[0]) * INITIAL_CASH

    plt.figure(figsize=(12,6))
    plt.plot(df['timestamp'][:-1], equity_curve, label='Bot ML', linewidth=2)
    plt.plot(df['timestamp'], bh_curve, label='Buy & Hold', linestyle='dashed', alpha=0.7)
    plt.title('Comparativa equity: Bot ML vs. Buy & Hold')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if args.save_plot:
        plt.savefig(args.save_plot)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default=None)
    parser.add_argument('--commission', type=float, default=0.001)
    parser.add_argument('--initial-cash', type=float, default=1.0)
    parser.add_argument('--prob-threshold', type=float, default=None)
    parser.add_argument('--stop-loss', type=float, default=0.03, help="Stop-loss en %, ej: 0.03 para 3%")
    parser.add_argument('--take-profit', type=float, default=0.05, help="Take-profit en %")
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--save-plot', type=str, default=None)
    args = parser.parse_args()
    # Usa ruta fija para tu archivo:
    args.pred_file = 'evals/eval_20251015T163112Z/predictions.csv'
    run_backtest(args)
