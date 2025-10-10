import argparse, os
from .fetch_ohlcv import fetch_ohlcv_full, save_csv
from ..features.indicators import compute_indicators, make_labeled_dataset, select_feature_cols

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def build_one(symbol: str, timeframe: str, since: str, out_dir: str, horizon: int):
    raw = fetch_ohlcv_full(symbol, timeframe, since_iso=since)
    ensure_dir(out_dir)
    sym_clean = symbol.replace("/", "")
    raw_path = os.path.join(out_dir, f"{sym_clean}_{timeframe}.csv")
    save_csv(raw, raw_path)

    feats = compute_indicators(raw)
    labeled = make_labeled_dataset(feats, horizon=horizon)
    feat_cols = ["ts","open","high","low","close","volume"] + select_feature_cols(labeled) + ["fwd_ret","y"]
    labeled = labeled[feat_cols]

    feat_path = os.path.join(out_dir, f"{sym_clean}_{timeframe}_features_h{horizon}.csv")
    save_csv(labeled, feat_path)
    return raw_path, feat_path, len(labeled)

def main():
    ap = argparse.ArgumentParser(description="Fetch OHLCV and build feature dataset(s)")
    ap.add_argument("--symbols", type=str, default="BNB/USDT", help="Comma-separated symbols, e.g., BNB/USDT,BTC/USDT")
    ap.add_argument("--timeframes", type=str, default="1h", help="Comma-separated tfs, e.g., 15m,1h,4h")
    ap.add_argument("--since", type=str, default="2019-01-01T00:00:00Z", help="ISO start time in UTC")
    ap.add_argument("--out", type=str, default="data", help="Output directory")
    ap.add_argument("--horizon", type=int, default=1, help="Label horizon in bars (default 1)")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tfs = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    for s in symbols:
        for tf in tfs:
            raw_path, feat_path, n = build_one(s, tf, args.since, args.out, args.horizon)
            print(f"OK: {s} {tf} -> {n} rows\n  raw:  {raw_path}\n  feats: {feat_path}")

if __name__ == "__main__":
    main()
