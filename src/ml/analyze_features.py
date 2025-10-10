import pandas as pd

def main(path=\"data/BNBUSDT_1h_features_h3.csv\"):
    df = pd.read_csv(path).sort_values(\"ts\")
    print(\"\n==== INFO ====\")
    print(\"shape:\", df.shape)
    print(\"cols:\", len(df.columns))

    print(\"\n==== CLASS BALANCE (y) ====\")
    print(df[\"y\"].value_counts(normalize=True).rename(\"proportion\"))

    print(\"\n==== NaNs (top 15) ====\")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    corr = df[num_cols].corr(numeric_only=True)[\"y\"].sort_values(ascending=False)
    print(\"\n==== TOP 20 corr con y ====\")
    print(corr.head(20))
    print(\"\n==== BOTTOM 20 corr con y ====\")
    print(corr.tail(20))

if __name__ == \"__main__\":
    main()
