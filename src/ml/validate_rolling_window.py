import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report
import joblib

# Carga y procesamiento
df = pd.read_csv("BNBUSDT_1h_features_full.csv", parse_dates=['timestamp'])
df.sort_values('timestamp', inplace=True)
target = 'signal'
features = [col for col in df.columns if col != target and col != 'timestamp']

X = df[features].fillna(df[features].median())
y = df[target]

n_splits = 5  # Puedes variar este valor
tscv = TimeSeriesSplit(n_splits=n_splits)
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Imputador simple + Modelo ML
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_val_imp = imputer.transform(X_val)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_tr_imp, y_tr)
    preds = model.predict(X_val_imp)
    score = f1_score(y_val, preds)
    f1_scores.append(score)

    print(f"\nFold {fold+1}:")
    print(classification_report(y_val, preds))

print(f"\nF1 Rolling Window Average: {np.mean(f1_scores):.4f}")

# Exporta modelo final entrenado y imputador
model.fit(imputer.fit_transform(X), y)
joblib.dump(model, "models/FINAL_MODEL.joblib")
joblib.dump(imputer, "models/FINAL_IMPUTER.joblib")