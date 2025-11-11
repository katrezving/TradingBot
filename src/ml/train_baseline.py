import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path, target, features):
    df = pd.read_csv(path)
    required = set(features + [target])
    missing = required - set(df.columns)
    if missing:
        logger.error(f'Columnas faltantes: {missing} en {path}')
        raise ValueError('Columnas faltantes en dataset.')
    df = df.dropna(subset=features + [target])
    logger.info(f'Dataset cargado: {len(df)} filas, {len(features)} features')
    return df

def scale_features(df, features, method="none"):
    if method == "std":
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        logger.info('Features escalados: StandardScaler')
    elif method == "minmax":
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
        logger.info('Features escalados: MinMaxScaler')
    else:
        logger.info('Features sin escalado adicional')
    return df

def get_split(df, test_size, split_type="ts", random_state=42):
    X, y = df.drop(columns=["ts"]), df["y"]
    if split_type == "ts":
        tscv = TimeSeriesSplit(n_splits=int(1/test_size))
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    logger.info(f'Split realizado: train={len(X_train)}, test={len(X_test)}')
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, params):
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model

def eval_model(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_proba > threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "conf_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    logger.info(f"Métricas test: {metrics}")
    return metrics, y_pred, y_proba

def export_metrics(metrics, path):
    pd.DataFrame([metrics]).to_csv(path, index=False)
    logger.info(f"Métricas guardadas en {path}")

def save_model(model, path):
    joblib.dump(model, path)
    logger.info(f"Modelo guardado en {path}")

def export_predictions(preds, probs, out_path):
    pd.DataFrame({"y_pred": preds, "y_proba": probs}).to_csv(out_path, index=False)
    logger.info(f"Predicciones guardadas en {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento robusto LightGBM baseline.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--target", default="y")
    parser.add_argument("--model-out", default="models/lgb.pkl")
    parser.add_argument("--metrics-out", default="models/lgb_metrics.csv")
    parser.add_argument("--preds-out", default="models/lgb_preds.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-type", choices=["ts", "random"], default="ts")
    parser.add_argument("--scale", choices=["none", "std", "minmax"], default="none")
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    params = {"learning_rate": args.learning_rate, "n_estimators": args.n_estimators, "random_state": 42}
    df = load_data(args.data, args.target, args.features)
    df = scale_features(df, args.features, method=args.scale)
    X_train, X_test, y_train, y_test = get_split(df, args.test_size, split_type=args.split_type)
    model = train_model(X_train[args.features], y_train, params)
    metrics, y_pred, y_proba = eval_model(model, X_test[args.features], y_test, threshold=args.threshold)
    export_metrics(metrics, args.metrics_out)
    export_predictions(y_pred, y_proba, args.preds_out)
    save_model(model, args.model_out)
    logger.info("Entrenamiento y evaluación completados.")

if __name__ == "__main__":
    main()
