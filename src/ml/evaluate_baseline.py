import argparse
import joblib
import logging
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path, features, target):
    try:
        df = pd.read_csv(path)
        for col in features + [target]:
            if col not in df.columns:
                logger.error(f"Columna {col} faltante en {path}")
                raise ValueError
        return df
    except Exception as e:
        logger.error(f"Error cargando datos: {e}")
        raise

def load_model(path):
    try:
        model = joblib.load(path)
        logger.info(f"Modelo cargado: {path}")
        return model
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise

def evaluate(model, df, features, target, threshold=0.5):
    y_true = df[target]
    y_proba = model.predict_proba(df[features])[:,1]
    y_pred = (y_proba > threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "conf_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    logger.info(f"Métricas en test: {metrics}")
    return metrics, y_pred, y_proba

def export_results(metrics, preds, probas, out_metrics, out_preds):
    try:
        # GUARDA MÉTRICAS COMO JSON
        with open(out_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        # Predicciones como CSV (ok)
        pd.DataFrame({"y_pred": preds, "y_proba": probas}).to_csv(out_preds, index=False)
        logger.info(f"Métricas guardadas en {out_metrics}, predicciones en {out_preds}")
    except Exception as e:
        logger.error(f"Error guardando resultados: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evalúa modelo trenado en dataset nuevo")
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--target", default="y")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-metrics", default="models/eval_metrics.csv")
    parser.add_argument("--out-preds", default="models/eval_preds.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = load_data(args.data, args.features, args.target)
    model = load_model(args.model)
    metrics, preds, probas = evaluate(model, df, args.features, args.target, threshold=args.threshold)
    export_results(metrics, preds, probas, args.out_metrics, args.out_preds)
    logger.info("Evaluación de modelo completada.")

if __name__ == "__main__":
    main()
