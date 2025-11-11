import argparse
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(path):
    try:
        model = joblib.load(path)
        logger.info(f"Modelo cargado correctamente: {path}")
        return model
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise

def load_features(path, features):
    try:
        df = pd.read_csv(path)
        if not all(col in df.columns for col in features):
            logger.error(f"Features faltantes en {path}")
            raise ValueError("Features faltantes")
        logger.info(f"Archivo {path} cargado; filas: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error cargando archivo de features: {e}")
        raise

def predict_last(model, df, features):
    instance = df.iloc[[-1]][features]
    y_proba = model.predict_proba(instance)[0,1]
    y_pred = int(y_proba > 0.5)
    logger.info(f"Predicción última: proba={y_proba:.4f}, class={y_pred}")
    return y_pred, y_proba

def export_prediction(y_pred, y_proba, path):
    res = pd.DataFrame({"y_pred": [y_pred], "y_proba": [y_proba]})
    res.to_csv(path, index=False)
    logger.info(f"Predicción guardada en {path}")

def main():
    parser = argparse.ArgumentParser(description="Predice la clase y proba en la última fila del archivo de features")
    parser.add_argument("--features-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--out", default="latest_pred.csv")
    args = parser.parse_args()

    df = load_features(args.features_file, args.features)
    model = load_model(args.model_file)
    y_pred, y_proba = predict_last(model, df, args.features)
    export_prediction(y_pred, y_proba, args.out)

if __name__ == "__main__":
    main()
