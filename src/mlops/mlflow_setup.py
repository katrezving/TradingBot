import mlflow, os

def ensure_mlflow_tracking():
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(uri)
    print("MLflow tracking URI:", uri)

if __name__ == "__main__":
    ensure_mlflow_tracking()
