from fastapi import FastAPI
from loguru import logger
from .settings import Settings

settings = Settings()
app = FastAPI(title=settings.APP_NAME)

@app.get("/health")
def health():
    return {"status": "ok", "env": settings.APP_ENV}

@app.get("/version")
def version():
    return {"app": settings.APP_NAME, "env": settings.APP_ENV}

@app.post("/predict")
def predict_stub():
    return {"detail": "Predictor no inicializado (Fase 2)."}

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando API...")
    uvicorn.run(app, host="0.0.0.0", port=settings.APP_PORT)
