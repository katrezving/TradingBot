# ChatGPT‑Trading — Fase 0

## Requisitos
- Docker + Docker Compose
- Python 3.10+ (opcional, para utilidades)

## 1) Configura .env
Copia `.env.example` a `.env` y edita al menos los campos de DB y `APP_SECRET`.

```bash
python scripts/generate_secret.py >> tmp && tail -n1 tmp && rm tmp
# pega el secreto en .env (APP_SECRET)
```

## 2) Levanta servicios
```bash
docker compose up -d --build
```

## 3) Verifica salud
- API: http://localhost:8000/health
- MLflow: http://localhost:5000
- Postgres: puerto local 5433 (usuario/DB en .env)

## 4) Próximas fases
- **Fase 1**: Ingesta OHLCV + features deterministas (train/live)
- **Fase 2**: Modelos (LightGBM baseline + validación temporal)
- **Fase 3**: Serving real del modelo + paper trading + alertas Telegram
- **Fase 4**: Ejecución real con políticas de riesgo y canary release

## 5) Entorno

- Crear entorno: python -m venv TradingBot
- Activar entorno WIN: TradingBot\Scripts\activate
- Activar entorno MAC/LINUX: source TradingBot/bin/activate
- Instalar dependiencias: pip install -r requirements.txt