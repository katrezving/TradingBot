# ==== APP ====
APP_NAME="chatgpt-trading"
APP_ENV=dev
APP_SECRET="JyFFmbFKZAcViG6LtS1kxRx0D-WFMl3WUPQo4Ta-0hI"   # usa scripts/generate_secret.py
APP_PORT=8000

# ==== EXCHANGE API (solo trading, SIN retiros) ====
EXCHANGE_ID=binance
EXCHANGE_API_KEY=""
EXCHANGE_API_SECRET=""
EXCHANGE_PASSWORD=""        # si aplica
EXCHANGE_TESTNET=true        # true: testnet cuando aplique (futuros)

# ==== DATABASE ====
POSTGRES_USER=trader
POSTGRES_PASSWORD=traderpwd
POSTGRES_DB=trading
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# ==== REDIS ====
REDIS_HOST=redis
REDIS_PORT=6379

# ==== MLFLOW ====
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_ARTIFACT_ROOT=/mlruns

# ==== LOGGING ====
LOG_LEVEL=INFO

# ==== INTEGRATIONS (Data, News, Sentiment) ====

# CryptoPanic - noticias y sentimiento
CRYPTOPANIC_API_KEY=c47c7bd7d52339eda9f16dfd06512072b74cc361

# CoinmarketCap - 

CMC_API_KEY=0191d1aa-05d7-4839-8666-e5c4f9264d96

# CoinGlass - datos de Open Interest, Dominance, Liquidaciones, Funding, etc.
COINGLASS_API_KEY=c97edb4c7aef48928a743c94747880da

# Alternative.me - Fear & Greed Index (sin API key)
# (no requiere clave, solo referencia)
FEAR_GREED_SOURCE=https://api.alternative.me/fng/

# Reddit y X (Twitter) - para scraping o APIs de sentimiento (cuando se integren)
REDDIT_CLIENT_ID=tu_id_aqui
REDDIT_CLIENT_SECRET=tu_secret_aqui
REDDIT_USER_AGENT=TradingBotSentiment/1.0

X_BEARER_TOKEN=tu_bearer_token_aqui
