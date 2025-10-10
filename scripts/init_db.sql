-- Ejecutado automáticamente en el arranque del contenedor Postgres
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Tabla sencilla para logs de ejecución (ampliaremos en Fase 3)
CREATE TABLE IF NOT EXISTS trade_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  level TEXT NOT NULL,
  message TEXT NOT NULL
);
