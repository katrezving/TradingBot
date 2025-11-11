# src/ml/predict_latest.py
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import timezone


def load_features(features_path: Path):
    with open(features_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # puede venir como lista o dict {"features":[...]}
    if isinstance(obj, dict):
        if "features" in obj and isinstance(obj["features"], list):
            return obj["features"]
        # fallback: si es dict plano (col:desc), tomar keys
        return list(obj.keys())
    elif isinstance(obj, list):
        return obj
    else:
        raise ValueError("Formato de features.json no reconocido")


def auto_pick_model_and_features(model_prefix: str, models_dir="models"):
    models_dir = Path(models_dir)
    files = sorted(models_dir.glob(f"{model_prefix}*.joblib"),
                   key=lambda p: p.stat().st_mtime,
                   reverse=True)
    if not files:
        raise FileNotFoundError(f"No se encontró modelo con prefijo '{model_prefix}' en {models_dir}.")
    model_path = files[0]
    feat_path = model_path.with_suffix("").with_suffix(".features.json")
    if not feat_path.exists():
        # prueba basename + .features.json
        feat_path = Path(models_dir, f"{model_path.stem}.features.json")
    if not feat_path.exists():
        raise FileNotFoundError(f"No se encontró features.json acompañando al modelo {model_path.name}")
    return model_path, feat_path


def resolve_threshold(args, default=0.5):
    # Prioridades:
    # 1) --threshold explícito
    # 2) --metrics-file + --metrics-key
    # 3) default
    if args.threshold is not None:
        return float(args.threshold)
    if args.metrics_file and args.metrics_key:
        with open(args.metrics_file, "r", encoding="utf-8") as f:
            met = json.load(f)
        # resolver key tipo "a.b.c"
        cur = met
        for k in args.metrics_key.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                raise KeyError(f"No se encontró metrics_key='{args.metrics_key}' en {args.metrics_file}")
        val = float(cur)
        return val
    return float(default)


def sync_to_common_timestamp(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Mantiene sólo filas del timestamp más común (último en la mayoría de símbolos)."""
    if ts_col not in df.columns:
        return df
    # normaliza a pandas datetime (no estrellarse si ya es datetime)
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.copy()
    df[ts_col] = ts
    # mode del último timestamp por símbolo:
    # tomamos el timestamp máximo por símbolo y luego el modo entre símbolos
    last_ts_by_sym = df.groupby("symbol")[ts_col].max().dropna()
    if last_ts_by_sym.empty:
        return df
    common_ts = last_ts_by_sym.mode().iloc[0]
    return df[df[ts_col] == common_ts].copy()


def apply_btc_regime_filter(df: pd.DataFrame,
                            level: str = "none",
                            prob_col: str = "prob_1") -> pd.DataFrame:
    """
    Filtro de régimen BTC (opcional). Reglas:
      - Requiere columnas: btc_trend (±1/0), btc_rsi, btc_vol_z (si existen).
      - level in {none, lax, moderate, strict}

    Devuelve df con columnas:
      - take_signal (0/1)
      - post_filter_reason (str)
    """
    out = df.copy()
    if "post_filter_reason" not in out.columns:
        out["post_filter_reason"] = ""

    # si no se piden filtros o no hay columnas, permitir todo
    needed_any = {"btc_trend", "btc_rsi", "btc_vol_z"}.intersection(out.columns)
    if level == "none" or not needed_any:
        if "take_signal" not in out.columns:
            out["take_signal"] = 1
        else:
            out["take_signal"] = out["take_signal"].fillna(1).astype(int)
        return out

    # Reglas base por nivel (ajustables)
    # btc_trend: +1 alcista, -1 bajista, 0 neutro (si tu build_dataset lo define así)
    # Tolerancias RSI y volatilidad para permitir señal
    lvl = level.lower()
    if lvl not in {"lax", "moderate", "strict"}:
        lvl = "moderate"

    # defaults tolerantes por si faltan columnas
    trend = out["btc_trend"] if "btc_trend" in out.columns else 0
    rsi = out["btc_rsi"] if "btc_rsi" in out.columns else 50
    volz = out["btc_vol_z"] if "btc_vol_z" in out.columns else 0

    # Criterios por nivel
    if lvl == "lax":
        # sólo bloquea cuando hay señales extremas de volatilidad
        favorable = ~((volz > 2.5) | (volz < -2.5))
    elif lvl == "moderate":
        # evitar trades contra tendencia fuerte y en volatilidad alta
        favorable = ~((trend == -1) & (rsi > 60)) & ~((trend == 1) & (rsi < 40)) & (volz.abs() <= 2.0)
    else:  # strict
        # exige tendencia y RSI coherentes + baja vol
        favorable = ((trend == 1) & (rsi >= 45)) | ((trend == -1) & (rsi <= 55))
        favorable = favorable & (volz.abs() <= 1.5)

    # Inicializa si no existe
    if "take_signal" not in out.columns:
        out["take_signal"] = 1

    out.loc[~favorable, "take_signal"] = 0
    # construir string por fila sin .astype sobre str
    bad_mask = ~favorable
    if bad_mask.any():
        prev = out.loc[bad_mask, "post_filter_reason"].fillna("").astype(str)
        add = np.where(prev == "", "bad_btc_regime", prev + "|bad_btc_regime")
        out.loc[bad_mask, "post_filter_reason"] = add

    return out


def decide_signals_single_side(out: pd.DataFrame, side: str, thr: float, prob_col="prob_1") -> pd.DataFrame:
    """Decisión clásica: sólo long o sólo short."""
    out = out.copy()
    p = out[prob_col].values
    if side == "long":
        pred = (p >= thr).astype(int)
        signal = pred  # 1 o 0
    elif side == "short":
        pred = (p >= thr).astype(int)  # clase "1" como evento -> si queremos short cuando prob_1 alta? NO.
        # Interpretación para short: entrar cuando prob_1 < (1 - thr)
        pred = (p < (1.0 - thr)).astype(int)
        signal = -pred  # -1 o 0
    else:
        raise ValueError("side inválido para single-side")
    out["pred"] = pred
    out["signal_live"] = signal
    return out


def decide_signals_longshort(out: pd.DataFrame,
                             thr_long: float | None,
                             thr_short: float | None,
                             prob_col="prob_1") -> pd.DataFrame:
    """
    Lado automático:
      - LONG si prob_1 >= thr_long
      - SHORT si prob_1 <= (1 - thr_short)
      - Resolver conflictos por "confianza".
    """
    if thr_long is None and thr_short is None:
        raise ValueError("Para --side longshort define al menos --thr-long o --thr-short")

    out = out.copy()
    p = out[prob_col].values
    sig_long = np.zeros(len(out), dtype=int)
    sig_short = np.zeros(len(out), dtype=int)

    if thr_long is not None:
        sig_long = (p >= float(thr_long)).astype(int)
    if thr_short is not None:
        sig_short = (p <= (1.0 - float(thr_short))).astype(int)

    # Resolver conflictos ambos=1
    both = (sig_long == 1) & (sig_short == 1)
    if both.any():
        conf_long = p - (thr_long if thr_long is not None else 1.0)
        conf_short = ((1.0 - (thr_short if thr_short is not None else 1.0)) - p)
        choose_long = conf_long >= conf_short
        # Si no gana long, apaga long y deja short. Si gana long, apaga short.
        sig_long[both & (~choose_long)] = 0
        sig_short[both & choose_long] = 0

    out["signal_long"] = sig_long
    out["signal_short"] = -sig_short  # -1
    out["signal_live"] = out["signal_long"] + out["signal_short"]
    return out


def main():
    ap = argparse.ArgumentParser()
    # Datos / modelo
    ap.add_argument("--data", required=True, help="CSV con el dataset de features ya construido.")
    ap.add_argument("--model-path", default=None, help="Ruta explicita del modelo .joblib")
    ap.add_argument("--model-prefix", default=None, help="Prefijo para auto-elegir el modelo más reciente en ./models")
    ap.add_argument("--features-path", default=None, help="Ruta del features.json. Si no, se infiere del modelo.")
    ap.add_argument("--id-cols", default="symbol,timestamp,fwd_ret", help="Columnas ID separadas por coma.")
    ap.add_argument("--outdir", default="signals/live", help="Directorio de salida para csv/json")

    # Decisión umbrales
    ap.add_argument("--threshold", type=float, default=None, help="Umbral para single side.")
    ap.add_argument("--metrics-file", default=None, help="JSON con métricas (para extraer umbral).")
    ap.add_argument("--metrics-key", default=None, help="Clave dentro del JSON para el umbral (ej: optimized_threshold.best_threshold)")

    # Lado de operación
    ap.add_argument("--side", choices=["long", "short", "longshort"], default="long",
                    help="Dirección: solo long, solo short o longshort (decide automáticamente).")
    ap.add_argument("--thr-long", type=float, default=None, help="Umbral para long en modo longshort.")
    ap.add_argument("--thr-short", type=float, default=None, help="Umbral para short en modo longshort.")

    # Compatibilidad hacia atrás (deprecated)
    ap.add_argument("--trade-direction", choices=["long", "short"], default=None,
                    help="DEPRECATED. Usa --side. Si se pasa, sobreescribe --side.")

    # Filtros / sync
    ap.add_argument("--btc-regime-level", choices=["none", "lax", "moderate", "strict"], default="none",
                    help="Filtro de régimen BTC.")
    ap.add_argument("--sync-common-timestamp", action="store_true", help="Mantiene sólo el timestamp más común entre símbolos.")

    args = ap.parse_args()

    # Resolver modelo y features
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"No existe --model-path: {model_path}")
        if args.features_path:
            features_path = Path(args.features_path)
        else:
            features_path = model_path.with_suffix("").with_suffix(".features.json")
            if not features_path.exists():
                features_path = Path("models", f"{model_path.stem}.features.json")
        if not features_path.exists():
            raise FileNotFoundError(f"No se encontró features.json: {features_path}")
    else:
        if not args.model_prefix:
            raise ValueError("Define --model-path o --model-prefix")
        model_path, features_path = auto_pick_model_and_features(args.model_prefix)

    print(f"[predict_latest] Using model: {model_path.as_posix()}")
    print(f"[predict_latest] Using features: {features_path.as_posix()}")

    # Carga modelo/feats
    model = joblib.load(model_path)
    features = load_features(features_path)
    if not isinstance(features, list) or not features:
        raise ValueError("La lista de features está vacía o no es válida.")

    # Carga data
    df = pd.read_csv(args.data)
    # Asegurar timestamp en tz-aware (para impresión/consistencia). No forzamos int.
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Subset features presentes (evitar error si alguna falta)
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"[WARN] Faltan {len(missing)} features en data, se omiten: {missing[:10]}{'...' if len(missing)>10 else ''}")
    used_feats = [c for c in features if c in df.columns]
    if not used_feats:
        raise ValueError("Ninguna feature del modelo está en el CSV de data.")

    X = df[used_feats]
    # Predict proba (clase 1)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        # convertir margin a 0..1 con sigmoide
        margin = model.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-margin))
    else:
        # fallback a pred binaria -> probas 0/1
        pred_bin = model.predict(X)
        proba = pred_bin.astype(float)

    out_cols = []
    # Añadir columnas ID si existen
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    for c in id_cols:
        if c in df.columns and c not in out_cols:
            out_cols.append(c)
    # Si hay close, útil para revisar
    if "close" in df.columns:
        out_cols.append("close")

    out = df[out_cols].copy() if out_cols else pd.DataFrame(index=df.index)
    out["prob_1"] = proba

    # Sincroniza timestamp común si se pide
    if args.sync_common_timestamp and "timestamp" in out.columns:
        out = sync_to_common_timestamp(out, "timestamp")

    # Determinar lado (compatibilidad vieja)
    side = args.side
    if args.trade_direction:
        side = args.trade_direction  # override
    # Resolver thresholds según modo
    threshold_used = None
    if side in {"long", "short"}:
        threshold_used = resolve_threshold(args, default=0.5)
        print(f"[predict_latest] threshold_used={threshold_used}")
        out = decide_signals_single_side(out, side=side, thr=threshold_used, prob_col="prob_1")
    else:
        # longshort
        print("[predict_latest] side=longshort")
        # Si no se dieron, usar defaults razonables
        thr_long = args.thr_long if args.thr_long is not None else 0.28
        thr_short = args.thr_short if args.thr_short is not None else 0.60
        print(f"[predict_latest] thr_long={thr_long} | thr_short={thr_short}")
        out = decide_signals_longshort(out, thr_long=thr_long, thr_short=thr_short, prob_col="prob_1")

    # take_signal base (1 si hay señal, 0 si no)
    if "take_signal" not in out.columns:
        # cualquier no-cero en signal_live cuenta como tomar
        if "signal_live" in out.columns:
            out["take_signal"] = (out["signal_live"] != 0).astype(int)
        else:
            # single side fallback: si pred==1 en long o pred==1 en short (que es señal válida)
            out["take_signal"] = out.get("pred", 0).astype(int)

    # aplicar filtro régimen BTC
    out = apply_btc_regime_filter(out, level=args.btc_regime_level, prob_col="prob_1")

    # threshold_used: para trazabilidad
    if threshold_used is None and side == "longshort":
        # guardamos ambos como texto
        out["threshold_used"] = ""
        out.attrs["thr_long"] = args.thr_long
        out.attrs["thr_short"] = args.thr_short
    else:
        out["threshold_used"] = threshold_used

    # Orden de columnas amigable
    nice_cols = []
    for c in ["symbol", "timestamp", "close", "prob_1", "threshold_used", "pred",
              "signal_long", "signal_short", "signal_live", "take_signal", "post_filter_reason",
              "y_true", "btc_trend", "btc_rsi", "btc_vol_z"]:
        if c in out.columns:
            nice_cols.append(c)
    # añade las que falten al final
    for c in out.columns:
        if c not in nice_cols:
            nice_cols.append(c)
    out = out[nice_cols]

    # Crear outdir y guardar
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "signals_latest.csv"
    json_path = outdir / "signals_latest.json"

    out.to_csv(csv_path, index=False)

    # JSON compacto con sólo lo esencial para consumo externo
    # Cuidar tipos no serializables (timestamps a str ISO)
    out_json = out.copy()
    if "timestamp" in out_json.columns:
        out_json["timestamp"] = pd.to_datetime(out_json["timestamp"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")

    # Selecciona columnas más prácticas
    json_cols = [c for c in ["symbol", "timestamp", "close", "prob_1",
                             "threshold_used", "pred", "signal_long", "signal_short",
                             "signal_live", "take_signal", "post_filter_reason"] if c in out_json.columns]
    out_json[json_cols].to_json(json_path, orient="records", force_ascii=False)

    print(f"✔ Saved {csv_path.as_posix()} and {json_path.as_posix()}")

    # Imprime un preview tipo tabla simple
    preview_cols = [c for c in ["symbol", "timestamp", "close", "prob_1",
                                "threshold_used", "pred", "signal_long", "signal_short",
                                "signal_live", "take_signal", "post_filter_reason",
                                "y_true", "btc_trend", "btc_rsi", "btc_vol_z"] if c in out.columns]
    try:
        # mostrar primeras 6 filas por legibilidad
        disp = out[preview_cols].head(6)
        # formateo amigable
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(disp.to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
