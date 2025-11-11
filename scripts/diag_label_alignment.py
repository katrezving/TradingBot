# scripts/diag_label_alignment.py
import pandas as pd

PRED_PATH = r"evals/hgb_ts_no_leak/predictions.csv"
DATA_PATH = r"data/ALLPAIRS_ctx_1h_h6_ts.csv"
THR = 0.5794

preds = pd.read_csv(PRED_PATH)
data  = pd.read_csv(DATA_PATH)

# merge por keys si están; si no, alineación por posición (fallback)
if {"timestamp","symbol"}.issubset(preds.columns) and {"timestamp","symbol"}.issubset(data.columns):
    m = pd.merge(preds, data[["timestamp","symbol","ret1"]], on=["timestamp","symbol"], how="left")
else:
    m = preds.copy()
    if "ret1" in data.columns and len(data)==len(preds):
        m["ret1"] = data["ret1"].values

print("Columnas en merged:", list(m.columns))
print("N filas:", len(m))
print()

# Medias de ret1 por etiqueta
if "y_true" in m.columns:
    print("ret1 mean | y_true=1:", m.loc[m["y_true"]==1, "ret1"].mean())
    print("ret1 mean | y_true=0:", m.loc[m["y_true"]==0, "ret1"].mean())
else:
    print("No hay y_true en predictions.csv")

# Medias de ret1 para prob>=THR
if "prob_1" in m.columns:
    hi = m["prob_1"] >= THR
    print("ret1 mean | prob>=thr (interprete LONG):", m.loc[hi, "ret1"].mean())
    print("ret1 mean | prob>=thr (interprete SHORT):", -m.loc[hi, "ret1"].mean())
    print("Coverage (% barras >=thr):", hi.mean()*100)
else:
    print("predictions.csv no tiene prob_1")

# Guarda un resumen chico por si quieres revisar en Excel
out_cols = [c for c in ["timestamp","symbol","prob_1","y_true","ret1"] if c in m.columns]
m[out_cols].head(500).to_csv("evals/diag_alignment_sample.csv", index=False)
print("\nMuestra guardada en evals/diag_alignment_sample.csv")
