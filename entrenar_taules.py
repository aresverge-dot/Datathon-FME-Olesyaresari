"""
entrenar_taules.py  --  MODE DIFICIL
Genera el diccionari TAULES que fa servir agent.py a partir de data/train.
Execucio (des de starter_kit/, amb agent.py al costat):
    python entrenar_taules.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from agent import BatteryAgent

SOLAR_X = 12            # en Mode Dificil la planta solar es 12 vegades mes gran
N_ESCENARIS = 5
PERCENTILS = [10, 30, 50, 70, 90]   # 5 escenaris igual de probables

# Taula horaria 
DATA_DIR = Path("../data/train") if Path("../data/train").exists() else Path("data/train")

df_contador = pd.read_csv(DATA_DIR / "contador.csv", parse_dates=["Fecha"])
df_inversor = pd.read_csv(DATA_DIR / "inversor.csv", parse_dates=["Fecha"])
df_pvpc = pd.read_csv(DATA_DIR / "precio_pvpc.csv", parse_dates=["Fecha"])
df_co2 = pd.read_csv(DATA_DIR / "co2_red.csv", parse_dates=["Fecha"])

# Consum del campus (kW)
df_contador = df_contador.set_index("Fecha")
load_5min = (-df_contador["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(lower=0, upper=5000)
load_hourly = load_5min.resample("1h").mean().dropna()

# Solar total (kW), multiplicada x12
df_inversor = df_inversor.set_index("Fecha")
p1 = [c for c in df_inversor.columns if "INVERSOR 1" in c.upper() and "POTENCIA ACTIVA" in c.upper() and "FASE" not in c.upper()][0]
p2 = [c for c in df_inversor.columns if "INVERSOR 2" in c.upper() and "POTENCIA ACTIVA" in c.upper() and "FASE" not in c.upper()][0]
p3 = [c for c in df_inversor.columns if "INVERSOR 3" in c.upper() and "POTENCIA ACTIVA" in c.upper() and "FASE" not in c.upper()][0]
pv_total_5min = (df_inversor[p1].fillna(0) + df_inversor[p2].fillna(0) + df_inversor[p3].fillna(0)).clip(lower=0)
pv_hourly = pv_total_5min.resample("1h").mean().dropna() * SOLAR_X

# Unim amb preus i CO2
df_pvpc = df_pvpc.set_index("Fecha")
df_co2 = df_co2.set_index("Fecha")
df = pd.DataFrame({
    "load_kw": load_hourly,
    "pv_kw": pv_hourly,
    "precio_eur_per_kwh": df_pvpc["precio_eur_per_kwh"],
    "co2_kg_per_kwh": df_co2["co2_kg_per_kwh"],
}).dropna()

# Columnes per agrupar
df["mes"] = df.index.month - 1                              # 0 = gener ... 11 = desembre
df["cap_setmana"] = (df.index.dayofweek >= 5).astype(int)   # 0 = laborable, 1 = cap de setmana
df["hora"] = df.index.hour
print(f"Taula horaria: {len(df)} hores, de {df.index.min()} a {df.index.max()}")


def a_taula(serie, forma):
    """Converteix una serie agrupada (index [mes, ..., hora]) en una llista niada.
    Si falta algun mes sencer (a train no hi ha novembre de consum), l'omple
    amb la mitjana del mes anterior i el seguent."""
    index = pd.MultiIndex.from_product([range(n) for n in forma])
    taula = serie.reindex(index).to_numpy(dtype=float, copy=True).reshape(forma)
    for mes in range(12):
        if np.isnan(taula[mes]).all():
            taula[mes] = (taula[(mes - 1) % 12] + taula[(mes + 1) % 12]) / 2
    return np.round(taula, 4).tolist()

# Perfils tipics: [mes][laborable/cap de setmana][hora]

def claus(index):
    """Claus d'agrupacio [mes, laborable/cap de setmana, hora] d'un index de dates."""
    return [index.month - 1, (index.dayofweek >= 5).astype(int), index.hour]

# Preu i CO2: des dels seus propis CSV (cobreixen mes hores que la taula conjunta)
t_preu = a_taula(df_pvpc["precio_eur_per_kwh"].groupby(claus(df_pvpc.index)).mean(), (12, 2, 24))
t_co2 = a_taula(df_co2["co2_kg_per_kwh"].groupby(claus(df_co2.index)).mean(), (12, 2, 24))

# Consum: des de la taula conjunta.
#  Mediana: no es deixa enganyar pels talls a 0 kW
t_load = a_taula(df.groupby(["mes", "cap_setmana", "hora"])["load_kw"].median(), (12, 2, 24))

# Solar en dia clar: [mes][hora] (percentil 90 = dia sense nuvols)
t_pv_env = a_taula(df.groupby(["mes", "hora"])["pv_kw"].quantile(0.90), (12, 24))

# Errors de la previsio de consum net: [antelacio][hora][escenari]
#   Fem correr la previsio de l'agent sobre train com si fos en directe i
#   comparem cada previsio amb el que va passar de debo.
taules_provisionals = {"preu": t_preu, "co2": t_co2, "load": t_load, "pv_env": t_pv_env,
                       "errors": np.zeros((24, 24, N_ESCENARIS)).tolist()}
agent = BatteryAgent(taules=taules_provisionals)

net_real = (df["load_kw"] - df["pv_kw"]).values
dates = df.index
errors = [[[] for _ in range(24)] for _ in range(24)]       # errors[antelacio][hora]

for t in range(1, len(df) - 24):
    # Nomes si les 24 hores seguents son consecutives (sense forats a les dades)
    if dates[t + 23] - dates[t - 1] != pd.Timedelta(hours=24):
        continue
    ts = dates[t]
    obs = {
        "timestamp": ts,
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "preu_actual": float(df["precio_eur_per_kwh"].iloc[t]),       # l'agent tambe preveu el mercat,
        "co2_actual": float(df["co2_kg_per_kwh"].iloc[t]),            # tot i que aqui no ho fem servir
        "preus_properes_24h": df["precio_eur_per_kwh"].iloc[t:t + 3].tolist(),
        "co2_properes_24h": df["co2_kg_per_kwh"].iloc[t:t + 3].tolist(),
        "noves_dades_5min": {                                 # el que va passar l'hora anterior
            "ultim_consum_kw": float(df["load_kw"].iloc[t - 1]),
            "ultima_solar_kw": float(df["pv_kw"].iloc[t - 1]),
        },
    }
    _, previsio, hores = agent.previsions(obs)
    for k in range(24):
        errors[k][hores[k]].append(net_real[t + k] - previsio[k])

t_errors = np.zeros((24, 24, N_ESCENARIS))
for k in range(24):
    for h in range(24):
        if errors[k][h]:
            t_errors[k, h] = np.percentile(errors[k][h], PERCENTILS)
t_errors = np.round(t_errors, 1).tolist()

TAULES = {"preu": t_preu, "co2": t_co2, "load": t_load, "pv_env": t_pv_env, "errors": t_errors}

# Escriure TAULES al final d'agent.py (substitueix l'anterior)
cami = Path(__file__).parent / "agent.py"
codi = cami.read_text(encoding="utf-8")
nova_linia = "TAULES = " + json.dumps(TAULES, separators=(",", ":"))
codi = re.sub(r"^TAULES = .*$", lambda m: nova_linia, codi, flags=re.M)
cami.write_text(codi, encoding="utf-8")
print("TAULES escrites a agent.py")
