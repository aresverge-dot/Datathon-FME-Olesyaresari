"""
agent.py - Repte UPC Sostenible (Datathon 2026)

Controlador predictiu (MPC) amb programació lineal.

Cada hora:
  1. Construeix el cost combinat w_t = 0.7*co2_t/CO2_REF + 0.3*preu_t/PREU_REF
     per a les 24 h conegudes i, opcionalment, una extensió prevista (perfil per hora).
  2. Preveu el consum net (consum - solar) de cada hora vinent amb l'historial.
  3. Resol un LP: minimitzar sum(w_t * importacio_t) - valor_final * SOC_final
     amb importacio_t >= consum_net_t + carrega_t - descarrega_t, importacio_t >= 0.
  4. Aplica la decisió de la primera hora i la retalla per seguretat (0 infraccions).
"""

from collections import deque
import numpy as np
from scipy.optimize import linprog


class BatteryAgent:
    def __init__(self):
        """Inicialització de l'agent.
        S'executa una sola vegada a l'inici del període d'avaluació.
        """
        # Paràmetres físics de la bateria
        self.capacitat_kwh = 2000.0           # Capacitat màxima (kWh)
        self.potencia_nominal_kw = 500.0      # Potència màxima de l'inversor (kW)
        self.rendiment_eta = 0.9 ** 0.5       # Rendiment exacte de càrrega/descàrrega (0.948683...)
        self.marge_kwh = 0.05                  # Marge de seguretat per no tocar mai els límits

        # Pesos de la puntuació oficial (70% CO2, 30% cost) i valors mitjans del train
        # per posar el CO2 i el preu a la mateixa escala
        self.pes_co2 = 0.7
        self.pes_preu = 0.3
        self.co2_mitja = 0.0865                # kg/kWh (mitjana ponderada pel consum)
        self.preu_mitja = 0.143               # €/kWh  (mitjana ponderada pel consum)

        # Horitzó de planificació
        self.hores_extra = 24                 # Hores previstes més enllà de les 24 conegudes

        # Memòria històrica: per a cada hora del dia, els últims 7 consums nets observats
        self.historial_consum = {}

    def act(self, obs: dict) -> float:
        """Determina la consigna de potència activa de la bateria per a l'hora actual.

        Retorna:
            > 0  -> Càrrega (fins a +500.0 kW)
            < 0  -> Descàrrega (fins a -500.0 kW)
            = 0  -> Bateria en repòs
        """
        hora = obs["hour"]
        soc_kwh = float(obs["soc_kwh"])
        preus = list(obs["preus_properes_24h"]) or [obs["preu_actual"]]
        co2 = list(obs["co2_properes_24h"]) or [obs["co2_actual"]]

        # 1. Guardem el consum net de l'hora anterior (consum - solar)
        dades = obs["noves_dades_5min"]
        consum_net_ultim = dades["ultim_consum_kw"] - dades["ultima_solar_kw"]
        hora_anterior = (hora - 1) % 24
        if hora_anterior not in self.historial_consum:
            self.historial_consum[hora_anterior] = deque(maxlen=7)
        self.historial_consum[hora_anterior].append(consum_net_ultim)

        # 2. Cost de cada hora vinent per a la puntuació (CO2 i preu combinats)
        n_coneguts = min(len(preus), len(co2))
        cost_hores = [self.pes_co2 * co2[k] / self.co2_mitja + self.pes_preu * preus[k] / self.preu_mitja
                      for k in range(n_coneguts)]

        # Si tenim menys de 24 hores, som al final del període: l'energia que quedi no val res
        final_periode = n_coneguts < 24

        # 3. Allarguem l'horitzó suposant que demà serà com avui
        if not final_periode:
            cost_hores = cost_hores + cost_hores[:self.hores_extra]

        # 4. Previsió del consum net de cada hora vinent
        consum_previst = [self._previsio_consum((hora + k) % 24, consum_net_ultim)
                          for k in range(len(cost_hores))]

        # 5. Planificació òptima i decisió de l'hora actual
        try:
            consigna = self._planifica(soc_kwh, cost_hores, consum_previst, final_periode)
        except Exception:
            consigna = 0.0

        # 6. Clamping de seguretat física per evitar infraccions
        max_carrega = min(self.potencia_nominal_kw,
                          max(0.0, self.capacitat_kwh - self.marge_kwh - soc_kwh) / self.rendiment_eta)
        max_descarrega = min(self.potencia_nominal_kw,
                             max(0.0, soc_kwh - self.marge_kwh) * self.rendiment_eta)

        return float(max(-max_descarrega, min(max_carrega, consigna)))

    def _previsio_consum(self, hora, valor_per_defecte):
        """Consum net previst per a una hora del dia: mediana dels últims 7 dies,
        ignorant les lectures errònies properes a 0 (fallades del comptador)."""
        valors = [x for x in self.historial_consum.get(hora, []) if x > 50]
        if not valors:
            return max(valor_per_defecte, 0.0)
        return max(float(np.median(valors)), 0.0)

    def _planifica(self, soc0, cost_hores, consum_previst, final_periode):
        """Resol un problema de programació lineal sobre l'horitzó.

        Variables per a cada hora t:  carrega_t, descarrega_t, importacio_t  (kW, >= 0)
        Objectiu: minimitzar  sum(cost_t * importacio_t) - valor_final * SOC_final
        Restriccions:
            importacio_t >= consum_t + carrega_t - descarrega_t
            marge <= SOC_t <= capacitat - marge
        """
        n = len(cost_hores)
        eta = self.rendiment_eta
        w = np.array(cost_hores)
        consum = np.array(consum_previst)

        # Valor de l'energia que queda a la bateria al final de l'horitzó
        valor_final = 0.0 if final_periode else eta * float(np.median(w))

        # Funció objectiu
        cost = np.concatenate([np.full(n, -valor_final * eta),   # carrega
                               np.full(n, valor_final / eta),    # descarrega
                               w])                               # importacio

        # SOC_t = soc0 + sum_{k<=t} (eta * carrega_k - descarrega_k / eta)
        L = np.tril(np.ones((n, n)))
        Z = np.zeros((n, n))
        I = np.eye(n)
        A_soc = np.hstack([eta * L, -L / eta, Z])
        A_importacio = np.hstack([I, -I, -I])      # carrega - descarrega - importacio <= -consum

        A_ub = np.vstack([A_soc, -A_soc, A_importacio])
        b_ub = np.concatenate([np.full(n, self.capacitat_kwh - self.marge_kwh - soc0),
                               np.full(n, soc0 - self.marge_kwh),
                               -consum])
        limits = ([(0.0, self.potencia_nominal_kw)] * (2 * n)) + [(0.0, None)] * n

        resultat = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=limits, method="highs")
        if resultat.status != 0:
            return 0.0

        # Consigna de la primera hora: càrrega - descàrrega
        return float(resultat.x[0] - resultat.x[n])
