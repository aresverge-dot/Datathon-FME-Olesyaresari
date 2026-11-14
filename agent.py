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
    
    CAP = 2000.0
    PMAX = 500.0
    ETA = 0.9 ** 0.5
    MARGE = 1.0

    
    CO2_REF = 0.084
    PREU_REF = 0.143
    W_CO2 = 0.7
    W_PREU = 0.3

    
    H_EXTRA = 24 # hores extra previstes més enllà de les 24 conegudes
    DESCOMPTE_EXTRA = 1.0 # confiança en la previsió extra
    QUANTIL_TERMINAL = 0.5

    def __init__(self):
        self.net_hist = {} # (hora) -> deque de consums nets observats
        self.w_hist = {} # (hora) -> deque de w observats
        self.ultim_ts = None

    def _registra(self, obs, w_ara):
        hora = int(obs["hour"])
        dades = obs.get("noves_dades_5min") or {}
        net = float(dades.get("ultim_consum_kw", 0.0)) - float(dades.get("ultima_solar_kw", 0.0))
        h_ant = (hora - 1) % 24
        self.net_hist.setdefault(h_ant, deque(maxlen=7)).append(net)
        self.w_hist.setdefault(hora, deque(maxlen=7)).append(w_ara)
        return net

    def _previsio_net(self, hora0, n, ultim_net):
        out = np.empty(n)
        for k in range(n):
            hk = (hora0 + k) % 24
            hist = self.net_hist.get(hk)
            if hist:
                out[k] = min(hist) # conservador: el mínim de la setmana
            else:
                out[k] = ultim_net
        return np.maximum(out, 0.0)

    def _previsio_w(self, hora0, n, w_conegut):
        out = np.empty(n)
        for k in range(n):
            hk = (hora0 + k) % 24
            hist = self.w_hist.get(hk)
            # persistència: el valor conegut 24 h abans (mateixa hora) o la mitjana
            out[k] = w_conegut[k % len(w_conegut)] if len(w_conegut) >= 24 else (
                np.mean(hist) if hist else float(np.mean(w_conegut)))
        return out

    def act(self, obs: dict) -> float:
        soc = float(obs["soc_kwh"])
        preus = np.asarray(obs.get("preus_properes_24h") or [obs["preu_actual"]], dtype=float)
        co2 = np.asarray(obs.get("co2_properes_24h") or [obs["co2_actual"]], dtype=float)
        h = int(min(len(preus), len(co2)))
        if h == 0:
            return 0.0
        w = self.W_CO2 * co2[:h] / self.CO2_REF + self.W_PREU * preus[:h] / self.PREU_REF

        ultim_net = self._registra(obs, float(w[0]))
        hora0 = int(obs["hour"])
        final_dades = h < 24 # final del període: l'energia restant no val res

        if not final_dades and self.H_EXTRA > 0:
            w_ext = self._previsio_w(hora0 + h, self.H_EXTRA, w) * self.DESCOMPTE_EXTRA
            w_tot = np.concatenate([w, w_ext])
        else:
            w_tot = w
        n = len(w_tot)
        net = self._previsio_net(hora0, n, ultim_net)

        try:
            p = self._resol(soc, w_tot, net, final_dades)
        except Exception:
            p = 0.0
        return self._seguretat(p, soc)

    def _resol(self, soc0, w, net, final_dades):
        n = len(w)
        eta = self.ETA
        smin, smax = self.MARGE, self.CAP - self.MARGE
        v = 0.0 if final_dades else eta * float(np.quantile(w, self.QUANTIL_TERMINAL))

        # Variables: c (n), d (n), g (n)
        # SOC_h = soc0 + sum(eta c - d/eta) ; objectiu: sum w g - v*SOC_n
        cost = np.concatenate([np.full(n, -v * eta), np.full(n, v / eta), w])

        L = np.tril(np.ones((n, n)))
        Z = np.zeros((n, n))
        I = np.eye(n)
        A_soc = np.hstack([eta * L, -L / eta, Z])
        # g >= net + c - d   ->   c - d - g <= -net
        A_g = np.hstack([I, -I, -I])
        A_ub = np.vstack([A_soc, -A_soc, A_g])
        b_ub = np.concatenate([np.full(n, smax - soc0), np.full(n, soc0 - smin), -net])
        bounds = [(0.0, self.PMAX)] * (2 * n) + [(0.0, None)] * n

        res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status != 0:
            return 0.0
        return float(res.x[0] - res.x[n])

    def _seguretat(self, p, soc):
        max_c = max(0.0, (self.CAP - self.MARGE - soc) / self.ETA)
        max_d = max(0.0, (soc - self.MARGE) * self.ETA)
        p = max(-min(self.PMAX, max_d), min(min(self.PMAX, max_c), p))
        return 0.0 if abs(p) < 1e-6 else float(p)