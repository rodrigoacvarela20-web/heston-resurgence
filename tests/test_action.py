import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftfinance.heston import HestonParams
from ftfinance.action import discrete_freidlin_wentzell_action


def test_deterministic_path_has_small_action():
    params = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.4, rho=-0.5, v0=0.04)
    n = 20
    T = 1.0
    dt = T / n
    v = np.full(n + 1, 0.04)
    x = np.zeros(n + 1)
    for i in range(n):
        x[i + 1] = x[i] + (params.mu - 0.5 * v[i]) * dt
    action = discrete_freidlin_wentzell_action(x, v, params, T=T)
    assert action < 1e-10
