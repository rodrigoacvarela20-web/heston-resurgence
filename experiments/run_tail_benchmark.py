from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftfinance.heston import HestonParams
from ftfinance.benchmarks import compare_tail_estimators


params = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
result = compare_tail_estimators(params, threshold=-0.45, T=1.0, n_steps=60, n_paths=30000, seed=11, control_scale=0.85)

print("Naive MC probability:", result["p_mc"], "+/-", result["se_mc"])
print("Instanton-IS probability:", result["p_is"], "+/-", result["se_is"])
print("Estimated variance-reduction factor:", result["variance_reduction"])
print("Instanton action:", result["instanton_action"])
print("Optimizer success:", result["optimization_success"])

out = ROOT / "figures"
out.mkdir(exist_ok=True)
t = np.linspace(0.0, 1.0, len(result["x_star"]))

plt.figure()
plt.plot(t, result["x_star"])
plt.xlabel("t")
plt.ylabel("x*(t)")
plt.title("Minimum-action log-return path")
plt.tight_layout()
plt.savefig(out / "instanton_logreturn.png", dpi=180)
plt.close()

plt.figure()
plt.plot(t, result["v_star"])
plt.xlabel("t")
plt.ylabel("v*(t)")
plt.title("Minimum-action variance path")
plt.tight_layout()
plt.savefig(out / "instanton_variance.png", dpi=180)
plt.close()
