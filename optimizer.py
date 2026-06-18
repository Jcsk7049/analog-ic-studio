# -*- coding: utf-8 -*-
"""
optimizer.py — VCO 控制電壓 Vctrl 混合優化器 (任務二)

針對 sky130 真實 VCO: 固定振盪鏈尺寸, 自動尋找最佳「控制電壓 Vctrl」使輸出
振盪頻率逼近目標 (誤差 < 0.5%)。

為應對 sky130 複雜製程的非線性曲面與低壓死區 (Vctrl 過低不振盪), 採混合策略:
  第一階段: Differential Evolution 全域粗調 (跑 3 次不同種子, 跳出局部陷阱)
  第二階段: Nelder-Mead 單純形局部精微調適
不收斂的死區由 eda_control 安全護欄攔截, 並回傳極大懲罰 Loss=99999 迫使避開。
"""

import sys
import numpy as np
from scipy.optimize import differential_evolution, minimize

import eda_control as eda
from eda_control import CIRCUITS

sys.stdout.reconfigure(encoding="utf-8")

PENALTY = 99999.0      # 不收斂死區懲罰分
TOL = 0.005            # 收斂門檻 0.5%


def run_vco_optimization(circuit, target=None):
    c = CIRCUITS[circuit]
    key = c["param_keys"][0]                       # 'vctrl'
    lo, hi = c["ranges"][key]
    if target is None:
        target = c["target_default"]
    tgt = target * c.get("target_scale", 1.0)      # GHz -> Hz

    cache, trace = {}, []

    def sim(vctrl):
        vr = round(float(vctrl), 5)
        if vr in cache:
            return cache[vr]
        r = eda.run_circuit(circuit, {key: vr})
        freq = r.get("freq") if r.get("ok") else None
        cache[vr] = (freq, r)
        return cache[vr]

    def loss(x):
        vctrl = float(np.clip(x[0], lo, hi))
        freq, r = sim(vctrl)
        if freq is None or freq <= 0:              # 死區 / 不收斂 -> 懲罰
            trace.append({"vctrl": vctrl, "freq": None, "loss": PENALTY})
            return PENALTY
        L = ((freq - tgt) / tgt) ** 2
        trace.append({"vctrl": vctrl, "freq": freq, "loss": L})
        return L

    # ---- 第一階段: Differential Evolution 全域粗調 x3 (小預算, 靠快取去重) ----
    best_x, best_L = None, np.inf
    for seed in (1, 2, 3):
        de = differential_evolution(loss, [(lo, hi)], seed=seed, maxiter=2,
                                    popsize=3, tol=1e-2, polish=False, init="sobol")
        if de.fun < best_L:
            best_L, best_x = de.fun, float(de.x[0])
        if best_L < (TOL * 0.5) ** 2:               # 已很接近 -> 提前結束全域階段
            break

    # ---- 第二階段: Nelder-Mead 局部精微調適 ----
    nm = minimize(loss, x0=[best_x], method="Nelder-Mead",
                  options={"xatol": 5e-5, "fatol": 1e-12, "maxiter": 30})
    if nm.fun < best_L:
        best_L, best_x = float(nm.fun), float(nm.x[0])

    best_v = float(np.clip(best_x, lo, hi))
    freq, ver = sim(best_v)

    # KVCO (dFreq/dVctrl) 局部斜率 -> 線性度指標
    dv = 0.05 * (hi - lo)
    fa, _ = sim(np.clip(best_v - dv, lo, hi))
    fb, _ = sim(np.clip(best_v + dv, lo, hi))
    kvco = ((fb - fa) / (2 * dv) / 1e9) if (fa and fb) else None   # GHz/V

    err_pct = 100 * (freq - tgt) / tgt if freq else 100.0
    converged = freq is not None and abs(err_pct) < TOL * 100

    infl = {key: {"pct": 100.0, "sign": "+"}}      # Vctrl 對 freq 正相關 (單一旋鈕)
    valid = [t for t in trace if t["freq"] is not None]
    steps = []
    for i, t in enumerate(valid[-12:], 1):         # 取最後 12 個有效評估展示
        steps.append({"step": i, "params": {key: t["vctrl"]},
                      "metrics": {"freq": t["freq"]},
                      "err_pct": 100 * (t["freq"] - tgt) / tgt,
                      "influence": infl})

    n_heal = sum(1 for t in trace if t["freq"] is None)
    final = {"params": {key: best_v}, "metrics": {"freq": freq},
             "primary": freq, "step": len(valid), "err_pct": err_pct,
             "kvco": kvco, "n_eval": len(trace), "n_heal": n_heal}

    return {"circuit": circuit, "target": target,
            "status": "converged" if converged else "best",
            "method": "vco_hybrid", "steps": steps, "final": final,
            "multi_influence": {key: [{"spec": "freq", "pct": 100.0, "sign": "+"}]}}


if __name__ == "__main__":
    import time
    tgt = float(sys.argv[1]) if len(sys.argv) > 1 else 1.8
    t0 = time.perf_counter()
    r = run_vco_optimization("ringosc_sky130", tgt)
    f = r["final"]
    print(f"目標 {tgt} GHz -> Vctrl={f['params']['vctrl']:.4f}V  "
          f"freq={f['metrics']['freq']/1e9:.4f} GHz  誤差={f['err_pct']:+.3f}%  "
          f"KVCO={f['kvco']:.3f} GHz/V  評估={f['n_eval']}次(死區{f['n_heal']})  "
          f"耗時={time.perf_counter()-t0:.1f}s  status={r['status']}")
