# -*- coding: utf-8 -*-
"""
mc_yield.py — 製程 Corner 變異與蒙地卡羅良率預測 (任務三)

真實晶圓代工會有隨機物理變異, 同一設計做出來規格會散開。
本模組在「已收斂的最佳參數」附近注入高斯製程變異, 統計:
    平均值 Mean / 標準差 σ / 規格達標良率 Yield %

加速: 若該電路已有 DNN 替代模型, 蒙地卡羅直接在 DNN 上向量化預測 (瞬間完成),
否則退回逐次 ngspice 模擬。

對外介面:
    monte_carlo(circuit, params, target, n=50, sigma=0.10) -> dict
    corner_analysis(circuit, params, target) -> dict   (TT / FF / SS)
"""

import os
import sys
import numpy as np

import eda_control as eda
from eda_control import CIRCUITS

sys.stdout.reconfigure(encoding="utf-8")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YIELD_TOL = 0.05      # target 模式: 主指標落在 ±5% 視為良品
TC_SPEC = 50.0        # bandgap: TC < 50 ppm/°C 視為良品
PM_SPEC = 45.0        # opa: 相位裕度需 >= 45°

# --- 變異模型 (Pelgrom): 良率 = 全域製程 + 隨機失配, 失配 σ ∝ 1/sqrt(W·L) ---
AVT_REL = 0.025       # 1µm² 元件的等效 Vth 失配相對 σ (~AVT/Vov); 元件越大越匹配
SIGMA_PROC = 0.03     # 全域製程變異 (ΔL/Δtox/ΔVth, 同一晶片所有元件共用)
L_ASSUMED = 0.5e-6    # 範本將 L 焊死 (未列為參數) 的元件假設通道長


def _mismatch_sigma(circuit, params):
    """每個參數的隨機失配相對 σ。寬度型 ∝ 1/sqrt(W·L) (Pelgrom: 大元件更匹配);
       長度型與被動元件 (R/N) 不計隨機失配, 僅受全域製程。回傳與 param_keys 同序陣列。"""
    keys = CIRCUITS[circuit]["param_keys"]
    sig = []
    for k in keys:
        if not (k.startswith("W_") or k.startswith("w_")):
            sig.append(0.0); continue
        base = k[2:]
        lk = next((x for x in keys if x in (f"L_{base}", f"l_{base}")), None)
        L = params[lk] if lk else L_ASSUMED
        area_um2 = max(params[k] * L * 1e12, 1e-3)        # W·L -> µm²
        sig.append(AVT_REL / (area_um2 ** 0.5))
    return np.array(sig)


def _has_surrogate(circuit):
    # 僅 sky130 精準模式用 DNN 加速 (ngspice 慢); 快速電路 ngspice 本就快又準
    return (CIRCUITS[circuit].get("model") == "sky130"
            and os.path.exists(os.path.join(BASE_DIR, "data", f"surrogate_{circuit}.pth")))


def _clamp_matrix(circuit, M):
    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    lo = np.array([c["ranges"][k][0] for k in keys])
    hi = np.array([c["ranges"][k][1] for k in keys])
    return np.clip(M, lo, hi)


def _good_mask(circuit, metrics_cols, target):
    """向量化良品判定。metrics_cols: dict 名稱->陣列。"""
    c = CIRCUITS[circuit]
    if c["objective"] == "target":
        val = metrics_cols[c["metric"]]
        ok = np.abs(val - target) / (abs(target) + 1e-12) <= YIELD_TOL
        if c.get("pm_constraint") and "pm" in metrics_cols:
            ok = ok & (metrics_cols["pm"] >= PM_SPEC)
        return ok
    else:
        return np.abs(metrics_cols["tc"]) < TC_SPEC


def monte_carlo(circuit, params, target=None, n=50, sigma=0.10, seed=0, engine="auto"):
    """對設計參數注入 ±sigma 高斯變異 (σ 代表製造容差), 跑 n 次, 統計良率。"""
    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    metric = c["metric"]
    if target is None:
        target = c["target_default"]
    tgt = target * c.get("target_scale", 1.0)

    rng = np.random.RandomState(seed)
    base = np.array([params[k] for k in keys])
    # Pelgrom 雙成分: 全域製程 (n×1, 全元件共用) + 隨機失配 (n×K, 各自獨立, σ∝1/√WL)
    mm = _mismatch_sigma(circuit, params)
    scale = sigma / 0.10                                   # 沿用 sigma 當整體變異倍率 (預設 1.0)
    proc = SIGMA_PROC * scale * rng.randn(n, 1)
    dev = (mm * scale) * rng.randn(n, len(keys))
    M = _clamp_matrix(circuit, base * (1 + proc + dev))

    use_sur = (engine == "surrogate") or (engine == "auto" and _has_surrogate(circuit))
    cols = {}
    if use_sur:
        import dl_surrogate as dl
        sur = dl.Surrogate(circuit)
        pred = sur.predict(M)                              # (n, t) 瞬間預測
        for j, t in enumerate(sur.targets):
            cols[t] = pred[:, j]
        method = "DNN"
    else:
        recs = [eda.run_circuit(circuit, {k: M[i, j] for j, k in enumerate(keys)})
                for i in range(n)]
        for t in ("gain", "pm", "ugf", "tc", "vref", "freq"):
            vals = [r.get(t) for r in recs if r.get("ok") and r.get(t) is not None]
            if vals:
                cols[t] = np.array(vals)
        method = "ngspice"

    if metric not in cols or len(cols[metric]) == 0:
        return {"error": "全部樣本不收斂"}
    v = cols[metric]
    good = _good_mask(circuit, cols, tgt)
    return {
        "circuit": circuit, "metric": metric, "method": method,
        "n": n, "n_ok": int(len(v)), "sigma_pct": sigma * 100,
        "mean": float(v.mean()), "std": float(v.std()),
        "min": float(v.min()), "max": float(v.max()),
        "cv_pct": float(v.std() / (abs(v.mean()) + 1e-12) * 100),
        "yield_pct": float(100.0 * np.count_nonzero(good) / n),
        "samples": [float(x) for x in v],
        "target": target,
    }


def corner_analysis(circuit, params, target=None):
    """三製程角落 TT / FF / SS (參數級一階近似: FF 元件偏強, SS 偏弱)。"""
    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    metric = c["metric"]
    corners = {"TT": 0.0, "FF": +0.10, "SS": -0.10}
    use_sur = _has_surrogate(circuit)
    sur = None
    if use_sur:
        import dl_surrogate as dl
        sur = dl.Surrogate(circuit)

    out = {}
    for name, d in corners.items():
        p = _clamp_matrix(circuit, np.array([params[k] for k in keys]) * (1 + d)).reshape(1, -1)
        if use_sur:
            y = sur.predict(p)[0]
            row = {t: float(y[j]) for j, t in enumerate(sur.targets)}
            row["ok"] = True
        else:
            m = eda.run_circuit(circuit, {k: p[0, j] for j, k in enumerate(keys)})
            row = {metric: m.get(metric), "pm": m.get("pm"), "vref": m.get("vref"), "ok": bool(m.get("ok"))}
        out[name] = row
    return {"circuit": circuit, "metric": metric, "corners": out, "target": target}


if __name__ == "__main__":
    p = {"w_diff": 8.59e-6, "w_stage2": 26e-6, "r_bias": 62e3}
    r = monte_carlo("opa", p, 60, n=50)
    print(f"MC OPA@60: method={r['method']} Mean={r['mean']:.2f} σ={r['std']:.3f} 良率={r['yield_pct']:.0f}%")
