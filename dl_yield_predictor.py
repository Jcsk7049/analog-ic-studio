# -*- coding: utf-8 -*-
"""
dl_yield_predictor.py — 萬點級深度學習良率預測引擎 (任務一)

痛點: 真實產線評估 Six-Sigma 極端良率需 >10,000 次模擬, 傳統 ngspice 蒙地卡羅
      (50~100 點要數秒~數十秒) 在萬點時計算成本爆炸。
解法: 用已訓練的 PyTorch 替代模型 (dl_surrogate), 一次向量化推論 10,000 組
      帶製程高斯變異的參數矩陣 -> 毫秒級得到 Mean / σ / Yield / Sigma 等級。

對外介面:
    yield_predict(circuit, params, target, n=10000, sigma=0.10) -> dict
"""

import os
import sys
import time
import numpy as np

import eda_control as eda
from eda_control import CIRCUITS
import mc_yield

sys.stdout.reconfigure(encoding="utf-8")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_model(circuit):
    """若該電路尚無替代模型, 動態用 ngspice 累積資料即時訓練一個。"""
    import dl_surrogate as dl
    pth = os.path.join(BASE_DIR, "data", f"surrogate_{circuit}.pth")
    if not os.path.exists(pth):
        dl.train(circuit, n=300, epochs=400, regen=True, verbose=False)
        return True
    return False


def yield_predict(circuit, params, target=None, n=10000, sigma=0.10, seed=0):
    import dl_surrogate as dl
    trained_now = _ensure_model(circuit)
    sur = dl.Surrogate(circuit)

    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    metric = c["metric"]
    if target is None:
        target = c["target_default"]
    tgt = target * c.get("target_scale", 1.0)

    base = np.array([params[k] for k in keys])
    rng = np.random.RandomState(seed)

    t0 = time.perf_counter()
    # 10,000 組製程高斯變異 (σ 代表元件容差) -> 一次推論
    M = mc_yield._clamp_matrix(circuit, base * (1 + sigma * rng.randn(n, len(keys))))
    pred = sur.predict(M)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    cols = {t: pred[:, j] for j, t in enumerate(sur.targets)}
    v = cols[metric]
    good = mc_yield._good_mask(circuit, cols, tgt)
    n_good = int(np.count_nonzero(good))

    # 製程能力 / Sigma 等級: 規格邊界距均值有幾個 σ
    std = float(v.std())
    if c["objective"] == "target":
        tol = mc_yield.YIELD_TOL * abs(tgt)             # 規格半寬
        sigma_level = float(tol / (std + 1e-12))
    else:                                                # bandgap: 距 TC 上限
        sigma_level = float((mc_yield.TC_SPEC - v.mean()) / (std + 1e-12))

    return {
        "circuit": circuit, "metric": metric, "method": "DNN-10k",
        "n": n, "n_ok": n, "sigma_pct": sigma * 100,
        "mean": float(v.mean()), "std": std,
        "min": float(v.min()), "max": float(v.max()),
        "cv_pct": float(std / (abs(v.mean()) + 1e-12) * 100),
        "yield_pct": float(100.0 * n_good / n),
        "sigma_level": sigma_level,
        "infer_ms": infer_ms,
        "trained_now": trained_now,
        "surrogate_r2": sur.r2,
        "samples": v,                                    # numpy, 供繪圖 (web_app 用後移除)
        "target": target,
    }


if __name__ == "__main__":
    p = {"w_diff": 8.59e-6, "w_stage2": 26e-6, "r_bias": 62e3}
    r = yield_predict("opa", p, 60, n=10000)
    print(f"萬點 DNN: {r['n']} 點 推論={r['infer_ms']:.1f}ms "
          f"Mean={r['mean']:.2f} σ={r['std']:.3f} Yield={r['yield_pct']:.2f}% "
          f"≈{r['sigma_level']:.1f}σ")
