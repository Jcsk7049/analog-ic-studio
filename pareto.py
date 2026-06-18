# -*- coding: utf-8 -*-
"""
pareto.py — 多目標帕雷托最佳解推薦 (任務二)

真實類比設計各指標存在 Trade-off (調大 W 提升 Gain 但增功耗 Power/面積 Area)。
本模組用替代模型在參數空間快速搜尋, 在「滿足規格」的可行解中, 找出三組
位於帕雷托前緣的代表設計:
    套餐 A 高效能優先 (Performance): 衝極限頻寬/增益, PM 剛過安全線
    套餐 B 極致穩定優先 (Stability) : 達標且 PM 最佳化 (>60°) / 變異最不敏感
    套餐 C 低功耗平衡 (Balanced)    : 功耗 + 面積最小

功耗/面積採一階解析模型 (快速、可向量化), 最後選定的 3 組以 ngspice 校正。

對外介面: pareto_packages(circuit, target) -> [pkgA, pkgB, pkgC]
"""

import os
import sys
import math
import numpy as np

import eda_control as eda
from eda_control import CIRCUITS
import mc_yield

sys.stdout.reconfigure(encoding="utf-8")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# 一階解析: 面積 (ΣW·L) 與 功耗
# ----------------------------------------------------------------------
def area_um2(circuit, p):
    # 實體化版 (有 W_x/L_x): 真實矽面積 Σ W·L (µm²)
    wl = sum(p[k] * p["L_" + k[2:]] for k in p if k.startswith("W_") and "L_" + k[2:] in p)
    if wl:
        return wl * 1e12
    # 舊參數電路回退
    if circuit.startswith("opa"):
        return (2 * p.get("w_diff", 0) + p.get("w_stage2", 0) + 220e-6) * 1e6 * 0.7
    if circuit.startswith("ringosc"):
        return 5 * (p.get("w_p", 0) + p.get("w_n", 0)) * 1e6 * 0.5
    return p.get("n_bjt", 1) * 11.6 + p.get("r_trim", 0) * 1e-3


def power_uw(circuit, p, m):
    # 舊版有電源電流相關參數 -> 物理式; 實體化版 -> ΣW 代理 (∝ 電流容量)
    if "r_bias" in p:
        return 3.3 * max((3.3 - 0.9) / p["r_bias"], 1e-12) * 16 * 1e6
    if "r_trim" in p and not any(k.startswith("W_") for k in p):
        return 3.3 * 3 * 0.0259 * math.log(max(p["n_bjt"], 1.1)) / p["r_trim"] * 1e6
    if "w_p" in p:
        return 5 * 5e-15 * 1.8 * 1.8 * (m.get("freq") or 0) * 1e6
    return sum(v for k, v in p.items() if k.startswith("W_")) * 1e6   # ΣW (µm) 代理


# ----------------------------------------------------------------------
# 帕雷托套餐選取
# ----------------------------------------------------------------------
def pareto_packages(circuit, target=None, pop=120000, seed=0):
    import dl_surrogate as dl
    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    metric = c["metric"]
    if target is None:
        target = c["target_default"]
    tgt = target * c.get("target_scale", 1.0)

    lo = np.array([c["ranges"][k][0] for k in keys])
    hi = np.array([c["ranges"][k][1] for k in keys])
    rng = np.random.RandomState(seed)

    # 有效替代模型 (參數維度相符) -> 百萬級預測; 否則 (實體化新電路) -> 多線程 ngspice
    sur = None
    pth = os.path.join(BASE_DIR, "data", f"surrogate_{circuit}.pth")
    if os.path.exists(pth):
        try:
            s = dl.Surrogate(circuit)
            if len(s.features) == len(keys):
                sur = s
        except Exception:
            sur = None
    if sur is None:
        pop = min(pop, 600)                              # ngspice 路徑: 限制候選數
    X = lo + rng.random((pop, len(keys))) * (hi - lo)
    if sur is not None:
        pred = sur.predict(X)
        col = {t: pred[:, j] for j, t in enumerate(sur.targets)}
    else:
        import uuid
        from concurrent.futures import ThreadPoolExecutor
        def _ev(row):
            return eda.run_isolated(circuit, {k: float(row[i]) for i, k in enumerate(keys)},
                                    uuid.uuid4().hex[:8])
        with ThreadPoolExecutor(max_workers=8) as ex:
            recs = list(ex.map(_ev, X))
        col = {t: np.array([(r.get(t) if r.get("ok") and r.get(t) is not None else np.nan)
                            for r in recs]) for t in ("gain", "pm", "ugf", "tc", "vref", "freq")}

    # 可行解遮罩 (滿足規格)
    if c["objective"] == "target":
        feas = np.abs(col[metric] - tgt) / (abs(tgt) + 1e-12) <= 0.05
        if c.get("pm_constraint") and "pm" in col:
            feas = feas & (col["pm"] >= 45.0)
    else:                                                # bandgap: 低溫漂
        feas = np.abs(col["tc"]) < mc_yield.TC_SPEC
    idx = np.where(feas)[0]
    if len(idx) < 3:                                     # 放寬
        order = np.argsort(np.abs(col[metric] - tgt)) if c["objective"] == "target" else np.argsort(np.abs(col["tc"]))
        idx = order[:max(2000, pop // 50)]

    # DNN 候選的功耗/面積
    params_list = [{k: float(X[i, j]) for j, k in enumerate(keys)} for i in idx]
    _mt = list(sur.targets) if sur is not None else list(col.keys())
    mets_list = [{t: float(col[t][i]) for t in _mt} for i in idx]
    area = np.array([area_um2(circuit, p) for p in params_list])
    powr = np.array([power_uw(circuit, params_list[k], mets_list[k]) for k in range(len(idx))])

    def _topk(arr, k, largest):
        o = np.argsort(arr)
        return list(o[-k:] if largest else o[:k])

    # 蒐集各準則 top-K 候選 (聯集), 之後逐一 ngspice 校正
    # 快速電路 ngspice 便宜 -> 驗證更多候選使帕雷托前緣更豐富; sky130 慢 -> 少量
    K = 15 if "sky130" in circuit else 45
    cset = set()
    cset |= set(_topk(area + powr, K, False))             # 低成本
    if circuit.startswith("opa"):
        cset |= set(_topk(np.array([m.get("ugf", 0) for m in mets_list]), K, True))
        cset |= set(_topk(np.array([min(m["pm"], 85) for m in mets_list]), K, True))
    elif circuit.startswith("ringosc"):
        cset |= set(_topk(area, K, False)) | set(_topk(powr, K, False))
        cset |= set(_topk(np.abs(np.array([m["freq"] for m in mets_list]) - tgt), K, False))
    else:
        cset |= set(_topk(np.abs(np.array([m["tc"] for m in mets_list])), 2 * K, False))

    # ngspice 校正候選池, 重建真實可行集
    recs = []
    for li in cset:
        p = params_list[li]
        ver = eda.run_circuit(circuit, p)
        if not ver.get("ok"):
            continue
        m = {k: ver.get(k) for k in ("gain", "pm", "ugf", "tc", "vref", "freq") if ver.get(k) is not None}
        feasible = True
        if c["objective"] == "target":
            feasible = abs(m.get(metric, -9e9) - tgt) / (abs(tgt) + 1e-12) <= 0.05
            if c.get("pm_constraint"):
                pm = m.get("pm")
                feasible = feasible and (pm is not None and 45.0 <= pm <= 90.0)  # 排除量測假值
        else:
            feasible = m.get("tc") is not None and abs(m["tc"]) < mc_yield.TC_SPEC
        recs.append({"p": p, "m": m, "feasible": feasible,
                     "power": power_uw(circuit, p, ver), "area": area_um2(circuit, p)})

    feas = [r for r in recs if r["feasible"]] or recs    # 若全不可行則放寬用全部

    # 正規化功耗+面積成本 (兩者量級不同, 先各自 0~1 再相加)
    pw = np.array([r["power"] for r in feas]); ar = np.array([r["area"] for r in feas])
    def cost(r):
        return ((r["power"] - pw.min()) / (pw.ptp() + 1e-12)
                + (r["area"] - ar.min()) / (ar.ptp() + 1e-12))

    # --- 從真實校正過的可行解選 A / B / C (排除已選, 確保互異) ---
    chosen = []
    def pick(key, largest=False):
        avail = [r for r in feas if all(r is not x for x in chosen)] or feas
        r = (max if largest else min)(avail, key=key)
        chosen.append(r)
        return r
    if circuit.startswith("opa"):
        A = pick(lambda r: r["m"].get("ugf", 0), True)            # 高效能: 最大頻寬 (PM 自然較低)
        B = pick(lambda r: r["m"].get("pm", 0), True)             # 極致穩定: 最大 PM (>60)
        C = pick(cost)                                            # 低功耗平衡
    elif circuit.startswith("ringosc"):
        A = pick(lambda r: r["m"]["freq"], True)                 # 高效能: 最快頻率
        B = pick(lambda r: abs(r["m"]["freq"] - tgt))            # 穩定: 最貼目標頻率
        C = pick(lambda r: r["power"])                           # 低功耗平衡: 最低動態功耗
    else:
        A = pick(lambda r: abs(r["m"]["tc"]))                    # 高效能: 最低溫漂
        C = pick(cost)                                           # 低功耗平衡 (先選以免與 B 撞)
        B = pick(lambda r: abs(r["m"]["tc"]))                    # 穩定: 次低溫漂

    labels = [("A", "高效能優先", "Performance", A),
              ("B", "極致穩定優先", "Stability", B),
              ("C", "低功耗平衡", "Balanced", C)]
    pkgs = [{"id": pid, "label_zh": zh, "label_en": en,
             "params": r["p"], "metrics": r["m"],
             "power_uw": float(r["power"]), "area_um2": float(r["area"]),
             "feasible": bool(r["feasible"])}
            for (pid, zh, en, r) in labels]
    return {"circuit": circuit, "target": target, "packages": pkgs}


if __name__ == "__main__":
    import json
    r = pareto_packages("opa", 60)
    for pk in r["packages"]:
        m = pk["metrics"]
        print(f"套餐{pk['id']} {pk['label_zh']:8} gain={m.get('gain'):.2f} pm={m.get('pm'):.1f} "
              f"ugf={m.get('ugf',0)/1e6:.1f}MHz power={pk['power_uw']:.1f}µW area={pk['area_um2']:.1f}µm²")
