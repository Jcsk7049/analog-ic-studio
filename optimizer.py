# -*- coding: utf-8 -*-
"""
optimizer.py — VCO 多變量 W/L 混合優化器 (論文 VCO 任務三)

VCO 實體化後參數從 1 (Vctrl) 暴增到 14 (每顆 MOS 獨立 W/L + 偏壓群)。
單點搜尋失效, 改用具多變量處理能力的 Scipy Differential Evolution:
  第一階段: DE 全域粗調 (多線程平行評估整個族群, 跳出局部陷阱)
  第二階段: Nelder-Mead 局部精微調適
物理: 演算法協同壓低振盪管 L 至 0.15µm (最小化寄生電容), 並平衡 W
      (W 過大會 self-loading 反變慢), 兼顧驅動電流與極小寄生電容。
不收斂死區由 eda_control 安全護欄攔截 + 回傳 Loss=99999 迫使避開。
"""

import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import differential_evolution, minimize

import eda_control as eda
from eda_control import CIRCUITS

sys.stdout.reconfigure(encoding="utf-8")

PENALTY = 99999.0
TOL = 0.005            # 0.5%


# ----------------------------------------------------------------------
# 反相器 Wp/Wn 物理護欄 (電子遷移率 ~2-3x 電洞 -> 對稱反相器需 Wp = 2~3 x Wn)
# 在算法邊界硬鎖, 保證 Agent 不會跑出 NMOS>PMOS 的非物理解 (Duty 偏 50%)
# ----------------------------------------------------------------------
def _inv_pairs(keys):
    """找出反相器 P/N 配對: W_<base>p <-> W_<base>n (偏壓群 W_Mbias 等不成對, 自動排除)。"""
    bases = {k[2:-1] for k in keys if k.startswith("W_") and k.endswith("p")}
    return [(f"W_{b}p", f"W_{b}n") for b in sorted(bases) if f"W_{b}n" in keys]


def _inv_guard(keys, p, ranges=None, lo=2.0, hi=3.0):
    """把每個反相器的 Wp 夾進 [lo,hi]*Wn; 若給 ranges 再夾回參數範圍。"""
    out = dict(p)
    for wp, wn in _inv_pairs(keys):
        out[wp] = min(max(out[wp], lo * out[wn]), hi * out[wn])
        if ranges and wp in ranges:
            out[wp] = min(max(out[wp], ranges[wp][0]), ranges[wp][1])
    return out


def run_vco_optimization(circuit, target=None, max_workers=8):
    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    lo = np.array([c["ranges"][k][0] for k in keys])
    hi = np.array([c["ranges"][k][1] for k in keys])
    metric = c["metric"]
    if target is None:
        target = c["target_default"]
    tgt = target * c.get("target_scale", 1.0)

    cache, trace = {}, []

    def denorm(x):
        return lo + np.clip(np.asarray(x), 0, 1) * (hi - lo)

    def sim_freq(xn):
        key = tuple(np.round(xn, 4))
        if key in cache:
            return cache[key]
        params = _inv_guard(keys, {k: float(v) for k, v in zip(keys, denorm(xn))}, c["ranges"])
        r = eda.run_isolated(circuit, params, tag=f"de_{uuid.uuid4().hex[:10]}")
        f = r.get(metric) if r.get("ok") else None
        cache[key] = f
        return f

    def loss(xn):
        f = sim_freq(xn)
        if f is None or f <= 0:
            trace.append((None,))
            return PENALTY
        L = ((f - tgt) / tgt) ** 2
        trace.append((f, L))
        return L

    bounds = [(0.0, 1.0)] * len(keys)
    x0_norm = (np.array([c["start"][k] for k in keys]) - lo) / (hi - lo)

    # ---- 第一階段: 多變量 DE (多線程平行族群評估) ----
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        de = differential_evolution(
            loss, bounds, maxiter=12, popsize=4, tol=1e-3, mutation=(0.4, 1.2),
            recombination=0.8, polish=False, init="sobol", seed=1,
            updating="deferred", workers=ex.map)
    best_x, best_L = de.x, de.fun

    # ---- 第二階段: Nelder-Mead 局部精調 ----
    nm = minimize(loss, best_x, method="Nelder-Mead",
                  options={"xatol": 1e-3, "fatol": 1e-12, "maxiter": 120})
    if nm.fun < best_L:
        best_x, best_L = nm.x, nm.fun

    best_params = _inv_guard(keys, {k: float(v) for k, v in zip(keys, denorm(best_x))}, c["ranges"])
    freq = sim_freq(best_x)
    err_pct = 100 * (freq - tgt) / tgt if freq else 100.0
    converged = freq is not None and abs(err_pct) < TOL * 100

    # ---- 影響度: 各參數對頻率的局部敏感度 (正規化空間有限差分) ----
    base_f = freq or 0.0
    infl = {}
    g = {}
    for i, k in enumerate(keys):
        xp = np.array(best_x, float); xp[i] = min(xp[i] + 0.05, 1.0)
        fp = sim_freq(xp)
        g[k] = (fp - base_f) if fp else 0.0
    tot = sum(abs(v) for v in g.values()) or 1.0
    for k in keys:
        infl[k] = [{"spec": "freq", "pct": 100 * abs(g[k]) / tot,
                    "sign": "+" if g[k] >= 0 else "-"}]

    valid = [t for t in trace if t[0] is not None]
    n_heal = sum(1 for t in trace if t[0] is None)
    steps = [{"step": i + 1, "params": best_params,
              "metrics": {"freq": v[0]}, "err_pct": 100 * (v[0] - tgt) / tgt,
              "influence": infl}
             for i, v in enumerate(valid[-10:])]

    final = {"params": best_params, "metrics": {"freq": freq}, "primary": freq,
             "step": len(valid), "err_pct": err_pct,
             "n_eval": len(trace), "n_heal": n_heal}
    return {"circuit": circuit, "target": target,
            "status": "converged" if converged else "best",
            "method": "vco_de14", "steps": steps, "final": final,
            "multi_influence": infl}


if __name__ == "__main__":
    import time
    tgt = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    t0 = time.perf_counter()
    r = run_vco_optimization("ringosc_sky130", tgt)
    f = r["final"]
    print(f"目標 {tgt} GHz -> freq={f['metrics']['freq']/1e9:.3f} GHz "
          f"誤差={f['err_pct']:+.2f}% 評估={f['n_eval']}次(死區{f['n_heal']}) "
          f"耗時={time.perf_counter()-t0:.0f}s status={r['status']}")
    print("最佳 W/L (µm):")
    for k in CIRCUITS["ringosc_sky130"]["param_keys"]:
        print(f"  {k} = {f['params'][k]*1e6:.3f}")


# ----------------------------------------------------------------------
# 通用多變量混合優化 (DE 全域 + Nelder-Mead 局部) — 10+ 維 (任務三)
# ----------------------------------------------------------------------
import uuid
from concurrent.futures import ThreadPoolExecutor


def _mv_loss(circuit, r, tgt):
    c = CIRCUITS[circuit]
    if not r.get("ok"):
        return PENALTY
    if c["objective"] == "target":
        v = r.get(c["metric"])
        if v is None:
            return PENALTY
        L = ((v - tgt) / (abs(tgt) + 1e-12)) ** 2
        if c.get("pm_constraint"):
            pm = r.get("pm")
            if pm is None or not (0 < pm < 90):       # 非物理 (相位 wrap) -> 重罰避開
                L += 10.0
            elif pm < 45:
                L += 4.0 * ((45 - pm) / 45) ** 2
        return L
    tc = r.get("tc")                                  # minimize
    return PENALTY if tc is None else (tc / 50.0) ** 2


def run_multivar(circuit, target=None, workers=8):
    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    bounds = [c["ranges"][k] for k in keys]
    if target is None:
        target = c["target_default"]
    tgt = target * c.get("target_scale", 1.0)
    is_vco = c["template"].startswith("vco")          # 瞬態慢 -> 小預算

    def obj(x):
        p = _inv_guard(keys, {k: float(x[i]) for i, k in enumerate(keys)}, c["ranges"])
        return _mv_loss(circuit, eda.run_isolated(circuit, p, uuid.uuid4().hex[:8]), tgt)

    maxiter, popsize = (6, 4) if is_vco else (25, 12)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        de = differential_evolution(obj, bounds, maxiter=maxiter, popsize=popsize,
                                    tol=1e-3, seed=1, polish=False,
                                    updating="deferred", workers=ex.map)
    nm = minimize(obj, de.x, method="Nelder-Mead",
                  options={"maxiter": 40 if is_vco else 120, "fatol": 1e-9, "xatol": 1e-9})
    bx = nm.x if nm.fun <= de.fun else de.x
    best = _inv_guard(keys, {k: float(np.clip(bx[i], *bounds[i])) for i, k in enumerate(keys)}, c["ranges"])
    ver = eda.run_circuit(circuit, best)
    prim = ver.get(c["metric"])

    # 多維影響度: 主指標對各參數梯度 (歸一化)
    g = {}
    base = prim if prim is not None else 0.0
    for i, k in enumerate(keys):
        lo, hi = bounds[i]
        p2 = dict(best); p2[k] = min(best[k] + 0.04 * (hi - lo), hi)
        v2 = eda.run_isolated(circuit, p2, uuid.uuid4().hex[:8]).get(c["metric"])
        g[k] = (v2 - base) if (v2 is not None) else 0.0
    tot = sum(abs(x) for x in g.values()) or 1.0
    multi = {k: [{"spec": c["metric"], "pct": 100 * abs(g[k]) / tot,
                  "sign": "+" if g[k] >= 0 else "-"}] for k in keys}

    err = (100 * (prim - tgt) / (abs(tgt) + 1e-12)) if (prim is not None and c["objective"] == "target") else 0.0
    ok = ver.get("ok") and (abs(err) < 1.0 if c["objective"] == "target" else (ver.get("tc") or 99) < 30)
    final = {"params": best, "metrics": {k: ver.get(k) for k in ("gain", "pm", "ugf", "tc", "vref", "freq") if ver.get(k) is not None},
             "primary": prim, "step": int(de.nit), "err_pct": err}
    return {"circuit": circuit, "target": target, "status": "converged" if ok else "best",
            "method": "multivar", "steps": [{"step": 1, "params": best,
            "metrics": final["metrics"], "err_pct": err, "influence": {k: {"pct": multi[k][0]["pct"], "sign": multi[k][0]["sign"]} for k in keys}}],
            "final": final, "multi_influence": multi, "dims": len(keys)}
