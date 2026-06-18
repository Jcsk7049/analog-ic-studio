# -*- coding: utf-8 -*-
"""
pvt_scanner.py — 製程/電壓/溫度 三溫特性掃描 (PVT, 論文任務二)

任何類比 IC 論文都必須通過 PVT 全環境驗證才能畢業。本模組自動交叉:
    P (Process Corner): tt / ff / ss
    V (Voltage)       : 額定電源 ±10%
    T (Temperature)   : -40 / 25 / 125 °C
共 3×3×3 = 27 種極端環境組合, 多線程批量跑 ngspice, 疊加呈現魯棒性。

對外介面: pvt_scan(circuit, params, max_workers=8) -> {curves, metric, robustness}
僅支援 sky130 精準電路 (有 ff/ss 真實角落): opa_sky130 (Bode) / ringosc_sky130 (波形)
"""

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import eda_control as eda
from eda_control import CIRCUITS

sys.stdout.reconfigure(encoding="utf-8")

# 每電路: (電源替換 old, new 模板, 額定電壓, 匯出向量, 波形型態)
PVT_CFG = {
    "opa_sky130":     ("VDD_V = 1.8", "VDD_V = {v}", 1.8, "vdb(out) vp(out)", "bode"),
    "ringosc_sky130": ("Vdd   vdd 0 1.8", "Vdd   vdd 0 {v}", 1.8, "v(Vout)", "wave"),
}
CORNERS = ["tt", "ff", "ss"]
TEMPS = [-40, 25, 125]


def pvt_scan(circuit, params=None, max_workers=8):
    if circuit not in PVT_CFG:
        raise ValueError(f"{circuit} 不支援 PVT (需 sky130 精準模式)")
    c = CIRCUITS[circuit]
    params = params or dict(c["start"])
    metric = c["metric"]
    old, newt, vnom, dump_vec, kind = PVT_CFG[circuit]
    volts = [round(vnom * 0.9, 3), vnom, round(vnom * 1.1, 3)]

    combos = [(p, v, t) for p in CORNERS for v in volts for t in TEMPS]   # 27

    def _run(combo):
        p, v, t = combo
        replaces = [('.lib.spice" tt', f'.lib.spice" {p}'),
                    (old, newt.replace("{v}", str(v)))]
        r = eda.run_isolated(circuit, params, tag=f"pvt_{p}_{v}_{t}".replace(".", "p"),
                             inject=f".options temp={t}", dump_vec=dump_vec, replaces=replaces)
        rows = r.get("_wave", [])
        curve = {"label": f"{p.upper()} {v}V {t}°C", "p": p, "v": v, "t": t,
                 "metric": r.get(metric) if r.get("ok") else None, "ok": bool(r.get("ok"))}
        if kind == "bode":
            curve["freq"] = [x[0] for x in rows if len(x) >= 2]
            curve["mag"] = [x[1] for x in rows if len(x) >= 2]
        else:
            curve["t_ns"] = [x[0] * 1e9 for x in rows if len(x) >= 2]
            curve["v"] = [x[1] for x in rows if len(x) >= 2]
        return curve

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        curves = list(ex.map(_run, combos))

    mvals = [c0["metric"] for c0 in curves if c0["metric"] is not None]
    sc = 1e9 if metric == "freq" else 1.0
    robustness = None
    if mvals:
        mv = np.array(mvals) / sc
        robustness = {"mean": float(mv.mean()), "std": float(mv.std()),
                      "min": float(mv.min()), "max": float(mv.max()),
                      "spread_pct": float((mv.max() - mv.min()) / (abs(mv.mean()) + 1e-12) * 100)}
    return {"circuit": circuit, "kind": kind, "metric": metric,
            "curves": curves, "n_ok": len(mvals), "total": 27, "robustness": robustness}


if __name__ == "__main__":
    import time
    ckt = sys.argv[1] if len(sys.argv) > 1 else "opa_sky130"
    t0 = time.perf_counter()
    r = pvt_scan(ckt)
    rb = r["robustness"]
    print(f"[{ckt}] PVT 27 組: {r['n_ok']}/27 有效, 耗時 {time.perf_counter()-t0:.0f}s")
    if rb:
        u = "GHz" if r["metric"] == "freq" else ("dB" if r["metric"] == "gain" else "")
        print(f"  {r['metric']} 跨 PVT: mean={rb['mean']:.3f} σ={rb['std']:.3f} "
              f"範圍[{rb['min']:.3f},{rb['max']:.3f}]{u} 變異={rb['spread_pct']:.1f}%")
