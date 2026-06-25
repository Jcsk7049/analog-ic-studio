# -*- coding: utf-8 -*-
"""
ring_calc.py — 電流飢餓環形振盪器 (Ring VCO) 設計公式檢查器(純解析,零相依)

Ring VCO 沒有 LC tank:N 級反相器串成環,靠每級延遲決定頻率,以控制電流 I 調頻。
公式(電流飢餓):每級延遲 td = C·Vdd/I,半週期 = N·td → f0 = I/(2·N·C·Vdd)。
與 LC 互補:調諧範圍寬、面積小,但相位雜訊差。

主介面: analyze_ring_vco(design, spec=None) -> dict

design (SI):
  N           : 反相器級數(單端環需『奇數』才會振盪)
  C_load_F    : 每級負載電容
  I_A         : 控制/充電電流(設定頻率)
  Vdd_V       : 電源 (預設 1.8)
  Vswing_V    : 擺幅 (預設 = Vdd)
  I_min_A/I_max_A : 調諧電流範圍 (算 FTR/KVCO, 選填)
  Vctrl_span_V    : 控制電壓跨距 (算 KVCO, 預設 1.0)
  F_noise     : 雜訊因子 (環形較高, 預設 6)
spec:
  f0_GHz, ftr_pct_min, pn_dbc_1MHz_max
"""
import math

KT = 1.380649e-23 * 300.0


def _f_ring(I, N, C, V):
    return I / (2 * N * C * V) if (N > 0 and C > 0 and V > 0) else 0.0


def analyze_ring_vco(design, spec=None):
    d = dict(design)
    spec = dict(spec or {})
    findings = []

    def add(level, item, msg, suggest=None):
        findings.append({"level": level, "item": item, "msg": msg, "suggest": suggest})

    N = int(d.get("N", 0))
    C = d.get("C_load_F")
    I = d.get("I_A")
    if not (N and C and I):
        return {"error": "需 N(級數), C_load_F(每級電容), I_A(控制電流)"}
    Vdd = d.get("Vdd_V", 1.8)
    Vsw = d.get("Vswing_V", Vdd)

    f0 = _f_ring(I, N, C, Vsw)
    Imin = d.get("I_min_A"); Imax = d.get("I_max_A")
    f_min = _f_ring(Imin, N, C, Vsw) if Imin else None
    f_max = _f_ring(Imax, N, C, Vsw) if Imax else None
    ftr = (100 * (f_max - f_min) / (0.5 * (f_max + f_min))) if (f_min and f_max and f_max > f_min) else None
    vspan = d.get("Vctrl_span_V", 1.0)
    kvco = ((f_max - f_min) / vspan) if (f_min and f_max) else None
    Psig = N * I * Vdd                                       # 粗估總功耗
    F = d.get("F_noise", 6.0)
    # 環形相位雜訊 (Leeson, Q≈1, F 較高) — 估算, 環形本就比 LC 差 ~20dB
    df = 1e6
    pn_1m = 10 * math.log10((2 * F * KT / Psig) * (1 + (f0 / (2 * 1.0 * df)) ** 2)) if (Psig and f0) else None
    fom = (pn_1m - 20 * math.log10(f0 / df) + 10 * math.log10(Psig / 1e-3)) if pn_1m else None

    metrics = {
        "f0_GHz": f0 / 1e9, "stages_N": N,
        "f_min_GHz": (f_min / 1e9) if f_min else None, "f_max_GHz": (f_max / 1e9) if f_max else None,
        "ftr_pct": ftr, "kvco_MHz_V": (kvco / 1e6) if kvco else None,
        "Psig_mW": Psig * 1e3, "pn_1MHz_dBc": pn_1m, "fom_dBc": fom, "td_ps": (1 / (2 * N * f0) * 1e12) if f0 else None,
    }

    # --- 糾錯 ---
    if N % 2 == 0:
        add("error", "級數", f"N={N} 為偶數 → 單端環不會振盪!", "改為奇數級(3/5/7…);若差動環可偶數但需正確接線")
    else:
        add("ok", "級數", f"N={N} 為奇數,可振盪")

    if "f0_GHz" in spec:
        tgt = spec["f0_GHz"]
        if abs(f0 / 1e9 - tgt) / tgt > 0.05:
            I_need = tgt * 1e9 * 2 * N * C * Vsw
            add("error", "f0", f"頻率 {f0/1e9:.2f}GHz 偏離目標 {tgt}GHz",
                f"f0=I/(2N·C·Vdd):控制電流需 ≈{I_need*1e6:.0f}µA(目前 {I*1e6:.0f}µA),或減級數/降電容")
        else:
            add("ok", "f0", f"頻率 {f0/1e9:.2f}GHz 達標")

    if "ftr_pct_min" in spec and ftr is not None:
        if ftr < spec["ftr_pct_min"]:
            add("warn", "FTR", f"調諧範圍 {ftr:.0f}% < 需求 {spec['ftr_pct_min']}%", "加大控制電流範圍 I_max/I_min 比值")
        else:
            add("ok", "FTR", f"調諧範圍 {ftr:.0f}% 達標(環形通常很寬)")

    if "pn_dbc_1MHz_max" in spec and pn_1m is not None:
        if pn_1m > spec["pn_dbc_1MHz_max"]:
            add("warn", "相位雜訊", f"PN@1MHz {pn_1m:.0f} 差於需求 {spec['pn_dbc_1MHz_max']}dBc/Hz",
                "環形雜訊本就高:加大功耗/級數,或關鍵應用改用 LC VCO")
        else:
            add("ok", "相位雜訊", f"PN@1MHz {pn_1m:.0f}dBc/Hz 達標")

    return {"topology": "ring-vco", "metrics": metrics, "findings": findings,
            "ok": not any(f["level"] == "error" for f in findings)}


def demo():
    # 5 級環形 VCO, 每級 20fF, 控制電流 200µA
    d = {"N": 5, "C_load_F": 20e-15, "I_A": 200e-6, "Vdd_V": 1.8,
         "I_min_A": 50e-6, "I_max_A": 400e-6}
    r = analyze_ring_vco(d, {"f0_GHz": 1.0, "ftr_pct_min": 50, "pn_dbc_1MHz_max": -90})
    m = r["metrics"]
    assert m["f0_GHz"] > 0 and m["stages_N"] == 5
    assert m["ftr_pct"] and m["ftr_pct"] > 0
    print("demo OK")
    print(f"  f0={m['f0_GHz']:.3f}GHz  FTR={m['ftr_pct']:.0f}%  KVCO={m['kvco_MHz_V']:.0f}MHz/V  "
          f"PN@1MHz={m['pn_1MHz_dBc']:.0f}  td={m['td_ps']:.1f}ps")
    for f in r["findings"]:
        tag = {"ok": "✓", "info": "·", "warn": "!", "error": "✗"}[f["level"]]
        print(f"  [{tag}] {f['item']}: {f['msg']}" + (f"  → {f['suggest']}" if f["suggest"] else ""))
    # 偶數級應被擋
    r2 = analyze_ring_vco({"N": 4, "C_load_F": 20e-15, "I_A": 200e-6}, {})
    assert not r2["ok"], "偶數級應判定不振盪"
    print("  偶數級 N=4 -> 正確判定不會振盪 ✓")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    demo()
