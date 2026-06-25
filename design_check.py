# -*- coding: utf-8 -*-
"""
design_check.py — 類比設計檢查統一入口(零相依,零 token)

把 VCO / OPA / Bandgap 三顆設計公式檢查器收斂成一個函式,供夥伴 AI 工具
一行呼叫:讀完圖萃取參數後 -> analyze(topology, params, spec) -> 算性能+糾錯+給目標。
數值計算全在程式(不丟 LLM, 省 token, 不外送 PDK)。

    from design_check import analyze
    r = analyze("opa", params, spec)
    # r = {topology, metrics{...}, findings[{level,item,msg,suggest}], ok}

支援 topology(大小寫/別名皆可):
    "vco" / "lc-vco" / "lcvco"     -> LC-VCO          (vco_calc)
    "opa" / "opamp" / "op-amp"     -> 兩級米勒 OPA    (opa_calc)
    "bandgap" / "bg" / "reference" -> 帶隙基準        (bandgap_calc)
"""
from vco_calc import analyze_lc_vco
from opa_calc import analyze_two_stage_opa
from bandgap_calc import analyze_bandgap
from ring_calc import analyze_ring_vco

_VCO = {"vco", "lc-vco", "lcvco", "lc_vco"}
_RING = {"ring", "ring-vco", "ringvco", "ring_vco", "ro", "ring_oscillator"}
_OPA = {"opa", "opamp", "op-amp", "two-stage-opa", "amplifier"}
_BG = {"bandgap", "bg", "reference", "bgr", "vref"}


def supported():
    return {"vco": sorted(_VCO), "ring": sorted(_RING), "opa": sorted(_OPA), "bandgap": sorted(_BG)}


def analyze(topology, params, spec=None):
    """依拓樸分流到對應檢查器。回傳統一格式 dict;未知拓樸回 {error}。"""
    t = (topology or "").strip().lower()
    if t in _VCO:
        return analyze_lc_vco(params, spec)
    if t in _RING:
        return analyze_ring_vco(params, spec)
    if t in _OPA:
        return analyze_two_stage_opa(params, spec)
    if t in _BG:
        return analyze_bandgap(params, spec)
    return {"error": f"未知拓樸 '{topology}';支援: vco / ring / opa / bandgap",
            "supported": supported()}


def demo():
    cases = [
        ("vco", {"L_H": 2e-9, "C_fix_F": 200e-15, "Cvar_min_F": 100e-15, "Cvar_max_F": 400e-15,
                 "W_um": 40, "L_um": 0.18, "Id_A": 2e-3, "Q": 10}, {"f0_GHz": 6.0}),
        ("opa", {"Itail_A": 20e-6, "Id6_A": 60e-6, "Cc_F": 2e-12, "CL_F": 5e-12},
                {"gain_dB_min": 60, "pm_deg_min": 60}),
        ("bandgap", {"R1_ohm": 10e3, "R3_ohm": 89.3e3, "N": 8}, {"tc_ppm_max": 50}),
        ("xyz", {}, None),
    ]
    for topo, p, s in cases:
        r = analyze(topo, p, s)
        if "error" in r:
            print(f"[{topo}] -> error: {r['error'][:40]}")
            assert topo == "xyz"
        else:
            n_err = sum(1 for f in r["findings"] if f["level"] == "error")
            print(f"[{topo:8}] ok={r['ok']} metrics={len(r['metrics'])}項 findings={len(r['findings'])} (error {n_err})")
            assert "metrics" in r and "findings" in r
    print("demo OK — 統一入口三拓樸分流正常")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    demo()
