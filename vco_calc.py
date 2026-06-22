# -*- coding: utf-8 -*-
"""
vco_calc.py — LC-VCO 設計公式檢查器(純解析,零相依,零 token)

用途:已知拓樸(交叉耦合 LC-VCO)時,用閉式設計公式「一步步算出」應有的性能,
比對規格,標出哪個參數不對、該調到多少。完全不需要 LLM 也不需要 ngspice ——
把原本丟給 AI 算的數學(f0 / KVCO / FTR / 起振條件 / 相位雜訊 / FoM)改成程式,
省 token 又算得準,且不外送任何 PDK 機密。

對外主介面:
    analyze_lc_vco(design, spec=None) -> dict   # 算性能 + 糾錯 + 給目標值

design 欄位(SI 單位;缺的會用合理預設或從幾何/尺寸推):
    L_H            : 電感值 (亨利)。若沒有, 給 inductor geom 由 spiral_inductance 算
    C_fix_F        : 固定電容 (法拉)
    Cvar_min_F     : 變容最小值 (高 Vctrl)
    Cvar_max_F     : 變容最大值 (低 Vctrl)
    Cpar_F         : 其他寄生電容 (選填, 預設 0)
    Q              : tank 品質因數 (選填, 預設 由 L,Rs 算或給 10)
    Rs_ohm         : 電感串聯電阻 (選填, 用來算 Q)
    gm_S           : 交叉耦合單管跨導 (S)。若沒有, 給 W_um/L_um/Id_A 由 gm_sat 算
    Id_A           : 偏壓電流 (安培, 算 gm 與功耗用)
    Vdd_V          : 電源 (預設 1.8)
    Vctrl_span_V   : 調諧電壓跨距 (算 KVCO, 預設 1.0)
    ucox           : µ·Cox (A/V^2);TSMC0.18 NMOS~270u / sky130~ 類似 (預設 270e-6)
    F_noise        : 振盪器雜訊因子 (Leeson, 預設 4)

spec 欄位(目標, 全部選填):
    f0_GHz, ftr_pct_min, pn_dbc_1MHz_max(如 -110), startup_min(預設 1.5)
"""
import math

KT = 1.380649e-23 * 300.0      # 室溫 kT
MU0 = 4e-7 * math.pi


# ---------------------------------------------------------------- 幾何/尺寸輔助
def spiral_inductance(n, d_out_um, w_um, s_um=None):
    """方形螺旋電感 (Mohan current-sheet 公式) -> 亨利。
       n=圈數, d_out=外徑(µm), w=線寬(µm), s=線距(µm, 預設=w)。殘差約 ±5~10%。"""
    s = w_um if s_um is None else s_um
    d_in = d_out_um - 2 * n * (w_um + s) + s          # 內徑
    d_in = max(d_in, 0.1 * d_out_um)                  # 防呆
    d_avg = 0.5 * (d_out_um + d_in) * 1e-6            # 平均直徑 (m)
    rho = (d_out_um - d_in) / (d_out_um + d_in)        # 填充比
    rho = min(max(rho, 1e-3), 1.0)
    c1, c2, c3, c4 = 1.27, 2.07, 0.18, 0.13            # 方形係數
    return (c1 * MU0 * n * n * d_avg / 2) * (math.log(c2 / rho) + c3 * rho + c4 * rho * rho)


def gm_sat(W_um, L_um, Id_A, ucox=270e-6):
    """飽和區跨導 gm = sqrt(2·µCox·(W/L)·Id) (S)。"""
    return math.sqrt(2 * ucox * (W_um / L_um) * Id_A)


# ---------------------------------------------------------------- 核心分析
def _f0(L, C):
    return 1.0 / (2 * math.pi * math.sqrt(L * C)) if (L > 0 and C > 0) else 0.0


def analyze_lc_vco(design, spec=None):
    d = dict(design)
    spec = dict(spec or {})
    findings = []

    def add(level, item, msg, suggest=None):
        findings.append({"level": level, "item": item, "msg": msg, "suggest": suggest})

    # --- 電感 ---
    L = d.get("L_H")
    if L is None and "inductor_geom" in d:
        g = d["inductor_geom"]
        L = spiral_inductance(g["n"], g["d_out_um"], g["w_um"], g.get("s_um"))
        add("info", "L", f"由螺旋幾何 Mohan 公式估算 L = {L*1e9:.2f} nH (±~10%)")
    if not L:
        return {"error": "缺電感: 請給 L_H 或 inductor_geom{n,d_out_um,w_um}"}

    # --- 電容 / tank ---
    Cfix = d.get("C_fix_F", 0.0)
    Cpar = d.get("Cpar_F", 0.0)
    Cvmin = d.get("Cvar_min_F", 0.0)
    Cvmax = d.get("Cvar_max_F", Cvmin)
    C_mid = Cfix + Cpar + 0.5 * (Cvmin + Cvmax)
    C_hi = Cfix + Cpar + Cvmax          # 大電容 -> 低頻
    C_lo = Cfix + Cpar + Cvmin          # 小電容 -> 高頻
    if C_mid <= 0:
        return {"error": "缺電容: 至少給 C_fix_F 或 Cvar_min/max_F"}

    # --- 頻率 / 調諧 ---
    f0 = _f0(L, C_mid)
    f_max = _f0(L, C_lo)
    f_min = _f0(L, C_hi)
    fc = 0.5 * (f_max + f_min) or f0
    ftr = 100 * (f_max - f_min) / fc if fc else 0.0
    vspan = d.get("Vctrl_span_V", 1.0)
    kvco = (f_max - f_min) / vspan if vspan else 0.0     # 平均 (Hz/V)

    # --- tank Q / Rp / 起振 ---
    w0 = 2 * math.pi * f0
    Q = d.get("Q")
    if Q is None:
        Rs = d.get("Rs_ohm")
        Q = (w0 * L / Rs) if Rs else 10.0                 # 沒給就假設 Q=10
    Rp = Q * w0 * L                                        # 並聯 tank 損耗電阻
    gm = d.get("gm_S")
    if gm is None and all(k in d for k in ("W_um", "L_um", "Id_A")):
        gm = gm_sat(d["W_um"], d["L_um"], d["Id_A"], d.get("ucox", 270e-6))
    startup = (gm * Rp / 2) if gm else None               # >=1 起振, 設計建議 >=1.5

    # --- 功耗 / 相位雜訊 (Leeson) / FoM ---
    Vdd = d.get("Vdd_V", 1.8)
    Id = d.get("Id_A")
    Psig = (Vdd * Id) if Id else None
    F = d.get("F_noise", 4.0)
    pn_1m = fom = None
    if Psig and Q and f0:
        df = 1e6
        pn_1m = 10 * math.log10((2 * F * KT / Psig) * (1 + (f0 / (2 * Q * df)) ** 2))
        fom = pn_1m - 20 * math.log10(f0 / df) + 10 * math.log10(Psig / 1e-3)

    metrics = {
        "L_nH": L * 1e9, "C_tank_mid_fF": C_mid * 1e15,
        "f0_GHz": f0 / 1e9, "f_min_GHz": f_min / 1e9, "f_max_GHz": f_max / 1e9,
        "ftr_pct": ftr, "kvco_MHz_V": kvco / 1e6, "Q": Q, "Rp_ohm": Rp,
        "gm_mS": (gm * 1e3) if gm else None, "startup_factor": startup,
        "Psig_mW": (Psig * 1e3) if Psig else None,
        "pn_1MHz_dBc": pn_1m, "fom_dBc": fom,
    }

    # ---------------- 規格比對 + 糾錯(給目標值) ----------------
    if "f0_GHz" in spec:
        tgt = spec["f0_GHz"]
        if abs(f0 / 1e9 - tgt) / tgt > 0.05:
            C_need = 1 / ((2 * math.pi * tgt * 1e9) ** 2 * L)
            add("error", "f0", f"頻率 {f0/1e9:.2f}GHz 偏離目標 {tgt}GHz",
                f"固定 L={L*1e9:.2f}nH 下, 總電容應 ≈ {C_need*1e15:.1f}fF "
                f"(目前 {C_mid*1e15:.1f}fF);或改 L")
        else:
            add("ok", "f0", f"頻率 {f0/1e9:.2f}GHz 達標")

    if "ftr_pct_min" in spec:
        if ftr < spec["ftr_pct_min"]:
            add("error", "FTR", f"調諧範圍 {ftr:.1f}% < 需求 {spec['ftr_pct_min']}%",
                "加大變容 Cvar_max/Cvar_min 比值, 或降低固定電容 C_fix 佔比")
        else:
            add("ok", "FTR", f"調諧範圍 {ftr:.1f}% 達標")

    if startup is not None:
        smin = spec.get("startup_min", 1.5)
        if startup < 1.0:
            gm_need = 2 * smin / Rp
            add("error", "起振", f"gm·Rp/2 = {startup:.2f} < 1 → 不會起振!",
                f"需 gm ≥ {gm_need*1e3:.2f}mS (目前 {gm*1e3:.2f}mS):加大交叉耦合 W 或偏壓 Id")
        elif startup < smin:
            add("warn", "起振", f"起振餘裕 {startup:.2f} 偏低 (建議 ≥{smin})",
                f"PVT 下可能不穩, 建議 gm 提高至餘裕 ≥{smin}")
        else:
            add("ok", "起振", f"起振餘裕 {startup:.2f} 足夠")

    if "pn_dbc_1MHz_max" in spec and pn_1m is not None:
        if pn_1m > spec["pn_dbc_1MHz_max"]:
            add("warn", "相位雜訊", f"PN@1MHz {pn_1m:.1f} 差於需求 {spec['pn_dbc_1MHz_max']}dBc/Hz",
                "提高 tank Q 或加大功耗 Psig(Leeson:PN ∝ 1/(Q²·Psig))")
        else:
            add("ok", "相位雜訊", f"PN@1MHz {pn_1m:.1f}dBc/Hz 達標")

    return {"topology": "lc-vco", "metrics": metrics, "findings": findings,
            "ok": not any(f["level"] == "error" for f in findings)}


# ---------------------------------------------------------------- 自我測試
def demo():
    # 一顆 ~6GHz LC-VCO:L=2nH, C 給定, 交叉耦合 W=40µ/L=0.18µ, Id=2mA
    design = {
        "L_H": 2e-9, "C_fix_F": 200e-15, "Cvar_min_F": 100e-15, "Cvar_max_F": 400e-15,
        "W_um": 40, "L_um": 0.18, "Id_A": 2e-3, "Q": 10, "Vdd_V": 1.8, "Vctrl_span_V": 1.0,
    }
    spec = {"f0_GHz": 6.0, "ftr_pct_min": 10, "startup_min": 1.5, "pn_dbc_1MHz_max": -100}
    r = analyze_lc_vco(design, spec)
    m = r["metrics"]
    # 基本健全性
    assert 4 < m["f0_GHz"] < 9, m["f0_GHz"]
    assert m["f_max_GHz"] > m["f_min_GHz"]
    assert m["startup_factor"] > 0
    assert "fom_dBc" in m
    # 起振判定: gm·Rp/2, 這組應會起振
    assert m["startup_factor"] > 1
    print("demo OK")
    print(f"  f0={m['f0_GHz']:.2f}GHz  FTR={m['ftr_pct']:.1f}%  KVCO={m['kvco_MHz_V']:.0f}MHz/V")
    print(f"  起振餘裕={m['startup_factor']:.2f}  gm={m['gm_mS']:.2f}mS  Q={m['Q']:.0f}")
    print(f"  PN@1MHz={m['pn_1MHz_dBc']:.1f}dBc/Hz  FoM={m['fom_dBc']:.1f}")
    for f in r["findings"]:
        tag = {"ok": "✓", "info": "·", "warn": "!", "error": "✗"}[f["level"]]
        print(f"  [{tag}] {f['item']}: {f['msg']}" + (f"  → {f['suggest']}" if f["suggest"] else ""))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    demo()
