# -*- coding: utf-8 -*-
"""
bandgap_calc.py — 帶隙基準 (Kuijk/Brokaw) 設計公式檢查器(純解析,零相依)

原理: Vref = Vbe(CTAT, dVbe/dT≈-1.6mV/°C) + (R3/R1)·VT·ln(N)(PTAT, +)。
兩者溫度係數相消 -> 低溫漂。本模組算 Vref / TC、找出零溫漂的最佳 (R3/R1)·ln(N)。
不需 LLM、不需模擬。

主介面: analyze_bandgap(design, spec=None) -> dict

design:
  R1_ohm, R3_ohm : PTAT 設定電阻 / 輸出支路電阻
  N              : Q2/Q1 BJT 面積比 (PTAT 的 ln(N) 來源)
  Vbe0_V         : 27°C Vbe (預設 0.65)
  dVbe_dT        : Vbe 溫度係數 (V/°C, 預設 -1.6e-3)
spec:
  tc_ppm_max     : 溫漂上限 (預設 50)
  vref_target_V  : 目標基準電壓 (選填, 通常 ~1.20~1.25)
"""
import math

K_Q = 8.617333e-5          # k/q (V/K) = 0.08617 mV/°C
T0 = 300.15                # 27°C (K)
VT = K_Q * T0              # 熱電壓 @27°C ≈ 25.86 mV


def analyze_bandgap(design, spec=None):
    d = dict(design)
    spec = dict(spec or {})
    findings = []

    def add(level, item, msg, suggest=None):
        findings.append({"level": level, "item": item, "msg": msg, "suggest": suggest})

    R1 = d.get("R1_ohm")
    R3 = d.get("R3_ohm")
    N = d.get("N")
    if not (R1 and R3 and N and N > 1):
        return {"error": "需 R1_ohm, R3_ohm, N(>1)"}
    Vbe0 = d.get("Vbe0_V", 0.65)
    dVbe_dT = d.get("dVbe_dT", -1.6e-3)

    ratio = R3 / R1
    lnN = math.log(N)
    ptat_slope = ratio * K_Q * lnN                 # PTAT 溫度係數 (V/°C, 正)
    dVref_dT = dVbe_dT + ptat_slope                # 總溫度係數
    Vref = Vbe0 + ratio * VT * lnN                 # 27°C 基準電壓
    # 一階線性 TC (ppm/°C);實際因 Vbe 曲率有 ~10ppm 地板
    tc_lin = abs(dVref_dT) / Vref * 1e6

    # 零溫漂最佳條件: ptat_slope = -dVbe_dT  ->  (R3/R1)·ln(N) = |dVbe_dT|/(k/q)
    target_prod = abs(dVbe_dT) / K_Q               # 需要的 (R3/R1)·ln(N)
    ratio_opt = target_prod / lnN                  # 固定 N 下最佳 R3/R1
    N_opt = math.exp(target_prod / ratio)          # 固定 R3/R1 下最佳 N

    metrics = {
        "Vref_V": Vref, "tc_linear_ppm": tc_lin, "dVref_dT_uV": dVref_dT * 1e6,
        "ptat_slope_uV": ptat_slope * 1e6, "ctat_slope_uV": dVbe_dT * 1e6,
        "R3_R1": ratio, "lnN": lnN, "ratio_opt": ratio_opt, "N_opt": N_opt,
    }

    # --- 糾錯 ---
    tc_max = spec.get("tc_ppm_max", 50)
    if tc_lin > tc_max:
        if dVref_dT > 0:
            tip = f"PTAT 太強(過補償):降 R3/R1 至 ≈{ratio_opt:.2f}(目前 {ratio:.2f}),或降 N 至 ≈{N_opt:.1f}"
        else:
            tip = f"PTAT 太弱(欠補償):升 R3/R1 至 ≈{ratio_opt:.2f}(目前 {ratio:.2f}),或升 N 至 ≈{N_opt:.1f}"
        add("error", "溫漂", f"TC ≈ {tc_lin:.0f}ppm/°C > 需求 {tc_max}", tip)
    else:
        add("ok", "溫漂", f"TC ≈ {tc_lin:.0f}ppm/°C 達標(註:含 Vbe 曲率實際約再 +10~20ppm 地板)")

    if "vref_target_V" in spec:
        vt = spec["vref_target_V"]
        if abs(Vref - vt) > 0.05:
            add("warn", "基準電壓", f"Vref {Vref*1000:.0f}mV 偏離目標 {vt*1000:.0f}mV",
                "Vref 由 Vbe + PTAT 決定;若已校 TC, 微調靠輸出分壓或加總電阻")
        else:
            add("ok", "基準電壓", f"Vref {Vref*1000:.0f}mV 達標")

    # 健全性提醒
    if Vref < 1.0 or Vref > 1.4:
        add("warn", "Vref 合理性", f"Vref {Vref*1000:.0f}mV 偏離典型帶隙 ~1.2V, 檢查 Vbe0/比值")

    return {"topology": "bandgap", "metrics": metrics, "findings": findings,
            "ok": not any(f["level"] == "error" for f in findings)}


def demo():
    # N=8 (ln8=2.08), 找最佳 R3/R1。先給一個非最佳的看糾錯
    bad = {"R1_ohm": 10e3, "R3_ohm": 50e3, "N": 8}      # R3/R1=5, 過補償
    r = analyze_bandgap(bad, {"tc_ppm_max": 50, "vref_target_V": 1.2})
    m = r["metrics"]
    assert m["Vref_V"] > 0 and m["tc_linear_ppm"] >= 0
    print("demo OK")
    print(f"  Vref={m['Vref_V']*1000:.0f}mV  TC≈{m['tc_linear_ppm']:.0f}ppm/°C  "
          f"R3/R1={m['R3_R1']:.2f}  最佳 R3/R1≈{m['ratio_opt']:.2f}")
    for f in r["findings"]:
        tag = {"ok": "✓", "info": "·", "warn": "!", "error": "✗"}[f["level"]]
        print(f"  [{tag}] {f['item']}: {f['msg']}" + (f"  → {f['suggest']}" if f["suggest"] else ""))
    # 用最佳比值再算一次, TC 應接近 0
    good = {"R1_ohm": 10e3, "R3_ohm": m["ratio_opt"] * 10e3, "N": 8}
    r2 = analyze_bandgap(good, {"tc_ppm_max": 50})
    assert r2["metrics"]["tc_linear_ppm"] < 10, r2["metrics"]["tc_linear_ppm"]
    print(f"  用最佳比值 R3/R1={m['ratio_opt']:.2f} -> TC≈{r2['metrics']['tc_linear_ppm']:.1f}ppm/°C ✓")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    demo()
