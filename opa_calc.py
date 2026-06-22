# -*- coding: utf-8 -*-
"""
opa_calc.py — 兩級米勒補償 OPA 設計公式檢查器(純解析,零相依,零 token)

已知拓樸(NMOS 差動對 + PMOS 共源二級 + 米勒 Cc/Rz)時,用閉式公式算
增益 / GBW / 相位裕度 / 壓擺率 / 擺幅,比對規格,標錯誤並給目標值。
不需 LLM、不需 ngspice。把原本丟 AI 算的小訊號分析改成程式 -> 省 token、算得準。

主介面: analyze_two_stage_opa(design, spec=None) -> dict

design (SI 單位):
  Itail_A   : 差動對尾電流
  Id6_A     : 第二級電流
  Cc_F      : 米勒補償電容
  CL_F      : 負載電容
  Rz_ohm    : 零點電阻 (選填; 最佳 = 1/gm6)
  Vov_V     : 過驅動電壓 (算 gm 用; 預設 0.2)
  lambda    : 通道長度調變 (1/V, 算 ro; 預設 0.1)
  gm1_S/gm6_S : 可直接給, 否則由 Id+Vov 推 (gm = 2·Id/Vov)
  Vdd_V     : 電源 (預設 1.8)

spec (目標, 全選填):
  gain_dB_min, pm_deg_min(預設 60), gbw_Hz_min, slew_Vus_min
"""
import math


def analyze_two_stage_opa(design, spec=None):
    d = dict(design)
    spec = dict(spec or {})
    findings = []

    def add(level, item, msg, suggest=None):
        findings.append({"level": level, "item": item, "msg": msg, "suggest": suggest})

    Vov = d.get("Vov_V", 0.2)
    lam = d.get("lambda", 0.1)
    Itail = d.get("Itail_A")
    Id6 = d.get("Id6_A")
    Cc = d.get("Cc_F")
    CL = d.get("CL_F")
    if not (Itail and Id6 and Cc and CL):
        return {"error": "需 Itail_A, Id6_A, Cc_F, CL_F"}

    Id1 = Itail / 2.0                                    # 差動對單邊電流
    gm1 = d.get("gm1_S") or (2 * Id1 / Vov)              # 差動對跨導
    gm6 = d.get("gm6_S") or (2 * Id6 / Vov)              # 第二級跨導
    ro1 = 1.0 / (lam * Id1)                              # 第一級輸出阻抗 (近似)
    ro6 = 1.0 / (lam * Id6)                              # 第二級輸出阻抗

    # --- 小訊號 ---
    Av = gm1 * ro1 * gm6 * ro6                           # 兩級直流增益 (V/V)
    gain_dB = 20 * math.log10(Av)
    gbw = gm1 / (2 * math.pi * Cc)                       # 單位增益頻寬
    p2 = gm6 / (2 * math.pi * CL)                        # 第二極點 (Cc>>CL 近似)
    # 相位裕度: 主極點 -90°, 第二極點貢獻 -atan(GBW/p2); 零點以 Rz 消去假設已處理
    pm = 90 - math.degrees(math.atan(gbw / p2))
    slew = Itail / Cc                                    # 壓擺率 (V/s) = 尾電流/Cc
    Rz_opt = 1.0 / gm6                                   # 消 RHP 零點的最佳 Rz
    swing = d.get("Vdd_V", 1.8) - 2 * Vov                # 粗估輸出擺幅

    metrics = {
        "gain_dB": gain_dB, "gbw_MHz": gbw / 1e6, "p2_MHz": p2 / 1e6,
        "pm_deg": pm, "slew_V_us": slew / 1e6, "gm1_mS": gm1 * 1e3, "gm6_mS": gm6 * 1e3,
        "Rz_opt_ohm": Rz_opt, "swing_Vpp": swing, "Av_VV": Av,
    }

    # --- 規格比對 + 糾錯 ---
    if "gain_dB_min" in spec:
        if gain_dB < spec["gain_dB_min"]:
            add("error", "增益", f"增益 {gain_dB:.1f}dB < 需求 {spec['gain_dB_min']}dB",
                "加大 gm·ro:差動對/二級用較長 L 提高 ro,或加大 gm(W 或電流)")
        else:
            add("ok", "增益", f"增益 {gain_dB:.1f}dB 達標")

    pm_min = spec.get("pm_deg_min", 60)
    if pm < 45:
        # 穩定條件: p2 >= 2.2·GBW (PM~60); 推回需要的 gm6 或 Cc
        gm6_need = 2.2 * gm1 * CL / Cc
        add("error", "相位裕度", f"PM {pm:.0f}° < 45° → 不穩定!",
            f"需 p2 ≥ 2.2·GBW:把 gm6 提高至 ≥{gm6_need*1e3:.2f}mS (目前 {gm6*1e3:.2f}mS),"
            f"或加大 Cc 降 GBW")
    elif pm < pm_min:
        add("warn", "相位裕度", f"PM {pm:.0f}° < 建議 {pm_min}°",
            "加大 Cc(降 GBW)或加大 gm6(推遠 p2)")
    else:
        add("ok", "相位裕度", f"PM {pm:.0f}° 足夠")

    if "gbw_Hz_min" in spec:
        if gbw < spec["gbw_Hz_min"]:
            add("error", "GBW", f"GBW {gbw/1e6:.1f}MHz < 需求 {spec['gbw_Hz_min']/1e6:.1f}MHz",
                f"GBW=gm1/(2π·Cc):提高 gm1 或減小 Cc(注意 Cc 太小會傷 PM)")
        else:
            add("ok", "GBW", f"GBW {gbw/1e6:.1f}MHz 達標")

    if "slew_Vus_min" in spec:
        sv = slew / 1e6
        if sv < spec["slew_Vus_min"]:
            I_need = spec["slew_Vus_min"] * 1e6 * Cc
            add("error", "壓擺率", f"SR {sv:.1f}V/µs < 需求 {spec['slew_Vus_min']}V/µs",
                f"SR=Itail/Cc:尾電流需 ≥{I_need*1e6:.0f}µA(目前 {Itail*1e6:.0f}µA)或減小 Cc")
        else:
            add("ok", "壓擺率", f"SR {sv:.1f}V/µs 達標")

    # 零點電阻提醒
    if "Rz_ohm" in d:
        if abs(d["Rz_ohm"] - Rz_opt) / Rz_opt > 0.3:
            add("warn", "零點電阻", f"Rz={d['Rz_ohm']:.0f}Ω 偏離最佳 1/gm6≈{Rz_opt:.0f}Ω",
                f"設 Rz≈{Rz_opt:.0f}Ω 可把 RHP 零點推走、改善 PM")

    return {"topology": "two-stage-miller-opa", "metrics": metrics, "findings": findings,
            "ok": not any(f["level"] == "error" for f in findings)}


def demo():
    # 一顆典型兩級 OPA: 尾電流 20µA, 二級 60µA, Cc=2pF, CL=5pF
    design = {"Itail_A": 20e-6, "Id6_A": 60e-6, "Cc_F": 2e-12, "CL_F": 5e-12,
              "Vov_V": 0.2, "lambda": 0.1, "Vdd_V": 1.8}
    spec = {"gain_dB_min": 60, "pm_deg_min": 60, "gbw_Hz_min": 5e6, "slew_Vus_min": 5}
    r = analyze_two_stage_opa(design, spec)
    m = r["metrics"]
    assert 40 < m["gain_dB"] < 120, m["gain_dB"]
    assert m["gbw_MHz"] > 0 and m["pm_deg"] > 0
    assert m["slew_V_us"] > 0
    print("demo OK")
    print(f"  增益={m['gain_dB']:.1f}dB  GBW={m['gbw_MHz']:.1f}MHz  PM={m['pm_deg']:.0f}°  "
          f"SR={m['slew_V_us']:.1f}V/µs  gm1={m['gm1_mS']:.3f}mS")
    for f in r["findings"]:
        tag = {"ok": "✓", "info": "·", "warn": "!", "error": "✗"}[f["level"]]
        print(f"  [{tag}] {f['item']}: {f['msg']}" + (f"  → {f['suggest']}" if f["suggest"] else ""))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    demo()
