# -*- coding: utf-8 -*-
"""
web_app.py — 通用類比 IC 自動調參平台 視覺化網頁 (Flask)

啟動: python web_app.py  ->  http://127.0.0.1:5000

三電路模式 (opa / bandgap / ringosc) 共用同一看板:
  - 動態目標標籤、動態圖表 (波德圖 / Vref-溫度 / 方波)、動態影響度與 KiCad 參數
"""

import io
import os
import base64
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, render_template, Response

import eda_control as eda
from eda_control import CIRCUITS
import agent_main as agent
import mc_yield as mc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# 圖表配色 (Apple light 風格)
_AX = "#86868b"; _GRID = "#e5e5ea"; _BG = "#ffffff"
_BLUE = "#0071e3"; _TEAL = "#30b0c7"; _PURPLE = "#5e5ce6"; _ORANGE = "#ff9f0a"; _RED = "#ff3b30"


def _fig():
    # 圖表軸標籤一律用英文 (避免 matplotlib 缺中文字體時變成方框)
    plt.rcParams.update({"font.size": 9, "figure.facecolor": _BG,
                         "axes.facecolor": _BG, "savefig.facecolor": _BG,
                         "font.sans-serif": ["DejaVu Sans"],
                         "axes.unicode_minus": False})


def _style(ax):
    ax.grid(True, which="both", color=_GRID, lw=0.8)
    ax.tick_params(colors=_AX)
    for s in ax.spines.values():
        s.set_color(_GRID)
    ax.set_axisbelow(True)


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_chart(circuit, wave, metrics):
    """依電路型態產生對應圖表 PNG (base64)。"""
    _fig()
    kind = wave.get("kind")

    if kind == "bode" and wave.get("freq"):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 4.7), sharex=True)
        for ax in (a1, a2):
            ax.set_xscale("log"); _style(ax)
        a1.plot(wave["freq"], wave["mag_db"], color=_BLUE, lw=2)
        a1.axhline(0, color=_AX, lw=0.8, ls="--")
        a1.set_ylabel("Gain (dB)")
        ugf = metrics.get("ugf")
        if ugf:
            a1.axvline(ugf, color=_RED, lw=1, ls="--")
            a1.annotate(f"UGF~{ugf/1e6:.1f}MHz", xy=(ugf, 0), xytext=(6, 8),
                        textcoords="offset points", color=_RED, fontsize=8)
        a2.plot(wave["freq"], wave["phase_deg"], color=_PURPLE, lw=2)
        a2.axhline(-180, color=_AX, lw=0.8, ls="--")
        a2.set_ylabel("Phase (deg)"); a2.set_xlabel("Frequency (Hz)")
        return _b64(fig)

    if kind == "temp" and wave.get("temp"):
        fig, ax = plt.subplots(figsize=(6.6, 4.0)); _style(ax)
        ax.plot(wave["temp"], [v * 1e3 for v in wave["vref"]], color=_ORANGE, lw=2.2)
        ax.set_xlabel("Temperature (degC)"); ax.set_ylabel("Vref (mV)")
        tc = metrics.get("tc")
        if tc is not None:
            ax.set_title(f"Temp. Coeff ~ {tc:.2f} ppm/degC", color="#1d1d1f", fontsize=11)
        return _b64(fig)

    if kind == "wave" and wave.get("t"):
        t = wave["t"]; v = wave["v"]
        # 取穩態視窗 (跳過啟振), 顯示約 8 個週期
        n = len(t); i0 = int(n * 0.55); i1 = min(n, i0 + int(n * 0.18))
        fig, ax = plt.subplots(figsize=(6.6, 4.0)); _style(ax)
        ax.plot([x * 1e9 for x in t[i0:i1]], v[i0:i1], color=_TEAL, lw=2)
        ax.set_xlabel("Time (ns)"); ax.set_ylabel("Voltage (V)")
        f = metrics.get("freq")
        if f:
            ax.set_title(f"Osc. Frequency ~ {f/1e9:.3f} GHz", color="#1d1d1f", fontsize=11)
        return _b64(fig)

    return None


def make_hist_png(mcr):
    """蒙地卡羅樣本直方圖 (英文標籤)。"""
    s = np.array(mcr.get("samples", []))
    if len(s) == 0:
        return None
    metric = mcr["metric"]
    sc = 1e9 if metric == "freq" else 1.0
    unit = {"gain": "Gain (dB)", "freq": "Frequency (GHz)", "tc": "TC (ppm/degC)"}.get(metric, metric)
    s = s / sc
    mean, std = s.mean(), s.std()
    _fig()
    nb = int(np.clip(len(s) // 150, 12, 60))            # 樣本越多 bins 越細
    edge = "#fff" if len(s) < 2000 else "none"
    fig, ax = plt.subplots(figsize=(6.6, 3.6)); _style(ax)
    ax.hist(s, bins=nb, color=_BLUE, alpha=0.78, edgecolor=edge)
    ax.axvline(mean, color=_RED, lw=1.6, ls="-", label=f"Mean={mean:.3g}")
    ax.axvline(mean - std, color=_ORANGE, lw=1.1, ls="--", label=f"+/-1 sigma")
    ax.axvline(mean + std, color=_ORANGE, lw=1.1, ls="--")
    ax.set_xlabel(unit); ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return _b64(fig)


_CORNER_COLOR = {"tt": _BLUE, "ff": "#34c759", "ss": _RED}


def make_pvt_png(pvt):
    """27 條 PVT 曲線疊加圖 (依製程角落上色, 英文標籤)。"""
    _fig()
    kind = pvt["kind"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2)); _style(ax)
    seen = set()
    for cv in pvt["curves"]:
        col = _CORNER_COLOR.get(cv["p"], _AX)
        lbl = cv["p"].upper() if cv["p"] not in seen else None
        seen.add(cv["p"])
        if kind == "bode" and cv.get("freq"):
            ax.semilogx(cv["freq"], cv["mag"], color=col, lw=1, alpha=0.55, label=lbl)
        elif kind == "wave" and cv.get("t_ns"):
            t = cv["t_ns"]; v = cv["v"]
            n = len(t); i0 = int(n * 0.55); i1 = min(n, i0 + int(n * 0.22))
            ax.plot(t[i0:i1], v[i0:i1], color=col, lw=0.9, alpha=0.5, label=lbl)
    if kind == "bode":
        ax.axhline(0, color=_AX, lw=0.7, ls="--")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Gain (dB)")
    else:
        ax.set_xlabel("Time (ns)"); ax.set_ylabel("Voltage (V)")
    rb = pvt.get("robustness")
    if rb:
        ax.set_title(f"PVT 27 corners  |  spread = {rb['spread_pct']:.1f}%",
                     color="#1d1d1f", fontsize=10)
    ax.legend(loc="best", fontsize=8, framealpha=0.9, title="Process")
    return _b64(fig)


# KiCad 元件對照 (每個可調參數 -> 原理圖元件)
KICAD_MAP = {
    "opa": {"w_diff": ("M1 / M2", "差動對"), "w_stage2": ("M6", "第二級放大"), "r_bias": ("Rbias", "偏壓電阻")},
    "opa_sky130": {"w_diff": ("M1 / M2", "差動對"), "w_stage2": ("M6", "第二級放大"), "r_bias": ("Rbias", "偏壓電阻")},
    "bandgap": {"r_trim": ("R1", "PTAT 電流電阻"), "n_bjt": ("Q2", "BJT 面積比 (並聯數)")},
    "bandgap_sky130": {"r_trim": ("R1", "PTAT 電流電阻"), "n_bjt": ("Q2", "BJT 面積比 (並聯數)")},
    "ringosc": {"w_p": ("反相器 MP", "PMOS 寬度"), "w_n": ("反相器 MN", "NMOS 寬度")},
    "ringosc_sky130": {"vctrl": ("Vctrl 源", "VCO 控制電壓輸入")},
}


def _circuits_meta():
    """傳給前端的電路設定 (可 JSON 序列化)。"""
    meta = {}
    for k, c in CIRCUITS.items():
        meta[k] = {
            "label": c["label"], "objective": c["objective"],
            "family": c.get("family", k), "model": c.get("model", "fast"),
            "target_label": c["target_label"], "target_unit": c["target_unit"],
            "target_default": c["target_default"],
            "param_keys": c["param_keys"],
            "params": {pk: {"label": pv["label"], "unit": pv["unit"]}
                       for pk, pv in c["params"].items()},
        }
    return meta


@app.route("/")
def index():
    return render_template("index.html", circuits_json=json.dumps(_circuits_meta(), ensure_ascii=False))


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    data = request.get_json(force=True)
    circuit = data.get("circuit", "opa")
    if circuit not in CIRCUITS:
        return jsonify({"error": "未知電路"}), 400
    c = CIRCUITS[circuit]
    target = None
    if c["objective"] == "target":
        try:
            target = float(data.get("target"))
        except (TypeError, ValueError):
            return jsonify({"error": "目標必須是數字"}), 400

    result = agent.run_optimization(circuit, target)
    fin = result["final"]

    # 用最終參數跑波形並繪圖
    wave = eda.run_waveform(circuit, fin["params"])
    result["chart_png"] = make_chart(circuit, wave, fin["metrics"])

    # KiCad 參數對照: VCO (params 含 device/dim) -> 每顆 MOS 的 W/L 雙欄表; 其餘 -> ref/role 列
    if any("device" in v for v in c["params"].values()):
        devs = {}
        for k in c["param_keys"]:
            pv = c["params"][k]
            v = pv["fmt"].format(fin["params"][k] * pv["scale"])
            devs.setdefault(pv["device"], {})[pv["dim"]] = v
        order = list(dict.fromkeys(c["params"][k]["device"] for k in c["param_keys"]))
        result["wl_table"] = [{"device": d, "W": devs[d].get("W", "—"),
                               "L": devs[d].get("L", "—")} for d in order]
    else:
        rows = []
        for k in c["param_keys"]:
            pv = c["params"][k]; ref, role = KICAD_MAP[circuit][k]
            val = pv["fmt"].format(fin["params"][k] * pv["scale"])
            rows.append({"ref": ref, "role": role, "key": k,
                         "value": f"{val}{pv['unit']}", "label": pv["label"]})
        result["kicad"] = rows
    result["params_meta"] = {k: {"label": c["params"][k]["label"],
                                 "unit": c["params"][k]["unit"],
                                 "scale": c["params"][k]["scale"],
                                 "fmt": c["params"][k]["fmt"]} for k in c["param_keys"]}
    return jsonify(result)


@app.route("/api/yield", methods=["POST"])
def api_yield():
    data = request.get_json(force=True)
    circuit = data.get("circuit", "opa")
    if circuit not in CIRCUITS:
        return jsonify({"error": "未知電路"}), 400
    keys = CIRCUITS[circuit]["param_keys"]
    try:
        params = {k: float(data["params"][k]) for k in keys}
    except (TypeError, KeyError, ValueError):
        return jsonify({"error": "缺少參數"}), 400
    target = data.get("target")
    target = float(target) if target is not None else None

    mcr = mc.monte_carlo(circuit, params, target, n=int(data.get("n", 50)))
    if mcr.get("error"):
        return jsonify(mcr), 200
    mcr["hist_png"] = make_hist_png(mcr)
    mcr["corner"] = mc.corner_analysis(circuit, params, target)["corners"]
    return jsonify(mcr)


@app.route("/api/yield_dnn", methods=["POST"])
def api_yield_dnn():
    """AI 萬點高速良率預測 (PyTorch 替代模型, 毫秒級)。"""
    import dl_yield_predictor as dlp
    data = request.get_json(force=True)
    circuit = data.get("circuit", "opa")
    if circuit not in CIRCUITS:
        return jsonify({"error": "未知電路"}), 400
    keys = CIRCUITS[circuit]["param_keys"]
    try:
        params = {k: float(data["params"][k]) for k in keys}
    except (TypeError, KeyError, ValueError):
        return jsonify({"error": "缺少參數"}), 400
    target = data.get("target")
    target = float(target) if target is not None else None
    n = int(data.get("n", 10000))

    if not os.path.exists(os.path.join(BASE_DIR, "data", f"surrogate_{circuit}.pth")):
        return jsonify({"error": "此電路尚無 DNN 替代模型（如 VCO），請改用 50 點 ngspice 良率評估"}), 200
    r = dlp.yield_predict(circuit, params, target, n=n)
    r["hist_png"] = make_hist_png({"samples": r["samples"], "metric": r["metric"]})
    del r["samples"]                                    # 不回傳萬點原始陣列
    r["corner"] = mc.corner_analysis(circuit, params, target)["corners"]
    return jsonify(r)


@app.route("/api/pareto", methods=["POST"])
def api_pareto():
    """多目標帕雷托 3 套餐推薦 (每套餐附 KiCad 參數與多維影響度)。"""
    import pareto
    data = request.get_json(force=True)
    circuit = data.get("circuit", "opa")
    if circuit not in CIRCUITS:
        return jsonify({"error": "未知電路"}), 400
    c = CIRCUITS[circuit]
    if c.get("optimizer") == "vco_hybrid":
        return jsonify({"error": "VCO 僅單一控制電壓 Vctrl，不適用多目標帕雷托套餐"}), 200
    target = data.get("target")
    target = float(target) if target is not None else None

    r = pareto.pareto_packages(circuit, target)
    has_dev = any("device" in v for v in c["params"].values())
    for pk in r["packages"]:
        p = pk["params"]
        if has_dev:                                     # 實體化電路 -> 每顆 MOS W/L 雙欄
            devs = {}
            for k in c["param_keys"]:
                pv = c["params"][k]
                devs.setdefault(pv["device"], {})[pv["dim"]] = pv["fmt"].format(p[k] * pv["scale"])
            order = list(dict.fromkeys(c["params"][k]["device"] for k in c["param_keys"]))
            pk["wl_table"] = [{"device": d, "W": devs[d].get("W", "—"),
                               "L": devs[d].get("L", "—")} for d in order]
        else:
            pk["kicad"] = [{"ref": KICAD_MAP[circuit][k][0], "role": KICAD_MAP[circuit][k][1],
                            "value": f"{c['params'][k]['fmt'].format(p[k]*c['params'][k]['scale'])}{c['params'][k]['unit']}"}
                           for k in c["param_keys"]]
        try:
            pk["multi_influence"] = agent.compute_multi_influence(circuit, p)
        except Exception:
            pk["multi_influence"] = None
    return jsonify(r)


@app.route("/api/sweep2d", methods=["POST"])
def api_sweep2d():
    """雙變數交叉掃描 (20x20=400 多線程) -> 3D 曲面資料 (論文任務一)。"""
    import analyzer
    data = request.get_json(force=True)
    circuit = data.get("circuit", "opa")
    if circuit not in CIRCUITS:
        return jsonify({"error": "未知電路"}), 400
    keys = CIRCUITS[circuit]["param_keys"]
    if len(keys) < 2:
        return jsonify({"error": "此電路僅單一可調參數，無法做 2D 交叉掃描"}), 200
    kx = data.get("key_x") or keys[0]
    ky = data.get("key_y") or keys[1]
    if kx == ky:
        return jsonify({"error": "X / Y 需為不同參數"}), 200
    try:
        r = analyzer.sweep_2d(circuit, kx, ky, n=int(data.get("n", 20)))
    except ValueError as e:
        return jsonify({"error": str(e)}), 200
    return jsonify(r)


@app.route("/api/pvt", methods=["POST"])
def api_pvt():
    """PVT 27 組環境掃描 (論文任務二) -> 疊加曲線圖 + 魯棒性統計。"""
    import pvt_scanner
    data = request.get_json(force=True)
    circuit = data.get("circuit", "opa_sky130")
    if circuit not in pvt_scanner.PVT_CFG:
        return jsonify({"error": "PVT 僅支援 sky130 精準模式（OPA / VCO）"}), 200
    params = data.get("params")
    if params:
        params = {k: float(params[k]) for k in CIRCUITS[circuit]["param_keys"]}
    r = pvt_scanner.pvt_scan(circuit, params)
    png = make_pvt_png(r)
    return jsonify({"circuit": circuit, "kind": r["kind"], "metric": r["metric"],
                    "n_ok": r["n_ok"], "total": r["total"],
                    "robustness": r["robustness"], "pvt_png": png})


@app.route("/api/netlist")
def api_netlist():
    circuit = request.args.get("circuit", "opa")
    if circuit not in CIRCUITS:
        return "未知電路", 400
    try:
        params = {k: float(request.args.get(k)) for k in CIRCUITS[circuit]["param_keys"]}
    except (TypeError, ValueError):
        return "參數錯誤", 400
    eda.render_netlist(circuit, params)
    with open(eda.RUN_SP, "r", encoding="utf-8") as f:
        text = f.read()
    return Response(text, mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment; filename={circuit}_optimized.sp"})


if __name__ == "__main__":
    print("=" * 56)
    print("  通用類比 IC 自動調參平台 — 視覺化網頁")
    print("  http://127.0.0.1:5000")
    print("=" * 56)
    app.run(host="127.0.0.1", port=5000, debug=False)
