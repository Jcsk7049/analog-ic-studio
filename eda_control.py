# -*- coding: utf-8 -*-
"""
eda_control.py — 通用類比 IC 自動調參平台的「執行層」

支援三種電路模式 (CIRCUITS 註冊表):
  - opa      : 兩級 CMOS OPA      -> AC 分析, 取 Gain(dB)/PM(deg)/UGF(Hz)
  - bandgap  : 帶隙基準電壓源      -> DC 溫度掃描, 取 Vref(V)/TC(ppm/°C)
  - ringosc  : 5 級環形振盪器      -> 瞬態分析, 取 freq(Hz)

對外主要介面:
    run_circuit(circuit, params, dump=False) -> metrics dict
    run_waveform(circuit, params)            -> 圖表資料 dict
    CIRCUITS                                  -> 電路設定 (參數/範圍/標籤)
向後相容:
    run_simulation(w_diff, w_stage2, r_bias)  == run_circuit('opa', ...)
    run_bode(...)                             == run_waveform('opa', ...)
"""

import os
import re
import sys
import shutil
import subprocess

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_SP = os.path.join(BASE_DIR, "run.sp")
WAVE_PATH = os.path.join(BASE_DIR, "wave.txt")


def find_ngspice():
    local = os.path.join(BASE_DIR, "tools", "Spice64", "bin", "ngspice_con.exe")
    if os.path.exists(local):
        return local
    for name in ("ngspice_con", "ngspice"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(
        "找不到 ngspice。請確認 tools/Spice64/bin/ngspice_con.exe 存在, "
        "或將 ngspice 加入系統 PATH。")


NGSPICE = find_ngspice()


def _fmt(value):
    return f"{value:.6g}"


def _grab(pattern, log):
    m = re.search(pattern, log, re.MULTILINE)
    if not m:
        return None
    raw = m.group(1).strip()
    if "inf" in raw.lower() or "nan" in raw.lower():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _bad(log):
    """是否出現收斂/結構性錯誤。"""
    return ("singular matrix" in log
            or "iterations without convergence" in log
            or "Simulation interrupted" in log
            or "valid modelname" in log)


# ----------------------------------------------------------------------
# 各模式日誌解析
# ----------------------------------------------------------------------
def _parse_opa(log):
    gain = _grab(r"^gain\s*=\s*([-\d.eE+]+)", log)
    ugf = _grab(r"^ugf\s*=\s*([-\d.eE+]+)", log)
    pm = _grab(r"^pm\s*=\s*([-\d.eE+]+)", log)
    return {"gain": gain, "ugf": ugf, "pm": pm,
            "converged": (not _bad(log)) and gain is not None,
            "ok": gain is not None and not _bad(log)}


def _parse_bandgap(log):
    vref = _grab(r"^vref\s*=\s*([-\d.eE+]+)", log)
    tc = _grab(r"^tc\s*=\s*([-\d.eE+]+)", log)
    ok = vref is not None and tc is not None and not _bad(log) and 0.5 < (vref or 0) < 2.0
    return {"vref": vref, "tc": tc,
            "converged": ok, "ok": ok}


def _parse_ringosc(log):
    freq = _grab(r"^freq\s*=\s*([-\d.eE+]+)", log)
    ok = freq is not None and freq > 0 and not _bad(log)
    return {"freq": freq, "converged": ok, "ok": ok}


# ----------------------------------------------------------------------
# 電路註冊表
#   placeholders: 範本佔位符 -> 參數鍵
#   ranges      : 優化搜尋範圍 (SI: 公尺 / 歐姆 / 個數)
#   params      : 顯示用 (標籤, 單位, 換算倍率)
# ----------------------------------------------------------------------
CIRCUITS = {
    "opa": {
        "label": "兩級 OPA",
        "family": "opa", "model": "fast",
        "template": "two_stage_opa.sp.template",
        "param_keys": ["w_diff", "w_stage2", "r_bias"],
        "placeholders": {"w_diff": "{W_diff}", "w_stage2": "{W_stage2}", "r_bias": "{R_bias}"},
        "ranges": {"w_diff": (5e-6, 100e-6), "w_stage2": (20e-6, 200e-6), "r_bias": (20e3, 500e3)},
        "params": {
            "w_diff":   {"label": "W_diff (差動對)",  "unit": "µm", "scale": 1e6, "fmt": "{:.2f}"},
            "w_stage2": {"label": "W_stage2 (第二級)", "unit": "µm", "scale": 1e6, "fmt": "{:.2f}"},
            "r_bias":   {"label": "R_bias (偏壓阻)",   "unit": "kΩ", "scale": 1e-3, "fmt": "{:.2f}"},
        },
        "parser": _parse_opa,
        "dump": "wrdata wave.txt vdb(out) vp(out)",
        "objective": "target",            # 逼近目標
        "metric": "gain",
        "pm_constraint": True,            # 約束 PM >= 45°
        "target_label": "目標增益 (dB)",
        "target_unit": "dB",
        "target_scale": 1.0,              # 使用者目標 -> metric 原生單位 (dB=dB)
        "target_default": 60.0,
        "start": {"w_diff": 30e-6, "w_stage2": 80e-6, "r_bias": 80e3},
        "waveform": "bode",
    },
    "opa_sky130": {
        "label": "OPA · sky130 精準",
        "family": "opa", "model": "sky130",
        "template": "two_stage_opa_sky130.sp.template",
        "param_keys": ["w_diff", "w_stage2", "r_bias"],
        "placeholders": {"w_diff": "{W_diff}", "w_stage2": "{W_stage2}", "r_bias": "{R_bias}"},
        "ranges": {"w_diff": (4e-6, 30e-6), "w_stage2": (10e-6, 80e-6), "r_bias": (50e3, 400e3)},
        "params": {
            # net_scale=1e6: SI 公尺 -> 微米 (sky130 範本用 .option scale=1u)
            "w_diff":   {"label": "W_diff (差動對)",  "unit": "µm", "scale": 1e6, "net_scale": 1e6, "fmt": "{:.2f}"},
            "w_stage2": {"label": "W_stage2 (第二級)", "unit": "µm", "scale": 1e6, "net_scale": 1e6, "fmt": "{:.2f}"},
            "r_bias":   {"label": "R_bias (偏壓阻)",   "unit": "kΩ", "scale": 1e-3, "fmt": "{:.2f}"},
        },
        "parser": _parse_opa,
        "dump": "wrdata wave.txt vdb(out) vp(out)",
        "objective": "target",
        "metric": "gain",
        "pm_constraint": True,
        "target_label": "目標增益 (dB)",
        "target_unit": "dB",
        "target_scale": 1.0,
        "target_default": 65.0,
        "start": {"w_diff": 10e-6, "w_stage2": 30e-6, "r_bias": 100e3},
        "waveform": "bode",
    },
    "bandgap": {
        "label": "高精度 Bandgap",
        "family": "bandgap", "model": "fast",
        "template": "bandgap_reference.sp.template",
        "param_keys": ["r_trim", "n_bjt"],
        "placeholders": {"r_trim": "{R_trim}", "n_bjt": "{N_bjt}"},
        "ranges": {"r_trim": (4e3, 40e3), "n_bjt": (2, 24)},
        "params": {
            "r_trim": {"label": "R_trim (PTAT 電阻)", "unit": "kΩ", "scale": 1e-3, "fmt": "{:.2f}"},
            "n_bjt":  {"label": "N_bjt (BJT 面積比)", "unit": "x",  "scale": 1,    "fmt": "{:.1f}"},
        },
        "parser": _parse_bandgap,
        "dump": "wrdata wave.txt v(vref)",
        "objective": "minimize",          # 最小化 TC
        "metric": "tc",
        "target_label": "優化 TC 溫漂 (自動最小化)",
        "target_unit": "ppm/°C",
        "target_scale": 1.0,
        "target_default": 0.0,
        "start": {"r_trim": 10e3, "n_bjt": 8},
        "waveform": "temp",
    },
    "bandgap_sky130": {
        "label": "Bandgap · sky130 精準",
        "family": "bandgap", "model": "sky130",
        "template": "bandgap_reference_sky130.sp.template",
        "param_keys": ["r_trim", "n_bjt"],
        "placeholders": {"r_trim": "{R_trim}", "n_bjt": "{N_bjt}"},
        "ranges": {"r_trim": (6e3, 14e3), "n_bjt": (2, 24)},
        "params": {
            "r_trim": {"label": "R_trim (PTAT 電阻)", "unit": "kΩ", "scale": 1e-3, "fmt": "{:.2f}"},
            "n_bjt":  {"label": "N_bjt (BJT 面積比)", "unit": "x",  "scale": 1,    "fmt": "{:.1f}"},
        },
        "parser": _parse_bandgap,
        "dump": "wrdata wave.txt v(vref)",
        "objective": "minimize",
        "metric": "tc",
        "target_label": "優化 TC 溫漂 (自動最小化)",
        "target_unit": "ppm/°C",
        "target_scale": 1.0,
        "target_default": 0.0,
        "start": {"r_trim": 8e3, "n_bjt": 8},
        "waveform": "temp",
    },
    "ringosc": {
        "label": "高頻環形振盪器",
        "family": "ringosc", "model": "fast",
        "template": "ring_oscillator.sp.template",
        "param_keys": ["w_p", "w_n"],
        "placeholders": {"w_p": "{W_p}", "w_n": "{W_n}"},
        "ranges": {"w_p": (1e-6, 30e-6), "w_n": (0.5e-6, 20e-6)},
        "params": {
            "w_p": {"label": "W_p (PMOS 寬度)", "unit": "µm", "scale": 1e6, "fmt": "{:.2f}"},
            "w_n": {"label": "W_n (NMOS 寬度)", "unit": "µm", "scale": 1e6, "fmt": "{:.2f}"},
        },
        "parser": _parse_ringosc,
        "dump": "wrdata wave.txt v(n1)",
        "objective": "target",
        "metric": "freq",
        "target_label": "目標頻率 (GHz)",
        "target_unit": "GHz",
        "target_scale": 1e9,              # GHz -> Hz
        "target_default": 2.4,
        "start": {"w_p": 4e-6, "w_n": 2e-6},
        "waveform": "wave",
    },
    "ringosc_sky130": {
        # sky130 精準模式 = 真實電流飢餓型環形 VCO (固定振盪鏈尺寸, 只調控制電壓 Vctrl)
        "label": "VCO · sky130 精準",
        "family": "ringosc", "model": "sky130",
        "template": "vco_sky130.sp.template",
        "param_keys": ["vctrl"],
        "placeholders": {"vctrl": "{Vctrl_val}"},
        "ranges": {"vctrl": (0.2, 1.8)},
        "params": {
            "vctrl": {"label": "Vctrl (控制電壓)", "unit": "V", "scale": 1, "fmt": "{:.4f}"},
        },
        "parser": _parse_ringosc,
        "dump": "wrdata wave.txt v(Vout)",
        "objective": "target",
        "metric": "freq",
        "optimizer": "vco_hybrid",          # 用 Scipy 混合策略 (DE 全域 + 局部精調)
        "target_label": "目標頻率 (GHz)",
        "target_unit": "GHz",
        "target_scale": 1e9,
        "target_default": 1.8,
        "start": {"vctrl": 1.0},
        "waveform": "wave",
    },
}


# ----------------------------------------------------------------------
# 渲染 + 執行
# ----------------------------------------------------------------------
def render_netlist(circuit, params, dump=False):
    """讀對應範本, 替換參數與 {DUMP}, 寫出 run.sp。"""
    c = CIRCUITS[circuit]
    with open(os.path.join(BASE_DIR, c["template"]), "r", encoding="utf-8") as f:
        text = f.read()
    for k in c["param_keys"]:
        net_scale = c["params"][k].get("net_scale", 1.0)   # SI -> 網表單位 (sky130 用微米)
        text = text.replace(c["placeholders"][k], _fmt(params[k] * net_scale))
    text = text.replace("{DUMP}", c["dump"] if dump else "")
    with open(RUN_SP, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return RUN_SP


RUN_LOG = os.path.join(BASE_DIR, "run.log")
HEAL_VERBOSE = True                         # 是否印出治理安全護欄訊息
_CONV_KEYS = ("singular matrix", "iterations without convergence",
              "simulation interrupted", "no valid", "timeout",
              "valid modelname", "aborted")


def _self_heal_report():
    """讀 run.log 末 20 行, 辨識不收斂關鍵字, 印出治理訊息 (任務三)。"""
    if not HEAL_VERBOSE:
        return
    tail = ""
    try:
        with open(RUN_LOG, "r", encoding="utf-8", errors="replace") as f:
            tail = "".join(f.readlines()[-20:])
    except OSError:
        pass
    hit = next((k for k in _CONV_KEYS if k in tail.lower()), "未知收斂錯誤")
    print(f"[Agent 治理安全護欄] 偵測到電路不收斂異常 (關鍵字: {hit})，"
          f"正在跳出死區、自主調整 Vctrl 範圍進行 Self-healing...")


def _run_ngspice(timeout=8):
    """嚴密鎖定 ngspice 執行緒 (任務三): 任何中斷/不收斂都阻斷崩潰, 回傳含 ERROR 的 log。"""
    try:
        proc = subprocess.run([NGSPICE, "-b", RUN_SP], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=BASE_DIR, timeout=timeout)
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        try:
            with open(RUN_LOG, "w", encoding="utf-8") as f:
                f.write(log)
        except OSError:
            pass
        # Exit Code != 0 或出現不收斂關鍵字 -> 自我修復回報
        if proc.returncode != 0 or any(k in log.lower() for k in _CONV_KEYS):
            _self_heal_report()
        return log
    except subprocess.TimeoutExpired:
        with open(RUN_LOG, "w", encoding="utf-8") as f:
            f.write("ERROR: ngspice timeout (transient/operating point did not converge)\n")
        _self_heal_report()
        return "ERROR: ngspice timeout"
    except Exception as ex:                  # 任何其他例外都不讓它崩潰
        return f"ERROR: ngspice subprocess failed: {ex}"


def _crashed(log):
    """區分『瞬態崩潰(flaky, 可重試)』與『乾淨無振盪(確定性, 不重試)』。"""
    low = log.lower()
    return any(k in low for k in ("singular matrix", "iterations without convergence",
                                  "simulation interrupted", "timeout", "subprocess failed"))


def run_circuit(circuit, params, dump=False, retries=2):
    """跑一次模擬。若為瞬態 flaky 崩潰(非確定性無振盪)則自動重試 (sky130 瞬態非決定性)。"""
    result, log = None, ""
    for _ in range(retries + 1):
        render_netlist(circuit, params, dump=dump)
        log = _run_ngspice()
        result = CIRCUITS[circuit]["parser"](log)
        if result.get("ok") or not _crashed(log):
            break                               # 成功, 或確定性失敗(死區) -> 不再重試
    result.update({"circuit": circuit, "params": dict(params)})
    return result


def run_isolated(circuit, params, tag, inject="", dump_vec=None, replaces=None):
    """
    線程安全的單次模擬: 渲染到唯一檔名 (避免多線程共用 run.sp 競爭)。
      inject   : 在 .control 前插入的額外 SPICE 行 (PVT 用: .options temp / 改電源等)
      dump_vec : 若給定, 匯出該向量到唯一波形檔, 回傳於 result['_wave']
      replaces : [(old, new), ...] 額外文字替換 (PVT 用: 切 corner / 改電源電壓)
    回傳 metrics dict。
    """
    c = CIRCUITS[circuit]
    sp = os.path.join(BASE_DIR, f"sweep_{tag}.sp")
    wavef = os.path.join(BASE_DIR, f"wave_{tag}.txt")
    with open(os.path.join(BASE_DIR, c["template"]), "r", encoding="utf-8") as f:
        text = f.read()
    for k in c["param_keys"]:
        ns = c["params"][k].get("net_scale", 1.0)
        text = text.replace(c["placeholders"][k], _fmt(params[k] * ns))
    for old, new in (replaces or []):
        text = text.replace(old, new)
    dump_cmd = f"wrdata {os.path.basename(wavef)} {dump_vec}" if dump_vec else ""
    text = text.replace("{DUMP}", dump_cmd)
    if inject:                                   # PVT 注入 (插在 .control 之前)
        text = text.replace(".control", inject + "\n.control", 1)
    with open(sp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    try:
        proc = subprocess.run([NGSPICE, "-b", sp], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=BASE_DIR, timeout=20)
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        log = "ERROR: ngspice timeout"
    except Exception as ex:
        log = f"ERROR: {ex}"
    res = c["parser"](log)
    res.update({"circuit": circuit, "params": dict(params)})
    if dump_vec and os.path.exists(wavef):
        rows = []
        with open(wavef, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                p = line.split()
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    continue
        res["_wave"] = rows
    for fp in (sp, wavef):
        try:
            os.remove(fp)
        except OSError:
            pass
    return res


def run_waveform(circuit, params):
    """跑一次並回傳圖表資料; 波形為空且為 flaky 崩潰時自動重試。"""
    rows = []
    for _ in range(3):
        if os.path.exists(WAVE_PATH):
            os.remove(WAVE_PATH)
        r = run_circuit(circuit, params, dump=True, retries=0)
        if os.path.exists(WAVE_PATH):
            with open(WAVE_PATH, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.split()
                    try:
                        rows.append([float(x) for x in parts])
                    except ValueError:
                        continue
        if rows:
            break                               # 拿到波形即可

    kind = CIRCUITS[circuit]["waveform"]
    if kind == "bode":
        freq = [r[0] for r in rows if len(r) >= 4]
        mag = [r[1] for r in rows if len(r) >= 4]
        phase = [r[3] for r in rows if len(r) >= 4]
        return {"kind": "bode", "freq": freq, "mag_db": mag, "phase_deg": phase}
    if kind == "temp":
        temp = [r[0] for r in rows if len(r) >= 2]
        vref = [r[1] for r in rows if len(r) >= 2]
        return {"kind": "temp", "temp": temp, "vref": vref}
    if kind == "wave":
        t = [r[0] for r in rows if len(r) >= 2]
        v = [r[1] for r in rows if len(r) >= 2]
        return {"kind": "wave", "t": t, "v": v}
    return {"kind": kind}


# ----------------------------------------------------------------------
# 向後相容 (OPA)
# ----------------------------------------------------------------------
def run_simulation(w_diff, w_stage2, r_bias, verbose=False):
    r = run_circuit("opa", {"w_diff": w_diff, "w_stage2": w_stage2, "r_bias": r_bias})
    return r


def run_bode(w_diff, w_stage2, r_bias):
    return run_waveform("opa", {"w_diff": w_diff, "w_stage2": w_stage2, "r_bias": r_bias})


# ----------------------------------------------------------------------
# 自我測試: 三種電路各跑一次
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"ngspice: {NGSPICE}\n")
    tests = [
        ("opa", {"w_diff": 30e-6, "w_stage2": 80e-6, "r_bias": 80e3}),
        ("bandgap", {"r_trim": 9e3, "n_bjt": 8}),
        ("ringosc", {"w_p": 9e-6, "w_n": 4.5e-6}),
    ]
    for ckt, p in tests:
        r = run_circuit(ckt, p)
        print(f"[{ckt:8}] ok={r['ok']}  " +
              "  ".join(f"{k}={v}" for k, v in r.items()
                        if k not in ("circuit", "params", "ok", "converged")))
