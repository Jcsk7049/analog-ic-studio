# -*- coding: utf-8 -*-
"""
agent_main.py — 通用類比 IC 自動調參 Agent (閉環優化 + 中文面板)

支援三電路模式 (見 eda_control.CIRCUITS):
  - opa     : 逼近目標增益, 同時約束相位裕度 PM > 45° (低於則懲罰)
  - bandgap : 自動最小化溫漂係數 TC (ppm/°C)
  - ringosc : 逼近目標振盪頻率

統一優化核心 run_optimization(circuit, target) 供 CLI 與網頁共用。
用法:
    python agent_main.py                 # 互動選電路與目標
    python agent_main.py opa 60          # OPA 目標 60 dB
    python agent_main.py bandgap         # Bandgap 最小化 TC
    python agent_main.py ringosc 2.4     # RO 目標 2.4 GHz
"""

import os
import sys
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.align import Align
from rich import box

import eda_control as eda
from eda_control import CIRCUITS

sys.stdout.reconfigure(encoding="utf-8")
console = Console()

TOL = 0.01           # target 模式: 相對誤差 < 1%
TC_GOOD = 5.0        # bandgap: TC < 5 ppm/°C 視為達標
MAX_ITERS = 30
FD_STEP = 0.03       # 正規化空間有限差分步長
PM_MIN = 45.0        # OPA 相位裕度約束
PM_PENALTY = 4.0     # PM 不足的懲罰權重


# ----------------------------------------------------------------------
# 正規化參數空間 helpers (依電路)
# ----------------------------------------------------------------------
def _bounds(ckt):
    keys = CIRCUITS[ckt]["param_keys"]
    lo = np.array([CIRCUITS[ckt]["ranges"][k][0] for k in keys])
    hi = np.array([CIRCUITS[ckt]["ranges"][k][1] for k in keys])
    return keys, lo, hi


def to_real(ckt, p):
    keys, lo, hi = _bounds(ckt)
    vals = lo + np.clip(p, 0, 1) * (hi - lo)
    return {k: float(vals[i]) for i, k in enumerate(keys)}


def to_norm(ckt, real):
    keys, lo, hi = _bounds(ckt)
    arr = np.array([real[k] for k in keys])
    return (arr - lo) / (hi - lo)


def metrics_at(ckt, p):
    return eda.run_circuit(ckt, to_real(ckt, p))


def primary_val(ckt, m):
    return m.get(CIRCUITS[ckt]["metric"])


# ----------------------------------------------------------------------
# 目標 / 損失 / 收斂 (多目標約束)
# ----------------------------------------------------------------------
BIG = 1e6


def loss_of(ckt, m, target):
    obj = CIRCUITS[ckt]["objective"]
    if not m.get("ok"):
        return BIG
    if obj == "target":
        val = primary_val(ckt, m)
        if val is None:
            return BIG
        e = (val - target) / (abs(target) + 1e-12)
        loss = e * e
        if CIRCUITS[ckt].get("pm_constraint"):   # 相位裕度約束
            pm = m.get("pm")
            if pm is not None and pm < PM_MIN:
                loss += PM_PENALTY * ((PM_MIN - pm) / PM_MIN) ** 2
        return loss
    else:                                       # minimize TC
        tc = m.get("tc")
        if tc is None:
            return BIG
        return (tc / 50.0) ** 2


def is_converged(ckt, m, target):
    obj = CIRCUITS[ckt]["objective"]
    if not m.get("ok"):
        return False
    if obj == "target":
        val = primary_val(ckt, m)
        ok = abs(val - target) / (abs(target) + 1e-12) < TOL
        if CIRCUITS[ckt].get("pm_constraint"):
            ok = ok and (m.get("pm") is not None and m["pm"] >= PM_MIN)
        return ok
    else:
        return m.get("tc") is not None and abs(m["tc"]) < TC_GOOD


# ----------------------------------------------------------------------
# 梯度 (有限差分)
# ----------------------------------------------------------------------
def grad_primary(ckt, p, base_val):
    """主指標對各正規化參數的梯度 (供步進方向 + 影響度)。"""
    g = np.zeros(len(p))
    for i in range(len(p)):
        pp = p.copy()
        step = FD_STEP if pp[i] + FD_STEP <= 1 else -FD_STEP
        pp[i] += step
        v = primary_val(ckt, metrics_at(ckt, pp))
        g[i] = 0.0 if (v is None or base_val is None) else (v - base_val) / step
    return g


def grad_loss(ckt, p, target, base_loss):
    g = np.zeros(len(p))
    for i in range(len(p)):
        pp = p.copy()
        step = FD_STEP if pp[i] + FD_STEP <= 1 else -FD_STEP
        pp[i] += step
        L = loss_of(ckt, metrics_at(ckt, pp), target)
        g[i] = (L - base_loss) / step
    return g


def _gn_active(p, gm, val, tgt):
    """最小範數高斯-牛頓步 + active-set 投影:
    參數撞界後凍結, 把剩餘誤差重分配給自由參數 (避免主導參數夾死導致停滯)。"""
    err = tgt - val
    active = np.ones(len(p), dtype=bool)
    pcur = p.copy()
    for _ in range(len(p)):
        g = gm * active
        gg = float(g @ g)
        if gg < 1e-18:
            break
        step = np.clip(g * err / gg, -0.4, 0.4)
        cand = pcur + step
        viol = ((cand < 0) | (cand > 1)) & active
        if not viol.any():
            pcur = cand
            break
        for i in np.where(viol)[0]:
            pcur[i] = 0.0 if cand[i] < 0 else 1.0
            active[i] = False
    return pcur - p


def propose_direction(ckt, p, m, tgt, base_loss):
    """回傳建議步進向量 base 與主指標梯度 gm (tgt 為 metric 原生單位)。"""
    gm = grad_primary(ckt, p, primary_val(ckt, m))
    obj = CIRCUITS[ckt]["objective"]
    pm_violated = (CIRCUITS[ckt].get("pm_constraint") and m.get("pm") is not None and m["pm"] < PM_MIN)

    if obj == "target" and not pm_violated:
        val = primary_val(ckt, m)
        if val is None or float(gm @ gm) < 1e-18:
            base = -grad_loss(ckt, p, tgt, base_loss)
        else:                                   # 高斯-牛頓 + active-set
            base = _gn_active(p, gm, val, tgt)
    else:
        # 最小化 TC, 或 OPA 違反 PM 約束 -> 沿損失負梯度 (兼顧誤差+懲罰)
        gl = grad_loss(ckt, p, tgt, base_loss)
        n = np.linalg.norm(gl)
        base = (-gl / n * 0.3) if n > 1e-12 else np.zeros(len(p))
    return base, gm


# ----------------------------------------------------------------------
# 統一閉環優化核心
# ----------------------------------------------------------------------
def _use_surrogate(circuit):
    """sky130 精準模式且已訓練替代模型 -> 用 DNN 引擎。"""
    if CIRCUITS[circuit].get("model") != "sky130":
        return False
    return os.path.exists(os.path.join(os.path.dirname(__file__), "data", f"surrogate_{circuit}.pth"))


# 各電路要在面板顯示「多維度影響度」的規格
SHOW_SPECS = {
    "opa": ["gain", "pm"], "opa_sky130": ["gain", "pm"],
    "bandgap": ["tc", "vref"], "bandgap_sky130": ["tc", "vref"],
    "ringosc": ["freq"], "ringosc_sky130": ["freq"],
}


def compute_multi_influence(circuit, params):
    """各參數對「所有規格」的特徵重要性 (正規化%) 與相關方向 (↑/↓)。
    sky130 用 DNN 一次預測全規格; 快速電路用 ngspice 有限差分 (d+1 次)。"""
    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    specs = SHOW_SPECS[circuit]
    lo = np.array([c["ranges"][k][0] for k in keys])
    hi = np.array([c["ranges"][k][1] for k in keys])
    base = np.array([params[k] for k in keys])
    pts = [base] + [base + np.eye(len(keys))[i] * (hi - lo) * 0.04 for i in range(len(keys))]
    M = np.array(pts)

    if _use_surrogate(circuit):
        import dl_surrogate as dl
        sur = dl.Surrogate(circuit)
        pred = sur.predict(M)
        vals = {t: pred[:, j] for j, t in enumerate(sur.targets)}
    else:
        vals = {s: [] for s in specs}
        for row in M:
            r = eda.run_circuit(circuit, {k: row[j] for j, k in enumerate(keys)})
            for s in specs:
                vals[s].append(r.get(s))
        vals = {s: np.array([v if v is not None else np.nan for v in vals[s]], float) for s in specs}

    multi = {k: [] for k in keys}
    for s in specs:
        b = vals[s][0]
        g = np.array([(vals[s][i + 1] - b) for i in range(len(keys))])
        g = np.nan_to_num(g)
        tot = np.sum(np.abs(g)) or 1.0
        for i, k in enumerate(keys):
            multi[k].append({"spec": s, "pct": 100 * abs(g[i]) / tot,
                             "sign": "+" if g[i] >= 0 else "-"})
    return multi


def run_optimization(circuit, target=None, p0=None, engine="auto"):
    """引擎分派: VCO -> Scipy 混合; sky130 -> DNN 群體尋優; 否則傳統 ngspice 閉環。"""
    if CIRCUITS[circuit].get("optimizer") == "vco_hybrid":
        import optimizer
        return optimizer.run_vco_optimization(circuit, target)   # 已含 multi_influence
    if engine == "surrogate" or (engine == "auto" and _use_surrogate(circuit)):
        try:
            res = run_optimization_surrogate(circuit, target)
        except (FileNotFoundError, ImportError):
            res = _run_optimization_classic(circuit, target, p0)
    else:
        res = _run_optimization_classic(circuit, target, p0)
    try:
        res["multi_influence"] = compute_multi_influence(circuit, res["final"]["params"])
    except Exception:
        res["multi_influence"] = None
    return res


def run_optimization_surrogate(circuit, target=None):
    """DNN 替代模型引擎: 百萬級群體尋優 (秒級) + 1 次 ngspice 校正。"""
    import dl_surrogate as dl
    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    if target is None:
        target = c["target_default"]
    tgt = target * c.get("target_scale", 1.0)

    res = dl.surrogate_optimize(circuit, target, pop=300000, verify=True)
    best, pred, ver = res["params"], res["predicted"], res["verified"]

    # 影響度: DNN 對主指標的梯度 (正規化空間有限差分, 不耗 ngspice)
    sur = dl.Surrogate(circuit)
    lo = np.array([c["ranges"][k][0] for k in keys])
    hi = np.array([c["ranges"][k][1] for k in keys])
    base = np.array([best[k] for k in keys])
    mi = sur.targets.index(c["metric"])
    base_pred = sur.predict(base)[0][mi]
    g = np.zeros(len(keys))
    for i in range(len(keys)):
        xp = base.copy(); xp[i] += (hi[i] - lo[i]) * 0.03
        g[i] = sur.predict(xp)[0][mi] - base_pred           # 對正規化參數的梯度
    absum = np.sum(np.abs(g)) or 1.0
    infl = {k: {"pct": 100 * abs(g[j]) / absum, "sign": "+" if g[j] >= 0 else "-"}
            for j, k in enumerate(keys)}

    def _rec(metrics, step, note):
        r = {"step": step, "params": dict(best), "metrics": _clean(metrics),
             "influence": infl, "note": note}
        if c["objective"] == "target":
            v = metrics.get(c["metric"])
            r["err_pct"] = 100 * (v - tgt) / (abs(tgt) + 1e-12) if v is not None else 0.0
        return r

    steps = [_rec(pred, 1, "DNN 預測"), _rec(ver, 2, "ngspice 校正")]
    primary = ver.get(c["metric"])
    final = {"params": dict(best), "metrics": _clean(ver), "primary": primary,
             "step": 2, "predicted": pred, "surrogate_r2": res["surrogate_r2"]}
    if c["objective"] == "target":
        final["err_pct"] = 100 * (primary - tgt) / (abs(tgt) + 1e-12) if primary is not None else 0.0

    ok = ver.get("ok") and (is_converged(circuit, ver, tgt) if c["objective"] == "target" else True)
    return {"circuit": circuit, "target": target, "status": "converged" if ok else "best",
            "method": "surrogate", "steps": steps, "final": final}


def _run_optimization_classic(circuit, target=None, p0=None):
    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    if target is None:
        target = c["target_default"]
    tgt = target * c.get("target_scale", 1.0)   # 轉成 metric 原生單位
    if p0 is None:
        p0 = to_norm(circuit, c["start"])
    p = np.array(p0, dtype=float)

    steps, best = [], None
    status = "max_iters"

    for step in range(1, MAX_ITERS + 1):
        m = metrics_at(circuit, p)
        if not m.get("ok"):                     # 模擬失敗 -> 微擾重試
            steps.append({"step": step, "fail": True})
            p = np.clip(p + 0.04, 0, 1)
            continue

        L = loss_of(circuit, m, tgt)
        prim = primary_val(circuit, m)

        base, gm = propose_direction(circuit, p, m, tgt, L)
        absum = np.sum(np.abs(gm)) or 1.0
        infl = {k: {"pct": 100 * abs(gm[i]) / absum,
                    "sign": "+" if gm[i] >= 0 else "-"}
                for i, k in enumerate(keys)}

        rec = {"step": step, "params": to_real(circuit, p), "metrics": _clean(m),
               "loss": L, "primary": prim, "influence": infl}
        if c["objective"] == "target":
            rec["err_pct"] = 100 * (prim - tgt) / (abs(tgt) + 1e-12)
        steps.append(rec)

        if best is None or L < best[0]:
            best = (L, p.copy(), m)

        if is_converged(circuit, m, tgt):
            status = "converged"
            break

        # --- 含邊界夾限的 line search: 取候選中損失最低者 (避免過衝/原地踏步) ---
        cands = [(L, p)]
        for alpha in (1.0, 0.6, 0.35, 0.2, 0.12, 0.06, 0.03):
            pn = np.clip(p + alpha * base, 0, 1)
            if np.allclose(pn, p, atol=1e-4):
                continue
            cands.append((loss_of(circuit, metrics_at(circuit, pn), tgt), pn))
        bestL, bestP = min(cands, key=lambda t: t[0])
        if bestL < L - 1e-9:
            p = bestP
        else:
            # 最小化模式無法再改善 = 已抵局部最小 (視為成功)
            status = "converged" if c["objective"] == "minimize" else "stalled"
            break

    bL, bp, bm = best
    final = {"params": to_real(circuit, bp), "metrics": _clean(bm), "loss": bL,
             "primary": primary_val(circuit, bm),
             "step": len([s for s in steps if not s.get("fail")])}
    if c["objective"] == "target":
        final["err_pct"] = 100 * (final["primary"] - tgt) / (abs(tgt) + 1e-12)
    return {"circuit": circuit, "target": target, "status": status,
            "method": "classic", "steps": steps, "final": final}


def _clean(m):
    return {k: m.get(k) for k in ("gain", "pm", "ugf", "vref", "tc", "freq") if m.get(k) is not None}


# ----------------------------------------------------------------------
# rich 面板 (依電路動態)
# ----------------------------------------------------------------------
def _bar(pct, width=14):
    f = int(round(pct / 100 * width))
    return "█" * f + "░" * (width - f)


def _metric_rows(ckt, m, target):
    metric = CIRCUITS[ckt]["metric"]
    rows = []
    if metric == "gain":
        rows.append(("目標 Gain", f"[bold cyan]{target:.2f} dB[/]"))
        rows.append(("實測 Gain", f"[bold yellow]{m.get('gain', float('nan')):.2f} dB[/]"))
        pm = m.get("pm")
        pmc = "green" if (pm and pm >= PM_MIN) else "red"
        rows.append(("相位裕度 PM", f"[{pmc}]{pm:.1f}°  (約束 ≥{PM_MIN:.0f}°)[/]" if pm else "--"))
        if m.get("ugf"):
            rows.append(("單位增益頻率", f"[blue]{m['ugf']/1e6:.2f} MHz[/]"))
    elif metric == "tc":
        rows.append(("輸出 Vref", f"[bold yellow]{m.get('vref', float('nan')):.4f} V[/]"))
        tc = m.get("tc")
        tcc = "green" if (tc is not None and abs(tc) < TC_GOOD) else "amber"
        rows.append(("溫漂 TC", f"[bold {tcc}]{tc:.2f} ppm/°C[/]" if tc is not None else "--"))
        rows.append(("優化目標", "[cyan]最小化 TC → 0[/]"))
    elif metric == "freq":
        rows.append(("目標頻率", f"[bold cyan]{target:.3f} GHz[/]"))
        f = m.get("freq")
        rows.append(("實測頻率", f"[bold yellow]{f/1e9:.3f} GHz[/]" if f else "--"))
    return rows


def render_step(ckt, step, target, rec):
    c = CIRCUITS[ckt]
    header = Align.center(Text(f"  迭代步數  Step {step:>2d} / {MAX_ITERS}  ",
                               style="bold white on blue"))

    # 參數表
    tp = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", pad_edge=False)
    tp.add_column("參數"); tp.add_column("當前值", justify="right", style="bold yellow"); tp.add_column("單位", style="dim")
    for k in c["param_keys"]:
        pm = c["params"][k]
        tp.add_row(pm["label"], pm["fmt"].format(rec["params"][k] * pm["scale"]), pm["unit"])
    p_panel = Panel(tp, title="[bold]🔧 當前設計參數[/]", border_style="cyan", box=box.ROUNDED)

    # 量測表
    tm = Table.grid(padding=(0, 2)); tm.add_column(justify="right", style="dim"); tm.add_column()
    for label, val in _metric_rows(ckt, rec["metrics"], target):
        tm.add_row(label, val)
    if "err_pct" in rec:
        ec = "green" if abs(rec["err_pct"]) < 1 else ("yellow" if abs(rec["err_pct"]) < 5 else "red")
        tm.add_row("相對誤差", f"[bold {ec}]{rec['err_pct']:+.2f} %[/]")
    m_panel = Panel(tm, title="[bold]📊 模擬實測結果[/]", border_style="yellow", box=box.ROUNDED)

    # 影響度
    ti = Table(box=box.SIMPLE, show_header=True, header_style="bold green", pad_edge=False)
    ti.add_column("參數"); ti.add_column("影響度"); ti.add_column("%", justify="right", style="bold"); ti.add_column("方向", justify="center")
    for k in c["param_keys"]:
        info = rec["influence"][k]
        col = "green" if info["pct"] >= 40 else ("yellow" if info["pct"] >= 20 else "white")
        ti.add_row(c["params"][k]["label"], f"[{col}]{_bar(info['pct'])}[/]",
                   f"{info['pct']:4.1f}", "↑同向" if info["sign"] == "+" else "↓反向")
    i_panel = Panel(ti, title="[bold]🎯 各參數當前影響度[/]", border_style="green", box=box.ROUNDED)

    console.print(header)
    console.print(Columns([p_panel, m_panel], equal=True, expand=True))
    console.print(i_panel)
    console.rule(style="dim")


def optimize(circuit, target=None):
    c = CIRCUITS[circuit]
    if target is None:
        target = c["target_default"]
    console.print(Panel(
        f"[bold]電路: {c['label']}[/]\n"
        f"優化標的: {c['target_label']}"
        + (f"  →  目標 = {target:g} {c['target_unit']}" if c["objective"] == "target" else "")
        + f"\n收斂門檻: "
        + ("相對誤差 < 1%" + (" 且 PM ≥ 45°" if c.get("pm_constraint") else "")
           if c["objective"] == "target" else f"TC < {TC_GOOD:.0f} ppm/°C"),
        title="[bold blue]🤖 通用類比 IC 自動調參 Agent[/]", border_style="blue", box=box.DOUBLE))

    result = run_optimization(circuit, target)
    for rec in result["steps"]:
        if rec.get("fail"):
            console.print(f"[red]Step {rec['step']}: 模擬不收斂, 微調後重試[/]")
            continue
        render_step(circuit, rec["step"], target, rec)

    _summary(circuit, target, result)
    return result


def _summary(circuit, target, result):
    c = CIRCUITS[circuit]
    f = result["final"]
    m = f["metrics"]
    ok = result["status"] == "converged"
    lines = [f"[bold {'green' if ok else 'yellow'}]"
             + ("✔ 收斂成功!" if ok else "⚑ 已達邊界/上限, 回報最佳近似") + f"[/]  第 {f['step']} 步\n"]
    metric = c["metric"]
    if metric == "gain":
        lines.append(f"目標 Gain {target:.2f} dB → 實測 [bold]{m.get('gain'):.3f} dB[/] "
                     f"(誤差 {f['err_pct']:+.3f}%),  PM = {m.get('pm'):.1f}°")
    elif metric == "tc":
        lines.append(f"溫漂 TC = [bold]{m.get('tc'):.3f} ppm/°C[/],  Vref = {m.get('vref'):.4f} V")
    elif metric == "freq":
        lines.append(f"目標 {target:.3f} GHz → 實測 [bold]{m.get('freq')/1e9:.4f} GHz[/] "
                     f"(誤差 {f['err_pct']:+.3f}%)")
    lines.append("\n最終參數:")
    for k in c["param_keys"]:
        pm = c["params"][k]
        lines.append(f"  {pm['label']:<16} = {pm['fmt'].format(f['params'][k]*pm['scale'])} {pm['unit']}")
    console.print(Panel("\n".join(lines),
                        title=f"[bold {'green' if ok else 'yellow'}]🎉 自動調試完成[/]",
                        border_style="green" if ok else "yellow", box=box.DOUBLE))


# ----------------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:]]
    circuit = None
    target = None
    for a in args:
        if a in CIRCUITS:
            circuit = a
        else:
            try:
                target = float(a)
            except ValueError:
                pass
    if circuit is None:
        console.print("可用電路: " + " | ".join(f"[cyan]{k}[/] ({v['label']})" for k, v in CIRCUITS.items()))
        circuit = (console.input("[bold]選擇電路 (opa/bandgap/ringosc, 預設 opa): [/]").strip() or "opa")
        if circuit not in CIRCUITS:
            circuit = "opa"
    if target is None and CIRCUITS[circuit]["objective"] == "target":
        raw = console.input(f"[bold cyan]{CIRCUITS[circuit]['target_label']} "
                            f"(Enter 用預設 {CIRCUITS[circuit]['target_default']:g}): [/]")
        try:
            target = float(raw) if raw.strip() else None
        except ValueError:
            target = None
    optimize(circuit, target)


if __name__ == "__main__":
    main()
