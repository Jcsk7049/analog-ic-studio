# -*- coding: utf-8 -*-
"""
analyzer.py — 類比電路自動調試 Agent 的「資料/敏感度分析層」

職責:
  1. 在參數空間內用 Latin Hypercube 採樣 N 組 (W_diff, W_stage2, R_bias)
  2. 批量呼叫 eda_control.run_simulation 跑模擬, 存成 CSV
  3. 用線性回歸(標準化係數) 與 隨機森林(Feature Importance)
     計算三個輸入參數對 Gain 的影響度(%)

對外主要介面:
    collect_dataset(n=20)        -> pandas.DataFrame (同時存 CSV)
    compute_importance(df, 'gain') -> dict
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import os
from concurrent.futures import ThreadPoolExecutor

import eda_control as eda
from eda_control import CIRCUITS

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "samples.csv")

# 參數搜尋範圍 (SI 單位)        下限      上限
PARAM_RANGES = {
    "w_diff":   (5e-6,   100e-6),   # 差動對寬度  5u ~ 100u
    "w_stage2": (20e-6,  200e-6),   # 第二級寬度  20u ~ 200u
    "r_bias":   (20e3,   500e3),    # 偏壓電阻    20k ~ 500k
}
FEATURES = list(PARAM_RANGES.keys())


# ----------------------------------------------------------------------
# 步驟 3-1: 採樣
# ----------------------------------------------------------------------
def generate_samples(n=20, seed=42):
    """用 Latin Hypercube 在參數空間均勻採樣 n 組, 回傳 ndarray (n,3)。"""
    lows = np.array([PARAM_RANGES[f][0] for f in FEATURES])
    highs = np.array([PARAM_RANGES[f][1] for f in FEATURES])
    sampler = qmc.LatinHypercube(d=len(FEATURES), seed=seed)
    unit = sampler.random(n)               # [0,1)^3
    return qmc.scale(unit, lows, highs)     # 縮放到實際範圍


def collect_dataset(n=20, seed=42, save=True, progress=True):
    """採樣 + 批量模擬, 回傳 DataFrame 並存 CSV。"""
    samples = generate_samples(n, seed)
    rows = []
    for i, (wd, ws, rb) in enumerate(samples, 1):
        r = eda.run_simulation(wd, ws, rb)
        rows.append(r)
        if progress:
            g = f"{r['gain']:.2f}" if r["gain"] is not None else "  --  "
            pm = f"{r['pm']:.1f}" if r["pm"] is not None else " -- "
            print(f"  [{i:2d}/{n}] Wd={wd*1e6:6.1f}u Ws={ws*1e6:6.1f}u "
                  f"Rb={rb/1e3:6.1f}k -> Gain={g} dB  PM={pm}  ok={r['ok']}")

    df = pd.DataFrame(rows)[
        ["w_diff", "w_stage2", "r_bias", "gain", "ugf", "pm", "converged", "ok"]
    ]
    if save:
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"\n  已存檔: {CSV_PATH}  ({len(df)} 筆, "
              f"{int(df['ok'].sum())} 筆有效)")
    return df


# ----------------------------------------------------------------------
# 步驟 3-2: 影響度 / 敏感度
# ----------------------------------------------------------------------
def compute_importance(df, target="gain"):
    """
    用兩種方法量化 W_diff/W_stage2/R_bias 對 target 的影響度:
      - linear : 標準化線性回歸係數絕對值 (帶正負號方向) → 局部敏感度
      - forest : 隨機森林 Feature Importance
    回傳 dict, 影響度皆已正規化為百分比 (總和=100%)。
    """
    data = df[df["ok"]].dropna(subset=FEATURES + [target])
    if len(data) < 4:
        raise ValueError(f"有效樣本不足 ({len(data)} 筆), 無法做回歸分析。")

    X = data[FEATURES].values
    y = data[target].values

    # --- 標準化, 讓不同量級參數的係數可比較 ---
    Xs = StandardScaler().fit_transform(X)

    # --- 線性回歸: 標準化係數 = 局部敏感度 ---
    lin = LinearRegression().fit(Xs, y)
    coef = lin.coef_
    lin_pct = 100 * np.abs(coef) / np.sum(np.abs(coef))

    # --- 隨機森林: Feature Importance ---
    rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(Xs, y)
    rf_pct = 100 * rf.feature_importances_

    result = {
        "target": target,
        "n_used": len(data),
        "r2_linear": lin.score(Xs, y),
        "r2_forest": rf.score(Xs, y),
        "features": {},
    }
    for i, f in enumerate(FEATURES):
        result["features"][f] = {
            "linear_pct": float(lin_pct[i]),
            "linear_sign": "+" if coef[i] >= 0 else "-",   # 正=同向, 負=反向
            "forest_pct": float(rf_pct[i]),
            "coef_std": float(coef[i]),                    # 標準化係數(每 1σ 改變對 target 的影響量)
        }
    return result


def print_importance(imp):
    """文字版影響度報表。"""
    print(f"\n=== 參數對 {imp['target'].upper()} 的影響度 "
          f"(有效樣本 {imp['n_used']} 筆) ===")
    print(f"  擬合度  線性 R2={imp['r2_linear']:.3f}   "
          f"隨機森林 R2={imp['r2_forest']:.3f}")
    print(f"  {'參數':<10} {'線性敏感度':>12} {'方向':>5} {'森林重要度':>12} {'標準化係數':>12}")
    for f, v in imp["features"].items():
        print(f"  {f:<10} {v['linear_pct']:>10.1f} % {v['linear_sign']:>5} "
              f"{v['forest_pct']:>10.1f} % {v['coef_std']:>12.3f}")


# ----------------------------------------------------------------------
# 自我測試
# ----------------------------------------------------------------------
if __name__ == "__main__":
    n = 20
    if "-n" in sys.argv:
        n = int(sys.argv[sys.argv.index("-n") + 1])

    print(f"採樣 {n} 組參數並批量模擬 (Latin Hypercube)...\n")
    df = collect_dataset(n=n)

    for tgt in ("gain", "pm"):
        try:
            imp = compute_importance(df, tgt)
            print_importance(imp)
        except ValueError as e:
            print(f"\n[{tgt}] 跳過: {e}")


# ----------------------------------------------------------------------
# 雙變數交叉參數掃描 (2D Parametric Sweep) — 多線程 + 獨立網表 (論文任務一)
# ----------------------------------------------------------------------
_METRIC_DISP = {"gain": ("Gain (dB)", 1.0), "pm": ("Phase Margin (deg)", 1.0),
                "ugf": ("UGF (MHz)", 1e-6), "freq": ("Frequency (GHz)", 1e-9),
                "tc": ("TC (ppm/degC)", 1.0), "vref": ("Vref (V)", 1.0)}


def sweep_2d(circuit, key_x=None, key_y=None, n=20, metric=None, max_workers=8):
    """兩個引數各取 n 點, 組成 n×n 實驗矩陣, 多線程跑 ngspice, 回傳 3D 曲面資料。"""
    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    if len(keys) < 2:
        raise ValueError(f"{circuit} 僅 {len(keys)} 個可調參數, 無法做 2D 掃描")
    key_x = key_x or keys[0]
    key_y = key_y or keys[1]
    metric = metric or c["metric"]

    lox, hix = c["ranges"][key_x]
    loy, hiy = c["ranges"][key_y]
    xs = np.linspace(lox, hix, n)
    ys = np.linspace(loy, hiy, n)
    base = dict(c["start"])

    combos = [(i, j) for j in range(n) for i in range(n)]   # 共 n*n 組

    def _run(ij):
        i, j = ij
        p = dict(base); p[key_x] = float(xs[i]); p[key_y] = float(ys[j])
        r = eda.run_isolated(circuit, p, tag=f"{i}_{j}")
        v = r.get(metric) if r.get("ok") else None
        return (i, j, v)

    Z = [[None] * n for _ in range(n)]                       # Z[j][i]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, j, v in ex.map(_run, combos):
            Z[j][i] = v

    mlabel, mscale = _METRIC_DISP.get(metric, (metric, 1.0))
    pxs = c["params"][key_x]; pys = c["params"][key_y]
    Zs = [[(v * mscale if v is not None else None) for v in row] for row in Z]
    n_ok = sum(1 for row in Z for v in row if v is not None)
    return {
        "circuit": circuit, "metric": metric,
        "x": [float(v * pxs["scale"]) for v in xs],
        "y": [float(v * pys["scale"]) for v in ys],
        "z": Zs,
        "x_label": f"{key_x} ({pxs['unit']})",
        "y_label": f"{key_y} ({pys['unit']})",
        "z_label": mlabel,
        "n": n, "n_ok": n_ok, "total": n * n,
    }
