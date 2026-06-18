# -*- coding: utf-8 -*-
"""
dl_transfer.py — 跨製程工藝遷移學習 (任務二)

目的:
  學習「快速理想模型」與「sky130 真實製程」在相同參數下的物理行為差異
  (通道長度調變、遷移率 µCox 等非線性效應造成的系統性偏移)。
  以一個輕量級殘差映射 (Ridge / 線性變換) 表示:
        sky130_metric ≈ fast_metric + Residual(params)
  再用此校正模型把「快速模型解」遷移成「精準模型暖啟動點」,
  使精準模型的初始搜尋點直接落在收斂區間, 大幅縮短迭代步數。

對外介面:
  build_transfer(family)         -> 訓練殘差映射 (用兩邊替代模型, 不耗 ngspice)
  corrected_predict(family, X)   -> 快速模型 + 殘差 = sky130 估計
  warm_start(family, target)     -> 回傳精準模型的暖啟動正規化參數
  demo(family, target)           -> 比較冷啟動 vs 暖啟動的迭代步數
"""

import os
import sys
import numpy as np
from sklearn.linear_model import Ridge

import eda_control as eda
from eda_control import CIRCUITS
import dl_surrogate as dl

sys.stdout.reconfigure(encoding="utf-8")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# family -> (快速模型 key, sky130 精準模型 key)
PAIR = {"opa": ("opa", "opa_sky130"),
        "bandgap": ("bandgap", "bandgap_sky130"),
        "ringosc": ("ringosc", "ringosc_sky130")}


def _norm(ckt, X):
    keys = CIRCUITS[ckt]["param_keys"]
    lo = np.array([CIRCUITS[ckt]["ranges"][k][0] for k in keys])
    hi = np.array([CIRCUITS[ckt]["ranges"][k][1] for k in keys])
    return (X - lo) / (hi - lo)


def _ensure_fast_surrogate(fast, n=250):
    if not os.path.exists(os.path.join(BASE_DIR, "data", f"surrogate_{fast}.pth")):
        dl.train(fast, n=n, epochs=400, regen=True, verbose=False)


def build_transfer(family, n=4000, seed=3):
    """在 sky130 參數範圍內取樣, 用兩邊替代模型算殘差, 擬合 Ridge 線性殘差映射。"""
    fast, sky = PAIR[family]
    _ensure_fast_surrogate(fast)
    keys = CIRCUITS[sky]["param_keys"]
    metric = CIRCUITS[sky]["metric"]

    sur_f, sur_s = dl.Surrogate(fast), dl.Surrogate(sky)
    mi_f = sur_f.targets.index(metric)
    mi_s = sur_s.targets.index(metric)

    lo = np.array([CIRCUITS[sky]["ranges"][k][0] for k in keys])
    hi = np.array([CIRCUITS[sky]["ranges"][k][1] for k in keys])
    rng = np.random.RandomState(seed)
    X = lo + rng.random((n, len(keys))) * (hi - lo)

    fast_m = sur_f.predict(X)[:, mi_f]
    sky_m = sur_s.predict(X)[:, mi_s]
    resid = sky_m - fast_m                      # 系統性製程偏差

    Xn = _norm(sky, X)
    feat = np.column_stack([Xn, fast_m])        # 輸入: 正規化參數 + 快速指標
    model = Ridge(alpha=1.0).fit(feat, resid)
    pred = model.predict(feat)
    ss_res = np.sum((resid - pred) ** 2)
    ss_tot = np.sum((resid - resid.mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot

    np.savez(os.path.join(BASE_DIR, "data", f"transfer_{family}.npz"),
             coef=model.coef_, intercept=model.intercept_,
             keys=keys, metric=metric,
             resid_mean=float(resid.mean()), resid_std=float(resid.std()), r2=float(r2))
    return {"family": family, "metric": metric, "r2": float(r2),
            "resid_mean": float(resid.mean()), "resid_std": float(resid.std())}


class Transfer:
    def __init__(self, family):
        p = os.path.join(BASE_DIR, "data", f"transfer_{family}.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"尚未建立 {family} 遷移模型, 請先 build_transfer('{family}')")
        d = np.load(p, allow_pickle=True)
        self.family = family
        self.fast, self.sky = PAIR[family]
        self.coef, self.intercept = d["coef"], float(d["intercept"])
        self.keys = list(d["keys"]); self.metric = str(d["metric"])
        self.r2 = float(d["r2"])
        self.sur_f = dl.Surrogate(self.fast)
        self.mi_f = self.sur_f.targets.index(self.metric)

    def corrected_predict(self, X):
        """快速模型 + 殘差 -> sky130 主指標估計 (全用替代模型, 不耗 ngspice)。"""
        X = np.atleast_2d(np.asarray(X, float))
        fast_m = self.sur_f.predict(X)[:, self.mi_f]
        feat = np.column_stack([_norm(self.sky, X), fast_m])
        return fast_m + (feat @ self.coef + self.intercept)

    def warm_start(self, target=None, pop=200000, seed=0):
        """用校正後估計快速搜尋 sky130 暖啟動點 (正規化參數)。"""
        c = CIRCUITS[self.sky]
        if target is None:
            target = c["target_default"]
        tgt = target * c.get("target_scale", 1.0)
        lo = np.array([c["ranges"][k][0] for k in self.keys])
        hi = np.array([c["ranges"][k][1] for k in self.keys])
        rng = np.random.RandomState(seed)
        X = lo + rng.random((pop, len(self.keys))) * (hi - lo)
        est = self.corrected_predict(X)
        best = int(np.argmin(np.abs(est - tgt)))
        return _norm(self.sky, X[best]), float(est[best])


def demo(family, target=None):
    """比較精準模型: 冷啟動(預設中點) vs 遷移暖啟動 的迭代步數。"""
    import agent_main as agent
    tf = Transfer(family)
    sky = tf.sky
    cold = agent._run_optimization_classic(sky, target)
    p0, est = tf.warm_start(target)
    warm = agent._run_optimization_classic(sky, target, p0=p0)
    return {"family": family, "target": target,
            "cold_steps": cold["final"]["step"], "cold_status": cold["status"],
            "warm_steps": warm["final"]["step"], "warm_status": warm["status"],
            "warm_est": est, "transfer_r2": tf.r2}


if __name__ == "__main__":
    fam = sys.argv[1] if len(sys.argv) > 1 else "opa"
    tgt = float(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"建立 {fam} 遷移模型...")
    print(" ", build_transfer(fam))
    print(f"冷/暖啟動迭代比較 (target={tgt})...")
    print(" ", demo(fam, tgt))
