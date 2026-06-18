# -*- coding: utf-8 -*-
"""
dl_surrogate.py — 基於 PyTorch 的深度神經網路替代模型 (DNN Surrogate)

核心理念 (簡報用語: 「AI 替代模型預測引擎」):
  sky130 真實 BSIM4 模擬一次要 ~0.6~2 秒, 大規模尋優太慢。
  改為「離線」用 ngspice 生成數百組樣本訓練一個 MLP, 學會製程的非線性映射;
  「線上」尋優時改在 MLP 上做百萬次快速預測群體搜尋, 最後僅呼叫 1 次 ngspice 校正。

流程:
  1. generate_dataset(circuit, n)  -> 自動 LHS 採樣 + 批量 ngspice 模擬, 存 .npz
  2. train(circuit)                -> MLP 多輸出回歸, 存 .pth (含標準化參數)
  3. Surrogate(circuit).predict(X) -> 向量化快速預測
  4. surrogate_optimize(circuit, target) -> DNN 上群體尋優 + 1 次 ngspice 校正
"""

import os
import sys
import numpy as np

import torch
import torch.nn as nn
from scipy.stats import qmc

import eda_control as eda
from eda_control import CIRCUITS

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

torch.manual_seed(0)

# 每個電路要學習的輸出規格 (targets)
TARGET_KEYS = {
    "opa":            ["gain", "pm", "ugf"],
    "opa_sky130":     ["gain", "pm", "ugf"],
    "bandgap":        ["tc", "vref"],
    "bandgap_sky130": ["tc", "vref"],
    "ringosc":        ["freq"],
    "ringosc_sky130": ["freq"],
}


def _ds_path(circuit):
    return os.path.join(DATA_DIR, f"surrogate_{circuit}.npz")


def _pth_path(circuit):
    return os.path.join(DATA_DIR, f"surrogate_{circuit}.pth")


# ----------------------------------------------------------------------
# 1) 資料生成: LHS 採樣 + 批量 ngspice
# ----------------------------------------------------------------------
def generate_dataset(circuit, n=500, seed=1, verbose=True):
    keys = CIRCUITS[circuit]["param_keys"]
    tkeys = TARGET_KEYS[circuit]
    lo = np.array([CIRCUITS[circuit]["ranges"][k][0] for k in keys])
    hi = np.array([CIRCUITS[circuit]["ranges"][k][1] for k in keys])

    unit = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    samples = qmc.scale(unit, lo, hi)

    X, Y = [], []
    for i, row in enumerate(samples, 1):
        params = {k: float(row[j]) for j, k in enumerate(keys)}
        r = eda.run_circuit(circuit, params)
        if r.get("ok") and all(r.get(t) is not None for t in tkeys):
            X.append([params[k] for k in keys])
            Y.append([r[t] for t in tkeys])
        if verbose and i % 25 == 0:
            print(f"  [{i:4d}/{n}] 有效樣本 {len(X)}")

    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    np.savez(_ds_path(circuit), X=X, Y=Y, features=keys, targets=tkeys)
    if verbose:
        print(f"  資料集: {len(X)} 筆有效 / {n} 採樣  -> {_ds_path(circuit)}")
    return X, Y, keys, tkeys


# ----------------------------------------------------------------------
# 2) MLP 多輸出回歸
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(64, 64, 32)):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(circuit, n=500, epochs=400, lr=2e-3, regen=False, verbose=True):
    if regen or not os.path.exists(_ds_path(circuit)):
        if verbose:
            print(f"[{circuit}] 生成 {n} 組訓練資料 (批量 ngspice)...")
        X, Y, keys, tkeys = generate_dataset(circuit, n, verbose=verbose)
    else:
        d = np.load(_ds_path(circuit), allow_pickle=True)
        X, Y, keys, tkeys = d["X"], d["Y"], list(d["features"]), list(d["targets"])
        if verbose:
            print(f"[{circuit}] 載入既有資料集 {len(X)} 筆")

    # 標準化 (z-score), 輸入輸出皆做
    xm, xs = X.mean(0), X.std(0) + 1e-12
    ym, ys = Y.mean(0), Y.std(0) + 1e-12
    Xn = (X - xm) / xs
    Yn = (Y - ym) / ys

    # 80/20 訓練/驗證
    idx = np.random.RandomState(0).permutation(len(Xn))
    cut = int(len(idx) * 0.8)
    tr, va = idx[:cut], idx[cut:]
    Xt = torch.tensor(Xn[tr], dtype=torch.float32)
    Yt = torch.tensor(Yn[tr], dtype=torch.float32)
    Xv = torch.tensor(Xn[va], dtype=torch.float32)
    Yv = torch.tensor(Yn[va], dtype=torch.float32)

    model = MLP(X.shape[1], Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xt), Yt)
        loss.backward()
        opt.step()
        if verbose and ep % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = lossf(model(Xv), Yv).item()
            print(f"  epoch {ep:4d}  train MSE={loss.item():.4f}  val MSE={vl:.4f}")

    # 驗證 R² (每個 target)
    model.eval()
    with torch.no_grad():
        pred_v = model(Xv).numpy() * ys + ym
    true_v = Yn[va] * ys + ym
    r2 = {}
    for j, t in enumerate(tkeys):
        ss_res = np.sum((true_v[:, j] - pred_v[:, j]) ** 2)
        ss_tot = np.sum((true_v[:, j] - true_v[:, j].mean()) ** 2) + 1e-12
        r2[t] = 1 - ss_res / ss_tot

    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": X.shape[1], "out_dim": Y.shape[1],
        "features": keys, "targets": tkeys,
        "xm": xm, "xs": xs, "ym": ym, "ys": ys,
        "r2": r2, "n": len(X),
    }, _pth_path(circuit))
    if verbose:
        print(f"[{circuit}] 訓練完成, 驗證 R²: " +
              "  ".join(f"{t}={v:.3f}" for t, v in r2.items()))
        print(f"  權重存檔: {_pth_path(circuit)}")
    return r2


# ----------------------------------------------------------------------
# 3) 載入與向量化預測
# ----------------------------------------------------------------------
class Surrogate:
    def __init__(self, circuit):
        if not os.path.exists(_pth_path(circuit)):
            raise FileNotFoundError(f"尚未訓練 {circuit} 的替代模型, 請先 train('{circuit}')")
        d = torch.load(_pth_path(circuit), weights_only=False)
        self.circuit = circuit
        self.features = list(d["features"])
        self.targets = list(d["targets"])
        self.xm, self.xs = d["xm"], d["xs"]
        self.ym, self.ys = d["ym"], d["ys"]
        self.r2 = d["r2"]
        self.model = MLP(d["in_dim"], d["out_dim"])
        self.model.load_state_dict(d["state_dict"])
        self.model.eval()

    def predict(self, X):
        """X: (N, d) 真實 SI 參數 -> (N, t) 預測規格 (真實單位)。"""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Xn = (X - self.xm) / self.xs
        with torch.no_grad():
            out = self.model(torch.tensor(Xn, dtype=torch.float32)).numpy()
        return out * self.ys + self.ym

    def predict_dict(self, params):
        X = np.array([[params[k] for k in self.features]])
        y = self.predict(X)[0]
        return {t: float(y[j]) for j, t in enumerate(self.targets)}


# ----------------------------------------------------------------------
# 4) 替代模型群體尋優 (DNN 上百萬次預測) + 1 次 ngspice 校正
# ----------------------------------------------------------------------
PM_MIN = 45.0


def _vectorized_loss(circuit, pred, tgt):
    """pred: (N, t) 預測規格; 回傳 (N,) loss。"""
    c = CIRCUITS[circuit]
    tk = TARGET_KEYS[circuit]
    col = {t: i for i, t in enumerate(tk)}
    if c["objective"] == "target":
        val = pred[:, col[c["metric"]]]
        loss = ((val - tgt) / (abs(tgt) + 1e-12)) ** 2
        if c.get("pm_constraint") and "pm" in col:
            pm = pred[:, col["pm"]]
            deficit = np.clip(PM_MIN - pm, 0, None) / PM_MIN
            loss = loss + 4.0 * deficit ** 2
        return loss
    else:  # minimize tc
        return (pred[:, col["tc"]] / 50.0) ** 2


def surrogate_optimize(circuit, target=None, pop=300000, seed=0, verify=True):
    """
    在替代模型上做大規模隨機群體尋優, 找最佳參數;
    verify=True 時最後呼叫 1 次 ngspice 做真實校正。
    回傳 dict: {params, predicted, verified, surrogate_r2}
    """
    c = CIRCUITS[circuit]
    keys = c["param_keys"]
    if target is None:
        target = c["target_default"]
    tgt = target * c.get("target_scale", 1.0)

    sur = Surrogate(circuit)
    lo = np.array([c["ranges"][k][0] for k in keys])
    hi = np.array([c["ranges"][k][1] for k in keys])

    rng = np.random.RandomState(seed)
    cand = lo + rng.random((pop, len(keys))) * (hi - lo)   # 百萬級候選
    pred = sur.predict(cand)                                # DNN 一次預測全部
    loss = _vectorized_loss(circuit, pred, tgt)
    best = int(np.argmin(loss))

    best_params = {k: float(cand[best, j]) for j, k in enumerate(keys)}
    predicted = {t: float(pred[best, j]) for j, t in enumerate(sur.targets)}

    result = {"circuit": circuit, "target": target,
              "params": best_params, "predicted": predicted,
              "surrogate_r2": sur.r2, "pop": pop}
    if verify:
        result["verified"] = eda.run_circuit(circuit, best_params)
    return result


# ----------------------------------------------------------------------
# CLI: python dl_surrogate.py train <circuit> [n]
# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("用法: python dl_surrogate.py train <circuit> [n]")
        print("      python dl_surrogate.py opt   <circuit> [target]")
        print("circuits:", ", ".join(CIRCUITS.keys()))
        sys.exit(0)
    cmd = args[0]
    ckt = args[1] if len(args) > 1 else "opa_sky130"
    if cmd == "train":
        n = int(args[2]) if len(args) > 2 else 500
        train(ckt, n=n, regen=True)
    elif cmd == "opt":
        tgt = float(args[2]) if len(args) > 2 else None
        r = surrogate_optimize(ckt, tgt)
        print("最佳參數:", {k: round(v, 9) for k, v in r["params"].items()})
        print("DNN 預測:", {k: round(v, 4) for k, v in r["predicted"].items()})
        if "verified" in r:
            v = r["verified"]
            print("ngspice 校正:", {k: v.get(k) for k in TARGET_KEYS[ckt]})
