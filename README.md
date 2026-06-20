---
title: Analog IC Studio
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Analog IC Studio — 類比 IC 自動調參與論文實驗加速平台

以 **Python + Ngspice + PyTorch** 打造的類比積體電路自動化設計平台：輸入規格目標，
Agent 透過閉環優化／AI 替代模型自動回推元件參數，並提供良率、帕雷托、2D 曲面、
VCO 等論文級實驗工具。前端為 Apple 風格單頁看板（全中文 UI、英文圖表標籤）。

> 三拓撲 × 快速／sky130 精準對照 · AI 萬點良率 · 帕雷托多套餐 · 真實 sky130 VCO

---

## 支援電路 (六種，三拓撲 × 兩模型)

| 拓撲 | 快速模型 | sky130 精準模型 | 優化標的 |
|------|---------|----------------|---------|
| 兩級 CMOS OPA | Level-1 | BSIM4 | 逼近增益 + PM≥45° 約束 |
| 帶隙基準 Bandgap | Level-1 + GP BJT | 萃取自 sky130 PNP | 最小化溫漂 TC |
| 環形振盪器 / VCO | Level-1 環形 | **真實電流飢餓 VCO**（調 Vctrl） | 逼近振盪頻率 |

## 核心功能

- **閉環優化器**：高斯-牛頓 + active-set；多目標約束（PM）；VCO 用 Scipy 混合（DE 全域 + Nelder-Mead 局部，誤差 <0.5%）
- **DNN 替代模型**（`dl_surrogate.py`）：sky130 BSIM4 行為的 PyTorch MLP，百萬級群體尋優 → 1 次 ngspice 校正，7~11× 加速
- **萬點良率引擎**（`dl_yield_predictor.py`）：10,000 點製程變異毫秒級推論 + Six-Sigma
- **跨製程遷移學習**（`dl_transfer.py`）：學習理想↔sky130 物理偏差，暖啟動
- **蒙地卡羅良率 / Corner**（`mc_yield.py`）：Mean/σ/Yield% + TT/FF/SS
- **帕雷托多套餐**（`pareto.py`）：高效能 / 極致穩定 / 低功耗三組設計
- **2D 交叉掃描 + 3D 曲面**（`analyzer.sweep_2d`）：20×20=400 點多線程，Plotly 互動曲面
- **Self-healing 安全護欄**：ngspice 不收斂自動攔截、讀 log、懲罰避開死區、flaky 重試
- **KiCad 連動**：輸出可貼回的元件參數 + 下載最佳化 `.sp` 網表

## 檔案結構

```
eda_control.py          執行層：電路註冊表 CIRCUITS + 多模式模擬 + 安全護欄
agent_main.py           優化層：閉環優化 + 引擎分派 + rich 中文面板 (CLI)
analyzer.py             敏感度分析 + 2D 交叉掃描 (多線程)
optimizer.py            VCO Vctrl 混合優化器 (DE + Nelder-Mead)
dl_surrogate.py         PyTorch DNN 替代模型 (訓練/預測/尋優)
dl_transfer.py          跨製程遷移學習 (殘差映射)
dl_yield_predictor.py   萬點 DNN 良率引擎
mc_yield.py             蒙地卡羅良率 + Corner 分析
pareto.py               多目標帕雷托套餐
web_app.py              Flask 後端 (REST API + matplotlib 圖表)
templates/index.html    單頁前端看板 (Apple 風格 + Plotly 3D)
*.sp.template           六種電路 SPICE 網表範本
pdk/sky130_minimal.lib.spice   sky130 精簡模型庫選擇器
```

## 環境安裝

### 1. Python 套件
```bash
pip install numpy pandas scipy scikit-learn torch matplotlib flask rich
```

### 2. Ngspice（未納入版控，~47MB）
下載 ngspice-46 Windows 版解壓到 `tools/Spice64/`：
```bash
# 解壓後路徑須為 tools/Spice64/bin/ngspice_con.exe
# 來源: https://sourceforge.net/projects/ngspice/files/ng-spice-rework/46/
```

### 3. sky130 模型（未納入版控，~11MB）
`eda_control` 會找 `pdk/sky130_fd_pr/cells/{nfet_01v8,pfet_01v8,pfet_01v8_hvt,pnp_05v5}/`
下的 `*__tt.corner.spice` 與 `*__tt.pm3.spice`，以及
`pdk/sky130_fd_pr/models/parameters/invariant.spice`。
（**PVT 三溫掃描**另需 nfet/pfet/pfet_hvt 的 `*__ff.*` 與 `*__ss.*` 角落檔。）
皆下載自 [google/skywater-pdk-libs-sky130_fd_pr](https://github.com/google/skywater-pdk-libs-sky130_fd_pr)（Apache-2.0）。

> 只需「快速模型」(Level-1) 的話不必裝 sky130；OPA/Bandgap/RingOsc 的快速模式可獨立運作。

## 用法

### 網頁看板（推薦）
```bash
python web_app.py        # 開 http://127.0.0.1:5000
```
切換拓撲 + 快速／sky130 精準 → 輸入目標 → 開始優化 → 良率 / 帕雷托 / 3D 曲面。

### 終端機（rich 面板）
```bash
python agent_main.py opa 60          # OPA 目標 60 dB
python agent_main.py bandgap         # Bandgap 最小化 TC
python agent_main.py ringosc 2.4     # 環形振盪器 2.4 GHz
```

### 訓練 DNN 替代模型
```bash
python dl_surrogate.py train opa_sky130 450
```

## 技術備註
- 圖表軸標籤一律英文（避免 matplotlib 缺中文字體的豆腐塊）
- sky130 用 `.option scale=1u`（W/L 微米）；`.option scale` 不可置於網表第一行（會被當標題）
- 多線程 2D 掃描用獨立網表檔避免 `run.sp` 競爭
