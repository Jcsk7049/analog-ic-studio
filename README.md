---
title: Analog IC Studio
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Analog IC Studio — 類比 IC 自動調參平台

> 輸入規格目標，演算法閉環模擬並自動回推**每顆電晶體的 W/L**，輸出可填回 KiCad 的元件參數。

以 **Python + Ngspice + SciPy** 打造的類比積體電路自動化設計平台：四大拓撲 × 快速 / sky130 真實 PDK 雙模式，內建 RFIC 看盤、Pelgrom 失配良率、真實製程角落與帕雷托多目標。前端為 Apple 風格單頁看板（中文 UI、英文圖表標籤），含即時優化進度條。

**🔗 線上 Demo（永久）：** https://huggingface.co/spaces/LANCELOT7049/analog-ic-studio

---

## 支援電路（四拓撲 × 兩模型 = 八電路，全部 per-instance W/L）

| 拓撲 | 快速模型 (Level-1) | sky130 精準模型 (BSIM4) | 優化標的 |
|------|------|------|------|
| 兩級 CMOS OPA | ✅ 10 維 W/L | ✅ 11 維 W/L | 逼近增益 + PM ≥ 45° |
| 帶隙基準 Bandgap | ✅ 真實 5T OTA + GP BJT | ✅ 真實 sky130 PNP | 最小化溫漂 TC |
| 環形振盪器 / VCO | ✅ Level-1（含寄生） | ✅ 電流飢餓 VCO（14 維） | 逼近振盪頻率 |
| **LC 壓控振盪器** | ✅ 理想 L + 二極體變容 | ✅ sky130 MOS-cap 變容 | 逼近振盪頻率 |

每顆 MOS 都有獨立的 **W 與 L**（最多 14 維），前端 KiCad 表逐顆顯示寬度 / 長度 / 佈局 finger 拆分。

## 核心功能

### 最佳化
- **多變量混合優化器**：每顆電晶體獨立 W/L，SciPy Differential Evolution 全域 + Nelder-Mead 局部，多線程平行評估
- **物理護欄**：強制 Wp = 2~3 × Wn（電子 / 電洞遷移率比），杜絕 NMOS > PMOS 的非物理解
- **Self-healing 安全護欄**：ngspice 不收斂自動攔截、讀 log、避開死區；sky130 瞬態 flaky 自動重試
- **即時進度條**：優化過程輪詢評估點數，前端顯示百分比 / 耗時

### RFIC 與類比指標
- **相位雜訊**（Leeson 解析估算）：由模擬 f₀ + 功耗推 L(Δf) 曲線與 FoM（環形 vs LC 對比）
- **FTR 調諧曲線 / Kvco**：掃描 Vctrl 提取調諧範圍與增益
- **OPA 進階指標**：Slew Rate、輸出擺幅、ICMR、PSRR
- **Bandgap**：真實 5T CMOS OTA + startup（非理想行為級運放）

### 良率與穩健性
- **Pelgrom 失配良率**（`mc_yield.py`）：σ ∝ 1/√(W·L)，大元件更匹配（非「不分大小灌 ±10%」）
- **真實製程角落**：sky130 電路切 tt / ff / ss BSIM4 角落（非參數縮放假角落）
- **PVT 27 角落掃描**：Process × Voltage × Temperature 疊加曲線
- **帕雷托多套餐**（`pareto.py`）：高效能 / 極致穩定 / 低功耗三組設計
- **2D 交叉掃描 + 3D 曲面**：20×20 多線程，Plotly 互動曲面
- **佈局 finger 拆分**：單指 ≤ 10µm，KiCad 表顯示 NF × 每指寬度

### 其他
- **KiCad 連動**：輸出可一對一填回的元件參數 + 下載最佳化 `.sp` 網表
- **DNN 萬點良率**（選用）：PyTorch MLP 替代模型，需預訓練；雲端未裝 torch 時優雅退回 ngspice

## 檔案結構

```
eda_control.py          執行層：CIRCUITS 註冊表 + 多模式模擬 + 安全護欄 + 分析函式
optimizer.py            多變量混合優化器 (DE + Nelder-Mead) + Wp/Wn 護欄 + 進度
agent_main.py           優化分派 + rich 中文面板 (CLI)
mc_yield.py             Pelgrom 失配良率 + 真實 corner 分析
pareto.py               多目標帕雷托套餐
analyzer.py             敏感度分析 + 2D 交叉掃描 (多線程)
web_app.py              Flask 後端 (REST API + matplotlib 圖表 + 進度端點)
templates/index.html    單頁前端看板 (Apple 風格 + Plotly 3D + 進度條)
*.sp.template           八種電路 SPICE 網表範本
pdk/sky130_minimal.lib.spice   sky130 精簡模型庫選擇器 (tt/ff/ss)
Dockerfile              Hugging Face Spaces 部署 (apt ngspice + Flask)
```

## 本機執行

```bash
pip install -r requirements.txt          # flask numpy scipy matplotlib rich pandas scikit-learn
python web_app.py                        # 開 http://127.0.0.1:5000
```

> ngspice：Linux 用 `apt install ngspice`；Windows 解壓 ngspice 到 `tools/Spice64/bin/ngspice_con.exe`，或加入 PATH。
> sky130 PDK 已納入版控（`pdk/sky130_fd_pr/`，12MB minimal 子集），雲端 / 本機皆可跑 sky130 精準模式。

### 終端機（rich 面板）
```bash
python agent_main.py opa 60          # OPA 目標 60 dB
python agent_main.py bandgap         # Bandgap 最小化 TC
python agent_main.py ringosc 2.4     # 環形振盪器 2.4 GHz
```

## 雲端部署（Hugging Face Spaces）

以 Docker SDK 部署：`Dockerfile` 基於 `python:3.11-slim`，`apt` 裝 ngspice，帶 minimal PDK，跑 Flask（port 7860）。推送即自動 build。

```bash
git push hf main                     # 觸發 HF 重新 build + 部署
```

## 誠實標註的簡化邊界

本平台定位為教學 / 設計探索工具，以下簡化已在程式與 UI 明確標註，不誇大模擬能力：

- **相位雜訊為 Leeson 解析估算**（ngspice 無振盪器 pnoise 引擎），非直接模擬
- **LC-VCO 電感理想化**（sky130 spiral inductor 模型不在 minimal lib；交叉耦合對與變容已是真 sky130）
- **bandgap_sky130 的鏡像 / OTA 用 Level-1**（minimal lib 無 3.3V sky130 元件；溫漂準確度由真實 sky130 PNP 主導）
- **負載 CL / 供電 VDD 維持固定**（屬設計規格，非優化變數）

## 技術備註
- 圖表軸標籤一律英文（避免 matplotlib 缺中文字體的豆腐塊）
- sky130 用 `.option scale=1u`（W/L 微米）；`.option scale` 不可置於網表第一行（會被當標題）
- 多線程模擬用獨立網表檔避免 `run.sp` 競爭
- PM 相位 unwrap 正規化到 (−180, 180]，從源頭消除相位 wrap 假值
- 振盪器以**穩態峰峰值**驗證持續振盪（< 100mV 判定假振盪 → 頻率作廢）：避免 `.meas` 對已衰減的死振盪（outp=outn）在數值噪聲上誤觸發、算出「波形是直線卻有頻率」的假解，優化器才不會收斂到不會振盪的設計

---

> sky130 PDK 來源：[google/skywater-pdk-libs-sky130_fd_pr](https://github.com/google/skywater-pdk-libs-sky130_fd_pr)（Apache-2.0）
