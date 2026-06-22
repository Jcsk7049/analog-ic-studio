# Analog IC Studio — 整合 / API 串接說明

給「用 AI 判斷」的夥伴系統串接用。本平台是一個 **REST API 服務**:給規格目標 → 真實 ngspice 模擬 + 數值優化 → 回推每顆電晶體 W/L 與完整指標。你的 AI 只要呼叫這些端點,就能把「真實物理模擬」當成工具。

## Base URL

| 環境 | URL |
|------|-----|
| 線上 (Hugging Face) | `https://lancelot7049-analog-ic-studio.hf.space` |
| 本機 | `http://127.0.0.1:5000` |

所有 POST 端點都收/回 `application/json`。

## 電路代碼 (circuit)

| circuit | 拓撲 | 模型 | objective | 目標單位 |
|---------|------|------|-----------|---------|
| `opa` / `opa_sky130` | 兩級 OPA | 快速 / sky130 | `target` | 增益 dB |
| `bandgap` / `bandgap_sky130` | 帶隙基準 | 快速 / sky130 | `minimize`(無需 target) | — |
| `ringosc` | 環形振盪器 | 快速 | `target` | 頻率 GHz |
| `ringosc_sky130` | 電流飢餓 VCO | sky130 | `target` | 頻率 GHz |
| `lcvco` / `lcvco_sky130` | LC-VCO | 快速 / sky130 | `target` | 頻率 GHz |

> 完整電路設定(每顆參數鍵 / 範圍)定義在 `eda_control.CIRCUITS`;前端用的精簡版見 `web_app._circuits_meta()`。

---

## 主要端點

### 1. 優化(核心):`POST /api/optimize`
給目標 → 自動回推 W/L。

**Request**
```json
{ "circuit": "lcvco_sky130", "target": 7.0 }
```
- `target`:`objective=target` 的電路必填,單位見上表(增益 dB / 頻率 GHz);`bandgap` 不需 target。

**Response(節錄關鍵欄位)**
```json
{
  "circuit": "lcvco_sky130",
  "status": "converged",          // 或 "best"(未達標)
  "final": {
    "params": { "W_sw": 4.0e-5, "L_sw": 1.8e-7, "...": "...(SI 單位: W/L=公尺, C=法拉, R=歐姆)" },
    "metrics": { "freq": 7.004e9 },// gain(dB)/pm(deg)/ugf(Hz)/tc(ppm)/vref(V)/freq(Hz)
    "primary": 7.004e9,
    "err_pct": 0.06
  },
  "multi_influence": { "W_sw": [{"spec":"freq","pct":42.1,"sign":"+"}] },
  "wl_table": [ {"device":"M1,M2","W":"40.00","L":"0.180","fingers":4,"w_finger":"10.00"} ],
  "params_meta": { "W_sw": {"label":"...","unit":"µm","scale":1e6,"fmt":"{:.2f}"} }
}
```
- `final.params` 是 **SI 單位**;要顯示成 µm 等,乘 `params_meta[key].scale`。
- `status="best"` 代表該目標物理上達不到(會給最接近解),不是失敗。

### 2. VCO 調諧:`POST /api/tuning`(僅 VCO/LC-VCO)
```json
{ "circuit": "lcvco_sky130", "params": { "...": "選填, 省略則用最佳/起始參數" } }
```
回:`{ "points":[[vctrl,f_hz]...], "kvco_mhz_v":..., "ftr_pct":..., "f_min":..., "f_max":... }`

### 3. 相位雜訊:`POST /api/phase_noise`(僅 VCO/LC-VCO)
回:`{ "f0":..., "psig_mw":..., "curve":[[offset_hz,L_dbc]...], "pn_1m":..., "fom":... }`(Leeson 估算)

### 4. OPA 進階指標:`POST /api/opa_metrics`(僅 `opa`)
回:`{ "slew_v_us":..., "swing_vpp":..., "icmr_vpp":..., "psrr_db":... }`

### 5. 良率:`POST /api/yield`
```json
{ "circuit": "opa", "params": {...SI...}, "target": 60, "n": 50 }
```
回:`{ "mean":..., "std":..., "yield_pct":..., "samples":[...], "corner":{"TT":...,"FF":...,"SS":...} }`(Pelgrom 失配模型)

### 6. 帕雷托:`POST /api/pareto` → 三套餐(高效能/穩定/低功耗)
### 7. PVT:`POST /api/pvt`(僅 sky130 OPA/VCO,27 角落)
### 8. 網表:`GET /api/netlist?circuit=opa&W_M1=1e-5&...`(回 `.sp` 文字)
### 9. 進度:`GET /api/progress` → `{ "running":bool, "pct":int, "evals":int, "total":int }`(優化時輪詢做進度條)

---

## 給你 AI 的 Function-Calling Tool Schema

把這顆工具丟給你的 LLM,讓它呼叫真實模擬:

```json
{
  "name": "analog_optimize",
  "description": "給類比電路規格目標,回推每顆電晶體 W/L 並回傳真實 ngspice 模擬指標。用於設計/驗證類比 IC。",
  "parameters": {
    "type": "object",
    "properties": {
      "circuit": {
        "type": "string",
        "enum": ["opa","opa_sky130","bandgap","bandgap_sky130","ringosc","ringosc_sky130","lcvco","lcvco_sky130"],
        "description": "電路代碼"
      },
      "target": {
        "type": "number",
        "description": "目標值;增益電路用 dB, 振盪器用 GHz。bandgap(最小化溫漂)不需此欄。"
      }
    },
    "required": ["circuit"]
  }
}
```

對應的呼叫(你的後端 / AI 工具實作):
```python
import requests
BASE = "https://lancelot7049-analog-ic-studio.hf.space"
def analog_optimize(circuit, target=None):
    r = requests.post(f"{BASE}/api/optimize",
                      json={"circuit": circuit, "target": target}, timeout=600)
    d = r.json()
    return {
        "status": d.get("status"),
        "metrics": d["final"]["metrics"],     # freq(Hz)/gain(dB)/pm(deg)/tc(ppm)...
        "params_SI": d["final"]["params"],     # W/L 公尺
        "wl_table": d.get("wl_table"),         # 每顆 MOS W/L + finger (人類可讀)
        "err_pct": d["final"]["err_pct"],
    }
```

> 注意:sky130 模式每次跑真實 BSIM4 模擬,優化可能需 1~5 分鐘(免費雲端 CPU 慢),`timeout` 請設大;可同時輪詢 `/api/progress` 顯示進度。快速模型(`opa`/`lcvco`…)快很多,建議迭代用快速、最後用 sky130 驗證。

---

## ⭐ 設計檢查模組(零 token,取代 LLM 算數學)

**這是省 token 的關鍵**:已知拓樸(VCO/OPA/Bandgap)的數值分析改用程式算,不丟 LLM。
四個單檔、只用 `math`、無任何相依,可直接放進你的專案 import:

| 檔案 | 功能 |
|------|------|
| `design_check.py` | **統一入口** `analyze(topology, params, spec)` → 自動分流 |
| `vco_calc.py` | LC-VCO:f0/FTR/KVCO/起振餘裕/相位雜訊/FoM + Mohan 電感換算 |
| `opa_calc.py` | 兩級米勒 OPA:增益/GBW/相位裕度/壓擺率/擺幅 |
| `bandgap_calc.py` | 帶隙:Vref/TC + 找零溫漂最佳 (R3/R1)·ln(N) |

**用法**(AI 讀完圖萃取參數後,交給程式算與糾錯):
```python
from design_check import analyze

# topology: "vco" / "opa" / "bandgap"(別名見 design_check.supported())
r = analyze("opa",
            {"Itail_A": 20e-6, "Id6_A": 60e-6, "Cc_F": 2e-12, "CL_F": 5e-12},
            {"gain_dB_min": 60, "pm_deg_min": 60})

# r["metrics"]  -> 所有算好的性能數字
# r["findings"] -> [{level:"ok|warn|error", item, msg, suggest}]
#                  level=error 的就是「這參數錯了 → 該調到多少」
# r["ok"]       -> 是否無 error
```

`findings` 的 `suggest` 會給**具體目標值**,例如:
- OPA PM 不足 → 「把 gm6 提高至 ≥X mS,或加大 Cc」
- Bandgap TC 超標 → 「升 R3/R1 至 ≈8.93(目前 5.00)」
- VCO 起振餘裕 <1 → 「需 gm ≥ X mS:加大交叉耦合 W 或偏壓 Id」

→ LLM 只負責「讀圖 → 參數」與「把 findings 寫成建議文字」,**數學與糾錯全 0 token、且精確**。
（這些 calc 模組全在本機算,**不外送任何 PDK 機密**,也避開 NDA 風險。）

## 想要的話可加的端點(目前沒有,需要再說)

- `POST /api/simulate {circuit, params}`:給任意 W/L → 直接回指標(讓你的 AI 自己提參數、本平台只當「裁判」評分)。目前只有 target 驅動的 `/api/optimize`,若你的 AI 要自己提案 W/L,我可以加這顆。

有問題或要加 `simulate` 裁判端點,跟作者說。
