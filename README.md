# S2S-ERA5 多模式氣象降尺度與偏差校正系統

基於深度學習的統一框架，支援兩個可切換任務：

| 任務 | 代號 | 描述 |
|------|------|------|
| **Task A** | `history2era5` | 多模式歷史序列 → 未來 ERA5 風場預測 |
| **Task B** | `fcst2era5fit` | 多模式未來預報 → ERA5 偏差校正/擬合 |

---

## 目錄結構

```
downsampling_focas/
├── configs/
│   ├── task_a.yaml              # Task A 配置（歷史→ERA5）
│   └── task_b.yaml              # Task B 配置（預報→ERA5擬合）
├── models/
│   ├── __init__.py              # 模型工廠 build_model()
│   ├── common/                  # 共用模組
│   │   ├── wavelet.py           # 2D Haar 小波分解/重建
│   │   ├── attention.py         # 多頭自注意力、交叉注意力、空間/時序注意力
│   │   ├── conditioning.py      # FiLM 條件化（地形/LCZ）
│   │   ├── lead_embed.py        # Lead-time 正弦+學習嵌入
│   │   ├── conv_blocks.py       # ResBlock, DoubleConv, Up/Down, MultiScaleConv
│   │   └── output_head.py       # UV + 輻合場輸出頭（含診斷 conv）
│   ├── history2era5/
│   │   └── model.py             # Task A: 雙分支編碼 + TSDU + 多模式融合
│   └── fcst2era5fit/
│       └── model.py             # Task B: Lead 加權 + 殘差校正 UNet
├── datasets/
│   ├── __init__.py              # 資料集工廠 build_dataset()
│   ├── base_dataset.py          # ERA5↔S2S 降採樣、靜態特徵生成
│   ├── task_a_dataset.py        # Task A 真實/Dummy 資料集
│   └── task_b_dataset.py        # Task B 真實/Dummy 資料集
├── losses/
│   ├── __init__.py
│   └── unified_loss.py          # 統一損失：UV + conv + consistency + direction + gradient
├── frontend/
│   ├── app.py                   # Flask 後端 API
│   ├── templates/index.html     # 前端 HTML
│   └── static/                  # CSS + JS
├── train.py                     # 訓練腳本（AMP、cosine scheduler、gradient clip）
├── infer.py                     # 推論腳本
├── main.py                      # 統一入口（train / infer / serve）
├── requirements.txt             # Python 依賴
└── README.md
```

---

## 環境安裝

### 系統需求

- Python >= 3.9
- PyTorch >= 2.0（CUDA 版本以使用 GPU）
- NVIDIA GPU（推薦 >= 8 GB 顯存）

### 安裝步驟

```bash
# 1. 建立虛擬環境（推薦）
conda create -n s2s_era5 python=3.10 -y
conda activate s2s_era5

# 2. 安裝 PyTorch（根據你的 CUDA 版本選擇，以下以 CUDA 12.1 為例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安裝其餘依賴
cd D:\downsampling_focas
pip install -r requirements.txt
```

### 驗證 GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## 快速驗證（Demo 模式）

Demo 模式使用隨機資料和縮小的模型，無需真實資料即可驗證整條流程。

```bash
# Task A: 歷史→ERA5（demo，3個epoch）
python main.py train --config configs/task_a.yaml --demo --epochs 3

# Task B: 預報→ERA5擬合（demo，3個epoch）
python main.py train --config configs/task_b.yaml --demo --epochs 3
```

---

## 訓練

### Task A：歷史序列預測未來 ERA5

```bash
python main.py train \
    --config configs/task_a.yaml \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --device auto
```

### Task B：多模式預報偏差校正

```bash
python main.py train \
    --config configs/task_b.yaml \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --device auto
```

### 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--config` | 配置檔路徑（必填） | — |
| `--epochs` | 訓練輪數（覆蓋配置檔） | 配置檔中的值 |
| `--batch_size` | Batch 大小 | 配置檔中的值 |
| `--lr` | 學習率 | 配置檔中的值 |
| `--device` | 設備：`auto`（自動偵測GPU）/ `cuda` / `cuda:0` / `cpu` | `auto` |
| `--demo` | 啟用 Demo 模式（小模型 + 隨機資料） | 關閉 |

### 訓練輸出

```
checkpoints/
├── task_a/
│   ├── best_model.pt          # 最佳驗證損失的權重
│   ├── epoch_010.pt           # 每10個epoch存檔
│   └── training_log.json      # 完整訓練日誌
└── task_b/
    ├── best_model.pt
    └── training_log.json
```

### GPU 使用說明

- 當 `--device auto` 時，系統自動偵測並優先使用 GPU
- GPU 模式下自動啟用：
  - **FP16 混合精度訓練** (AMP) — 節省約 40% 顯存、提速約 2x
  - **pin_memory** — 加速 CPU→GPU 資料搬運
  - **4 個 DataLoader workers** — 平行資料讀取
  - **persistent_workers** — 避免 worker 重複啟動開銷

### 配置檔自訂

編輯 `configs/task_a.yaml` 或 `configs/task_b.yaml` 來調整模型架構和訓練參數：

```yaml
model:
  d_model: 128        # 特徵維度（GPU 可加大到 256）
  n_heads: 8          # 注意力頭數
  n_encoder_layers: 3 # 編碼器層數
  n_tsdu_layers: 3    # TSDU 動力學層數（僅 Task A）
  n_correction_blocks: 4  # 殘差校正層數（僅 Task B）

loss:
  uv_weight: 1.0
  conv_weight: 0.5
  conv_consistency_weight: 0.3
  direction_weight: 0.2
  gradient_weight: 0.1
  lead_decay: 0.95     # 短 lead 加權（指數衰減）
```

---

## 推論 / 評估

### 使用訓練好的權重進行推論

```bash
# Task A 推論
python main.py infer \
    --config configs/task_a.yaml \
    --checkpoint checkpoints/task_a/best_model.pt \
    --output_dir outputs/task_a \
    --max_samples 20 \
    --device auto

# Task B 推論
python main.py infer \
    --config configs/task_b.yaml \
    --checkpoint checkpoints/task_b/best_model.pt \
    --output_dir outputs/task_b \
    --max_samples 20 \
    --device auto
```

### Demo 推論（無需真實資料）

```bash
python main.py infer --config configs/task_a.yaml --demo --max_samples 5
```

### 推論輸出

```
outputs/
├── task_a/
│   ├── pred_0000.npz          # 每個樣本的預測結果
│   ├── pred_0001.npz          # 包含 uv, direct_conv, diag_conv, lead_ids
│   └── inference_summary.json # 彙總統計
└── task_b/
    └── ...
```

載入預測結果：

```python
import numpy as np
data = np.load("outputs/task_a/pred_0000.npz")
uv = data["uv"]               # (K, 2, H, W) — K個lead time的u,v風場
direct_conv = data["direct_conv"]  # (K, 1, H, W) — 直接預測的輻合場
diag_conv = data["diag_conv"]      # (K, 1, H, W) — 從u,v診斷計算的輻合場
```

---

## 前端 Web UI

系統附帶一個玻璃態暗色主題的 Web 控制台，可在瀏覽器中操作訓練和推論。

### 啟動前端

```bash
python main.py serve --port 5000
```

然後在瀏覽器開啟 **http://localhost:5000**。

### 前端功能

1. **任務選擇** — 點選 Task A 或 Task B 卡片
2. **訓練模式** — 設定 epochs、batch size、learning rate，點擊「開始訓練」
3. **推論模式** — 選擇 checkpoint、設定樣本數，點擊「開始推論」
4. **即時監控面板**：
   - 進度環（百分比）
   - Epoch / Train Loss / Val Loss / 耗時 指標卡片
   - 損失分量展示（UV、conv、consistency、direction、gradient）
   - 即時訓練日誌
   - 損失曲線圖表（Train + Val 雙線）
5. **GPU 狀態指示** — 右上角顯示 GPU 是否可用

### 前端截圖流程

```
選擇任務 → 設定參數 → 點擊開始 → 即時監控 → 查看結果
```

---

## 資料格式

### ERA5（標籤資料）

```
D:\downsampling_focas\ERA5\
├── u10\          # 10m U 風分量, 0.25° (721×1440)
│   ├── 2025_01.nc
│   └── ...
├── v10\          # 10m V 風分量
├── UWND\         # 高空 U 風（含氣壓層，本專案僅用地面）
└── VWND\         # 高空 V 風
```

每個 `.nc` 檔案維度：`(time=天數, lat=721, lon=1440)`，變數名如 `uwnd10`。

### S2S（多模式輸入資料）

```
D:\downsampling_focas\S2S\
├── CMA\          # 中國氣象局模式
├── ECMWF\        # 歐洲中期天氣預報中心
├── NECP\         # NCEP 模式
└── UNKO\         # 第四個模式
    ├── wind10m\  # 10m 風速
    ├── wind100m\ # 100m 風速
    └── ssrd\     # 地表太陽輻射
        └── 2025\01\20250102_ys_mean.nc
```

每個 `.nc` 檔案維度：`(time=60, lat=121, lon=240)` (1.5° 解析度，60天 lead time)。

系統會自動將 ERA5 從 0.25° 降採樣到 1.5° 以匹配 S2S 網格。

---

## 模型架構

### Task A: History2ERA5

```
多模式歷史序列 [B, M=4, T=7, C=3, H, W]
        │
        ├── Per-Model Dual-Branch Encoder (Raw + Haar Wavelet)
        │       │
        │       └── Temporal Attention → 歷史聚合
        │
        ├── LCZ / Terrain FiLM Conditioning
        │
        ├── Multi-Model Dynamic Fusion (Attention-based weighting)
        │
        ├── Lead-Time Embedding (Sinusoidal + Learned)
        │
        ├── TSDU (Temporal-Scale Dynamics Unit)
        │       └── MultiScale Conv → Spatial Attention → ResBlock × N
        │
        └── Output Head → UV [B,K,2,H,W] + Conv [B,K,1,H,W]
```

### Task B: Forecast2ERA5Fit

```
多模式未來預報 [B, M=4, K=4, C=3, H, W]
        │
        ├── Per-Model Forecast Encoder
        │
        ├── Lead-Aware Model Weighting (Cross-Attention)
        │
        ├── Terrain / LCZ FiLM Conditioning
        │
        ├── Residual Correction Backbone (UNet)
        │       └── Encoder → Bottleneck (SpatialAttn) → Decoder
        │
        ├── Residual: corrected = model_mean + learned_residual
        │
        └── Output Head → UV [B,K,2,H,W] + Conv [B,K,1,H,W]
```

### 損失函數

| 損失 | 權重 | 說明 |
|------|------|------|
| UV Loss | 1.0 | 0.7×MSE + 0.3×MAE |
| Conv Loss | 0.5 | 輻合場 MSE |
| Conv Consistency | 0.3 | 直接預測 conv vs 從 u,v 診斷計算的 conv |
| Direction Loss | 0.2 | 風向餘弦相似度損失 |
| Gradient Loss | 0.1 | 空間梯度一致性 |

所有損失均加入 lead-dependent 指數衰減權重（短期 lead 權重更高）。

---

## 常見問題

**Q: 如何切換到僅用 CPU 訓練？**
```bash
python main.py train --config configs/task_a.yaml --device cpu
```

**Q: 如何指定特定 GPU？**
```bash
python main.py train --config configs/task_a.yaml --device cuda:1
```

**Q: demo_mode 和正式訓練有什麼區別？**

| | Demo 模式 | 正式訓練 |
|--|----------|---------|
| 資料 | 隨機生成 | 真實 ERA5 + S2S |
| 網格 | 16×32 | 121×240 |
| d_model | 64 | 128 |
| 用途 | 驗證流程 | 實際訓練 |

**Q: 如何用真實資料訓練？**

在 `configs/task_a.yaml` 中設定 `demo_mode: false`，確保 ERA5 和 S2S 資料路徑正確即可。
