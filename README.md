# 基於卷積神經網路的桌球影像分類研究 / Ping-Pong CNN Classifier

> **Table Tennis Image Classification Using Convolutional Neural Networks**

這是一個使用深度學習技術進行桌球影像分類的研究專案，比較了 VGG16 和 ResNet34 兩種卷積神經網路架構在桌球圖像識別任務上的性能表現。

VGG16訓練資料雲端連結：https://drive.google.com/drive/folders/1udNGmyyF2Sl8-jO9T46f7UeHSsleIKT0?usp=drive_link
ResNet34訓練資料雲端連結：https://drive.google.com/drive/folders/1e-V52fyxuxOKOJ9VlSzuUiXQrhMf5L0s?usp=drive_link

## 🏓 專案概述 / Project Overview

本專案旨在開發一個能夠自動識別影像中是否包含桌球相關內容的深度學習模型。我們使用了兩種經典的卷積神經網路架構：

- **VGG16**：使用預訓練模型進行遷移學習
- **ResNet34**：從頭開始訓練的殘差神經網路

### 🎯 研究目標

- 比較不同 CNN 架構在桌球影像分類任務上的性能
- 探索遷移學習與從頭訓練的效果差異
- 建立可靠的桌球影像自動分類系統

## 📊 資料集 / Dataset

### 資料結構
```
訓練資料集/
├── train/
│   ├── pingpong/        # 桌球相關影像
│   └── not_pingpong/    # 非桌球影像
└── val/
    ├── pingpong/        # 驗證集桌球影像
    └── not_pingpong/    # 驗證集非桌球影像
```

### 資料集特徵
- **影像尺寸**：224×224 像素
- **分類標籤**：二元分類（桌球/非桌球）
- **資料前處理**：歸一化至 [0,1] 範圍

## 🏗️ 模型架構 / Model Architecture

### VGG16 模型
- **基礎模型**：ImageNet 預訓練 VGG16
- **遷移學習**：凍結預訓練權重
- **分類器**：
  - Flatten 層
  - 全連接層（256 神經元，ReLU 激活）
  - Dropout（0.5）
  - 輸出層（1 神經元，Sigmoid 激活）

### ResNet34 模型
- **架構**：自定義 ResNet34 實現
- **特色**：殘差連接（Residual Connections）
- **訓練策略**：從頭開始訓練
- **資料增強**：旋轉、平移、剪切、縮放、水平翻轉

### 殘差塊結構
```python
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    if stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = add([x, shortcut])
    x = ReLU()(x)
    return x
```

## ⚙️ 訓練配置 / Training Configuration

### VGG16 設定
- **優化器**：Adam（學習率 0.001）
- **損失函數**：Binary Crossentropy
- **訓練輪數**：20 epochs
- **早停機制**：5 epochs patience
- **批次大小**：32

### ResNet34 設定
- **優化器**：Adam（學習率 0.00001）
- **損失函數**：Binary Crossentropy
- **訓練輪數**：100 epochs
- **批次大小**：32
- **Dropout**：0.5

### 類別平衡
兩個模型都使用了計算類別權重來處理資料不平衡問題：
```python
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(classes), 
    y=classes
)
```

## 📈 實驗結果 / Results

### 模型性能比較

| 模型 | 特點 | 訓練策略 |
|------|------|----------|
| VGG16 | 預訓練模型 + 遷移學習 | 凍結特徵提取器 |
| ResNet34 | 殘差網路 + 資料增強 | 端到端訓練 |

### 執行結果
- 詳細的訓練歷史記錄保存在 `training_history.json`
- 驗證集預測結果保存在 `results.csv`
- 視覺化結果請參考：
  - `VGG16執行結果.png`
  - `ResNet34執行結果.png`

## 🚀 如何使用 / Usage

### 環境需求
```bash
pip install tensorflow
pip install scikit-learn
pip install numpy
pip install pandas
pip install keras
```

### 執行 VGG16 模型
```bash
cd VGG16/
python "Pingpong Image Model.py"
```

### 執行 ResNet34 模型
```bash
cd ResNet34/
python table_tennis_resnet34.py
```

### GPU 測試（可選）
```bash
cd VGG16/
python GPU-test.py
```

## 📁 專案結構 / Project Structure

```
Ping-Pong-CNN-Classifier/
├── README.md                              # 專案說明
├── 基於卷積神經網路的桌球影像分類研究.pdf      # 研究報告
├── 訓練資料集/                             # 資料集目錄
│   ├── train/                            # 訓練集
│   └── val/                              # 驗證集
├── VGG16/                               # VGG16 模型實驗
│   ├── Pingpong Image Model.py         # 主要訓練腳本
│   ├── GPU-test.py                     # GPU 測試腳本
│   ├── best_model.keras               # 最佳模型
│   ├── trained_model.keras           # 訓練完成模型
│   ├── results.csv                   # 預測結果
│   └── training_history.json        # 訓練歷史
├── ResNet34/                          # ResNet34 模型實驗
│   ├── table_tennis_resnet34.py     # 主要訓練腳本
│   ├── best_model.keras            # 最佳模型
│   ├── trained_model.keras        # 訓練完成模型
│   ├── results.csv               # 預測結果
│   └── training_history.json    # 訓練歷史
├── VGG16執行結果.png               # VGG16 結果圖表
└── ResNet34執行結果.png            # ResNet34 結果圖表
```

## 🔍 技術特色 / Technical Features

### 1. 遷移學習 (Transfer Learning)
- 利用 ImageNet 預訓練權重
- 加速訓練過程並提高性能

### 2. 殘差連接 (Residual Connections)
- 解決深層網路梯度消失問題
- 提高模型訓練穩定性

### 3. 資料增強 (Data Augmentation)
- 旋轉、平移、剪切變換
- 增加資料多樣性，提高模型泛化能力

### 4. 類別平衡處理
- 自動計算類別權重
- 解決資料不平衡問題

### 5. 模型保存與載入
- 自動保存最佳模型
- 支援 Keras 格式模型檔案

## 🎛️ 調參策略 / Hyperparameter Tuning

### VGG16 調參
- 使用較高學習率（0.001）進行快速收斂
- 早停機制避免過擬合

### ResNet34 調參
- 使用較低學習率（0.00001）進行精細調整
- 增加訓練輪數獲得更好性能
- 豐富的資料增強提高泛化能力

## 📝 評估指標 / Evaluation Metrics

- **準確率 (Accuracy)**
- **精確率 (Precision)**
- **召回率 (Recall)**
- **F1 分數 (F1-Score)**
- **混淆矩陣 (Confusion Matrix)**

所有評估結果都會輸出到終端並保存到 CSV 文件中。

**關鍵字**: 深度學習, 卷積神經網路, VGG16, ResNet34, 影像分類, 桌球識別, 遷移學習, TensorFlow, Keras


**Keywords**: Deep Learning, Convolutional Neural Networks, VGG16, ResNet34, Image Classification, Table Tennis Recognition, Transfer Learning, TensorFlow, Keras
