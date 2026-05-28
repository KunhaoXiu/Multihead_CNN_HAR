# 人体动作识别 — ANN / CNN / LSTM 对比实验

使用 UCI HAR Dataset 对人体 6 种日常动作（走路、上楼、下楼、坐、站、躺）进行分类，对比 ANN、单分支 CNN、多分支 CNN 和 LSTM 四种模型在两种输入（手工特征 / 原始信号）上的表现。

## 数据集

[UCI Human Activity Recognition](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) — 30 名受试者佩戴手机采集的加速度计和陀螺仪数据。

```text
UCI HAR Dataset/
├── train/
│   ├── X_train.txt            ← 561 维手工提取特征（features 系列使用）
│   ├── y_train.txt            ← 标签（1~6）
│   └── Inertial Signals/      ← 9 个传感器文件，每个 (7352, 128)（raw 系列使用）
└── test/
    └── ...
```

两种输入的形状：

| 输入类型 | 形状 | 说明 |
| --- | --- | --- |
| 手工特征 | (561, 1) | 561 维特征向量，reshape 为 (561, 1) 供 Conv1D/LSTM 使用 |
| 原始信号 | (128, 9) | 128 个时间步 × 9 个传感器通道 |

## 模型对比

| 模型 | 原始信号 (raw) | 手工特征 (features) |
| --- | --- | --- |
| **ANN** | `ann_raw.ipynb` | `ann_features.ipynb` |
| **单分支 CNN** | `cnn_raw.ipynb` | `cnn_features.ipynb` |
| **多分支 CNN** | `multicnn_raw.ipynb` | `multicnn_features.ipynb` |
| **LSTM** | `lstm_raw.ipynb` | `lstm_features.ipynb` |

## 网络结构

### ANN

```text
Dense(128, relu) → Dropout(0.5)
  → Dense(64, relu) → Dropout(0.5)
  → Softmax(6)
```

- raw 版本：输入 (128, 9) 先经 Flatten 展平为 (1152,)
- features 版本：直接输入 561 维特征

### 单分支 CNN

```text
Conv1D(64, 3) → BN → Conv1D(64, 3) → BN → MaxPool → Dropout(0.3)
  → Conv1D(128, 3) → BN → MaxPool → Dropout(0.3) → GAP
  → Dense(128) → BN → Dropout(0.5)
  → Dense(64) → BN → Dropout(0.5)
  → Softmax(6)
```

### 多分支 CNN

```text
输入
  ├─ Conv1D(64, 3) → Conv1D(64, 3) → MaxPool → Conv1D(128, 3) → MaxPool → GAP
  ├─ Conv1D(64, 5) → Conv1D(64, 5) → MaxPool → Conv1D(128, 5) → MaxPool → GAP
  └─ Conv1D(64, 7) → Conv1D(64, 7) → MaxPool → Conv1D(128, 7) → MaxPool → GAP
       ↓
  Concat → Dense(128) → BN → Dropout(0.5)
         → Dense(64) → BN → Dropout(0.5)
         → Softmax(6)
```

多分支设计用不同大小的卷积核同时捕获不同尺度的特征组合模式。

### LSTM

```text
LSTM(128, return_sequences=True) → Dropout(0.3)
  → LSTM(64) → Dropout(0.3)
  → Dense(128, relu) → BN → Dropout(0.5)
  → Dense(64, relu) → BN → Dropout(0.5)
  → Softmax(6)
```

## 训练策略

- **标准化**：`StandardScaler` 沿特征维度标准化（raw 系列）
- **优化器**：Adam（lr=1e-3）
- **正则化**：卷积层、LSTM 层和全连接层均使用 `l2(1e-4)` 权重衰减
- **早停**：`val_loss` 连续 15 轮不降即停止，恢复最佳权重
- **学习率调度**：验证损失停滞时学习率减半（patience=5，min_lr=1e-6）
- **验证集**：训练集中留出 20%

## 运行

```bash
# 数据准备：下载并解压 UCI HAR Dataset 到项目目录
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip "UCI HAR Dataset.zip"

# 启动 notebook
jupyter notebook
```

## 环境

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas, numpy
