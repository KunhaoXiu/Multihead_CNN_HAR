# 人体动作识别 — 1D-CNN

使用 UCI HAR Dataset 对人体 6 种日常动作（走路、上楼、下楼、坐、站、躺）进行分类，基于一维卷积神经网络从 9 通道惯性传感器信号中提取时序特征。

## 数据集

[UCI Human Activity Recognition](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) — 30 名受试者佩戴手机采集的加速度计和陀螺仪数据，128 个时间步，9 个传感器通道。

```
UCI HAR Dataset/
├── train/
│   ├── Inertial Signals/   ← 9 个传感器文件（body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z）
│   └── y_train.txt         ← 标签（1~6）
└── test/
    └── ...
```

## 模型对比

| | 单分支 CNN | 多分支 CNN |
|---|---|---|
| 文件 | `1dcnn_tf.ipynb` | `multi_1dcnn_tf.ipynb` |
| 卷积核 | k=3 | k=3, 5, 7 并行 |
| 特征提取 | 单一路径 | 三路并行后 concat |
| 参数量 | 较少 | 约 3 倍 |

两个模型共享相同的后处理层（Dense + BatchNorm + Dropout），训练策略完全一致。

## 网络结构

### 单分支 CNN

```
Conv1D(64,3) → BN → Conv1D(64,3) → BN → MaxPool → Dropout(0.3)
  → Conv1D(128,3) → BN → MaxPool → Dropout(0.3) → GAP
  → Dense(128) → BN → Dropout(0.5)
  → Dense(64) → BN → Dropout(0.5)
  → Softmax(6)
```

### 多分支 CNN

```
输入 (128, 9)
  ├─ Conv1D(64,3) → Conv1D(64,3) → MaxPool → Conv1D(128,3) → MaxPool → GAP
  ├─ Conv1D(64,5) → Conv1D(64,5) → MaxPool → Conv1D(128,5) → MaxPool → GAP
  └─ Conv1D(64,7) → Conv1D(64,7) → MaxPool → Conv1D(128,7) → MaxPool → GAP
       ↓
  Concat → Dense(128) → BN → Dropout(0.5)
         → Dense(64) → BN → Dropout(0.5)
         → Softmax(6)
```

多分支设计用不同大小的卷积核同时捕获不同时间尺度的运动模式（短/中/长窗口）。

## 训练策略

- **标准化**：`StandardScaler` 沿特征维度标准化
- **优化器**：Adam（lr=1e-3）
- **正则化**：卷积层和全连接层均使用 `l2(1e-4)` 权重衰减
- **早停**：`val_loss` 连续 15 轮不降即停止，恢复最佳权重
- **学习率调度**：验证损失停滞时学习率减半（patience=5，min_lr=1e-6）
- **验证**：训练集中留出 20% 作为验证集
- **显存**：启用 memory growth，按需分配不占满

## 运行

```bash
# 数据准备：下载并解压 UCI HAR Dataset 到项目目录
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip "UCI HAR Dataset.zip"

# 启动 notebook
jupyter notebook
```

然后依次执行 `1dcnn_tf.ipynb` 或 `multi_1dcnn_tf.ipynb` 的 cell。

## 结果示例

```
单分支 CNN:  测试集准确率 ~93%
多分支 CNN:  测试集准确率 ~94%
```

## 环境

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas, numpy
