# UCI HAR Dataset — 数据集说明

## 1. 基本信息

- **来源**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **论文**: Anguita et al. *Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine.* IWAAL 2012
- **机构**: Smartlab, Università degli Studi di Genova
- **许可**: 非商业使用需引用论文

## 2. 实验设置

30 名志愿者（19~48 岁），腰部佩戴 Samsung Galaxy S II，执行 6 项动作：

| 标签 | 动作 | 中文 |
| ---- | ---- | ---- |
| 1 | WALKING | 走路 |
| 2 | WALKING_UPSTAIRS | 上楼 |
| 3 | WALKING_DOWNSTAIRS | 下楼 |
| 4 | SITTING | 坐 |
| 5 | STANDING | 站 |
| 6 | LAYING | 躺 |

- 采样率: **50 Hz**
- 滑动窗口: **2.56 秒**，50% 重叠 → 每窗口 **128 个采样点**
- 训练 / 测试划分: 70% 志愿者（21 人）训练，30%（9 人）测试

## 3. 数据量

| 数据集 | 样本数 | 窗口数 |
| ------ | ------ | ------ |
| 训练集 | 7352 | 7352 个 128 时间步窗口 |
| 测试集 | 2947 | 2947 个 128 时间步窗口 |

## 4. 文件结构

```
UCI HAR Dataset/
├── activity_labels.txt       # 标签映射（1~6 → 活动名称）
├── features.txt              # 561 个特征名称列表
├── features_info.txt         # 特征计算方法说明
│
├── train/
│   ├── X_train.txt           # 训练特征矩阵 (7352 × 561)
│   ├── y_train.txt           # 训练标签 (7352 × 1)
│   ├── subject_train.txt     # 志愿者编号 (7352 × 1, 1~30)
│   └── Inertial Signals/
│       ├── total_acc_x_train.txt    # X 轴总加速度 (7352 × 128)
│       ├── total_acc_y_train.txt    # Y 轴总加速度 (7352 × 128)
│       ├── total_acc_z_train.txt    # Z 轴总加速度 (7352 × 128)
│       ├── body_acc_x_train.txt     # X 轴身体加速度 (7352 × 128)
│       ├── body_acc_y_train.txt     # Y 轴身体加速度 (7352 × 128)
│       ├── body_acc_z_train.txt     # Z 轴身体加速度 (7352 × 128)
│       ├── body_gyro_x_train.txt    # X 轴角速度 (7352 × 128)
│       ├── body_gyro_y_train.txt    # Y 轴角速度 (7352 × 128)
│       └── body_gyro_z_train.txt    # Z 轴角速度 (7352 × 128)
│
└── test/
    ├── X_test.txt             # 测试特征矩阵 (2947 × 561)
    ├── y_test.txt             # 测试标签 (2947 × 1)
    ├── subject_test.txt       # 志愿者编号 (2947 × 1, 1~30)
    └── Inertial Signals/      # 结构同 train，样本数 2947
```

## 5. 传感器通道（Inertial Signals）

9 个 `.txt` 文件，来自两种传感器三轴数据：

| 传感器 | 前缀 | 信号含义 | 单位 |
| ------ | ---- | -------- | ---- |
| 加速度计 | `total_acc` | 总加速度（含重力） | g |
| 加速度计 | `body_acc` | 身体加速度（减重力） | g |
| 陀螺仪 | `body_gyro` | 角速度 | rad/s |

每个前缀有 `_x` `_y` `_z` 三个方向，共 3 × 3 = 9 个通道。数据形状为 `(样本数, 128, 9)`。

## 6. 信号预处理

原始加速度和陀螺仪信号按以下步骤处理：

1. **采集**: 50 Hz 固定采样
2. **去噪**: 中值滤波 + 三阶低通 Butterworth 滤波器（截止频率 20 Hz）
3. **加速度分离**: 用低通 Butterworth 滤波器（截止频率 0.3 Hz）将加速度分为：
   - **身体加速度** `tBodyAcc` — 运动分量（高频）
   - **重力加速度** `tGravityAcc` — 重力分量（低频）
4. **Jerk（加加速度）**: 对身体加速度和角速度求导 → `tBodyAccJerk`、`tBodyGyroJerk`
5. **幅度**: 对三轴信号计算欧氏范数 → `tBodyAccMag`、`tGravityAccMag`、`tBodyAccJerkMag`、`tBodyGyroMag`、`tBodyGyroJerkMag`
6. **频域变换**: 部分信号做 FFT，前缀 `f` → `fBodyAcc`、`fBodyAccJerk`、`fBodyGyro` 及对应 Mag

## 7. 特征工程（561 维）

从以上信号对每个 128 点窗口提取统计量，共产生 **561 个特征**。时域和频域信号对照：

| 时域 | 频域 |
| ---- | ---- |
| tBodyAcc-XYZ | fBodyAcc-XYZ |
| tGravityAcc-XYZ | — |
| tBodyAccJerk-XYZ | fBodyAccJerk-XYZ |
| tBodyGyro-XYZ | fBodyGyro-XYZ |
| tBodyGyroJerk-XYZ | — |
| tBodyAccMag | fBodyAccMag |
| tGravityAccMag | — |
| tBodyAccJerkMag | fBodyAccJerkMag |
| tBodyGyroMag | fBodyGyroMag |
| tBodyGyroJerkMag | fBodyGyroJerkMag |

对每个信号提取以下 17 类统计量：

| 函数 | 含义 |
| ---- | ---- |
| `mean()` | 均值 |
| `std()` | 标准差 |
| `mad()` | 绝对中位差 |
| `max()` / `min()` | 最大 / 最小值 |
| `sma()` | 信号幅度区域 |
| `energy()` | 能量 |
| `iqr()` | 四分位距 |
| `entropy()` | 信号熵 |
| `arCoeff()` | 自回归系数（Burg 阶数=4） |
| `correlation()` | 两信号相关系数 |
| `maxInds()` / `meanFreq()` | 频率分量索引 / 加权平均 |
| `skewness()` / `kurtosis()` | 频域偏度 / 峰度 |
| `bandsEnergy()` | 频带能量（64 bin） |
| `angle()` | 两向量夹角 |

另有 5 个附加向量（`gravityMean`、`tBodyAccMean` 等）用于 `angle()` 计算。完整特征名称见 `features.txt`。

## 8. 使用建议

| 数据类型 | 适用模型 |
| -------- | -------- |
| `X_train.txt` / `X_test.txt`（561 维特征） | SVM、随机森林等传统 ML |
| `Inertial Signals/`（N × 128 × 9 原始信号） | CNN、LSTM 等深度学习 |

本项目使用 Inertial Signals 的 9 通道原始信号直接输入 1D-CNN。

> 特征值已归一化至 [-1, 1]；Inertial Signals 数据为原始浮点值，使用前需自行标准化。
