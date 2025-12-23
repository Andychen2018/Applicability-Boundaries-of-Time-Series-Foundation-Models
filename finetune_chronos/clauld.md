太OK了！下面是\*\*结合你最新要求（区分 ShengYing / ZhenDong 域）\*\*重写后的“给 Agent 执行”的项目说明文档。你可以直接放进 `README.md` 或发给 agent 使用。

---

# 🎯 Chronos微调执行 电机异常检测 | 任务执行说明（使用虚拟环境atuoG）

## 0. 项目背景与目标

基于 **Chronos** 时间序列模型，对 1s（65536 点，采样率 65536 Hz）的电机时序信号进行**多分类**：

* `normal`（正常）、`spark`（火花异常）、`vibrate`（振动异常）。

**三种互补方案（均需落地并对比）**：

1. **方案A**：仅用正常数据微调 Chronos → 用**预测残差**构造特征 → 训练**多分类器**（normal/spark/vibrate）。
2. **方案B**：仅用正常数据微调 Chronos → 抽取**embedding** → 训练**多分类器**。
3. **方案C**：用三类数据共同微调 Chronos → 抽取**embedding** → 训练**多分类器**。

> A、B 的微调为 **normal-only**；C 的微调为 **all-class**。A 是“残差做特征”，B/C 是“embedding 做特征”。

---

## 1. 数据与预处理（已完成）

* 原始数据根目录：

```
/home/deep/TimeSeries/Zhendong/data3/
├── ShengYing/
│   ├── normal/   ├── spark/   └── vibrate/
└── ZhenDong/
    ├── normal/   ├── spark/   └── vibrate/
```

* 预处理输出（Chronos 可直接使用）：

```
/home/deep/TimeSeries/Zhendong/data3/processed_motor_data.csv
```

数据列：`item_id, timestamp, target, label`（1868 条序列；单条长度 65536）。

**统一基础设定**

* 随机种子：`123`
* 预测步长 `prediction_length = 1024`
* 上下文窗口 `context_length = 4096`（可对比 2048/8192）

---

## 2. 模型与资源

* 本地 Chronos 模型（已下载）

  * `/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base`
  * `/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-small`（可选）
* 统一输出根目录

  ```
  /home/deep/TimeSeries/Zhendong/output/
  ```

---

## 3. 训练按域（传感器）策略

**强制**做两条线对比：

* **分域（Per-Sensor）**：对 **ShengYing** 与 **ZhenDong** 各自**单独微调**、预测与评估（允许不同标准化/阈值/超参）。
* **合训（Joint）**：将两域合并训练一个模型，但**必须显式加入“域标识”**（静态特征 one-hot：`sensor_type ∈ {ShengYing, ZhenDong}`）。

> 合训场景下，**标准化、阈值与评估依然按域分别进行**。

---

## 4. 数据增强（训练期使用，测试期禁用）

**目的**：增强泛化、放大正常/异常可分性。

### 4.1 通用增强（适用于 normal-only 与 all-class）

* 随机裁剪：从 65536 中随机截取 `4096` 连续片段；每条样本每 epoch 采样 1–3 个片段
* 幅值缩放：`Uniform(0.9, 1.1)`
* 高斯抖动：`σ = 0.01 * std(signal)`
* 时间遮挡：随机不连续小段，总长度约序列的 1–3%
* 时间平移：环形平移不超过 1% 序列长度
* 频域轻扰动（可选）：对 STFT/FFT 高频区域 5–10% 增益调整

### 4.2 类别特定增强（仅用于 all-class）

* `spark`：极少量稀疏尖脉冲（幅度 1–2×σ，占比 <1%）
* `vibrate`：叠加窄带正弦（200–500Hz，幅值 0.05–0.1×std）

> 增强强度需 **小而稳**，避免破坏原分布；**增强策略按域可微调**。

---

## 5. 三种方案（训练+推理）与目录组织

### 5.1 方案A：Normal-only 微调 → 残差特征多分类

**目标**：仅用 `normal` 微调 Chronos 的预测能力；对测试样本做未来 1024 点预测，用**残差**构造特征做三分类。

**数据**

* 训练：`label == normal`（分域或合训）
* 测试：`normal + spark + vibrate`（按 motor\_id 分组防泄漏）

**Chronos 微调建议**

* `prediction_length = 1024`，`context_length = 4096`
* `fine_tune=True`，从 `chronos-bolt-base` 启动
* `lr=5e-5 ~ 1e-4`；`steps=5k → 20k` 逐步放大；`dropout=0.1`
* 指标使用 WQL/MAE（内部参考），下游以分类效果为准

**残差特征（供分类器）**

* 整体：`MAE/MSE/RMSE/MAPE/MdAE`
* 分段：1024 分为 4 段（每段 256）各自 MAE/MSE（8 维）
* 分位残差（可选）：`QL@{0.1,0.5,0.9}`
* ACF/PACF：前 10 阶
* 频域能量比：高频/低频能量比（阈设 200Hz 或 Nyquist 0.2）
* 跨传感器一致性（可配对）：双测点残差相关系数

> 将以上特征拼接（建议 20–60 维）→ 分类器（LightGBM/MLP/SVM）三分类。

**落盘目录**

```
/home/deep/TimeSeries/Zhendong/output/methodA_residual_normal_ft/
├── ShengYing/
│   ├── predictor/                    # ⬅️ 分域 Chronos（normal-only）模型（必须保存）
│   ├── residual_features_train.csv
│   ├── residual_features_test.csv
│   ├── clf_model.pkl
│   ├── metrics.json
│   └── figures/
├── ZhenDong/
│   └── ...（同上）
└── Joint/                            # 合训（带域标识）
    ├── predictor/
    └── ...
```

---

### 5.2 方案B：Normal-only 微调 → 抽 embedding → 多分类

**目标**：仅用 `normal` 微调 Chronos，使编码器贴合正常模式；对全部样本抽取 **embedding**，用分类器做三分类。

**数据**

* 微调：`label == normal`（分域/合训）
* 抽特征 & 训练分类器：使用全部样本（normal/spark/vibrate）

**embedding 抽取**

* 每条样本截取若干 `4096` 片段（滑窗/随机裁剪）
* 取 Encoder 最后一层 hidden states 的**平均池化**（或等效 CLS）为 `[D]` 向量
* 多片段做平均/注意力融合；**拼接域 one-hot**（ShengYing/ZhenDong）

**落盘目录**

```
/home/deep/TimeSeries/Zhendong/output/methodB_embed_normal_ft/
├── ShengYing/
│   ├── predictor/                   # ⬅️ 分域 Chronos（normal-only）模型
│   ├── embeddings_train.npy(.csv)
│   ├── embeddings_test.npy(.csv)
│   ├── clf_model.pkl
│   ├── metrics.json
│   └── figures/tsne.png
├── ZhenDong/
│   └── ...（同上）
└── Joint/
    ├── predictor/
    └── ...
```

---

### 5.3 方案C：All-class 微调 → 抽 embedding → 多分类

**目标**：用三类数据共同微调 Chronos（仍是预测任务），令编码器“见过三类模式”，提升细粒度区分能力；再抽 **embedding** 做三分类。

**数据**

* 微调：`normal + spark + vibrate`（分域/合训）
* 其余流程：同 B（embedding 抽取 → 分类器）

**两阶段（可选，推荐做对比）**

* 阶段1：按方案B先用 normal-only 微调
* 阶段2：小学习率、只解冻后几层，用 all-class 再微调少量步

> 两阶段模型请**单独存档**，便于横向对比。

**落盘目录**

```
/home/deep/TimeSeries/Zhendong/output/methodC_embed_all_ft/
├── ShengYing/
│   ├── predictor/
│   ├── embeddings_train.npy(.csv)
│   ├── embeddings_test.npy(.csv)
│   ├── clf_model.pkl
│   ├── metrics.json
│   └── figures/tsne.png
├── ZhenDong/
│   └── ...（同上）
└── Joint/
    ├── predictor/
    └── ...
```

---

## 6. 划分与防泄漏（必须遵守）

* **按电机编号（motor\_id）分组划分**：同一 motor\_id 的样本不得同时出现在训练与测试。
* 建议比例：`train:val:test = 7:1:2` 或 5-fold CV。
* **标准化**：**按域**拟合 scaler（优先用该域 normal 训练集），推理时分域应用；必要时细化到 `item_id`。
* 训练期滑窗：每条序列每 epoch 采样 1–3 个 4096 窗口（配合增强）。

---

## 7. 统一超参（起步值，可在各方案内调参）

* `prediction_length = 1024`
* `context_length = 4096`（可对比 2048/8192）
* `fine_tune_lr = 5e-5`（all-class 可降到 3e-5）
* `fine_tune_steps = 10k`（先小规模验证，再扩至 20k）
* `dropout = 0.1`（数据较少可到 0.2）
* `batch_size`：视显存（4096×base 模型，单卡 16–32GB 常用 4–16）
* `quantile_levels=[0.1,…,0.9]`（方案A 若需分位残差则开启）

---

## 8. 模型保存规范（**必须严格执行**）

* 每个方案、每个域（或合训）的 **Chronos 微调后模型**必须保存到该方案目录下的 `predictor/`。
* 使用 AutoGluon 时，设置 `TimeSeriesPredictor(path=...)`，即可自动将权重与元数据写入 `predictor/`。
* 分类器、embedding/残差特征、指标与图形按前述目录落盘，便于复现实验与对比。

---

## 9. 结果汇总与文档

* 每个方案产出 `metrics.json` 与关键图（残差分布、ROC/PR、混淆矩阵、t-SNE）。
* 在根目录生成总表：

  ```
  /home/deep/TimeSeries/Zhendong/output/result.md
  ```

  汇总内容包括：

  * 三方案配置（context\_length / steps / augment 摘要）
  * 指标对比（Accuracy、Macro-F1、per-class F1、二分类时 ROC-AUC）
  * 分域（ShengYing/ZhenDong）与合训（Joint）的横向对比
  * 主要可视化图引用与结论建议（如是否采用两阶段、是否分域上线等）

---

## 10. Agent 执行顺序（建议）

1. 读取 `/data3/processed_motor_data.csv`，按 **motor\_id 分组**完成数据划分；
2. **分域（ShengYing、ZhenDong）与合训（Joint）** 三条线并行落地 **方案A**（normal-only 微调 → 残差特征 → 三分类）；
3. 同样三条线落地 **方案B**（normal-only 微调 → embedding → 三分类）；
4. 同样三条线落地 **方案C**（all-class 微调 → embedding → 三分类；可加两阶段变体）；
5. 将各自结果、模型与中间产物按约定目录落盘，并更新 `/output/result.md` 总结对比。

---

## 11. 验收标准

* A/B/C 三方案均在 **ShengYing/ZhenDong/Joint** 下产出：

  * `predictor/`（微调后的 Chronos 模型）
  * 特征（residual\_\* 或 embeddings\_\*）
  * 分类器（`clf_model.pkl`）
  * 指标（`metrics.json`）与图（`figures/`）
* `/output/result.md` 含完整对比与结论，可一键复现。
* 脚本可重复运行且幂等（已有产物时不重复训练，支持 `--force` 重训）。

---

分类器用LightGBM / MLP / SVM
