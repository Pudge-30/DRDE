# π0.5 LIBERO 训练总结与复现指南

生成日期：2025-10-27

---

## 📊 一、当前状态

### 1.1 已有的模型

| 模型 | 位置 | 训练步数 | Batch Size | 状态 |
|------|------|---------|-----------|------|
| **pi05_libero** (基础) | `/home/kemove/hyp/lerobot/pretrained_models/pi05_libero` | - | - | ✅ 已下载 (14GB) |
| **pi05_libero_finetuned** (官方) | `/home/kemove/hyp/lerobot/pretrained_models/pi05_libero_finetuned` | 6000 | 32 | ✅ 已下载 (7GB) |
| **你的训练** | `/home/kemove/hyp/lerobot/outputs/pi05_libero_finetuned_withjson/checkpoints/002000` | 2000 | 100 | ✅ 已完成 |

### 1.2 评估结果

- **你的模型 (2000步)** 在 `libero_spatial task_1` 上的成功率：**40%** (20/50 episodes)
- 官方模型尚未评估
- 基础模型尚未评估

---

## 🔍 二、训练配置对比分析

### 2.0 官方训练配置的获取方法

**问题**：如何知道官方 `pi05_libero_finetuned` 模型是用什么参数训练的？

**答案**：官方模型在 HuggingFace Hub 上会保存完整的训练配置。

#### 📍 官方配置文件位置

下载官方模型后，可以在以下位置找到完整的训练配置：

```bash
/home/kemove/hyp/lerobot/pretrained_models/pi05_libero_finetuned/train_config.json
```

#### 🔍 如何获取官方配置

**方法1：下载后查看本地文件**
```bash
# 下载官方模型
huggingface-cli download lerobot/pi05_libero_finetuned \
    --local-dir /home/kemove/hyp/lerobot/pretrained_models/pi05_libero_finetuned

# 查看训练配置
cat /home/kemove/hyp/lerobot/pretrained_models/pi05_libero_finetuned/train_config.json
```

**方法2：在线查看（不下载）**
- 访问 HuggingFace Hub: https://huggingface.co/lerobot/pi05_libero_finetuned/blob/main/train_config.json
- 直接在线浏览配置文件

**方法3：使用 Python 读取**
```python
from lerobot.configs.train import TrainPipelineConfig

# 从 HuggingFace Hub 加载配置
config = TrainPipelineConfig.from_pretrained("lerobot/pi05_libero_finetuned")

# 查看关键参数
print(f"Steps: {config.steps}")              # 6000
print(f"Batch size: {config.batch_size}")    # 32
print(f"Save freq: {config.save_freq}")      # 2000
print(f"Learning rate: {config.optimizer.lr}")  # 2.5e-05
```

#### 📄 官方配置文件示例（关键部分）

```json
{
  "policy": {
    "pretrained_path": "lerobot/pi05_libero",
    "type": "pi05"
  },
  "batch_size": 32,
  "steps": 6000,
  "num_workers": 4,
  "save_freq": 2000,
  "log_freq": 200,
  "optimizer": {
    "type": "adamw",
    "lr": 2.5e-05,
    "weight_decay": 0.01
  },
  "scheduler": {
    "type": "cosine_decay_with_warmup",
    "num_warmup_steps": 1000,
    "num_decay_steps": 6000,
    "peak_lr": 2.5e-05,
    "decay_lr": 2.5e-06
  }
}
```

**说明**：所有在 HuggingFace Hub 上发布的 LeRobot 模型都会包含 `train_config.json` 文件，记录了完整的训练配置。这使得训练过程完全可复现。

### 2.1 核心参数对比

| 参数 | 官方配置 | 你之前的配置 | 影响 |
|------|---------|------------|------|
| **预训练模型** | `lerobot/pi05_libero` | `lerobot/pi05_libero` | ✓ 相同 |
| **训练步数** | **6000** | **2000** | ⚠️ 你只训练了 33% |
| **Batch Size** | **32** | **100** | ⚠️ 你的大 3 倍 |
| **学习率** | 2.5e-05 | 2.5e-05 | ✓ 相同 |
| **Warmup Steps** | 1000 | 1000 | ✓ 相同 |
| **Decay Steps** | 6000 | 6000 | ✓ 相同 |
| **Workers** | 4 | 15 | 差异（影响加载速度） |
| **Save Freq** | 2000 | 2000 | ✓ 相同 |
| **Log Freq** | 200 | 200 | ✓ 相同 |

### 2.2 数据量对比

```
官方训练：6000 steps × 32 samples/step = 192,000 样本
你的训练：2000 steps × 100 samples/step = 200,000 样本
```

**结论**：
- 你看到的**数据总量**和官方相当（甚至略多）
- 但**模型更新次数少**（2000 vs 6000 次梯度更新）
- 这可能导致模型收敛不充分

---

## 🚀 三、官方配置复现训练命令

### 3.1 多卡训练命令（强烈推荐）🚀

使用 2 张 GPU 进行分布式训练，速度最快，显存压力最小：

```bash
cd /home/kemove/hyp/lerobot

torchrun --nproc_per_node=2 \
    $(which lerobot-train) \
    --policy.path=/home/kemove/hyp/lerobot/pretrained_models/pi05_libero \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root=$HOME/.cache/huggingface/smolvla_datasets/libero \
    --batch_size=32 \
    --steps=6000 \
    --save_freq=2000 \
    --output_dir=outputs/pi05_libero_official_reproduce \
    --wandb.enable=true
```

**⚠️ 关键参数说明**：
- `--policy.dtype=bfloat16`：**必需**！使用半精度减半显存（float32 → bfloat16）
- `--policy.gradient_checkpointing=true`：**必需**！启用梯度检查点节省显存
- 这两个参数可以将显存占用从 ~95GB 降低到 ~40GB/卡

**为什么推荐多卡训练**：
- ⚡ **速度最快**：2 张卡并行训练，预计 4-9 小时完成
- 💾 **显存友好**：每张卡只需处理 batch_size/2 = 16 的数据
- ✅ **保持官方配置**：总 batch_size=32，与官方一致
- 🔧 **自动优化**：PyTorch 自动处理梯度同步

**多卡训练参数说明**：
- `torchrun --nproc_per_node=2`：使用 2 张 GPU（GPU 0 和 GPU 1）
- `$(which lerobot-train)`：获取 lerobot-train 脚本的完整路径
- 有效 batch_size = 32（每张卡 16，梯度会自动汇总）
- `--policy.path`：使用本地模型路径
- `--policy.dtype=bfloat16`：必须显式指定，否则会使用 float32 导致 OOM
- `--policy.gradient_checkpointing=true`：必须显式指定节省显存

**⚠️ 如果遇到 "Output directory already exists" 错误**：

输出目录已存在时，需要先删除或更改目录名：

```bash
# 方案1：删除旧目录（推荐）
cd /home/kemove/hyp/lerobot
rm -rf outputs/pi05_libero_official_reproduce

# 然后重新运行训练命令

# 方案2：一条命令搞定（删除 + 训练）
cd /home/kemove/hyp/lerobot && \
rm -rf outputs/pi05_libero_official_reproduce && \
torchrun --nproc_per_node=2 \
    $(which lerobot-train) \
    --policy.path=/home/kemove/hyp/lerobot/pretrained_models/pi05_libero \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root=$HOME/.cache/huggingface/smolvla_datasets/libero \
    --batch_size=32 \
    --steps=6000 \
    --save_freq=2000 \
    --output_dir=outputs/pi05_libero_official_reproduce \
    --wandb.enable=true

# 方案3：使用新的目录名
torchrun --nproc_per_node=2 \
    $(which lerobot-train) \
    --policy.path=/home/kemove/hyp/lerobot/pretrained_models/pi05_libero \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root=$HOME/.cache/huggingface/smolvla_datasets/libero \
    --batch_size=32 \
    --steps=6000 \
    --save_freq=2000 \
    --output_dir=outputs/pi05_libero_6000steps_2gpu \
    --wandb.enable=true
```

**⚠️ 如果仍然遇到 OOM（显存不足）错误**：

即使是多卡训练，如果不指定 `dtype` 和 `gradient_checkpointing`，每张卡仍会占用 ~95GB 显存导致 OOM。

**OOM 错误特征**：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.89 GiB.
GPU has a total capacity of 94.97 GiB of which 884.62 MiB is free.
Including non-PyTorch memory, this process has 94.08 GiB memory in use.
```

**解决方案**：必须添加这两个参数
```bash
--policy.dtype=bfloat16                  # 将 float32 改为 bfloat16，减半显存
--policy.gradient_checkpointing=true     # 启用梯度检查点，牺牲少量速度换取显存
```

添加这两个参数后，每张卡显存占用约 40GB，可以在 95GB 显存的卡上正常训练。

### 3.2 单卡训练命令（显存充足时）

如果你只想用一张卡，且显存充足（>= 80GB）：

```bash
cd /home/kemove/hyp/lerobot

CUDA_VISIBLE_DEVICES=0 lerobot-train \
    --policy.path=/home/kemove/hyp/lerobot/pretrained_models/pi05_libero \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root=$HOME/.cache/huggingface/smolvla_datasets/libero \
    --batch_size=32 \
    --steps=6000 \
    --save_freq=2000 \
    --output_dir=outputs/pi05_libero_official_reproduce \
    --wandb.enable=true
```

**⚠️ 注意**：单卡训练 batch=32 需要约 95GB 显存，可能会 OOM。如果遇到显存不足，使用方法 3.1（多卡）或方法 3.3（单卡小batch）。

### 3.3 单卡小 Batch 训练（显存不足时）

如果只有一张卡且遇到 OOM（显存不足）：

```bash
cd /home/kemove/hyp/lerobot

CUDA_VISIBLE_DEVICES=0 lerobot-train \
    --policy.path=/home/kemove/hyp/lerobot/pretrained_models/pi05_libero \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root=$HOME/.cache/huggingface/smolvla_datasets/libero \
    --batch_size=16 \
    --steps=12000 \
    --save_freq=4000 \
    --output_dir=outputs/pi05_libero_official_reproduce \
    --wandb.enable=true
```

**关键修改**：
- `--policy.dtype=bfloat16`：使用半精度，减半显存占用
- `--policy.gradient_checkpointing=true`：启用梯度检查点，节省显存
- `--batch_size=16`：减小 batch（从 32 → 16）
- `--steps=12000`：因为 batch 减半，步数翻倍以保持总数据量一致
- `--save_freq=4000`：相应调整保存频率

**省略的参数（使用默认值）**：
- `--num_workers=4` → 默认就是 4
- `--log_freq=200` → 默认就是 200
- `--policy.push_to_hub=false` → 默认就是 false
- `--wandb.project=lerobot` → 默认就是 lerobot

**⚠️ 重要说明**：
- 使用 `--policy.path` 而不是 Hub ID，避免重复下载 14GB 模型
- 使用 `--dataset.root` 指定本地数据集路径，避免重新下载数据集

### 3.4 后台运行版本

如果需要在后台运行训练（推荐多卡版本）：

```bash
cd /home/kemove/hyp/lerobot

# 多卡后台运行
nohup torchrun --nproc_per_node=2 \
    $(which lerobot-train) \
    --policy.path=/home/kemove/hyp/lerobot/pretrained_models/pi05_libero \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root=$HOME/.cache/huggingface/smolvla_datasets/libero \
    --batch_size=32 \
    --steps=6000 \
    --save_freq=2000 \
    --output_dir=outputs/pi05_libero_official_reproduce \
    --wandb.enable=true \
    > train_official_reproduce.log 2>&1 &

# 查看训练进程
ps aux | grep lerobot-train

# 实时查看日志
tail -f train_official_reproduce.log

# 查看 GPU 使用情况
watch -n 1 nvidia-smi
```

### 3.5 policy.path 参数详解 ⚠️

`--policy.path` 参数用于指定预训练模型位置，有两种用法：

#### 📋 两种用法对比

| 用法 | 格式 | 行为 | 模型位置 | 优缺点 |
|------|------|------|---------|--------|
| **Hub ID** | `lerobot/pi05_libero` | 从 HuggingFace 自动下载 | `~/.cache/huggingface/hub/` | ✓ 简洁<br>✗ 每次都会检查/下载（耗时） |
| **本地路径** | `/home/kemove/.../pi05_libero` | 直接使用本地文件 | 指定的本地目录 | ✓ 快速，不重复下载<br>✓ 节省空间 |

#### 🔍 详细说明

**方式1：HuggingFace Hub ID（会自动下载）**
```bash
--policy.pretrained_path=lerobot/pi05_libero
```
- 这是 HuggingFace Hub 的仓库标识符
- LeRobot 会从网络下载模型（约14GB）
- 下载到 HuggingFace 缓存目录
- 即使你已经手动下载到 `/home/kemove/hyp/lerobot/pretrained_models/`，也**不会使用**那个文件

**方式2：本地绝对路径（直接使用）**
```bash
--policy.pretrained_path=/home/kemove/hyp/lerobot/pretrained_models/pi05_libero
```
- 这是文件系统的绝对路径
- 直接加载本地模型文件
- **不会联网下载**
- 节省时间和磁盘空间

#### ✅ 推荐做法

由于你已经下载了模型到本地，**强烈建议使用本地路径**：

```bash
--policy.pretrained_path=/home/kemove/hyp/lerobot/pretrained_models/pi05_libero
```

**优势**：
- ⚡ 启动更快（无需检查和下载）
- 💾 节省磁盘空间（不重复存储）
- 🎯 确定性更强（明确使用哪个模型）

#### 🔍 如何验证使用的是哪个模型？

训练开始时，查看日志输出：

```bash
# 使用本地路径时
Loading policy from: /home/kemove/hyp/lerobot/pretrained_models/pi05_libero

# 使用 Hub ID 时
Downloading model from lerobot/pi05_libero...
Fetching 9 files: 100%|██████████| 9/9 [XX:XX<00:00]
```

### 3.5 预计训练时间

根据你之前的训练经验：
- **你的配置**：2000 steps (batch=100) ≈ **6 小时**
- **官方配置**：6000 steps (batch=32) ≈ **9-18 小时**

**说明**：
- 步数增加 3 倍
- Batch size 减小 3 倍（每步更快）
- 预计总时间在 9-18 小时之间

### 3.5 输出位置

```
/home/kemove/hyp/lerobot/outputs/pi05_libero_official_reproduce/
├── checkpoints/
│   ├── 002000/          # 2000步 checkpoint
│   │   └── pretrained_model/
│   ├── 004000/          # 4000步 checkpoint
│   │   └── pretrained_model/
│   └── 006000/          # 6000步 最终模型
│       └── pretrained_model/
├── logs/
│   └── training.log
└── wandb/
    └── latest-run/
```

### 3.6 默认参数说明

LeRobot 训练脚本提供了许多默认参数，了解这些可以帮助你简化命令。

#### 🔍 默认参数来源

这些默认值来自 LeRobot 源码配置文件：

**主要配置文件**：
1. **`src/lerobot/configs/train.py`** - 训练流程配置
   - 定义了 `TrainPipelineConfig` 类
   - 包含核心训练参数的默认值

```python
# 来自 src/lerobot/configs/train.py (line 37-66)
@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    resume: bool = False
    seed: int | None = 1000              # 默认 1000
    num_workers: int = 4                 # 默认 4
    batch_size: int = 8                  # 默认 8
    steps: int = 100_000                 # 默认 100000
    eval_freq: int = 20_000              # 默认 20000
    log_freq: int = 200                  # 默认 200
    save_checkpoint: bool = True
    save_freq: int = 20_000              # 默认 20000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
```

2. **`src/lerobot/configs/default.py`** - 基础配置
   - 定义了 `DatasetConfig`, `EvalConfig`, `WandBConfig`

```python
# 来自 src/lerobot/configs/default.py (line 23-37)
@dataclass
class DatasetConfig:
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False

# 来自 src/lerobot/configs/default.py (line 40-49)
@dataclass
class WandBConfig:
    enable: bool = False                 # 默认 False
    disable_artifact: bool = False
    project: str = "lerobot"            # 默认 "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None
```

**如何查看默认参数**：
```bash
# 方法1: 查看配置文件
cat /home/kemove/hyp/lerobot/src/lerobot/configs/train.py
cat /home/kemove/hyp/lerobot/src/lerobot/configs/default.py

# 方法2: 运行时查看（如果有 help）
lerobot-train --help  # 可能不完整

# 方法3: 查看实际运行的配置
# 训练后会保存完整配置到 output_dir/checkpoints/*/pretrained_model/train_config.json
```

#### 📋 参数默认值表

| 参数 | 默认值 | 官方配置 | 是否需要指定 |
|------|--------|---------|------------|
| **核心参数** | | | |
| `policy.pretrained_path` | 无 | `lerobot/pi05_libero` | ❌ 必须 |
| `dataset.repo_id` | 无 | `HuggingFaceVLA/libero` | ❌ 必须 |
| `batch_size` | **8** | **32** | ❌ 必须（不同） |
| `steps` | **100000** | **6000** | ❌ 必须（不同） |
| **频率与保存** | | | |
| `save_freq` | **20000** | **2000** | ❌ 必须（不同） |
| `log_freq` | **200** | **200** | ✅ 可省略（相同） |
| `eval_freq` | **20000** | **20000** | ✅ 可省略（相同） |
| **数据加载** | | | |
| `num_workers` | **4** | **4** | ✅ 可省略（相同） |
| `seed` | **1000** | **1000** | ✅ 可省略（相同） |
| **输出与Hub** | | | |
| `output_dir` | 自动生成 | 自定义 | ⚠️ 建议指定 |
| `push_to_hub` | **False** | **False** | ✅ 可省略（相同） |
| **WandB** | | | |
| `wandb.enable` | **False** | **True** | ⚠️ 建议指定 |
| `wandb.project` | **"lerobot"** | **"lerobot"** | ✅ 可省略（相同） |
| `wandb.disable_artifact` | **False** | **False** | ✅ 可省略（相同） |

#### 💡 参数选择建议

**必须指定的参数**（与默认值不同）：
```bash
--policy.pretrained_path=/home/kemove/hyp/lerobot/pretrained_models/pi05_libero  # 无默认值，使用本地路径
--dataset.repo_id=HuggingFaceVLA/libero      # 无默认值
--batch_size=32                               # 默认8，官方用32
--steps=6000                                  # 默认100000，官方用6000
--save_freq=2000                              # 默认20000，官方用2000
```

**注意**：`pretrained_path` 也可以使用 Hub ID (`lerobot/pi05_libero`)，但会从网络重新下载模型。详见 [3.4 pretrained_path 参数详解](#34-pretrained_path-参数详解-)。

**建议指定的参数**（虽然有默认值但最好明确）：
```bash
--output_dir=outputs/pi05_libero_official_reproduce  # 方便找到输出
--wandb.enable=true                                  # 启用训练监控
```

**可以省略的参数**（默认值即为官方配置）：
```bash
--num_workers=4        # 默认就是4
--log_freq=200         # 默认就是200
--push_to_hub=false    # 默认就是false
--wandb.project=lerobot  # 默认就是lerobot
```

---

## 📈 四、你之前的训练结果分析

### 4.1 Loss 曲线

- **初始 Loss**: 1.386
- **最终 Loss**: 0.197
- **降幅**: 85.8%
- **训练时间**: ~6 小时
- **总步数**: 2000

**Loss 数据点**：
- Steps 1-37: 每步记录（来自 wandb cache）
- Steps 200-2000: 每 200 步记录（来自 logs）
- 总共 47 个数据点

可视化文件：
- `/home/kemove/hyp/lerobot/outputs/pi05_libero_finetuned_withjson/loss_complete_all_available.png`

### 4.2 评估结果（libero_spatial task_1）

```
总体成功率：40% (20/50 episodes)
平均总奖励：0.4
评估时间：913 秒 (~15 分钟)
```

**按时间段的成功率**：
- Episodes 1-10: 40% (4/10)
- Episodes 11-20: 40% (4/10)
- Episodes 21-30: 50% (5/10)
- Episodes 31-40: 40% (4/10)
- Episodes 41-50: 30% (3/10)

**观察**：
- 性能在后期有下降趋势（50% → 30%）
- 可能是模型训练不够充分

评估输出：`/home/kemove/hyp/lerobot/outputs/eval/2025-10-27/19-52-31_libero_pi05/`

---

## 🎯 五、接下来的计划

### 5.1 训练阶段

1. **从头开始训练 6000 步**（官方配置）
   - 使用上面提供的命令
   - 预计耗时 9-18 小时
   - 监控 wandb 上的 loss 曲线

2. **保存的 Checkpoints**
   - 2000 步
   - 4000 步
   - 6000 步（最终）

### 5.2 评估阶段

训练完成后，评估三个模型的性能：

#### 评估 1：基础模型（未微调）
```bash
CUDA_VISIBLE_DEVICES=1 lerobot-eval \
    --policy.path=/home/kemove/hyp/lerobot/pretrained_models/pi05_libero \
    --env.type=libero_spatial \
    --eval.batch_size=50 \
    --eval.n_episodes=50 \
    --policy.device=cuda
```

#### 评估 2：官方 6000 步模型
```bash
CUDA_VISIBLE_DEVICES=1 lerobot-eval \
    --policy.path=/home/kemove/hyp/lerobot/pretrained_models/pi05_libero_finetuned \
    --env.type=libero_spatial \
    --eval.batch_size=50 \
    --eval.n_episodes=50 \
    --policy.device=cuda
```

#### 评估 3：你的新训练 6000 步模型
```bash
CUDA_VISIBLE_DEVICES=1 lerobot-eval \
    --policy.path=/home/kemove/hyp/lerobot/outputs/pi05_libero_official_reproduce/checkpoints/006000/pretrained_model \
    --env.type=libero_spatial \
    --eval.batch_size=50 \
    --eval.n_episodes=50 \
    --policy.device=cuda
```

#### 评估 4：你之前的 2000 步模型（已完成）
```bash
# 已完成，成功率：40%
# 路径：/home/kemove/hyp/lerobot/outputs/pi05_libero_finetuned_withjson/checkpoints/002000/pretrained_model
```

### 5.3 预期成功率

根据论文和官方数据：
- **基础模型** (pi05_libero): ~30-40%
- **官方 6000 步**: ~60-75%
- **你的 2000 步**: ~40% ✅ 已测试
- **你的新 6000 步**: 预期 ~60-75%

---

## 📝 六、重要文件路径

### 6.1 模型文件

```
基础模型：
/home/kemove/hyp/lerobot/pretrained_models/pi05_libero/

官方 6000 步模型：
/home/kemove/hyp/lerobot/pretrained_models/pi05_libero_finetuned/

你的 2000 步训练：
/home/kemove/hyp/lerobot/outputs/pi05_libero_finetuned_withjson/checkpoints/002000/

你的新训练（即将开始）：
/home/kemove/hyp/lerobot/outputs/pi05_libero_official_reproduce/
```

### 6.2 配置文件

```
官方训练配置：
/home/kemove/hyp/lerobot/pretrained_models/pi05_libero_finetuned/train_config.json

你的 2000 步训练配置：
/home/kemove/hyp/lerobot/outputs/pi05_libero_finetuned_withjson/checkpoints/002000/pretrained_model/train_config.json
```

### 6.3 评估结果

```
你的 2000 步模型评估：
/home/kemove/hyp/lerobot/outputs/eval/2025-10-27/19-52-31_libero_pi05/
├── eval_info.json
├── episode_successes.json
└── video/ (如果有录制)
```

### 6.4 可视化文件

```
Loss 曲线：
/home/kemove/hyp/lerobot/outputs/pi05_libero_finetuned_withjson/loss_complete_all_available.png

评估结果可视化：
/home/kemove/hyp/lerobot/outputs/pi05_libero_finetuned_withjson/eval_summary.png
```

---

## 🔧 七、监控训练进度

### 7.1 查看实时日志

```bash
# 如果是后台运行
tail -f train_official_reproduce.log

# 如果在 tmux/screen 中运行
# 直接查看终端输出
```

### 7.2 WandB 监控

登录 WandB 查看：
- Loss 曲线
- Learning rate 变化
- Gradient norm
- 训练速度（steps/s）

### 7.3 检查 Checkpoint

```bash
# 查看已保存的 checkpoints
ls -lh /home/kemove/hyp/lerobot/outputs/pi05_libero_official_reproduce/checkpoints/

# 查看具体 checkpoint 内容
ls -lh /home/kemove/hyp/lerobot/outputs/pi05_libero_official_reproduce/checkpoints/002000/pretrained_model/
```

### 7.4 GPU 使用情况

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 查看训练进程
ps aux | grep lerobot-train
```

---

## 💡 八、常见问题与解决方案

### 8.1 训练中断怎么办？

如果训练意外中断，可以从最近的 checkpoint 继续：

```bash
CUDA_VISIBLE_DEVICES=1 lerobot-train \
    --policy.pretrained_path=/home/kemove/hyp/lerobot/outputs/pi05_libero_official_reproduce/checkpoints/002000/pretrained_model \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --batch_size=32 \
    --steps=4000 \
    --num_workers=4 \
    --save_freq=2000 \
    --log_freq=200 \
    --output_dir=outputs/pi05_libero_official_reproduce_continue \
    --policy.push_to_hub=false \
    --wandb.enable=true
```

### 8.2 WandB 存储配额满了

如果 wandb 存储满了：
1. 关闭 wandb：`--wandb.enable=false`
2. 或者清理旧的 runs
3. 训练日志会保存在本地 logs 目录

### 8.3 OOM（显存不足）

如果遇到显存不足：
- 减小 `batch_size`（32 → 16 或 8）
- 增加 `--policy.gradient_checkpointing=true`（已默认开启）
- 调整 `num_workers`

### 8.4 数据加载慢

如果数据加载成为瓶颈：
- 增加 `num_workers`
- 使用 `--dataset.streaming=true`（在线流式加载）

---

## 📚 九、参考资源

### 9.1 LeRobot 文档

- 官方文档：https://github.com/huggingface/lerobot
- π0.5 模型：https://huggingface.co/collections/lerobot/pi05
- LIBERO 数据集：https://huggingface.co/datasets/HuggingFaceVLA/libero

### 9.2 模型页面

- `pi05_libero`: https://huggingface.co/lerobot/pi05_libero
- `pi05_libero_finetuned`: https://huggingface.co/lerobot/pi05_libero_finetuned

### 9.3 LIBERO 基准测试

LIBERO 包含 5 个套件，共 130 个任务：
- `libero_spatial`: 10 个空间推理任务
- `libero_object`: 10 个物体操作任务
- `libero_goal`: 10 个目标达成任务
- `libero_10`: 10 个综合任务
- `libero_90`: 90 个长期学习任务

---

## ✅ 十、检查清单

训练前：
- [ ] 确认 GPU 可用：`nvidia-smi`
- [ ] 确认基础模型已下载：`ls -lh /home/kemove/hyp/lerobot/pretrained_models/pi05_libero/`
- [ ] 确认数据集可访问：`HuggingFaceVLA/libero`
- [ ] 确认有足够磁盘空间（至少 50GB）
- [ ] 确认 wandb 已登录（如果使用）

训练中：
- [ ] 监控 loss 是否正常下降
- [ ] 检查 GPU 利用率
- [ ] 查看 checkpoint 是否正常保存（每 2000 步）
- [ ] 记录训练时间

训练后：
- [ ] 验证最终模型文件完整性
- [ ] 在 libero_spatial 上评估
- [ ] 对比三个模型的性能
- [ ] 保存评估结果和可视化

---

## 🎉 十一、总结

你现在有：
1. ✅ **3 个模型**：基础、官方 6000 步、你的 2000 步
2. ✅ **完整的训练配置**：可以严格复现官方训练
3. ✅ **评估框架**：可以系统对比模型性能
4. ✅ **详细文档**：本总结文件

下一步：
1. 运行上面的训练命令
2. 等待 9-18 小时训练完成
3. 评估并对比所有模型
4. 分析性能差异

祝训练顺利！ 🚀
