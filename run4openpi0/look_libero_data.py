"""探索 LIBERO 数据集 - 使用本地已下载数据"""
import sys
from pathlib import Path
import os

# 添加 lerobot 源码路径
lerobot_root = Path(__file__).parent.parent
src_path = lerobot_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 使用完整的导入路径
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

print("=" * 80)
print("加载 LIBERO 数据集（使用本地缓存）...")
print("=" * 80)

# 本地数据路径
local_data_path = Path.home() / ".cache/huggingface/smolvla_datasets/libero"
print(f"\n数据路径: {local_data_path}")
print(f"数据存在: {local_data_path.exists()}")

if local_data_path.exists():
    # 计算数据集大小
    import subprocess
    size_result = subprocess.run(['du', '-sh', str(local_data_path)],
                                capture_output=True, text=True)
    print(f"数据大小: {size_result.stdout.split()[0]}")

# 加载元数据（使用本地路径）
print(f"\n{'=' * 80}")
print("加载数据集元数据...")
metadata = LeRobotDatasetMetadata(
    "HuggingFaceVLA/libero",
    root=local_data_path  # 指定本地路径
)

print(f"\n【数据集基本信息】")
print(f"  - 总剧集数: {metadata.info['total_episodes']}")
print(f"  - 总帧数: {metadata.info['total_frames']}")
print(f"  - 采样率: {metadata.info['fps']} FPS")
print(f"  - 机器人类型: {metadata.info.get('robot_type', 'N/A')}")

print(f"\n【数据特征】")
for key, value in metadata.info['features'].items():
    print(f"  - {key}: {value}")

# 加载完整数据集（使用本地路径）
print(f"\n{'=' * 80}")
print("加载完整数据集（从本地缓存，速度很快）...")
dataset = LeRobotDataset(
    "HuggingFaceVLA/libero",
    root=local_data_path  # 指定本地路径
)

# 查看第一个样本
print(f"\n数据集大小: {len(dataset)} 帧")
sample = dataset[0]
print(f"\n【第一个样本的数据结构】")
for key in sorted(sample.keys()):
    if hasattr(sample[key], 'shape'):
        print(f"  - {key}: shape={sample[key].shape}, dtype={sample[key].dtype}")
    else:
        print(f"  - {key}: {sample[key]}")

print(f"\n{'=' * 80}")
print("✓ 数据集加载成功！")
print("=" * 80)