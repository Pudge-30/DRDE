#!/usr/bin/env python3
"""
CMP vs Baseline 评估结果综合分析脚本。

统计内容:
  1. 成功率、成功步数、replan 次数、评估时间
  2. Chunk 利用分析: replan chunk vs full chunk, chunk 利用率
  3. 自动生成对比大表

计算方式:
  - Baseline replan: sum(ceil(ep_steps / 10)), 固定每 10 步 replan
  - CMP replan: sum(replan_counts), drift 阈值触发时 _replan_count++ (从 JSON 直接读取)
  - Chunk 分析: replan chunk 只用 ~10 步 (seg 0), full chunk 用满 50 步

用法:
  python src/lerobot/scripts/analyze_eval.py                    # 分析所有配置
  python src/lerobot/scripts/analyze_eval.py --config goal      # 只分析 GOAL
  python src/lerobot/scripts/analyze_eval.py --json /path/to/eval_info.json  # 分析单个文件
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# ── 路径配置 ──────────────────────────────────────────────────────
BASE = Path("/home/kemove/hyp/lhm-heritage")
BASELINE_DIR = Path("/home/kemove/qyh/DRDE/DRDE/outputs")
CMP_DIR = BASE / "outputs/comparasion1"

# ── 所有实验配置 ──────────────────────────────────────────────────
# 格式: (任务集标签, 配置名, JSON路径, 是否Baseline)
CONFIGS = [
    # LIBERO-10
    ("LIBERO-10", "Baseline", BASE / "outputs/eval_baseline/eval_info_libero10.json", True),
    ("LIBERO-10", "h55/m50", CMP_DIR / "eval_6000_adaptive_h0.055_m0.050/libero10/eval_info.json", False),
    ("LIBERO-10", "h45/m40", CMP_DIR / "eval_6000_adaptive_h0.045_m0.040/libero_10/eval_info.json", False),
    # LIBERO-SPATIAL
    ("LIBERO-SPATIAL", "Baseline", BASELINE_DIR / "pi05_libero_official_reproduce_eval_spatial/eval_info.json", True),
    ("LIBERO-SPATIAL", "h55/m50", CMP_DIR / "eval_6000_adaptive_h0.055_m0.050/libero_spatial/eval_info.json", False),
    ("LIBERO-SPATIAL", "h45/m40", CMP_DIR / "eval_6000_adaptive_h0.045_m0.040/libero_spatial/eval_info.json", False),
    # LIBERO-OBJECT
    ("LIBERO-OBJECT", "Baseline", BASELINE_DIR / "pi05_libero_official_reproduce_eval_object/eval_info.json", True),
    ("LIBERO-OBJECT", "h55/m50", CMP_DIR / "eval_6000_adaptive_h0.055_m0.050/libero_object/eval_info.json", False),
    ("LIBERO-OBJECT", "h45/m40", CMP_DIR / "eval_6000_adaptive_h0.045_m0.040/libero_object/eval_info.json", False),
    ("LIBERO-OBJECT", "h35/m30", CMP_DIR / "eval_6000_adaptive_h0.035_m0.030/libero_object/eval_info.json", False),
    # LIBERO-GOAL
    ("LIBERO-GOAL", "Baseline", BASELINE_DIR / "pi05_libero_official_reproduce_eval_goal/eval_info.json", True),
    ("LIBERO-GOAL", "h55/m50", CMP_DIR / "eval_6000_adaptive_h0.055_m0.050/libero_goal/eval_info.json", False),
    ("LIBERO-GOAL", "h45/m40", CMP_DIR / "eval_6000_adaptive_h0.045_m0.040/libero_goal/eval_info.json", False),
    ("LIBERO-GOAL", "h35/m30", CMP_DIR / "eval_6000_adaptive_h0.035_m0.030/libero_goal/eval_info.json", False),
]

N_ACTION_STEPS = 10
CHUNK_SIZE = 50


# ── 核心分析函数 ──────────────────────────────────────────────────
def analyze_eval_json(json_path: str | Path, is_baseline: bool = False) -> dict:
    """分析单个 eval_info.json，返回统计结果。

    Args:
        json_path: eval_info.json 文件路径
        is_baseline: 是否为 Baseline (影响 replan 计算方式)

    Returns:
        dict: 包含所有统计指标
    """
    with open(json_path) as f:
        data = json.load(f)

    total_episodes = 0
    total_success = 0
    total_steps = 0
    total_success_steps = 0
    total_replan = 0
    per_task_results = []

    for task in data["per_task"]:
        tid = task.get("task_id", len(per_task_results))

        # 兼容两种 JSON 格式
        if "metrics" in task:
            m = task["metrics"]
            rewards = m.get("sum_rewards", [])
            ep_steps = m.get("ep_steps", [])
            replan_counts = m.get("replan_counts", [])
        else:
            # 旧格式 (baseline libero10)
            n_succ = task.get("successes", 0)
            n_fail = task.get("failures", 0)
            rewards = [1.0] * n_succ + [0.0] * n_fail
            ep_steps = []
            replan_counts = []

        n_ep = len(rewards)
        n_succ = sum(1 for r in rewards if r > 0)

        # 成功 episode 步数
        success_steps = []
        all_ep_steps = []
        for i, r in enumerate(rewards):
            steps = ep_steps[i] if i < len(ep_steps) else 0
            all_ep_steps.append(steps)
            if r > 0 and steps > 0:
                success_steps.append(steps)

        # Replan 计算
        if is_baseline:
            # Baseline: 固定每 10 步 replan = sum(ceil(ep_steps / 10))
            if ep_steps:
                task_replan = sum(math.ceil(s / N_ACTION_STEPS) for s in ep_steps)
            else:
                # 无 ep_steps 数据 (如 LIBERO-10 Baseline), 用估算值
                # 估算: 成功 episode ~246 步, 失败 ~520 步
                est_succ_steps = 246
                est_fail_steps = 520
                task_replan = (n_succ * math.ceil(est_succ_steps / N_ACTION_STEPS)
                               + (n_ep - n_succ) * math.ceil(est_fail_steps / N_ACTION_STEPS))
        else:
            # CMP: drift 阈值触发 = sum(replan_counts)
            task_replan = sum(replan_counts) if replan_counts else 0

        total_episodes += n_ep
        total_success += n_succ
        total_steps += sum(all_ep_steps)
        total_success_steps += sum(success_steps)
        total_replan += task_replan

        per_task_results.append({
            "task_id": tid,
            "n_episodes": n_ep,
            "n_success": n_succ,
            "success_rate": n_succ / n_ep * 100 if n_ep > 0 else 0,
            "avg_success_steps": np.mean(success_steps) if success_steps else 0,
            "replan": task_replan,
            "ep_steps": all_ep_steps,
            "replan_counts": replan_counts,
        })

    # 评估时间
    overall = data.get("overall", {})
    eval_time_s = overall.get("eval_s", 0)
    eval_time_min = eval_time_s / 60 if eval_time_s else 0

    # 成功率 & 成功步数
    success_rate = total_success / total_episodes * 100 if total_episodes > 0 else 0
    avg_success_steps = total_success_steps / total_success if total_success > 0 else 0

    # VLA Inference Count & Chunk 利用分析
    # VLA inference = 每次调用 predict_action_chunk / sample_actions 的次数
    if is_baseline:
        # Baseline: 每 10 步调一次 VLA = replan 次数就是 VLA 推理次数
        vla_inference_count = total_replan
        steps_per_inference = N_ACTION_STEPS  # 固定 10 步/次
        chunk_analysis = None
    elif total_steps > 0 and total_replan >= 0:
        # CMP: VLA 推理次数 = total_chunks (初始 + 用满续期 + replan)
        # replan chunk: 在 seg=1 触发 replan, 只用了 seg 0 = 10 步
        # full chunk: 用满全部 5 段 = 50 步
        # total_steps = full_chunks * 50 + replan_chunks * 10
        replan_chunks = total_replan
        full_chunks = (total_steps - replan_chunks * N_ACTION_STEPS) / CHUNK_SIZE
        total_chunks = full_chunks + replan_chunks
        vla_inference_count = int(round(total_chunks))
        avg_steps_per_chunk = total_steps / total_chunks if total_chunks > 0 else 0
        steps_per_inference = avg_steps_per_chunk
        utilization = avg_steps_per_chunk / CHUNK_SIZE * 100

        chunk_analysis = {
            "replan_chunks": replan_chunks,
            "full_chunks": int(round(full_chunks)),
            "total_chunks": vla_inference_count,
            "avg_steps_per_chunk": avg_steps_per_chunk,
            "utilization": utilization,
        }
    else:
        vla_inference_count = 0
        steps_per_inference = 0
        chunk_analysis = None

    return {
        "json_path": str(json_path),
        "is_baseline": is_baseline,
        "total_episodes": total_episodes,
        "total_success": total_success,
        "success_rate": success_rate,
        "avg_success_steps": avg_success_steps,
        "total_steps": total_steps,
        "total_replan": total_replan,
        "vla_inference_count": vla_inference_count,
        "steps_per_inference": steps_per_inference,
        "eval_time_min": eval_time_min,
        "chunk_analysis": chunk_analysis,
        "per_task": per_task_results,
    }


# ── 输出格式化 ────────────────────────────────────────────────────
def print_single_analysis(result: dict, label: str = ""):
    """打印单个配置的详细分析。"""
    print(f"\n{'=' * 70}")
    print(f"  {label or result['json_path']}")
    print(f"{'=' * 70}")
    print(f"  成功率:     {result['success_rate']:.1f}% ({result['total_success']}/{result['total_episodes']})")
    print(f"  成功步数:   {result['avg_success_steps']:.1f}")
    print(f"  总 replan:  {result['total_replan']}")
    if result['eval_time_min'] > 0:
        print(f"  评估时间:   {result['eval_time_min']:.1f} 分钟")
    print(f"  总步数:     {result['total_steps']}")

    if result["chunk_analysis"]:
        ca = result["chunk_analysis"]
        print(f"\n  ── Chunk 利用分析 ──")
        print(f"  full chunk (用满50步):     {ca['full_chunks']}")
        print(f"  replan chunk (~10步):      {ca['replan_chunks']}")
        print(f"  总 chunk:                  {ca['total_chunks']}")
        print(f"  平均步数/chunk:            {ca['avg_steps_per_chunk']:.1f}")
        print(f"  chunk 利用率:              {ca['utilization']:.1f}%")

    print(f"\n  ── Per-task ──")
    print(f"  {'Task':>4}  {'成功率':>7}  {'成功步数':>8}  {'replan':>7}")
    print(f"  {'─' * 35}")
    for t in result["per_task"]:
        print(f"  {t['task_id']:>4}  {t['success_rate']:>6.1f}%  {t['avg_success_steps']:>8.1f}  {t['replan']:>7}")


def print_comparison_table(all_results: list[tuple[str, str, dict]]):
    """打印完整对比大表。

    Args:
        all_results: [(任务集, 配置名, 分析结果), ...]
    """
    # 找到每个任务集的 Baseline 结果
    baselines = {}
    for task_set, config_name, result in all_results:
        if config_name == "Baseline":
            baselines[task_set] = result

    print(f"\n{'=' * 120}")
    print(f"{'Baseline vs CMP 完整对比大表':^120}")
    print(f"{'=' * 120}")
    print()
    header = (f"  {'任务集':<16} {'配置':<10} {'成功率':>6} {'成功步数':>8} "
              f"{'replan':>7} {'VLA推理':>7} {'步/推理':>7} "
              f"{'vs Baseline':>12} {'推理减少':>10}")
    print(header)
    print(f"  {'─' * 90}")

    prev_task_set = None
    for task_set, config_name, result in all_results:
        if prev_task_set and task_set != prev_task_set:
            print(f"  {'─' * 90}")
        prev_task_set = task_set

        sr = result["success_rate"]
        steps = result["avg_success_steps"]
        replan = result["total_replan"]
        vla_inf = result["vla_inference_count"]
        spi = result["steps_per_inference"]

        # vs Baseline
        bl = baselines.get(task_set)
        if bl and config_name != "Baseline":
            vs_bl = f"{sr - bl['success_rate']:+.1f}pp"
            if bl["vla_inference_count"] > 0:
                inf_reduction = f"{(1 - vla_inf / bl['vla_inference_count']) * 100:.1f}%"
            else:
                inf_reduction = "--"
        else:
            vs_bl = "--"
            inf_reduction = "--"

        steps_str = f"{steps:.1f}" if steps > 0 else "~246"
        spi_str = f"{spi:.1f}" if spi > 0 else "10.0"

        print(f"  {task_set:<16} {config_name:<10} {sr:>5.1f}% {steps_str:>8} "
              f"{replan:>7} {vla_inf:>7} {spi_str:>7} "
              f"{vs_bl:>12} {inf_reduction:>10}")

    print(f"  {'─' * 90}")
    print()
    print(f"  指标说明:")
    print(f"    成功率:   成功 episode / 总 episode (500 per task set)")
    print(f"    成功步数: 成功 episode 平均步数")
    print(f"    replan:   drift 阈值触发的重规划次数 (Baseline = 固定每10步)")
    print(f"    VLA推理:  VLA 模型总推理次数 (predict_action_chunk 调用次数)")
    print(f"              Baseline = sum(ceil(ep_steps/10)), CMP = total_chunks")
    print(f"    步/推理:  平均每次 VLA 推理执行的动作步数 (越高越高效)")
    print(f"              Baseline 固定 10, CMP 越高说明 chunk 复用越好")
    print(f"    推理减少: 1 - CMP_VLA推理 / Baseline_VLA推理 (Inference Reduction)")

    # 综合结论表
    print(f"\n{'─' * 95}")
    print(f"  {'综合结论: 每个任务集的最佳 CMP 配置':^95}")
    print(f"{'─' * 95}")
    print(f"  {'任务集':<16} {'Baseline':>8} {'最佳CMP':>8} {'配置':<10} "
          f"{'成功率差':>8} {'推理减少':>8} {'步/推理':>10}")
    print(f"  {'─' * 75}")

    for task_set in dict.fromkeys(ts for ts, _, _ in all_results):
        bl = baselines.get(task_set)
        if not bl:
            continue

        # 找最佳 CMP (成功率最高, 同成功率取 VLA 推理更少的)
        best = None
        best_config = None
        for ts, cn, r in all_results:
            if ts == task_set and cn != "Baseline":
                if best is None or r["success_rate"] > best["success_rate"] or \
                   (r["success_rate"] == best["success_rate"] and r["vla_inference_count"] < best["vla_inference_count"]):
                    best = r
                    best_config = cn

        if best:
            diff = best["success_rate"] - bl["success_rate"]
            inf_red = (1 - best["vla_inference_count"] / bl["vla_inference_count"]) * 100 if bl["vla_inference_count"] > 0 else 0
            spi = best["steps_per_inference"]
            marker = " ✓" if diff >= -1.0 else ""
            print(f"  {task_set:<16} {bl['success_rate']:>7.1f}% {best['success_rate']:>7.1f}% {best_config:<10} "
                  f"{diff:>+7.1f}pp {inf_red:>7.1f}% {spi:>7.1f}步/次{marker}")

    print()


# ── 主函数 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CMP vs Baseline 评估结果综合分析")
    parser.add_argument("--json", type=str, help="分析单个 eval_info.json 文件")
    parser.add_argument("--baseline", action="store_true", help="标记 --json 为 Baseline 格式")
    parser.add_argument("--config", type=str, help="只分析指定任务集 (10/spatial/object/goal)")
    parser.add_argument("--detail", action="store_true", help="显示 per-task 详情")
    args = parser.parse_args()

    # 模式 1: 分析单个文件
    if args.json:
        path = Path(args.json)
        if not path.exists():
            print(f"[ERROR] 文件不存在: {path}")
            sys.exit(1)
        result = analyze_eval_json(path, is_baseline=args.baseline)
        print_single_analysis(result, label=str(path))
        return

    # 模式 2: 分析所有/指定配置
    filter_map = {
        "10": "LIBERO-10",
        "libero10": "LIBERO-10",
        "spatial": "LIBERO-SPATIAL",
        "object": "LIBERO-OBJECT",
        "goal": "LIBERO-GOAL",
    }
    task_filter = filter_map.get(args.config.lower()) if args.config else None

    all_results = []
    for task_set, config_name, json_path, is_baseline in CONFIGS:
        if task_filter and task_set != task_filter:
            continue
        if not json_path.exists():
            print(f"[SKIP] {task_set} {config_name}: {json_path} 不存在")
            continue

        result = analyze_eval_json(json_path, is_baseline=is_baseline)
        all_results.append((task_set, config_name, result))

        if args.detail:
            print_single_analysis(result, label=f"{task_set} {config_name}")

    if not all_results:
        print("[ERROR] 没有找到任何有效的评估数据")
        sys.exit(1)

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
