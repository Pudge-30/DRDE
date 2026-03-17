#!/usr/bin/env python3
"""对指定 checkpoints 目录下所有模型进行批量评估，并生成对比报告。

功能：
1. 对每个 checkpoint 运行 lerobot-eval，计算 success rate、retrieval、CKA、similarity
2. 为每个模型的 z1/z2 模态绘制 t-SNE 图（由 eval 流程自动生成）
3. 根据各指标绘制折线图
4. 在输出目录生成评估报告

用法示例：

    PYTHONPATH=/home/kemove/qyh/DRDE/DRDE/src python -m lerobot.scripts.lerobot_compare_eval \
        --checkpoints_dir=/home/kemove/qyh/DRDE/DRDE/outputs/pi05_libero_base_ft/checkpoints \
        --output_dir=/home/kemove/qyh/DRDE/DRDE/outputs/pi05_libero_compare_eval \
        --env.type=libero \
        --env.task=libero_10 \
        --eval.n_episodes=50 \
        --eval.batch_size=12
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# 兼容无 matplotlib 环境
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def discover_checkpoints(checkpoints_dir: Path) -> list[tuple[str, Path]]:
    """发现 checkpoints 目录下所有有效的 pretrained_model 路径。

    Returns:
        List of (checkpoint_name, pretrained_model_path), 按名称排序（如 001000, 002000, ...）
    """
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints 目录不存在: {checkpoints_dir}")

    results: list[tuple[str, Path]] = []
    for sub in sorted(checkpoints_dir.iterdir()):
        if not sub.is_dir():
            continue
        pretrained = sub / "pretrained_model"
        config_path = pretrained / "config.json"
        if config_path.exists():
            results.append((sub.name, pretrained))
    return results


def run_eval_for_checkpoint(
    policy_path: Path,
    output_dir: Path,
    base_args: list[str],
    pythonpath: str | None = None,
) -> bool:
    """对单个 checkpoint 运行 lerobot-eval。

    Returns:
        True 表示成功，False 表示失败。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_eval",
        f"--policy.path={policy_path}",
        f"--output_dir={output_dir}",
        *base_args,
    ]
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath

    logging.info("Running: %s", " ".join(cmd))
    try:
        with open(log_path, "w") as f:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=72000,  # 2 hours per checkpoint
            )
        if result.returncode != 0:
            logging.warning("Eval 失败 (exit=%d): %s", result.returncode, policy_path)
            return False
        return True
    except subprocess.TimeoutExpired:
        logging.error("Eval 超时: %s", policy_path)
        return False
    except Exception as e:
        logging.exception("Eval 异常: %s", policy_path)
        return False


def parse_base_args(extra: list[str]) -> list[str]:
    """将用户传入的 --key=value 转换为 lerobot-eval 可用的参数列表。"""
    args = []
    for s in extra:
        if s.startswith("--") and "=" in s:
            args.append(s)
        elif s.startswith("--"):
            args.append(s)
    return args


def collect_results(output_dir: Path, checkpoint_names: list[str]) -> list[dict]:
    """从各 checkpoint 的 eval_info.json 收集指标。"""
    results = []
    for name in checkpoint_names:
        info_path = output_dir / name / "eval_info.json"
        if not info_path.exists():
            results.append({
                "checkpoint": name,
                "success": False,
                "error": f"eval_info.json 不存在",
            })
            continue

        with open(info_path) as f:
            info = json.load(f)

        overall = info.get("overall", info)
        pc_success = overall.get("pc_success", float("nan"))
        modal = overall.get("modal_alignment", {}) or {}

        tsne_path = modal.get("tsne_plot_path", "")
        if tsne_path and not Path(tsne_path).exists():
            tsne_path = ""

        row = {
            "checkpoint": name,
            "success": True,
            "pc_success": float(pc_success) if pc_success is not None else float("nan"),
            "mean_cosine_similarity": modal.get("mean_cosine_similarity", float("nan")),
            "linear_cka": modal.get("linear_cka", float("nan")),
            "tsne_plot_path": tsne_path,
            "itm_auc_roc": modal.get("itm_auc_roc", float("nan")),
            "ranking_kendall_tau": modal.get("ranking_kendall_tau", float("nan")),
            "ranking_spearman": modal.get("ranking_spearman", float("nan")),
            "effective_dim_z1": modal.get("effective_dim_z1", float("nan")),
            "effective_dim_z2": modal.get("effective_dim_z2", float("nan")),
            "mAP": modal.get("mAP", float("nan")),
            "pos_sim_mean": modal.get("pos_sim_mean", float("nan")),
            "pos_sim_std": modal.get("pos_sim_std", float("nan")),
            "neg_sim_mean": modal.get("neg_sim_mean", float("nan")),
            "neg_sim_std": modal.get("neg_sim_std", float("nan")),
            "pos_neg_sim_gap": modal.get("pos_neg_sim_gap", float("nan")),
            "sim_dist_plot_path": modal.get("sim_dist_plot_path", ""),
        }
        # 收集所有 retrieval_* 指标（Recall@1/5/10/20 等）
        for key, val in modal.items():
            if key.startswith("retrieval_") and isinstance(val, (int, float)):
                row[key] = float(val)
        results.append(row)
    return results


def _recall_avg_for_k(results: list[dict], k: int) -> list[float]:
    """对每个 result 取 Recall@k 的 (z1→z2 + z2→z1)/2，缺失则 nan。"""
    fwd_key = f"retrieval_z1_to_z2_recall@{k}"
    rev_key = f"retrieval_z2_to_z1_recall@{k}"
    out = []
    for r in results:
        a, b = r.get(fwd_key, float("nan")), r.get(rev_key, float("nan"))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not (np.isnan(a) or np.isnan(b)):
            out.append((a + b) / 2)
        else:
            out.append(float("nan"))
    return out


def plot_metrics_line_charts(results: list[dict], output_dir: Path) -> None:
    """根据 retrieval、CKA、similarity、success rate 绘制折线图（含 Recall@1/5/10 等）。"""
    if not HAS_MATPLOTLIB:
        logging.warning("未安装 matplotlib，跳过折线图绘制")
        return

    valid = [r for r in results if r.get("success")]
    if not valid:
        logging.warning("无有效结果，跳过折线图")
        return

    labels = [r["checkpoint"] for r in valid]
    x = np.arange(len(labels))

    # 确定有哪些 Recall@K 有数据（从第一个有效结果取）
    recall_ks: list[int] = []
    for key in valid[0]:
        if key.startswith("retrieval_") and "@" in key:
            try:
                k = int(key.split("@")[1])
                if k not in recall_ks:
                    recall_ks.append(k)
            except ValueError:
                pass
    recall_ks = sorted(recall_ks)[:3]  # 最多画 3 个 Recall 子图：@1, @5, @10（或 @20）

    # Success, CosSim, CKA + Recall@K + AUC-ROC, Kendall, Spearman, mAP, EffDim, Pos-Neg Gap
    n_plots = 3 + len(recall_ks) + 6
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    idx = 0
    # 1. Success Rate
    ax = axes_flat[idx]
    idx += 1
    vals = [r.get("pc_success", float("nan")) for r in valid]
    ax.plot(x, vals, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # 2. Mean Cosine Similarity
    ax = axes_flat[idx]
    idx += 1
    vals = [r.get("mean_cosine_similarity", float("nan")) for r in valid]
    ax.plot(x, vals, "o-", color="coral", linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Similarity (z1 ↔ z2)")
    ax.grid(True, alpha=0.3)

    # 3. Linear CKA
    ax = axes_flat[idx]
    idx += 1
    vals = [r.get("linear_cka", float("nan")) for r in valid]
    ax.plot(x, vals, "o-", color="seagreen", linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Linear CKA")
    ax.set_title("Linear CKA (z1 ↔ z2)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 4+ Recall@K
    colors = ["purple", "darkorange", "brown"]
    for i, k in enumerate(recall_ks):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        idx += 1
        r_fwd = [r.get(f"retrieval_z1_to_z2_recall@{k}", float("nan")) for r in valid]
        r_rev = [r.get(f"retrieval_z2_to_z1_recall@{k}", float("nan")) for r in valid]
        avg = _recall_avg_for_k(valid, k)
        color = colors[i % len(colors)]
        ax.plot(x, avg, "o-", color=color, linewidth=2, markersize=8, label=f"R@{k} (avg)")
        ax.plot(x, r_fwd, "s--", color="gray", linewidth=1, markersize=4, alpha=0.7, label="z1→z2")
        ax.plot(x, r_rev, "^--", color="gray", linewidth=1, markersize=4, alpha=0.7, label="z2→z1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(f"Recall@{k}")
        ax.set_title(f"Retrieval Recall@{k}")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    # AUC-ROC
    if idx < len(axes_flat):
        ax = axes_flat[idx]
        idx += 1
        vals = [r.get("itm_auc_roc", float("nan")) for r in valid]
        ax.plot(x, vals, "o-", color="darkviolet", linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("ITM AUC-ROC")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # Kendall τ
    if idx < len(axes_flat):
        ax = axes_flat[idx]
        idx += 1
        vals = [r.get("ranking_kendall_tau", float("nan")) for r in valid]
        ax.plot(x, vals, "o-", color="teal", linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Kendall τ")
        ax.set_title("Ranking Kendall τ")
        ax.grid(True, alpha=0.3)

    # Spearman
    if idx < len(axes_flat):
        ax = axes_flat[idx]
        idx += 1
        vals = [r.get("ranking_spearman", float("nan")) for r in valid]
        ax.plot(x, vals, "o-", color="darkcyan", linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Ranking Spearman")
        ax.grid(True, alpha=0.3)

    # mAP
    if idx < len(axes_flat):
        ax = axes_flat[idx]
        idx += 1
        vals = [r.get("mAP", float("nan")) for r in valid]
        ax.plot(x, vals, "o-", color="chocolate", linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("mAP")
        ax.set_title("Mean Average Precision")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # Effective Dimension (z1, z2)
    if idx < len(axes_flat):
        ax = axes_flat[idx]
        idx += 1
        v1 = [r.get("effective_dim_z1", float("nan")) for r in valid]
        v2 = [r.get("effective_dim_z2", float("nan")) for r in valid]
        ax.plot(x, v1, "o-", color="navy", linewidth=2, markersize=8, label="z1")
        ax.plot(x, v2, "s-", color="orangered", linewidth=2, markersize=8, label="z2")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Effective Dim")
        ax.set_title("Representation Effective Dimension")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    # Pos-Neg Sim Gap
    if idx < len(axes_flat):
        ax = axes_flat[idx]
        idx += 1
        vals = [r.get("pos_neg_sim_gap", float("nan")) for r in valid]
        ax.plot(x, vals, "o-", color="forestgreen", linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Pos-Neg Gap")
        ax.set_title("Positive vs Negative Similarity Gap")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

    for j in range(idx, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plot_path = output_dir / "metrics_line_charts.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("折线图已保存至 %s", plot_path)


def _sorted_recall_keys(result: dict) -> list[str]:
    """从单条 result 中取出所有 retrieval_* 键并按 @K 与方向排序。"""
    keys = [k for k in result if k.startswith("retrieval_") and isinstance(result.get(k), (int, float))]
    def order(k: str):
        # retrieval_z1_to_z2_recall@10 -> (10, 0), retrieval_z2_to_z1_recall@10 -> (10, 1)
        if "@" not in k:
            return (999, 1)
        try:
            k_num = int(k.split("@")[1])
        except ValueError:
            k_num = 999
        direction = 0 if "z1_to_z2" in k else 1
        return (k_num, direction)
    return sorted(keys, key=order)


def _recall_key_short_label(key: str) -> str:
    """将 retrieval_z1_to_z2_recall@5 转为简短表头，如 R@5 (z1→z2)。"""
    if "z1_to_z2" in key:
        direction = "z1→z2"
    else:
        direction = "z2→z1"
    if "@" in key:
        try:
            k_num = key.split("@")[1]
            return f"R@{k_num} ({direction})"
        except Exception:
            pass
    return key


def copy_tsne_plots(results: list[dict], output_dir: Path) -> None:
    """将各 checkpoint 的 t-SNE 图与正负相似度分布图复制到输出目录的 tsne_plots 子目录。"""
    tsne_dir = output_dir / "tsne_plots"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        if not r.get("success"):
            continue
        src = r.get("tsne_plot_path")
        if src and Path(src).exists():
            dst = tsne_dir / f"modal_alignment_tsne_{r['checkpoint']}.png"
            try:
                shutil.copy2(src, dst)
                logging.info("复制 t-SNE: %s -> %s", src, dst)
            except Exception as e:
                logging.warning("复制 t-SNE 失败: %s", e)
        src_sim = r.get("sim_dist_plot_path")
        if src_sim and Path(src_sim).exists():
            dst_sim = tsne_dir / f"modal_alignment_sim_dist_{r['checkpoint']}.png"
            try:
                shutil.copy2(src_sim, dst_sim)
                logging.info("复制相似度分布图: %s -> %s", src_sim, dst_sim)
            except Exception as e:
                logging.warning("复制相似度分布图失败: %s", e)


def generate_report(results: list[dict], output_dir: Path, checkpoints_dir: Path) -> None:
    """生成 Markdown 评估报告。表中包含从 JSON 读取的所有 Recall 指标。"""
    report_path = output_dir / "evaluation_report.md"
    valid = [r for r in results if r.get("success")]

    # 从任意一条成功结果得到所有 Recall 列顺序（保证表头与列数一致）
    recall_keys: list[str] = []
    for r in valid:
        recall_keys = _sorted_recall_keys(r)
        if recall_keys:
            break

    extra_metric_headers = [
        "AUC-ROC", "Kendall τ", "Spearman", "mAP",
        "EffDim z1", "EffDim z2", "Pos-Neg Gap",
    ]
    header_cells = (
        ["Checkpoint", "Success Rate (%)", "Mean Cos Sim", "Linear CKA"]
        + extra_metric_headers
        + [_recall_key_short_label(k) for k in recall_keys]
    )
    n_cols = len(header_cells)
    sep = "|" + "|".join(["------------"] * n_cols) + "|"
    table_header = "| " + " | ".join(header_cells) + " |"
    # 失败行：列数 = n_cols，最后一列写错误，其余列 "-"
    fail_last_col = "*(失败: {err})*"

    lines = [
        "# Checkpoint 批量评估报告",
        "",
        "## 概览",
        f"- **Checkpoints 目录**: `{checkpoints_dir}`",
        f"- **输出目录**: `{output_dir}`",
        f"- **评估模型数**: {len(results)}",
        f"- **成功完成**: {len(valid)}",
        "",
        "## 指标汇总表",
        "",
        table_header,
        sep,
    ]

    for r in results:
        ckpt = r["checkpoint"]
        if r.get("success"):
            sr = r.get("pc_success", float("nan"))
            mcs = r.get("mean_cosine_similarity", float("nan"))
            cka = r.get("linear_cka", float("nan"))
            def _fmt(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "-"
                return f"{float(v):.4f}"
            cells = [ckpt, f"{sr:.2f}", f"{mcs:.4f}", f"{cka:.4f}"]
            cells += [
                _fmt(r.get("itm_auc_roc")),
                _fmt(r.get("ranking_kendall_tau")),
                _fmt(r.get("ranking_spearman")),
                _fmt(r.get("mAP")),
                _fmt(r.get("effective_dim_z1")),
                _fmt(r.get("effective_dim_z2")),
                _fmt(r.get("pos_neg_sim_gap")),
            ]
            for k in recall_keys:
                val = r.get(k, float("nan"))
                cells.append(f"{val:.4f}" if isinstance(val, (int, float)) and not np.isnan(val) else "-")
            lines.append("| " + " | ".join(cells) + " |")
        else:
            err = r.get("error", "unknown")
            cells = [ckpt] + ["-"] * (n_cols - 2) + [fail_last_col.format(err=err)]
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## 图表",
        "",
        "- **指标折线图**: `metrics_line_charts.png`",
        "- **t-SNE 图**: 见 `tsne_plots/` 目录下各 checkpoint 的 `modal_alignment_tsne_<checkpoint>.png`",
        "- **正负相似度分布图**: 见 `tsne_plots/` 下 `modal_alignment_sim_dist_<checkpoint>.png`",
        "",
        "## 指标说明",
        "",
        "- **Success Rate**: 任务成功率 (%)",
        "- **Mean Cos Sim**: z1（观测）与 z2（动作）嵌入的成对余弦相似度均值",
        "- **Linear CKA**: Centered Kernel Alignment，度量 z1 与 z2 表示空间的相似度",
        "- **AUC-ROC**: ITM 二分类能力（正对 vs 负对相似度作为得分）的 ROC 曲线下面积",
        "- **Kendall τ / Spearman**: 每个 query 的相似度排序与理想排序（正样本排第一）的相关系数均值",
        "- **mAP**: Mean Average Precision，检索正样本时的平均精度均值",
        "- **EffDim z1/z2**: 表示有效维度 (Σλ)²/Σλ²，衡量表示空间利用率",
        "- **Pos-Neg Gap**: 正对与负对余弦相似度均值之差",
        "- **R@K (z1→z2)**: 给定 z1 检索 top-K 最相似 z2 时配对命中的比例",
        "- **R@K (z2→z1)**: 给定 z2 检索 top-K 最相似 z1 时配对命中的比例",
        "",
        "## 正负样本相似度分布",
        "",
        "- 各 checkpoint 的正负对相似度分布直方图见 `tsne_plots/` 下 `modal_alignment_sim_dist_<checkpoint>.png`",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("评估报告已保存至 %s", report_path)


def build_lerobot_eval_args(parsed: argparse.Namespace) -> list[str]:
    """根据解析后的参数构建 lerobot-eval 的额外参数。"""
    args = []
    # 移除 compare_eval 专用参数，只保留 env / eval / policy 相关
    skip = {"checkpoints_dir", "output_dir", "pythonpath", "skip_eval"}
    for k, v in vars(parsed).items():
        if k in skip or v is None:
            continue
        if k == "env_type":
            args.append(f"--env.type={v}")
        elif k == "env_task":
            args.append(f"--env.task={v}")
        elif k == "eval_batch_size":
            args.append(f"--eval.batch_size={v}")
        elif k == "eval_n_episodes":
            args.append(f"--eval.n_episodes={v}")
        elif k == "eval_max_episodes_rendered":
            args.append(f"--eval.max_episodes_rendered={v}")
        elif k == "eval_use_async_envs":
            args.append(f"--eval.use_async_envs={str(bool(v)).lower()}")
        elif k == "policy_replan_mode":
            args.append(f"--policy.replan_mode={v}")
        elif k == "policy_n_action_steps":
            args.append(f"--policy.n_action_steps={v}")
        elif k == "policy_device":
            args.append(f"--policy.device={v}")
        elif k == "seed":
            args.append(f"--seed={v}")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="对 checkpoints 目录下所有模型进行批量评估并生成对比报告"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Checkpoints 根目录，例如 .../outputs/pi05_libero_base_ft/checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="评估结果输出目录",
    )
    parser.add_argument(
        "--pythonpath",
        type=str,
        default=None,
        help="PYTHONPATH，默认使用当前项目 src 路径",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="跳过 eval，仅根据已有 eval_info.json 生成报告和图表",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="只评估指定名称的 checkpoint（支持带/不带前导零，如 60000 或 060000）；不指定则评估所有",
    )
    # 评估参数（与参考命令一致，使用下划线避免 argparse 与点的冲突）
    parser.add_argument("--env_type", default="libero", help="env.type")
    parser.add_argument("--env_task", default="libero_10", help="env.task")
    parser.add_argument("--eval_batch_size", type=int, default=12, help="eval.batch_size")
    parser.add_argument("--eval_n_episodes", type=int, default=50, help="eval.n_episodes")
    parser.add_argument("--eval_max_episodes_rendered", type=int, default=10)
    parser.add_argument("--eval_use_async_envs", action="store_true")
    parser.add_argument("--policy_replan_mode", default="fixed")
    parser.add_argument("--policy_n_action_steps", type=int, default=10)
    parser.add_argument("--policy_device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)

    parsed, extra = parser.parse_known_args()
    checkpoints_dir = Path(parsed.checkpoints_dir).resolve()
    output_dir = Path(parsed.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pythonpath = parsed.pythonpath
    if not pythonpath:
        # 尝试推断：通常 lerobot 在 src 下
        candidate = Path(__file__).resolve().parent.parent.parent
        if (candidate / "lerobot").exists():
            pythonpath = str(candidate)
        else:
            pythonpath = ""

    pairs = discover_checkpoints(checkpoints_dir)
    if not pairs:
        logging.error("未发现任何 checkpoint")
        sys.exit(1)
    logging.info("发现 %d 个 checkpoint: %s", len(pairs), [p[0] for p in pairs])

    if parsed.checkpoint is not None:
        target = parsed.checkpoint.strip()
        # 同时接受 "60000" 和 "060000" 等带/不带前导零的写法
        candidates = {target, target.lstrip("0") or "0"}
        try:
            candidates.add(f"{int(target):06d}")
        except ValueError:
            pass
        filtered = [(name, path) for name, path in pairs if name in candidates]
        if not filtered:
            logging.error(
                "未找到名称匹配 %r 的 checkpoint（可用: %s）",
                target, [p[0] for p in pairs],
            )
            sys.exit(1)
        logging.info("仅评估指定 checkpoint: %s", [p[0] for p in filtered])
        pairs = filtered

    base_args = build_lerobot_eval_args(parsed)
    # 用户传入的额外参数（如 --env.type=libero）追加到 base_args
    for a in extra:
        if a.startswith("--") and "=" in a:
            base_args.append(a)

    if not parsed.skip_eval:
        for name, policy_path in pairs:
            ckpt_out = output_dir / name
            ok = run_eval_for_checkpoint(
                policy_path=policy_path,
                output_dir=ckpt_out,
                base_args=base_args,
                pythonpath=pythonpath,
            )
            if not ok:
                logging.warning("Checkpoint %s 评估失败，将继续处理", name)

    checkpoint_names = [p[0] for p in pairs]
    results = collect_results(output_dir, checkpoint_names)

    copy_tsne_plots(results, output_dir)
    plot_metrics_line_charts(results, output_dir)
    generate_report(results, output_dir, checkpoints_dir)

    logging.info("批量评估完成，结果目录: %s", output_dir)


if __name__ == "__main__":
    main()
