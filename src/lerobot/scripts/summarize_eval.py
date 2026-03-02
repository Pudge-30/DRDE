#!/usr/bin/env python3
"""
评估结果统计脚本：从 eval_info.json 和 replan log 中提取精确数据，
生成汇总表追加到 eval_record.txt。

统计内容：
  1. 准确的 replan 次数（模拟计算精确推理次数）
  2. 成功/失败 episode 的实际 step 数
  3. 成功率

用法：
  python -m lerobot.scripts.summarize_eval
  # 或
  python src/lerobot/scripts/summarize_eval.py
"""

import json
import re
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────
BASE = Path("/home/kemove/hyp/lhm-heritage")
RECORD_PATH = BASE / "src/lerobot/scripts/eval_record.txt"

EPISODE_LENGTH = 520
CHUNK_SIZE = 50
N_ACTION_STEPS = 10

EXPERIMENTS = {
    "baseline": {
        "label": "baseline (纯BC, 每10步replan)",
        "json": BASE / "outputs/eval_baseline/eval_info.json",
        "log": None,
        "replan_strategy": "固定每10步",
        "infer_per_ep": EPISODE_LENGTH // N_ACTION_STEPS,  # 52
    },
    "no_replan": {
        "label": "neg_imprv 无replan (ckpt 6000)",
        "json": BASE / "outputs/neg_imprv/eval_6000/eval_info.json",
        "log": BASE / "outputs/neg_imprv/eval_6000.log",
        "replan_strategy": "50步跑完",
        "infer_per_ep": EPISODE_LENGTH // CHUNK_SIZE + 1,  # 11
    },
    "drift_replan": {
        "label": "neg_imprv drift replan=0.10 (ckpt 6000)",
        "json": BASE / "outputs/neg_imprv/eval_6000_replan_0.1/eval_info.json",
        "log": BASE / "outputs/neg_imprv/eval_6000_replan_0.1.log",
        "replan_strategy": "drift>0.10",
        "infer_per_ep": None,  # 需要精确计算
    },
}


# ── 工具函数 ──────────────────────────────────────────────────────
def load_eval_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_per_task(data: dict) -> list[dict]:
    """
    从 eval_info.json 提取 per-task 统计。
    兼容两种格式:
      - baseline: per_task[i] = {task_id, success_rate, successes(int), failures(int)}
      - neg_imprv: per_task[i] = {task_id, metrics: {sum_rewards: [...], ...}}
    """
    results = []
    for t in data.get("per_task", []):
        tid = t.get("task_id", len(results))
        if "metrics" in t:
            m = t["metrics"]
            rewards = m.get("sum_rewards", [])
            successes_list = [r == 1.0 for r in rewards]
            n_succ = sum(successes_list)
            n_total = len(rewards)
            # ep_steps
            ep_steps_all = m.get("ep_steps", [])
            # 如果没有 ep_steps 字段，尝试从 per_episode 推断
            drift_means = m.get("drift_means", [])
            drift_maxs = m.get("drift_maxs", [])
            replan_counts = m.get("replan_counts", [])
        else:
            n_succ = t.get("successes", 0)
            n_total = n_succ + t.get("failures", 0)
            successes_list = [True] * n_succ + [False] * t.get("failures", 0)
            ep_steps_all = []
            drift_means = []
            drift_maxs = []
            replan_counts = []

        results.append({
            "task_id": tid,
            "n_succ": n_succ,
            "n_total": n_total,
            "rate": n_succ / n_total * 100 if n_total else 0,
            "successes_list": successes_list,
            "ep_steps": ep_steps_all,
            "drift_means": drift_means,
            "drift_maxs": drift_maxs,
            "replan_counts": replan_counts,
        })
    return results


def extract_ep_steps_from_per_episode(data: dict) -> list[dict]:
    """
    从 eval_info.json 的 per_task -> metrics 级别，
    尝试还原每个 episode 的 {success, n_steps}。
    如果 eval_info 里没有 ep_steps 字段，返回空。
    """
    all_eps = []
    for t in data.get("per_task", []):
        if "metrics" not in t:
            continue
        m = t["metrics"]
        rewards = m.get("sum_rewards", [])
        ep_steps = m.get("ep_steps", [])
        for i, r in enumerate(rewards):
            ep = {"success": r == 1.0, "task_id": t["task_id"]}
            if i < len(ep_steps) and ep_steps[i]:
                ep["n_steps"] = int(ep_steps[i])
            all_eps.append(ep)
    return all_eps


# ── Replan log 解析 ──────────────────────────────────────────────
def parse_replan_log(log_path: Path) -> dict:
    """解析 replan log, 返回详细统计"""
    if not log_path or not log_path.exists():
        return {}

    rollouts = []
    current = []

    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[Replan\] chunk=(\d+), seg=(\d+), drift=([\d.]+)', line)
            if m:
                current.append({
                    "chunk": int(m.group(1)),
                    "seg": int(m.group(2)),
                    "drift": float(m.group(3)),
                })
            m2 = re.search(r'\[Replan\] rollout: (\d+) replans', line)
            if m2:
                rollouts.append(current[:])
                current = []

    # 统计总 rollout 数：用 chunk=10 的 [ITM-fresh] 行（每个 rollout 恰好一行）
    all_rollout_replan_counts = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[ITM-fresh\] chunk=10, fresh_count=\d+, .+replan_count=(\d+)', line)
            if m:
                all_rollout_replan_counts.append(int(m.group(1)))

    # 模拟精确推理次数
    def simulate_inferences(events):
        replan_map = {e["chunk"]: e["seg"] for e in events}
        step, inferences, chunk_idx = 0, 0, 0
        while step < EPISODE_LENGTH:
            inferences += 1
            if chunk_idx in replan_map:
                step += replan_map[chunk_idx] * N_ACTION_STEPS
            else:
                step += CHUNK_SIZE
            chunk_idx += 1
        return inferences

    # 统计各分布
    all_events = [e for r in rollouts for e in r]
    seg_counter = Counter(e["seg"] for e in all_events)
    chunk_counter = Counter(e["chunk"] for e in all_events)
    first_chunks = [r[0]["chunk"] for r in rollouts if r]

    n_total_rollouts = len(all_rollout_replan_counts) if all_rollout_replan_counts else len(rollouts)
    n_no_replan = n_total_rollouts - len(rollouts)

    # 精确推理次数
    infer_with_replan = [simulate_inferences(r) for r in rollouts]
    all_infer = [11] * n_no_replan + infer_with_replan
    avg_infer = sum(all_infer) / len(all_infer) if all_infer else 11

    # 级联检测
    cascade_chains = []
    for events in rollouts:
        chain = []
        for e in events:
            if e["seg"] == 1:
                if chain and e["chunk"] == chain[-1] + 1:
                    chain.append(e["chunk"])
                else:
                    if len(chain) >= 2:
                        cascade_chains.append(chain[:])
                    chain = [e["chunk"]]
            else:
                if len(chain) >= 2:
                    cascade_chains.append(chain[:])
                chain = []
        if len(chain) >= 2:
            cascade_chains.append(chain[:])

    return {
        "rollouts": rollouts,
        "n_total_rollouts": n_total_rollouts,
        "n_replan_rollouts": len(rollouts),
        "n_no_replan": n_no_replan,
        "total_events": len(all_events),
        "seg_dist": dict(sorted(seg_counter.items())),
        "chunk_dist": dict(sorted(chunk_counter.items())),
        "first_chunks": Counter(first_chunks),
        "all_infer": all_infer,
        "avg_infer": avg_infer,
        "infer_dist": Counter(all_infer),
        "cascade_chains": cascade_chains,
        "per_rollout_replans": [len(r) for r in rollouts],
    }


# ── 生成报告 ──────────────────────────────────────────────────────
def generate_report() -> str:
    lines = []

    # 加载数据
    exp_data = {}
    for name, cfg in EXPERIMENTS.items():
        if not cfg["json"].exists():
            print(f"[WARN] {cfg['json']} 不存在, 跳过 {name}")
            continue
        data = load_eval_json(cfg["json"])
        per_task = extract_per_task(data)
        all_eps = extract_ep_steps_from_per_episode(data)
        overall = data.get("overall", data.get("summary", {}))
        replan_stats = parse_replan_log(cfg["log"]) if cfg.get("log") else {}

        # 精确推理次数
        if replan_stats and replan_stats["avg_infer"]:
            infer_per_ep = replan_stats["avg_infer"]
        else:
            infer_per_ep = cfg["infer_per_ep"]

        exp_data[name] = {
            "cfg": cfg,
            "data": data,
            "per_task": per_task,
            "all_eps": all_eps,
            "overall": overall,
            "replan": replan_stats,
            "infer_per_ep": infer_per_ep,
        }

    # ═══════════════════════════════════════════════════════════════
    lines.append("")
    lines.append("=" * 105)
    lines.append("评估结果汇总表 (精确统计)")
    lines.append("=" * 105)
    lines.append("")

    # ── 总览 ────────────────────────────────────────────────────
    lines.append("┌─ 总览对比 " + "─" * 93)
    lines.append(f"│ {'实验':<38} {'replan策略':<14} {'成功率':>7} {'推理次数/ep':>12} {'replan次数/rollout':>18}")
    lines.append("│" + "─" * 104)

    for name in ["baseline", "no_replan", "drift_replan"]:
        if name not in exp_data:
            continue
        e = exp_data[name]
        pc = e["overall"].get("pc_success", 0)
        infer = e["infer_per_ep"]
        strategy = e["cfg"]["replan_strategy"]
        rp = e["replan"]
        if rp:
            avg_rp = f"{sum(rp['per_rollout_replans'])/len(rp['per_rollout_replans']):.1f} (有replan的)" if rp["per_rollout_replans"] else "0"
        else:
            avg_rp = "N/A" if name == "baseline" else "0"
        infer_str = f"{infer:.1f}" if isinstance(infer, float) else str(infer)
        lines.append(f"│ {e['cfg']['label']:<38} {strategy:<14} {pc:>6.1f}% {infer_str:>12} {avg_rp:>18}")
    lines.append("└" + "─" * 104)
    lines.append("")

    # ── Per-task 成功率 + step 数 ────────────────────────────────
    lines.append("┌─ Per-task 对比 (50 episodes/task) " + "─" * 69)
    # 检查是否有 step 数据
    has_steps = {}
    for name in ["baseline", "no_replan", "drift_replan"]:
        if name not in exp_data:
            has_steps[name] = False
            continue
        all_eps = exp_data[name]["all_eps"]
        has_steps[name] = any(ep.get("n_steps") for ep in all_eps)

    any_has_steps = any(has_steps.values())

    col_names = [("baseline", "baseline"), ("no_replan", "无replan"), ("drift_replan", "drift replan")]
    header = f"│ {'Task':>4}"
    for name, label_short in col_names:
        if name in exp_data:
            header += f" │ {label_short:>12}"
            if has_steps.get(name):
                header += f" {'succ步数':>8} {'fail步数':>8}"
    header += f" │ {'replan提升':>10} {'vs baseline':>11}"
    lines.append(header)
    lines.append("│" + "─" * 104)

    for tid in range(10):
        row = f"│ {tid:>4}"
        rates = {}
        for name in ["baseline", "no_replan", "drift_replan"]:
            if name not in exp_data:
                continue
            pt = exp_data[name]["per_task"]
            if tid < len(pt):
                t = pt[tid]
                rate = t["rate"]
                rates[name] = rate
                row += f" │ {rate:>11.0f}%"

                if has_steps.get(name):
                    # 按 success/fail 分组计算平均 step
                    eps = [ep for ep in exp_data[name]["all_eps"] if ep["task_id"] == tid]
                    succ_steps = [ep["n_steps"] for ep in eps if ep["success"] and ep.get("n_steps")]
                    fail_steps = [ep["n_steps"] for ep in eps if not ep["success"] and ep.get("n_steps")]
                    s_avg = f"{sum(succ_steps)/len(succ_steps):.0f}" if succ_steps else "-"
                    f_avg = f"{sum(fail_steps)/len(fail_steps):.0f}" if fail_steps else "-"
                    row += f" {s_avg:>8} {f_avg:>8}"
            else:
                row += f" │ {'-':>12}"
                rates[name] = 0

        # replan 提升 & vs baseline
        gain = rates.get("drift_replan", 0) - rates.get("no_replan", 0) if "drift_replan" in rates and "no_replan" in rates else 0
        if "drift_replan" in rates and "baseline" in rates:
            vs_bl = rates["drift_replan"] - rates["baseline"]
            marker = " ★" if vs_bl > 0 else ""
            row += f" │ {gain:>+9.0f}pp {vs_bl:>+10.0f}pp{marker}"
        elif "drift_replan" in rates:
            row += f" │ {gain:>+9.0f}pp {'N/A':>11}"
        else:
            row += f" │ {'':>10} {'':>11}"
        lines.append(row)

    # 总计
    lines.append("│" + "─" * 104)
    row = f"│ {'总计':>4}"
    total_rates = {}
    for name in ["baseline", "no_replan", "drift_replan"]:
        if name not in exp_data:
            continue
        e = exp_data[name]
        pc = e["overall"].get("pc_success", 0)
        total_rates[name] = pc
        row += f" │ {pc:>10.1f}%"

        if has_steps.get(name):
            all_eps = e["all_eps"]
            succ_steps = [ep["n_steps"] for ep in all_eps if ep["success"] and ep.get("n_steps")]
            fail_steps = [ep["n_steps"] for ep in all_eps if not ep["success"] and ep.get("n_steps")]
            s_avg = f"{sum(succ_steps)/len(succ_steps):.0f}" if succ_steps else "-"
            f_avg = f"{sum(fail_steps)/len(fail_steps):.0f}" if fail_steps else "-"
            row += f" {s_avg:>8} {f_avg:>8}"

    gain = total_rates.get("drift_replan", 0) - total_rates.get("no_replan", 0) if "drift_replan" in total_rates and "no_replan" in total_rates else 0
    if "drift_replan" in total_rates and "baseline" in total_rates:
        vs_bl = total_rates["drift_replan"] - total_rates["baseline"]
        row += f" │ {gain:>+9.1f}pp {vs_bl:>+10.1f}pp"
    else:
        row += f" │ {gain:>+9.1f}pp {'N/A':>11}"
    lines.append(row)
    lines.append("└" + "─" * 104)

    if not any_has_steps:
        lines.append("  注：现有 eval 数据中无 ep_steps 字段，需重新跑 eval 后才有 step 统计")

    lines.append(f"  ★ = drift replan 超越 baseline")
    lines.append("")

    # ── Replan 精确统计 ──────────────────────────────────────────
    rp = exp_data.get("drift_replan", {}).get("replan", {})
    if rp:
        lines.append("┌─ Drift Replan 精确统计 (threshold=0.10) " + "─" * 63)
        lines.append(f"│ 总 replan 事件: {rp['total_events']}  |  "
                     f"rollout: {rp['n_total_rollouts']} 个 (有replan: {rp['n_replan_rollouts']}, 无: {rp['n_no_replan']})")
        lines.append("│")

        # 精确推理次数分布
        lines.append("│ ▸ 精确推理次数/episode (模拟计算):")
        lines.append(f"│   {'次数':>4} {'rollout数':>9} {'占比':>7}")
        lines.append(f"│   {'─'*28}")
        for k in sorted(rp["infer_dist"].keys()):
            pct = rp["infer_dist"][k] / len(rp["all_infer"]) * 100
            bar = "█" * max(1, int(pct / 2))
            lines.append(f"│   {k:>4} {rp['infer_dist'][k]:>9} {pct:>6.1f}% {bar}")
        lines.append(f"│   平均: {rp['avg_infer']:.2f} 次/episode")
        lines.append("│")

        # Segment 分布
        lines.append("│ ▸ 触发 segment 分布:")
        lines.append(f"│   {'seg':>5} {'次数':>5} {'占比':>7}  含义")
        lines.append(f"│   {'─'*45}")
        seg_labels = {1: "chunk 首个10步", 2: "第20步", 3: "第30步", 4: "末尾10步"}
        for seg in sorted(rp["seg_dist"].keys()):
            cnt = rp["seg_dist"][seg]
            pct = cnt / rp["total_events"] * 100
            lines.append(f"│   seg={seg} {cnt:>5} {pct:>6.1f}%  {seg_labels.get(seg, '')}")
        lines.append("│")

        # 首次触发 chunk
        lines.append("│ ▸ 首次 replan chunk:")
        for c in sorted(rp["first_chunks"].keys()):
            cnt = rp["first_chunks"][c]
            pct = cnt / sum(rp["first_chunks"].values()) * 100
            lines.append(f"│   chunk={c}: {cnt}次 ({pct:.1f}%)")
        lines.append("│")

        # 级联
        if rp["cascade_chains"]:
            chain_lens = [len(c) for c in rp["cascade_chains"]]
            lines.append(f"│ ▸ seg=1 级联链: {len(rp['cascade_chains'])}条, "
                        f"长度 {min(chain_lens)}-{max(chain_lens)}, "
                        f"平均 {sum(chain_lens)/len(chain_lens):.1f}")
        lines.append("└" + "─" * 104)
        lines.append("")

    return "\n".join(lines)


# ── 主函数 ────────────────────────────────────────────────────────
def main():
    report = generate_report()
    print(report)

    # 追加到 eval_record.txt
    with open(RECORD_PATH, "a", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")
    print(f"\n[Done] 已追加到 {RECORD_PATH}")


if __name__ == "__main__":
    main()
