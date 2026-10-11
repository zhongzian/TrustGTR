# 5.4 GPU smoke 独立验收

验收时间：2026-10-06T11:18:37.393292+08:00。只读核验已有实验与 runner，未启动或重跑训练，未加载 checkpoint，未修改科学代码。

## 5.4.1 完成结论

3/3 独立 smoke SUCCESS；每项均首次尝试、退出码 0、1 epoch、3 次 optimizer step、2 个最终评估 batch。7 项 smoke_checks 每项全通过（共 21/21 项）；OOM 重试总计 0，日志无 OOM/Traceback。6 个冻结源文件 × 3 个 manifest 共 18/18 匹配；当前源码也与冻结哈希一致。

| 实验 | 开始→结束（上海时间，2026-10-06） | 总耗时 s | 峰值 MiB | GPU |
|---|---|---:|---:|---|
| `gtr__etth1__smoke_recheck_v8__t96_h96__seed2024__20261005_221654` | 11:06:30 → 11:06:42 | 11.65 | 444 | NVIDIA GeForce RTX 4080，GPU-c88b779d-77c7-92b9-e031-943bae990121 |
| `trustgtr_a8__etth1__smoke__t96_h96__seed2024__20261005_221654` | 11:06:45 → 11:06:51 | 5.73 | 628 | NVIDIA GeForce RTX 4080，GPU-c88b779d-77c7-92b9-e031-943bae990121 |
| `trustgtr_a8__solar__smoke__t96_h96__seed2024__20261005_221654` | 11:14:45 → 11:15:00 | 14.73 | 1516 | NVIDIA GeForce RTX 4090，GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb |

## 5.4.2 链路与资产证据

检查项：forward、backward、optimizer_step、checkpoint_save、checkpoint_load、evaluation、metrics_write。

实际实现证据：`exp/exp_repro.py:200` 的 forward hook；`:208` 的 optimizer post-step hook 同时检查非空 gradient 并增加 global_step；`:57` 与 `:73` 原子保存且读取验证 checkpoint；`:305` 保存完整 last.pt；`:313` 在官方训练循环成功返回后登记 checkpoint_load，而各 run 的 `instrumented_train_source.py` 确实执行 `load_state_dict(torch.load(best_model_path))`；`:326` 在调用官方 metric 后登记 evaluation。`scripts/run_experiment.py:435` 要求全部检查通过，并在 `:438` 落盘 metrics.json 后才写 SUCCESS。三个 manifest 中这些 runner/训练循环源码哈希均与当前核验版本一致。

每个 run 均保存独立 ID、resolved_config.yaml、command.sh、env.txt、git_state.txt、code.diff、code_manifest.json、train.log、events.jsonl、evaluation_assets.json、resource_usage.json、status.json、metrics.json、best.pt 和 last.pt。train.log 含 Epoch 1 / Steps 3 / Saving model / 最终 mse、mae；events 含完成事件和一致指标。

预算明确限制为 smoke.enabled=true、train_epochs=1、max_train_batches=3、max_eval_batches=2，ID 含 smoke；same_evaluation_protocol=false。ETTh1 batch=256 / lr=.001，Solar batch=64 / lr=.003；输入和预测长度均96、seed2024。smoke 指标仅证明链路运行，不用于正式收益结论。

- `gtr__etth1__smoke_recheck_v8__t96_h96__seed2024__20261005_221654`：7/7 checks；best.pt 2,538,689 bytes，last.pt 10,175,851 bytes；证据目录 `runs/gtr__etth1__smoke_recheck_v8__t96_h96__seed2024__20261005_221654/`。
- `trustgtr_a8__etth1__smoke__t96_h96__seed2024__20261005_221654`：7/7 checks；best.pt 2,543,212 bytes，last.pt 10,195,906 bytes；证据目录 `runs/trustgtr_a8__etth1__smoke__t96_h96__seed2024__20261005_221654/`。
- `trustgtr_a8__solar__smoke__t96_h96__seed2024__20261005_221654`：7/7 checks；best.pt 2,621,420 bytes，last.pt 10,508,930 bytes；证据目录 `runs/trustgtr_a8__solar__smoke__t96_h96__seed2024__20261005_221654/`。

## 5.4.3 功能范围与限制

all-off run 的 A/B/C 均关闭、memory_mode=independent；A8 两数据集的 A 开启、B/C 关闭、memory_mode=historical、router_mode=per_bank_evidence。历史记忆 events 标记 train-only / causal end<=query origin。A8 metrics 持久化了 prior/posterior/effective bank mass，posterior 随匹配质量变化，证明新路由实际进入训练与评估链路；这些统计累计覆盖 train/val/test，不代表最佳 checkpoint 的验证集单独表现。

资源快照含 index、UUID、名称、total/used/free，CUDA_VISIBLE_DEVICES 与实际 GPU UUID 一致，前后资源快照及退出码已落盘。快速轮询 process peak 可能漏过短峰值；报告 peak_gpu_memory_mib 使用 Torch reserved 与进程观测的较大值，Solar 为1516MiB。

本次验收范围仅5.4；5.5正式训练状态由队列和正式 run 另行报告，本验收不停止队列，也不宣称5.5完成。
