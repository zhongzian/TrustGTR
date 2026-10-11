# 6.7 新数据集首次公平对照：准备验收

1. **范围与状态**：新增 Weather、ETTm2；原 ETTh1、Solar 结果保留。首次仅比较原 GTR 与历史检索 A6（per_bank），每个数据集各一项 smoke 和一项正式训练，共 8 项任务。仅准备配置；本报告生成时未启动 GPU。
2. **方法选择**：A8 在 Solar 已明显退化，因此不将 A8 作为通用默认。A6 使用已有成功实验的原配置，不依据新数据 test 结果选版本。B/C 在本轮归因对照中关闭，完整研究范围仍保留；不扩多 seed 或外部模型。
3. **官方协议**：输入/预测 96/96，seed=2024，最大 30 epoch、patience=5，完整训练/验证/测试数据，MSE 训练与原 MAE/MSE 评估、相同早停检查点选择。每项 smoke 仅改为 1 epoch、3 个 train batch/2 个评估 batch，batch 和模型不缩水。

| 数据集 | 官方 data | 通道 | cycle | batch | LR | dropout | RevIN |
|---|---|---:|---:|---:|---:|---:|---:|
| Weather | custom | 21 | 144 | 64 | 0.001 | 0.5 | 1 |
| ETTm2 | ETTm2 | 7 | 96 | 256 | 0.001 | 0.5 | 1 |

4. **继承默认值**：已从原 `run.py` AST 核验 d_model=512、label_len=0、model_type=mlp、loss=mse、lradj=type3、pct_start=0.3、num_workers=10、use_amp=false、use_multi_gpu=false。官方脚本未覆盖的 freq=h 保持原值；GTR 不消费时间标记。运行入口会保存完整解析后的 `resolved_config.yaml`。
5. **资产与划分**：asset manifest 为 `artifacts/expanded_data_assets.json`。Weather SHA256=`34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63`，52696 行；沿用官方 custom 的 floor(70%) train、floor(20%) test、remainder val。ETTm2 SHA256=`db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd`，69680 行；沿用官方前 57600 行的 12/4/4 月划分，其余行不参与训练。Weather 原数据中的一个重复时间和一个间隔异常保留，不去重、不重采样。
6. **同协议检查**：各数据集正式 GTR 与 A6 的 `official_args` 只差 model/model_id；资产、训练预算、指标与 loader 相同。A6 新增选项与旧 ETTh1 A6 原配置完全一致，基础周期由各自官方 cycle 决定。当前历史库是同基础相位加三档年龄，不声称已完成三个独立周期取模。
7. **任务依赖与资源**：GTR 正式依赖自己的 smoke；A6 smoke 依赖该数据 GTR smoke，A6 正式依赖自己的 smoke 加该数据 GTR 正式成功。最大三并发、后台非阻塞轮询、GPU UUID/锁和 OOM 退避回队列复用现有调度器。首次 GTR/A6 smoke 显存估计均未知；不将无历史 GTR 峰值冒充历史库峰值，首次未知任务按调度器保守模式避免同 GPU 冲突。发生 OOM 保持配置回队列，正式使用自己的成功 smoke 峰值估计。队列启动前移除了 A6 smoke 借用 baseline smoke 的调度估峰来源，科学配置和 ID 均未改变，修改已记录 JSON 审计。
8. **可恢复与不可变**：新队列仅有 8 项新 ID，不携带任何旧 job，也不重复旧实验。每项保留 source/protocol/hash/change ledger；不预先猜 baseline 指标。准备脚本再次执行会核对现有队列和资产后复用，不覆盖配置或生成重复 ID。科学代码哈希冻结在 queue preparation 和 JSON 审计中。
9. **已执行验收**：准备脚本编译/CLI、调度器 `load_config` 的 DAG/ID/路径校验、全部配置与 asset audit 的官方默认值逐项匹配、正式无 smoke 限制、A6 开关检查、未产生 runs 目录、重复准备不变，均通过，退出码 0。原论文新增数据集参考值未独立核验，MSE/MAE 保持 NA。
10. **下一步**：父任务完成 5.6 判断、确认新 runner 分割元数据和显存后，用现有总脚本启动 `configs/queue_four_datasets_v1.json`；各 smoke 成功才允许对应正式任务进入队列。配置 JSON 与 source hash 的完整证据在 `reports/four_dataset_protocol_audit.json`。

准备命令（已执行，退出 0）：

```sh
.conda/bin/python scripts/prepare_four_dataset_queue.py --assets artifacts/expanded_data_assets.json
```

启动命令（本子任务未执行）：

```sh
bash scripts/queue.sh start configs/queue_four_datasets_v1.json
```
