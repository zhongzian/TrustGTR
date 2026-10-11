# 6.5.2 / 6.5.5 C 诊断独立只读审查

结论：未发现阻塞正式实验或后续 best-checkpoint 完整验证诊断的问题。本次只读审查脚本及已有 preflight 输出，没有重新跑模型、测试、训练或修改科学源码。完整验证集诊断尚未在本审查中执行，以下验证路径结论为静态代码核验。

审查对象：`scripts/diagnose_step65_contrast.py`、`scripts/diagnose_step64_gate.py` 的归档/数据/聚合辅助函数、`reports/step_6_5_preflight_train64_20261006.json/.md`，以及冻结的 `TrustGTR.py` / `HistoricalMemory.py`。

1. 资产和来源：四数据集 raw 文件、train manifest、split、window_count、data_x/data_y、cycle_index、train scaler 的 mean/var/scale 均核验一致。静态记忆从核验后的 train.data_x 重建并匹配保存的 memory SHA。归档五个科学来源先按 code_manifest 核验；preflight 明确改为当前 6.5 模型读取 A6 best 权重、记录原配置 SHA 和当前三个模型 SHA，并标记 diagnostic-only。正式 `--runs` 路径使用归档模型严格加载 best.pt，不按当前未归档源偷偷推理。
2. 采样：独立重建 A6 per_bank 的 top-k 槽位及实际权重，检查新 collector 的 positions 完全一致、weights 最大误差为 0；槽位并列保持实际 top-k 的次序。负例需 selected 且权重 > 0、同 phase、因果、严格非重叠、不同于正例、满足原 future-regime gap；按其实际 key 距离取最近 eligible 者。独立参考索引/距离和模型 helper 匹配，独立欧氏两类 CE 与实际辅助 loss 匹配。距离 collector 的逐值正确性还由 6.5.1 新 CPU 单元检查覆盖。
3. 标签：train target 仅在诊断独立 reference 和训练辅助 loss 使用；模型 forward 只接收观察 x/origin。训练探针不新增记忆、不使用 val/test 标签、不插入 optimizer。两采样组读取同一 A6 checkpoint、同一64窗口索引、同一微批 dropout seed，主预测逐值相同。
4. 梯度：每8窗口计算主 MSE 和 `0.002 * auxiliary CE` 的 `autograd.grad`，没有 optimizer step 或参数 `.grad` 累积。nonencoder auxiliary 最大绝对梯度在全部4数据集×2采样×8微批均为0；可微索引选择不会训练 router/strength/GTR 等参数，最终 CE 仍训练真实 query/candidate 的 key encoder。报告 norm/cosine 是八个微批对应统计值的分布及均值，不能解释成官方训练 batch 或全训练集累计梯度比值/夹角。
5. 完整 val 路径：只构建固定 train/val，`model.eval()` + `torch.inference_mode()`；没有 test loader、auxiliary_losses、optimizer、backward、参数更新或 checkpoint 写入。重建 `local + strength*((retrieved-local)*valid)` 与实际 B-off 输出逐值比较，eval 不启用新 C collector。target 的误差和 forecast-regime 差异均在预测完成后统计，未用于采样、检索、模型输入或重新挑 best checkpoint。
6. 分支口径：`same_checkpoint_gtr_branch` 是该联合训练 checkpoint 内的 GTR 参数分支（等价去除检索校正），不能当作独立训练的官方 GTR baseline。`retrieval_alone` 同样是事后分解，不能冒充重训消融。full-val 使用固定全部窗口、无 shuffle/drop_last，故与官方早停验证日志的聚合范围不同；两组同样处理。

preflight 的分母是64窗口×数据集通道数；actual-retrieval membership 的分母是该采样版本的有效 channel-window 对。

| 数据集 | 旧 C 有效对 / channel-window | 新 C 有效对 / channel-window | 旧负例实际检索 membership | 新负例实际检索 membership |
|---|---:|---:|---:|---:|
| ETTh1 | 416 / 448 | 416 / 448 | 43 / 416 | 416 / 416 |
| Solar | 8422 / 8768 | 7959 / 8768 | 465 / 8422 | 7959 / 7959 |
| Weather | 1242 / 1344 | 1232 / 1344 | 127 / 1242 | 1232 / 1232 |
| ETTm2 | 428 / 448 | 425 / 448 | 52 / 428 | 425 / 425 |

新版本100% membership由采样定义保证，也在这些固定探针对中得到验证；这不是预测收益，不是全部训练窗口的监督有效率。部分数据新版本有效对数下降属于限制；两组正例规则仍完全一致，有效对筛选变化会使报告中的正例 membership 计数改变。

两个新的采样对照均为 euclidean/T=3/λ=0.002。旧 AC7 的 cosine/T=0.1 与这两组不同，直接比较旧 AC7 的收益不能只归因采样。当前 preflight 使用已训练 A6 权重，不代表新的正式 C 训练结果。

报告和当前脚本 SHA 一致：diagnostic `47db22561ff89c0ef726b1607900d84df57faa7bc76f1f5bed91d566bac124f0`；archive helper `6d80c65ccc15462a55ad06b980e0a8a567968cc03e534d8d23cdf1c5723dc1c6`。当前科学 SHA：TrustGTR `de07b0dd728c20111592624eac9fda1200bafe44090a73a174ee7dc4b464311b`；HistoricalMemory `60d848f2b087f7448e33006ab6f72e0dd21b935a86a2c5b94f57f82f114583bd`。

非阻塞文字精确化建议：聚合 helper 的实现是 float32 预测残差先提升到 float64，再平方和求和；当前 `float32 errors then float64 sums` 可进一步明确这一点。八微批结果与100% membership应保留上述范围限定。最终完整 val 实际运行后的重建误差、固定资产、总窗口数及 checkpoint hash 仍需复核。
