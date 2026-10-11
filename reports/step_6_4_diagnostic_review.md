# 6.4.5 诊断脚本独立复核

6.4.5.1【通过】本复核为只读源码审查及已存在 ETTh1 resolved config 一致性检查，未跑训练、未创建 test 数据、未修改科学文件或诊断脚本。审查时脚本 SHA256：`ad6e922cd922d701d76b2c930107f1c0b2882ba4ecafdac5aadc81b3a8c7444c`。其他数据集正式结果尚未全部落盘，不能把本审查冒充四数据集诊断已运行成功。

6.4.5.2【通过】GTR、HistoricalMemory、TrustGTR、data_loader、metrics 均核对各 run 的 code_manifest SHA；新模块优先归档副本，追踪原始文件则须当前文件哈希仍与归档一致。加载 TrustGTR 前绑定已核验的 GTR 与 Memory 类，结束后恢复 sys.modules；checkpoint 严格加载。

6.4.5.3【通过】数据类映射覆盖 ETTh1、Solar、Weather(custom)、ETTm2。仅构造 train/val；raw 数据 SHA、资产清单 SHA、data_x/y、整个 cycle_index 数组、窗口数、split 与 normalization 数值均校验。预测绝对起点 `split_start + index + seq_len` 与原数据相位生成规则一致。ETTh1 实际 resolved official_args.trustgtr 与顶层 trustgtr 一致，val split=[8544,11520]、cycle=24；其余类的运行时校验由正式诊断执行。

6.4.5.4【通过】模型置 eval 且整个计算位于 inference_mode。actual_dynamic 使用真正 `model(x,origin)`；decomposition 重建输入/历史专家及 gate，未调用辅助 teacher 或辅助 loss。重建沿用正式 forward 的乘法结合顺序，并逐批验证。

6.4.5.5【通过】g0 是同一联合训练 checkpoint 的原 GTR 分支；g1、constant 和 oracle 都是该 checkpoint 内的反事实。报告明确 g0 不等于独立训练 baseline；oracle 用 val 标签，只描述 MSE 可改善空间，不能用于部署或 test 结论。

6.4.5.6【通过】MSE/MAE 按元素总和流式聚合；第二遍重新读取窗口，实际计算 constant mixture 的 MSE/MAE，不以平均 MAE 或统计猜测替代。constant 是全窗口/通道的 dynamic gate 均值，属于 posthoc 对照，不是独立训练消融。长期仅保留 horizon-reduced 标量，未保留整段预测。

6.4.5.7【通过】g1_gain=当前分支MSE−固定g1MSE，正值表示检索校正有益；harmful/helpful 使用同定义。相关和五等分仅选 power>1e-6，与既有 utility_oracle meaningful 定义一致；五等分按门值从低到高，未按标签排序。标签仅用于事后效用/校准/oracle计算。全 val mean power 的校准权重口径已明确与训练 minibatch 不同。

6.4.5.8【无阻塞，边界说明】全窗口口径与官方 shuffle/drop_last early-stopping val 不同，不能混用排名；门值大量相同会形成任意稳定索引的等分，应结合相关、std 与 dynamic−constant 收益解读。Markdown 对无 helpful/harmful 样本的均值会显示 0，JSON 保留 None 与 counts；当前真实任务须查看 counts，空集合时不能把该 0 当测量值。完整机制结论仍等待实际诊断落盘。
