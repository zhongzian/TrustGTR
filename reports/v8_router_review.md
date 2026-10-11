# 5.2 A8 候选感知路由审查

审查日期：2026-10-06。结论：**未发现阻塞问题，可以进入 5.4 的 GPU smoke；本审查不证明预测收益。**

1. **公式与实际预测路径通过。** `models/HistoricalMemory.py:243` 对每库有限 top-k 平方距离使用 `v=1/(d+0.001)`、`evidence=sum(v)/finite_count`，再计算 `posterior ∝ masked query prior × evidence`。共同除以最大 evidence 仅用于数值稳定，不改变归一化后的公式。每个候选权重是库内逆距归一化乘 posterior；该权重在第 333 行直接混合历史 continuation，第 262–291 行的 `TrustGTR.forward_historical` 再将其写入最终 forecast 修正，posterior 不是只用于日志。
2. **边界通过。** `isfinite(distance)` 控制有效计数，避免把 `1/inf=0` 当作有效样本；NaN/inf 距离权重为零，零距由原有 0.001 保护。部分空库 posterior 为零，全空查询返回观察窗口均值，TrustGTR 的 `valid` 掩码使其最终不修正 GTR。训练记忆的来源、因果界与候选选择规则未改变。
3. **梯度、参数及保存加载通过。** 已落盘 6 项 evidence 路由测试验证非零 router/key-encoder 梯度、有限梯度、相同参数集合与数量、完整模型 forward/backward/optimizer step、state_dict 保存加载的一致性。没有新增可训练参数。原有零初始化 strength 会让第一步 forecast 先与基线相同，随后 strength 非零才将 forecast 梯度传入记忆路由，这是既有行为。
4. **旧模式精确回归通过。** 测试核验归档 A6 源文件 SHA256 `3d8f84247121808d30a1690b7f835b4519424e4cea2ae50703a1e54ba546d6c0`，对 `global_topk` / `per_bank` 各自 reliability 开关的四种组合比较 metadata、状态字段、预测张量、所有参数梯度、诊断和 RNG，均精确一致。另有原始基线回归 5 项与历史集成回归 3 项通过。证据为 `logs/v8_evidence_router_checks.log`、`logs/v8_baseline_regression.log`、`logs/v8_historical_integration.log`；尚不能将 CPU 回归替代实际 GPU smoke。
5. **统计语义通过。** `cycle_weights` 保持 query prior 原意；新模式另存 prior、posterior、最终按库累计的 effective weight，后三者避免混淆。runner 只在新模式导出新统计，并明确标注累计 train/validation/test 调用，不能误当作最佳 checkpoint 的纯验证统计。两组旧模式未增加该 metrics 字段。
6. **机制限制须保留。** 本改动利用近邻距离的绝对大小，但距离近不保证历史未来有用；零距附近可能让某库占优。当每库有效 top-k 数量相同时，最终候选权重等价于对各库 top-k 并集做带 query prior 的逆距归一化；这是 A 的改进，不应另算独立创新。新增诊断中每次 `.tolist()` 会同步 GPU，实际成本由正式日志记录；有效性与路由改善必须在 5.5–5.6 核验。

仅新建本审查文档；没有修改模型、runner、配置，没有启动 GPU 或重复训练。
