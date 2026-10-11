# 6.5.1 C 实际检索困难负例：实现审计

状态：模型实现和 CPU 科学不变量检查完成；本报告不包含 GPU smoke 或正式指标。

唯一方法因素是 `contrast_sampling`：默认 `regime_extremes` 精确保留旧 C，新增 `retrieval_hard` 改变负例来源。两组共同使用欧氏对比 logits、温度 3、loss 权重 0.002；本实现不新增参数、不改温度、loss 权重、距离阈值或 CE。

1. 正例仍由原 `base_candidate_*` 池中满足同基周期 phase、因果、非重叠条件的未来 regime 最近候选给出。原 candidate cap、三相对年龄 bank、`positive_max_distance`、`regime_distance_gap` 均保留；“全部”指本次 capped 候选池，而非扫描全部历史。
2. 新负例仅从当前 forward 实际选中的检索槽位取样，实际合成权重须大于 0，key 距离有限，历史与查询需非重叠 (`query_origin - historical_origin >= T+H`)，不得与正例相同，未来 regime 距离须至少为正例距离加原 gap。在满足条件者中选当前 key 平方欧氏距离最小者；并列时按实际 router 槽位顺序，不重新运行 top-k。
3. collector 仅在 `use_regime_contrast=true`、`contrast_sampling=retrieval_hard` 且 training 时生效。`selected_candidate_positions` 相对 `base_candidate_*` 索引，masked 槽位为 -1；`selected_candidate_distances`、`selected_candidate_weights` 为此次真实检索值。三个 collector 字段均 detached，eval/C-off/旧 C 无采集。
4. Query 未来 target 只用于辅助 loss；历史 continuation 来自 caller 已核验的固定 train-only 记忆库。索引选择无梯度，最终两类 CE 从原 differentiable query/candidate embeddings 取 logits；无可用对时返回可微 0。不把未完成历史、重叠窗口、零权重槽位或全池最远 regime 当作真实困难负例。
5. `retrieval_hard` 与启用 C 的非 historical 模式不兼容时明确报错；模块关闭的非 historical 模式允许保留该闲置开关并恢复基线。没有恢复/覆盖原 GTR 文件。

执行：`.conda/bin/python scripts/test_retrieval_hard_contrast.py`，12/12 tests 通过；日志 `logs/step_6_5_retrieval_hard_cpu.log`。`py_compile` 通过。

核验内容：18 组旧归档双科学源（3 routers × 2 contrast metrics × B-off/B-train/B-inference）与默认新源的初始化 state、forward、含辅助 loss、逐参数 backward、RNG、diagnostics、eval 均逐项 `torch.equal`；all-off 原官方基线和 C-off 精确兼容；新/旧 sampling 的主预测与主 loss 梯度/RNG 精确兼容；三个 router 的 collector 距离、权重与真实 differentiable retrieval 精确一致；近邻负例成员/最小距离、同 phase/因果、严格非重叠、零权重、masked sentinel、ties、gap、正例阈值、显式空槽与空库可微零、欧氏及余弦 CE、仅 embeddings 梯度、target 无梯度、静态记忆 buffer 不变均通过。

旧归档：`runs/trustgtr_ab9_train__etth1__forecast__t96_h96__seed2024__20261006_122231/untracked_code/models/`。

| 文件 | SHA256 |
|---|---|
| 旧 TrustGTR.py | `59fef362c5e61905bcd4db9f866ef9f30454ab63f511c797c72eff28eb540970` |
| 旧 HistoricalMemory.py | `c05182901b603689b4852fa4b569d41f9ba4db938e3d3fcd462f5dfd3f90b6b9` |
| 新 models/TrustGTR.py | `de07b0dd728c20111592624eac9fda1200bafe44090a73a174ee7dc4b464311b` |
| 新 models/HistoricalMemory.py | `60d848f2b087f7448e33006ab6f72e0dd21b935a86a2c5b94f57f82f114583bd` |
| 新 scripts/test_retrieval_hard_contrast.py | `dd76f87c01f3879ed81c6c7c4de99dcc54bb0a709934fc5c960c5eaa4675c622` |

限制：真实 top-k 槽位更少，严格非重叠和未来 gap 会进一步减少可用 anchors。不能把较低 loss 或可用率当作预测收益；后续固定 train 探针需报告有效率与实际负例 key 距离，正式四数据集需同协议比较。当前多 bank 仍是同一基周期 phase 下的相对年龄 bank，独立多周期候选规则留在 6.6；本次没有缩减 A/B/C 研究范围。
