# 6.5.5 机制解释与实际限制

四个数据集、每组两个归档 best，共 8 份 checkpoint 的完整固定 validation 已完成。4 个 CPU 子命令与总监测命令均退出 0，源代码、实际加载的模型文件、配置、训练库、split/scaler/phase/数组、checkpoint 与脚本哈希均有记录。实际推理预测和 B 关闭时的 `joint GTR + strength * (retrieval - joint GTR)` 分解逐元素精确相同。没有 test、反向、optimizer、训练 collector、C auxiliary loss 或 checkpoint 写入。

| 数据集 | actual MSE/MAE extremes→hard | retrieval MSE/MAE extremes→hard | 观察 |
|---|---:|---:|---|
| ETTh1 | 0.669014482/0.537928510→0.669430365/0.538213560 | 0.794620363/0.599689858→0.796751794/0.605582618 | actual、joint GTR 分支及检索双指标均略差 |
| Solar | 0.144189565/0.223868143→0.140885151/0.211305619 | 0.167977054/0.225263500→0.168660810/0.223482979 | actual 与 joint GTR 分支双指标改善；检索单独 MSE 略差、MAE 改善 |
| Weather | 0.378460597/0.259059674→0.378746281/0.259536397 | 0.473829048/0.322354579→0.474581792/0.320494030 | actual 双指标略差；检索 MAE 改善没有传到融合后的 actual |
| ETTm2 | 0.118558637/0.234850426→0.118453443/0.234503640 | 0.172068298/0.283917494→0.160219092/0.271428639 | 检索与 actual 双指标改善，joint GTR 分支略差 |

1. 固定训练 64 窗口预诊断支持“实际困难负例”得到落实：旧 far 负例只有约 5.5%–12.1% 属于当前非零检索集合，新 sampling 有效正负对的负例均属于实际非零检索集合。后者由设计保证，不能写成模型训练后学出了 100% 的可靠性。全部已选对的 phase/causal/nonoverlap 违规为 0；两 sampling 在相同 A6 参数与 dropout RNG 下预测精确相同，辅助梯度只进入 key_encoder。
2. 完整 val 的 actual 双指标在 Solar、ETTm2 改善，ETTh1、Weather 略差。该表仅用于机制诊断；正式 test 结果与同预算归因应读取 `step_6_5_paired_results.*`，不可用这张完整 val 表替代官方指标或重选 checkpoint。
3. Solar 的 joint GTR 分支也明显改善，而检索单独 MSE 略差；证据只支持联合学习/融合变化与总预测改善共同出现，不支持“检索自身 MSE 改善导致收益”的因果叙述。ETTm2 的检索分支双指标改善，与总预测改善一致；仍是同一联合训练 checkpoint 的分解，没有重新训练各个分支来验证因果。
4. hard 的 nearest key 平方距离在四数据集都下降，但两个 checkpoint 的 key_encoder 参数不同，学得的尺度与几何也会改变。跨模型的绝对 key 距离不能直接当作检索准确性指标，也不能凭距离下降断言表征坍塌。固定 A6 参数下的预诊断距离对比才具有同一几何口径。
5. `retrieved_forecast_regime_distance` 比较完成预测后检索混合 forecast 和真实 val future 的 level/trend（共同用 query history scale、各维 clip ±5），不是训练 positive/negative regime 距离或 auxiliary CE。该距离在 ETTm2、Weather 下降，在 ETTh1、Solar 略升，不能只靠 regime 距离下降宣称预测变好。
6. 完整 val 使用全部固定窗口和 CPU；float32 残差提升 float64 后平方/求和。它与官方 shuffle/drop_last 的早停 val 聚合不同，且 CPU/GPU浮点实现也可能有细小差异。两采样的正式 checkpoint 始终依照原协议选择。
7. 两本轮采样共同使用 euclidean/T=3/λ=0.002、A6 per_bank/bounded、B 关闭；旧 AC7 使用 cosine/T=0.1，不能把相对旧 AC7 的变化全归因于采样。单 seed 支持筛方向，不能作为统计稳定性或所有数据集通用收益的证据。

证据：[完整汇总](step_6_5_retrieval_diagnostics.json)、[训练预诊断](step_6_5_preflight_train64_20261006.json)、[监测与 CPU 退出状态](step_6_5_diagnostic_monitor_state_20261006.json)。旧与最终诊断脚本分别以 SHA256 归档在 `reports/step_6_5_diagnostic_sources/`；仅修正报告口径与实际加载文件元数据，数学未变，未重复任何已成功的训练或诊断。
