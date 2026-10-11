# 6.5 最终独立复核

6.5.5【完成】仅只读核对既有结论、五份 JSON 证据、八份正式 resolved config/metrics 及结果总表；未启动训练、GPU 或重复大型测试。本次仅新增本报告。复核范围内未发现 `step_6_5_conclusions.md` 的事实错误。

6.5.4【正式配对通过】八份原始 metrics 的数值及 SHA256 与 `step_6_5_paired_results.json` 一致。负值表示误差下降；对照为本轮新运行的 euclidean/T3/λ=.002 `regime_extremes`，旧 AC7 cosine/T.1 不能隔离采样影响。

| 数据集 | hard 相对 fresh extremes 的 MSE | MAE | 判断 |
|---|---:|---:|---|
| ETTh1 | +0.105651% | +0.071257% | 双指标微退化 |
| Solar | −2.988904% | −5.947492% | 双指标改善 |
| Weather | +0.153539% | +0.131416% | 双指标微退化 |
| ETTm2 | −0.016373% | −0.187189% | 双指标小幅改善 |

Solar hard 为 0.183669269/0.240148470，独立原 GTR 为 0.181172356/0.236659229，仍分别差 +1.378198%/+1.474374%。结论没有将其写成超过原 GTR。两采样组参数量相同，均较 A6 增加 0；共同 A 路径较原 GTR 增加 560，不能把整个模型称为对原 GTR 零新增参数。

6.5.5【预算与机制边界通过】四组正式配置同 seed2024、T96/H96、max30epochs/patience5，除独立 ID/保存路径和采样开关外，训练设置保持一致。Solar 实际 extremes 8epoch/best3、hard 14epoch/best9，最大预算一致不代表实际步数/计算量相同。共同前8epoch 日志最优验证损失 hard .1439436@6、extremes .1441822@3（−.165485%）；这是原日志的辅助对照，未重选 checkpoint、未重新计算 test，不能单独确定全部最终收益的原因。共享 GPU 的耗时也不能据此作严格速度排名。

完整固定 val 覆盖 ETTh1 2785、Solar 5161、Weather 5175、ETTm2 11425 个窗口，八个 best checkpoint 的实际预测分解重建误差均为 0；四组 CPU 诊断 exit0。`same_checkpoint_gtr_branch` 是联合训练 checkpoint 内的 GTR 分支，`retrieval_alone` 也是同 checkpoint 的事后分解，两者都不是独立训练基线或独立消融。完整 val 聚合与官方 shuffle/drop_last 早停值不同，不替代正式 test 指标。

Solar 的混合预测与联合 GTR 分支改善，但 retrieval-alone MSE .167977054→.168660810、regime 距离 .530803→.534795 反而变差；不能称检索 MSE 改善已解释其收益。ETTm2 的 retrieval-alone 双指标与混合预测改善提供局部机制支持，仍非唯一因果证明。四组 learned key 绝对距离均下降，但不同 checkpoint 的 embedding 尺度可变，不能直接据此比较检索质量；共享 A6 参数的 preflight 只检验困难负例选择确实落实。

6.5.5【运行与保留记录通过】`step_6_5_runtime_audit.json` 的 17 jobs/17 attempts 恰为 9 smoke+8 formal，全部 SUCCESS/exit0，63 smoke 链路检查通过、0 OOM、最大观察并发3、violations为空；宿主机清理审计无自任务残留/自GPU预留。结果总表132条为124 SUCCESS与8条旧 CODE_ERROR，旧失败仍保留，不能把累计总表称全成功。

`step_6_5_change_summary.json` 记录相对6.4仅两项科学源码差异：TrustGTR +51/−3、HistoricalMemory +21/−3；原 GTR 保持。空 tracked diff 不等于新增未跟踪代码没有改动。

6.5【收尾判断】单 seed 证据支持保留 hard 为可关闭的 C 候选，尚不足以设为通用默认或主张统计稳定提升。6.6【尚未启动】；A/B/C 完整范围继续保留。

证据：`step_6_5_conclusions.md`、`step_6_5_paired_results.json`、`step_6_5_runtime_audit.json`、`step_6_5_retrieval_diagnostics.json`、`step_6_5_equal_epoch_validation.json`、`step_6_5_change_summary.json`、`latest_results.csv`。
