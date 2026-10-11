# v7 组合归因与风险校准结果

seed2024，T96/H96，相同官方训练预算与固定资产；变化负数代表误差下降，所有任务成功。

| dataset | model | mse | mae | mse_relative_vs_baseline_pct | mae_relative_vs_baseline_pct | official_best_validation_loss | runtime_seconds | peak_gpu_memory_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| solar | TrustGTR_a6 | 0.185902 | 0.241035 | 2.610346 | 1.848988 | 0.142306 | 894.993980 | 1830.000000 |
| etth1 | TrustGTR_ac7 | 0.365637 | 0.390612 | -1.071695 | -0.507400 | 0.670129 | 53.517357 | 844.000000 |
| etth1 | TrustGTR_ab7 | 0.366594 | 0.391041 | -0.812980 | -0.398167 | 0.669007 | 111.980733 | 640.000000 |
| solar | TrustGTR_ab7 | 0.186810 | 0.239268 | 3.111956 | 1.102372 | 0.144779 | 1021.158233 | 2042.000000 |

ETTh1 A/C 的 2×2：

| router | C | MSE | MAE |
| --- | --- | --- | --- |
| global_topk | False | 0.366726667 | 0.391115844 |
| global_topk | True | 0.365943819 | 0.390748143 |
| per_bank | False | 0.365974873 | 0.390658081 |
| per_bank | True | 0.365637451 | 0.390612364 |

C 在 per_bank 上额外降低 test MSE0.000337422；对 fullval 也改善 A6，但整体 AC7 的 fullval 仍差于 global A5。两因子没有呈现超加性收益。
风险目标 ETT 门值均值0.9964；Solar几乎1。它改善部分指标，但没有获得可靠拒绝功能，不能按“安全门控已成功”定型。
Solar旧AB6在同checkpoint固定256val上优于g=mean/g=1/g=0，存在有限排序收益，仍未超过原始baseline正式test。
当前不扩多seed；完整source/metrics/hash/成本见JSON。原GTR保留，所有负结果在latest_results总表。
时间受GPU并发影响，本轮没有严格速度对照。
