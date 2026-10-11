# v6 机制对照

全部 seed2024 / T96 H96 / 相同官方预算和数据资产，负变化代表误差降低。

| dataset | model | mse | mae | mse_relative_vs_baseline_pct | mae_relative_vs_baseline_pct | official_best_validation_loss | runtime_seconds | peak_gpu_memory_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| etth1 | TrustGTR_a6 | 0.365975 | 0.390658 | -0.980401 | -0.495756 | 0.670597 | 69.373156 | 634.000000 |
| etth1 | TrustGTR_ab6 | 0.367206 | 0.391181 | -0.647292 | -0.362444 | 0.672018 | 97.767203 | 640.000000 |
| solar | TrustGTR_ab6 | 0.183593 | 0.239349 | 1.336095 | 1.136707 | 0.143244 | 493.852518 | 1868.000000 |
| etth1 | TrustGTR_ab5_nocal | 0.366722 | 0.391125 | -0.778202 | -0.376836 | 0.669113 | 86.761140 | 814.000000 |
| etth1 | TrustGTR_ac6 | 0.366548 | 0.390962 | -0.825341 | -0.418328 | 0.669608 | 69.910335 | 658.000000 |

完整验证集复核见 v6_fixed_full_val_20261005.md；不与随机shuffle/drop_last的官方val loss混算。
A6/AC5测试排名不能证明两个模块可叠加；补AC7关闭2×2表。Solar B6仍输原始baseline，继续最小机制迭代。
新校准目标仅为待验证假设，保留旧版本和全部负结果。硬件并发不同，耗时不是严格速度基准。
