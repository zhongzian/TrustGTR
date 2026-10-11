# v8 固定验证窗口路由诊断

仅验证集、归档科学源码与 best checkpoint；没有 test、反向或优化器更新。

| dataset | router | MSE | MAE | windows | posterior−prior mean abs |
|---|---|---:|---:|---:|---:|
| etth1 | per_bank | 0.401185393 | 0.421809971 | 4 | 0.000000 |

同数据集 indices 与资产逐项核对；此诊断的窗口聚合口径不同于官方早停验证损失，不能混为同一数值。
匹配距离接近不保证未来更有用；先看同协议指标，再决定是否采用候选证据路由。
