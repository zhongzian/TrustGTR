# 6.5.2 固定训练窗口：负例与梯度预诊断

四数据集各使用已有 A6 best 参数；每组固定 64 个覆盖完整训练集的窗口，包含早期无因果候选的窗口。两采样方式使用同一参数、同一训练输入及 dropout seed。CPU 2 threads、每 8 窗口调用 autograd.grad，未执行 optimizer、写入 checkpoint 或加载 test。总命令退出 0。

| 数据集 | 有效 anchor 旧 far→hard | 旧 far 负例实际检索覆盖 | hard 负例实际检索覆盖 | 负例 key 平方距离旧→hard | hard 加权辅助/main encoder 梯度 norm 微批均值 | hard/main encoder cosine 微批均值 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 92.86%→92.86% | 10.34% | 100% | 5.359→0.602 | 1.025 | −0.114 |
| Solar | 96.05%→90.77% | 5.52% | 100% | 6.118→0.224 | 0.197 | +0.092 |
| Weather | 92.41%→91.67% | 10.23% | 100% | 16.256→0.740 | 0.412 | +0.090 |
| ETTm2 | 95.54%→94.87% | 12.15% | 100% | 10.704→0.939 | 0.909 | +0.210 |

1. 本实现确实把负例从全候选中后续 regime 最远的样本，移到本次有非零真实检索权重的困难近邻。hard 负例 key 距离更小属于设计得到落实的证据，不等于预测改善。
2. 两方式的全部已选正负例同相位、历史未来已完成、完整窗口不重叠；违规计数均为 0。辅助 loss 对非 key_encoder 参数的梯度严格为 0；独立重建的 CE 与实际实现一致，collector 的 positions/weights 与真实检索精确一致。
3. 两采样方式在同参数、同 dropout RNG 下预测逐元素完全相同。监督只在训练辅助 loss 中读取 future target，不进入预测。
4. ETTh1 的加权辅助 encoder 梯度与主梯度规模相近且略冲突，Solar 有效监督覆盖下降，保留为风险证据。四组仍有约 91%–95% 可用 anchor，当前不据此私改 λ、gap、positive threshold 或预算；需要正式同协议配对结果判断。
5. 上述 norm/cosine 是每 8 窗口微批统计，且处于已训练 A6 参数上，不是官方训练 batch 或新 C 训练过程的真实梯度。新模型 strength=0 时主预测对 key_encoder 的梯度为 0，故使用 A6 已训练参数作预诊断；不将该预诊断作为正式模型成绩。
6. 两组共同使用 euclidean/T=3/λ=0.002。这与旧 AC7 的 cosine/T=0.1 有区别，不能将相对 AC7 的变化全部归因于采样。6.5 的采样归因只比较本轮新 control 与 hard 配对。

明细与哈希：[JSON](step_6_5_preflight_train64_20261006.json)，执行日志位于 `logs/step_6_5_preflight_train64_20261006.log`。下一步保持冻结配置，先 smoke 后正式四数据集配对；正式 best 的完整固定 val 使用归档源码与训练库做只读分解，不能用 test 重选 checkpoint。
