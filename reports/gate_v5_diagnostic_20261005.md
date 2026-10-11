# TrustGTR v5 门控与对比目标诊断

诊断范围：ETTh1 四个 best checkpoint，固定完整 validation 的 2785 个窗口；只在 CPU 推理。AC5 同样纳入。未构造 test dataset，未执行 optimizer step，未修改模型、正式配置或 checkpoint。数值源为 [gate_v5_diagnostic_20261005.json](gate_v5_diagnostic_20261005.json)，复现入口为 `scripts/diagnose_gate_v5.py`。

这里使用固定完整 validation，故不能与训练过程 shuffled/drop_last 的 validation 均值混用。强制门控是同一个联合训练 checkpoint 的事后分解，不能当作重新训练的开关消融。Oracle 使用 validation target，只作误差上限分析，不能部署，也不能列为正式性能。

## B 的直接证据

| checkpoint | 实际 MSE | 强制 g=1 MSE | 强制 g=0 MSE | 固定 g=动态均值 MSE | target-aware oracle MSE |
|---|---:|---:|---:|---:|---:|
| A5 | 0.668787 | 0.668787 | 0.673977 | 0.668787 | 0.662521 |
| AB5 | 0.671347 | 0.669765 | 0.674917 | 0.671038 | 0.663451 |
| AC5 | 0.669430 | 0.669430 | 0.673843 | 0.669430 | 0.663308 |
| ABC5 | 0.671706 | 0.670389 | 0.674851 | 0.671537 | 0.664362 |

- AB5/ABC5 的 gate 均值为 0.7034/0.6890，实际平均校正系数约 0.0492/0.0472。A5 系数为 0.0703。当前 B 明显降低整体校正力度。
- AB5 gate 与 oracle 的相关为 −0.0169，与固定 g=1 的 MSE 收益相关为 −0.0934。其有益样本 gate 均值 0.7023，低于有害样本 0.7049；当前样本排序没有带来 MSE 收益。ABC5 也出现同方向问题。
- ABC5 oracle 有 38.67% 为 0、55.93% 为 1、仅 5.40% 在内点。回归一个近似二元、噪声较大的标签，容易学出中间值而整体削弱校正。
- ABC5 固定 train256 上，gate 校准辅助梯度为 forecast 梯度的 0.678 倍，二者方向 cosine 为 −0.328。先做 `gate_loss_weight: 0.05 → 0` 的独立训练对照，可以区分校准目标冲突与 gate 描述不足。

**可靠性描述的限制：**ABC5 的检索距离与 oracle 相关仅 0.0407，neighbor continuation dispersion 与 oracle 相关仅 0.0145。不能把“距离大/分歧大”直接当作有害的单调规则；高分歧组的平均校正收益反而更大。若加入真实检索距离、权重熵、历史 continuation 分歧、选中库覆盖与年龄，应作为可学习联合特征，并保持标签只出现在训练 loss 中。JSON 保留逐 channel 指标与距离/门控/分歧分组结果。

## C 的训练数据尺度校准

固定同一 train256、同一 dropout seed、无 optimizer step；梯度只统计真实检索 key encoder，λ 始终为 0.002。

| 监督 metric / temperature | AC5 加权辅助/main 梯度比 | AC5 梯度 cosine | ABC5 比值 | ABC5 cosine |
|---|---:|---:|---:|---:|
| cosine / 0.1 | 0.831 | −0.018 | 1.279 | 0.089 |
| Euclidean / 0.1 | 21.191 | −0.166 | 33.051 | −0.102 |
| Euclidean / 0.3 | 6.769 | −0.163 | 10.413 | −0.102 |
| Euclidean / 1.0 | 1.488 | −0.148 | 2.245 | −0.096 |
| Euclidean / 3.0 | 0.292 | −0.028 | 0.522 | 0.014 |
| Euclidean / 10.0 | 0.130 | 0.110 | 0.233 | 0.100 |

当前实际检索用平方 Euclidean，C 用 cosine。ABC5 的有效 train query 上两者最近候选一致率为 78.80%，top3 重合 81.51%；监督确实没有严格对齐实际检索排名。有效对比 anchors 为 1645/1792。

建议第一项 C 对照明确记录 `cosine, temperature=0.1 → Euclidean, temperature=3.0`，λ 仍为 0.002，训练预算与 A/B 固定。温度缩放保留实际 Euclidean 排名，同时缓和 logit 饱和；纯降低 λ 只能整体缩放既有梯度，不能改变饱和所产生的样本权重。不要归一化 embedding 来冒充 Euclidean 对齐，归一化后排名将等价于 cosine。

这些尺度来自已训练的 best checkpoint，只用于选择最小验证方向；不能预先证明重新训练一定有效。B 校准关停、B 新特征、C 度量和路由应分别控制一个因素，通过 smoke 后才进入同预算正式队列。

## 复现命令

```sh
cd /mnt/data/zza/code/TrustGTR
.conda/bin/python scripts/diagnose_gate_v5.py --runs trustgtr_a5__etth1__forecast__t96_h96__seed2024__20261005_110620 trustgtr_ab5__etth1__forecast__t96_h96__seed2024__20261005_110620 trustgtr_ac5__etth1__forecast__t96_h96__seed2024__20261005_155314 trustgtr_abc5__etth1__forecast__t96_h96__seed2024__20261005_110620 --sample-count 0 --output reports/gate_v5_diagnostic_new_run.json
```

入口只支持冻结 v5 的 `global_topk` 和原 forecast gate 结构；新 retrieval gate 需要单独的后续诊断入口。重跑使用新 output 名称，保留原始报告。
