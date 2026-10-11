# 6.5 相同已完成 epoch 范围的验证日志诊断

仅提取官方 train.log 的 Vali Loss；未提取逐epoch Test Loss。每组共同范围为两边实际完成epoch的较小值，比较该范围内最低验证损失与日志最优epoch。负值表示下降。

| 数据集 | 共同epoch | extremes实际epoch | hard实际epoch | extremes最低Vali@epoch | hard最低Vali@epoch | 变化% |
|---|---:|---:|---:|---:|---:|---:|
| etth1 | 14 | 14 | 14 | 0.6695381 @9 | 0.6699216 @9 | 0.0573 |
| solar | 8 | 8 | 14 | 0.1441822 @3 | 0.1439436 @6 | -0.1655 |
| weather | 18 | 18 | 18 | 0.3764901 @13 | 0.3767537 @13 | 0.0700 |
| ettm2 | 8 | 8 | 8 | 0.1184711 @3 | 0.1183677 @3 | -0.0873 |

Solar共同前8epochs，hard最低Vali比extremes低0.1655%，说明共同范围内已经有很小的验证收益。hard全程最低Vali 0.1414100在epoch9，进一步收益在共同范围之外；正式结果同时伴随实际训练epoch增加，不能由这张表分解正式test收益的全部来源。

这是原shuffle=true/drop_last=true验证损失日志诊断，受batch/归一化/取样及日志四舍五入影响。未重新评估任何checkpoint，不是正式test结果，不能代替best checkpoint固定完整val诊断。
未修改epoch、patience、checkpoint选择、模型、配置或训练预算；未创建/训练额外实验。
