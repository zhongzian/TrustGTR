# 6.5 配对实验协议审计

准备时间：2026-10-06T21:46:46.788506+08:00。仅生成配置；未启动 GPU。

6.5.2：四数据集各一组 AC10_extremes / AC10_hard，A 固定已完成 A6 的 per_bank/bounded 历史检索，B 关闭以归因。
两组共同开启 C：euclidean、temperature=3、weight=.002、positive_max_distance=1、regime_distance_gap=.25。唯一科学差异为 contrast_sampling: regime_extremes → retrieval_hard。
此处为新建相同欧氏距离/温度的 control；不将历史 cosine/T=.1 的 AC7 冒充本轮 control。
9项 smoke（含全模块关闭回归）+8项正式；每项正式先依赖自身 smoke。hard 正式另依赖相应 extremes 正式成功。
正式沿用原 batch、30 epoch上限、patience5、seed2024、输入/预测96、原优化/评估协议。所有 TrustGTR 默认值均显式解析进入输入配置；启动 runner 再保存最终 resolved_config。
已完成 GTR/A6 只引用，不重新训练。固定数据、划分、scaler、代码、源配置和成功指标均校验哈希。

| 数据集 | batch | LR | head dropout | 周期 | extremes control | hard |
|---|---:|---:|---:|---:|---|---|
| etth1 | 256 | 0.001 | 0.5 | 24 | trustgtr_ac10_extremes__etth1__forecast__t96_h96__seed2024__20261006_214646 | trustgtr_ac10_hard__etth1__forecast__t96_h96__seed2024__20261006_214646 |
| solar | 64 | 0.003 | 0.0 | 144 | trustgtr_ac10_extremes__solar__forecast__t96_h96__seed2024__20261006_214646 | trustgtr_ac10_hard__solar__forecast__t96_h96__seed2024__20261006_214646 |
| weather | 64 | 0.001 | 0.5 | 144 | trustgtr_ac10_extremes__weather__forecast__t96_h96__seed2024__20261006_214646 | trustgtr_ac10_hard__weather__forecast__t96_h96__seed2024__20261006_214646 |
| ettm2 | 256 | 0.001 | 0.5 | 96 | trustgtr_ac10_extremes__ettm2__forecast__t96_h96__seed2024__20261006_214646 | trustgtr_ac10_hard__ettm2__forecast__t96_h96__seed2024__20261006_214646 |

启动命令（本脚本不执行）：

`bash scripts/queue.sh start configs/queue_step_6_5_ac10.json`

状态：`bash scripts/queue.sh status configs/queue_step_6_5_ac10.json`；日志：`tail -f runtime/step_6_5_ac10/queue.log`；安全停止/续跑使用同脚本 stop/resume。

显存：未知 smoke 单独估峰，正式使用各自 smoke。UUID、短时锁、动态补位、最多三并发、OOM退避回队列保持现有调度器；不改变任何正式参数。
