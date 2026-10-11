# 6.4 配对实验协议审计

准备时间：2026-10-06T12:22:31.733738+08:00。仅生成配置；未启动 GPU。

6.4.1：四数据集各一组 AB9_train / AB9_inference，A 固定已完成 A6 的 per_bank 历史检索，C 关闭以归因。
6.4.2：两组共同开启 B；retrieval 描述、detach=true、power_weighted、权重0.05、初始门值0.9完全一致。唯一科学变化是 gate_calibration_context: train → inference。
6.4.3：9项 smoke（含一项先行 TrustGTR 全关闭回归）+8项正式；每项正式先通过自身 smoke。每个 inference 正式还依赖相应 train 正式成功。
6.4.4：正式配置沿用各数据集原 batch、30 epoch上限、patience5、seed2024、输入/预测96、原优化与评估协议。新增确定性校准前向会增加计算，须单独比较耗时与峰值。
6.4.5：已完成 GTR/A6 仅作为落盘引用，不进入新队列；Weather/ETTm2 的论文指标保持 NA。源配置、指标、代码清单、数据、固定划分/归一化资产均记录哈希。

| 数据集 | batch | LR | dropout | 周期 | control | inference |
|---|---:|---:|---:|---:|---|---|
| etth1 | 256 | 0.001 | 0.5 | 24 | trustgtr_ab9_train__etth1__forecast__t96_h96__seed2024__20261006_122231 | trustgtr_ab9_inference__etth1__forecast__t96_h96__seed2024__20261006_122231 |
| solar | 64 | 0.003 | 0.0 | 144 | trustgtr_ab9_train__solar__forecast__t96_h96__seed2024__20261006_122231 | trustgtr_ab9_inference__solar__forecast__t96_h96__seed2024__20261006_122231 |
| weather | 64 | 0.001 | 0.5 | 144 | trustgtr_ab9_train__weather__forecast__t96_h96__seed2024__20261006_122231 | trustgtr_ab9_inference__weather__forecast__t96_h96__seed2024__20261006_122231 |
| ettm2 | 256 | 0.001 | 0.5 | 96 | trustgtr_ab9_train__ettm2__forecast__t96_h96__seed2024__20261006_122231 | trustgtr_ab9_inference__ettm2__forecast__t96_h96__seed2024__20261006_122231 |

启动命令（本脚本不执行）：

`bash scripts/queue.sh start configs/queue_step_6_4_ab9.json`

状态：`bash scripts/queue.sh status configs/queue_step_6_4_ab9.json`；日志：`tail -f runtime/step_6_4_ab9/queue.log`；停止/续跑使用同一脚本 stop/resume。

显存：未知 smoke 独立估峰；正式使用各自 smoke 峰值。现有统一调度器执行 UUID、短时锁、动态补位、最多三并发及 OOM 退避回队列，不改变正式配置。
