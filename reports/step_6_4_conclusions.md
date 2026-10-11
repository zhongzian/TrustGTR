# 6.4 B推理一致校准：完成与继续判断

6.4.1–6.4.5已完成。只改变辅助校准上下文train→inference；主训练dropout/main gate梯度、A6 per_bank、18维retrieval descriptor、power_weighted λ=0.05、init gate=0.9、C关闭，以及四数据集官方预算/seed2024/固定评估资产不变。

当前判断：本次单因素修改未显示跨数据集收益，不将B9推理上下文设为通用默认，不进入多seed扩展。Weather出现很小的test改善与val动态MSE收益；其val MAE仍变差。其余三数据集train-context与inference-context的正式MSE/MAE均微退化，不能以校准误差下降或门均值下降宣称可靠性成功。A/B/C完整创新范围继续保留。

## 6.4.4 正式结果

T96/H96、seed2024。下表自动读取持久化指标；括号内为MSE/MAE，负百分比代表误差下降。checkpoint均按原val规则选择，未用test重选。

| 数据集 | 原GTR | A6 | train校准 | inference校准 | inference相对train MSE/MAE变化 |
|---|---:|---:|---:|---:|---:|
| etth1 | 0.369598418/0.392604440 | 0.365974873/0.390658081 | 0.366705477/0.390909523 | 0.367007792/0.391029507 | +0.0824%/+0.0307% |
| solar | 0.181172356/0.236659229 | 0.185901582/0.241035029 | 0.185495690/0.253038049 | 0.185586348/0.253394067 | +0.0489%/+0.1407% |
| weather | 0.156770736/0.201339573 | 0.156595722/0.207328215 | 0.155600712/0.203520939 | 0.155492634/0.203499496 | -0.0695%/-0.0105% |
| ettm2 | 0.167993650/0.251111001 | 0.166146219/0.250517786 | 0.166486874/0.250503242 | 0.166505724/0.250529200 | +0.0113%/+0.0104% |

相对A6，两个B版本在ETTh1的双指标都较差，在ETTm2的MSE较差；Solar只小降MSE、MAE明显升高；Weather双指标改善，但其MAE仍高于原GTR。原GTR改善不能直接归因于本次inference修改。更完整的绝对/相对差、唯一ID、配置与checkpoint哈希见 [配对结果](step_6_4_paired_results.json)。

## 6.4.5 固定完整验证集机制

使用各自best checkpoint、归档科学源码、同一train-only历史库和固定val资产。actual为真实eval预测；g0、g1和constant均为同一joint checkpoint的事后固定门诊断，没有重新训练。g0不是独立原GTR。oracle使用val标签，只代表诊断上限。完整val的聚合与官方早停shuffle/drop_last聚合不同，不替换正式结果。

表中gain=constant MSE−dynamic MSE，正值表示动态门优于同均值常数门；相关为gate与g1校正MSE效用的Pearson。校准误差采用完整val的power归一化，不能当作训练batch辅助loss的相同数值。

| 数据集/val窗口 | actual MSE train→inference | dynamic对constant gain train→inference | gate效用相关 train→inference | 校准误差 train→inference |
|---|---:|---:|---:|---:|
| etth1/2785 | 0.671870843→0.672509798 | +0.000266154→+0.000092237 | +0.19840→+0.09860 | 0.243593→0.239292 |
| solar/5161 | 0.144903252→0.145005516 | -0.000349770→-0.000336840 | -0.09111→-0.08874 | 0.254646→0.251822 |
| weather/5175 | 0.379155312→0.378898771 | +0.000841672→+0.000979099 | +0.04237→+0.05148 | 0.273860→0.264366 |
| ettm2/11425 | 0.119043064→0.119064806 | +0.000114557→+0.000094251 | +0.07428→+0.06171 | 0.242001→0.237378 |

ETTh1、ETTm2：校准误差下降，但动态对常数门的收益和效用相关都减弱，两个版本在val MSE上也都输同checkpoint的g1。仍有弱效用信号，不能说完全无信号。

Solar：两版动态门val MSE都输constant/g1，gate效用相关仍为负；helpful/harmful二分类均值略有改善不等于效用幅度排序成功。预测头dropout=0，但原GTR内部dropout=0.1，因此不能称为无dropout对照。

Weather：两个版本的动态门均在val MSE/MAE优于constant/g1，推理校准增加动态MSE收益，但actual val MAE从0.256841782升到0.257205916，gate与oracle相关及helpful/harmful均门间隔下降。证据支持局部、很小且有指标取舍的效果，尚不能通用化。

推断：单独校正train/eval辅助专家不一致不足以解决门对检索效用的排序。标签oracle仍有余量，但不是可实现模型成绩；下一步应检查候选/状态监督与实际检索近邻是否一致，而不是扩大本版多seed。

## 6.4 成本与执行验收

本修改没有新增可训练参数；两context参数数相同。共同B配置相对A6增加321参数，相对原GTR增加881参数。inference-context每训练batch额外一次no_grad专家前向；eval不新增teacher前向。耗时受共享GPU/并发影响，不能用下列观测作严谨速度优劣结论。

| 数据集/context | 参数量 | 训练秒 | 观测秒/epoch | test推理秒 | 峰值MiB | 完成epoch/最佳epoch |
|---|---:|---:|---:|---:|---:|---:|
| etth1/train | 634635 | 34.64 | 2.47 | 0.25 | 652 | 14/9 |
| etth1/inference | 634635 | 126.62 | 9.04 | 0.36 | 656 | 14/9 |
| solar/train | 654195 | 600.90 | 75.11 | 3.67 | 1966 | 8/3 |
| solar/inference | 654195 | 361.53 | 45.19 | 3.21 | 1966 | 8/3 |
| weather/train | 637491 | 916.60 | 50.92 | 2.96 | 580 | 18/13 |
| weather/inference | 637491 | 397.16 | 22.06 | 1.59 | 582 | 18/13 |
| ettm2/train | 635139 | 250.12 | 27.79 | 1.85 | 668 | 9/4 |
| ettm2/inference | 635139 | 331.06 | 36.78 | 1.46 | 670 | 9/4 |

12项新CPU检查及旧模块回归通过；46组归档配置输出/梯度/RNG精确兼容。GPU smoke 9/9成功、63/63链路检查；正式8/8成功、退出0。Solar两个smoke首attempt OOM，各退避后资源改善再执行成功（累计2次OOM）；未改正式batch、预算或协议。最大并发3，未杀其他用户进程或用空Tensor占显存。

所有ID/config/env/代码快照/资产/指标/checkpoint/资源和失败记录已落盘。自动总表115项保留历史失败，不等于115成功。执行命令：bash scripts/queue.sh start configs/queue_step_6_4_ab9.json；结果命令：.conda/bin/python scripts/report_step64.py --queue configs/queue_step_6_4_ab9.json，以及 .conda/bin/python scripts/summarize_results.py，均退出0。

证据：[smoke验收](step_6_4_smoke_acceptance.json)、[运行审计](step_6_4_runtime_audit.md)、[完整val机制](step_6_4_gate_diagnostics.md)、[最终独立复核](step_6_4_final_review.md)、[自动总表](latest_results.csv)。独立审计和机制报告记录各自验证范围，不能用静态检查替代实际运行成功。

## 6.5 最短下一步

保留当前A/B/C及旧loss/context开关，先实现C实际检索近邻中的不同regime困难负例，并核验positive/negative来自真实train-only因果候选；future趋势标签仅用于训练监督，不进入预测或测试检索。只改变C采样/监督因素，固定其余训练协议，先CPU及baseline smoke再四数据集单seed配对。6.6完整多周期候选规则、6.7组合独立收益继续保留，当前不启动这些后续实验。
