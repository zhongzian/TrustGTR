# 6.5 实际检索困难负例：结果与继续判断

6.5.1–6.5.5全部完成。当前判断：retrieval_hard在Solar和ETTm2较本轮regime_extremes对照有双指标收益，ETTh1和Weather微退化；保留为C候选及独立开关，不设为通用默认、不扩大本版多seed。下一步6.6完整多周期候选规则，A/B/C研究范围不收缩。

## 6.5.1–6.5.2 实际改动与协议

正例仍为同phase、train-only因果且窗口不重叠候选中future regime最近者；新负例必须是当前实际检索选中且weight>0、满足原regime间隔的候选，再按当前key距离选最近者。原两类CE、positive_max_distance=1、gap=.25不变，无有效对返回可微0。标签只在训练辅助loss或预测后的验证诊断使用，不进入预测/检索选择。

唯一paired因素为contrast_sampling。两组共同A6 per_bank/bounded、B关闭、C开启euclidean/T3/λ=.002、T96/H96/seed2024；数据划分/归一化/官方batch、max30epochs、patience5、优化器/LR/scheduler/评价与best选择不变。旧AC7采用cosine/T.1，不能将C10与旧AC7的差异归因采样。对A6增加C是额外共同因素，新旧采样对照才隔离困难负例效果。

模型只改TrustGTR.py与HistoricalMemory.py；新C训练才采集actual selected位置/距离/权重，不添参数，eval不采集。原GTR及所有旧开关保留。12项新检查、5+3+12旧回归、18组冻结B9双源及46组归档输出/梯度/RNG检查通过；固定A6 train64预诊断/每8窗口微批，违规0、aux非encoder梯度0。hard实际检索membership100%是有效样本的设计验证，不是预测收益或完整train覆盖。

ETTh1微批weighted aux/main encoder norm均值约1.025、cosine−.114，Solar有效率96.05%→90.77%，风险已记录；这些不是官方训练batch统计，未据此私调λ或阈值。原始/最终测试日志与覆盖说明见实现审计。

## 6.5.4 正式结果

下表由持久化metrics自动生成，数值为MSE/MAE，负百分比表示误差下降。全部checkpoint沿原验证集规则选择。

| 数据集 | 原GTR | A6 | extremes对照 | retrieval_hard | hard相对对照 MSE/MAE变化 |
|---|---:|---:|---:|---:|---:|
| etth1 | 0.369598418/0.392604440 | 0.365974873/0.390658081 | 0.365748525/0.390593171 | 0.366134942/0.390871495 | +0.1057%/+0.0713% |
| solar | 0.181172356/0.236659229 | 0.185901582/0.241035029 | 0.189328104/0.255334467 | 0.183669269/0.240148470 | -2.9889%/-5.9475% |
| weather | 0.156770736/0.201339573 | 0.156595722/0.207328215 | 0.155262366/0.205132857 | 0.155500755/0.205402434 | +0.1535%/+0.1314% |
| ettm2 | 0.167993650/0.251111001 | 0.166146219/0.250517786 | 0.167001963/0.249974921 | 0.166974619/0.249506995 | -0.0164%/-0.1872% |

Solar hard比A6 MSE/MAE分别−1.2008%/−0.3678%，但比原GTR仍+1.3782%/+1.4744%；不能称超过原基线。ETTm2 hard比A6 MSE+0.4986%/MAE−0.4035%，属于指标取舍。ETTh1 hard比A6双指标微差；Weather hard比A6双指标改善，但比原GTR的MAE仍高2.0179%。不能只保留两项正结果。

## 6.5.5 完整固定验证集机制

4pair/8best，CPU四组退出0，覆盖所有固定val窗口；实际eval预测分解精确重建（最大误差0），无test、反向、优化器、aux loss或训练collector。联合GTR分支与retrieval-alone均是同checkpoint的事后分解，不是独立训练消融；聚合不同于官方shuffle/drop_last早停值。

| 数据集/窗口数 | actual MSE extremes→hard | actual MAE extremes→hard | retrieval MSE extremes→hard | joint GTR MSE extremes→hard |
|---|---:|---:|---:|---:|
| etth1/2785 | 0.669014482→0.669430365 | 0.537928510→0.538213560 | 0.794620363→0.796751794 | 0.672861442→0.673167835 |
| solar/5161 | 0.144189565→0.140885151 | 0.223868143→0.211305619 | 0.167977054→0.168660810 | 0.147451296→0.144220211 |
| weather/5175 | 0.378460597→0.378746281 | 0.259059674→0.259536397 | 0.473829048→0.474581792 | 0.386745328→0.387299345 |
| ettm2/11425 | 0.118558637→0.118453443 | 0.234850426→0.234503640 | 0.172068298→0.160219092 | 0.120908700→0.121003771 |

ETTm2：检索自身MSE/MAE与混合预测均改善，regime误差也下降，有局部机制支持；joint GTR分支略差。

Solar：混合预测改善同时伴随joint GTR分支改善及strength .18447→.19555，retrieval自身MSE与regime距离反而略差。不能将收益归为已证实检索MSE变好，分解也不证明唯一因果机制。

ETTh1、Weather：混合预测双指标略差；Weather的regime距离下降仍未带来预测改善。两维regime一致不保证完整未来轨迹预测质量。四组key绝对距离都下降，但各自训练后的embedding尺度不同，不能当作检索质量提高；preflight共享同A6参数的key比较只证明困难负例选择落实。

## 6.5.5 共同epoch与实际成本

最大预算/早停协议相同，实际早停epoch可以不同。Solar extremes完成8epoch/best3，hard完成14epoch/best9：不能称严格等实际训练步数或等计算量。只读取Vali Loss日志的前8epoch比较，hard最好.1439436@6，对照.1441822@3（−0.1655%）；更大的最终验证收益出现在hard epoch9。因此有早期小收益且伴随更晚最佳epoch，不能将全部正式test收益单独归因多训练或纯采样。未重评估/重选checkpoint，日志舍入及原val聚合限制见equal_epoch_validation报告。

两组C对A6新增参数均0，对原GTR共同新增560。实际耗时受共享GPU、并发及不同早停影响；只记录，不作严格速度优劣结论。

| 数据集/sampling | 参数量 | 训练秒 | 观测秒/epoch | test推理秒 | 峰值MiB | 完成/最佳epoch |
|---|---:|---:|---:|---:|---:|---:|
| etth1/regime_extremes | 634314 | 39.33 | 2.81 | 0.26 | 650 | 14/9 |
| etth1/retrieval_hard | 634314 | 106.50 | 7.61 | 0.33 | 650 | 14/9 |
| solar/regime_extremes | 653874 | 659.29 | 82.41 | 3.53 | 1928 | 8/3 |
| solar/retrieval_hard | 653874 | 641.45 | 45.82 | 3.25 | 1930 | 14/9 |
| weather/regime_extremes | 637170 | 919.61 | 51.09 | 2.32 | 612 | 18/13 |
| weather/retrieval_hard | 637170 | 494.54 | 27.47 | 1.56 | 610 | 18/13 |
| ettm2/regime_extremes | 634818 | 262.96 | 32.87 | 1.46 | 840 | 8/3 |
| ettm2/retrieval_hard | 634818 | 253.20 | 31.65 | 1.20 | 840 | 8/3 |

## 6.5 执行与收尾

9/9smoke、63/63链路检查、8/8正式均SUCCESS/退出0，17attempt、0OOM，观察最大并发3。daemon COMPLETED，宿主机核验无自任务残留/僵尸/自GPU预留或CUDA进程。14科学源码/17输入config哈希保持，所有独立ID/resolved配置/env/代码/资产/指标/资源及best/last恢复记录完整。自动总表132项保留8个旧CODE_ERROR，不能把总数称全成功。

纯调度偏差：本轮生成器多加hard formal依赖control正式完成，使最后Weather等待空slot261.15–261.49秒。原daemon不支持安全热reload，当前任务/queue/科学配置未动；未来生成器已去掉多余依赖，只保留alloff+ownsmoke，旧生成器按原SHA归档。该等待不是显存不足，也不等于已实测下一轮总完工加速。

启动 bash scripts/queue.sh start configs/queue_step_6_5_ac10.json；汇总 .conda/bin/python scripts/report_step65.py 与 .conda/bin/python scripts/summarize_results.py，均退出0。科学改动差异见step_6_5_model_changes.diff（TrustGTR +51/−3、HistoricalMemory +21/−3），原tracked git diff为空，不代表新增未跟踪代码无修改。未commit/push或删除旧成果。

证据：[正式配对](step_6_5_paired_results.json)、[完整val诊断](step_6_5_retrieval_diagnostics.md)、[机制解释](step_6_5_retrieval_interpretation.md)、[共同epoch日志](step_6_5_equal_epoch_validation.md)、[运行审计](step_6_5_runtime_audit.md)、[独立复核](step_6_5_final_review.md)、[调度纠正](step_6_5_scheduling_note.md)。

## 6.6 下一步（尚未启动）

当前A为同基础phase加三档相对年龄历史库，尚不是三个独立周期取模。保持完整短/中/长周期创新范围，下一步单因素实现独立candidate-bank周期规则及可关闭开关，先核因果/覆盖/收益空间与旧基线兼容，再smoke和四数据集公平对照。C新旧版本/B完整保留，与周期规则分开归因；6.7组合及7阶段多seed/强baseline须待有效版本证据充分。
