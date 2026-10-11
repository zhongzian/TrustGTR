# 6.4/B9 最终独立复核

日期：2026-10-06（Asia/Shanghai）。结论：**通过，无阻塞问题**。

1.1【范围】本次独立复核读取已完成的8项正式实验、4组配对报告、GTR/A6参考资产、完整4组验证机制报告及完成结论文档。未修改科学源码、配置、实验、指标、checkpoint或其他报告，未重新运行训练、原CPU测试或诊断。仅新建本审计记录。静态源码审查与实际已落盘结果核验分别记录，不以静态审查冒充实验运行。

1.2【正式结果】8项 formal 的 status/metrics 均为SUCCESS、退出码0、smoke关闭。报告 experiment ID集合与队列formal集合完全一致，MSE/MAE、seed、参数、训练/推理时间、峰值显存、完成/最佳epoch与各自metrics逐值一致。metrics、resolved config、checkpoint SHA逐项核对。所有run的科学源均匹配冻结TrustGTR SHA `59fef362c5e61905bcd4db9f866ef9f30454ab63f511c797c72eff28eb540970`。

1.3【公平性】四组配对完整选项并集的唯一差异均为 `gate_calibration_context: train → inference`；resolved顶层选项与official_args.trustgtr一致。seed=2024、T96/H96与正式官方参数，除model/ID/checkpoint路径及TrustGTR模块选项外，分别与独立GTR和A6参考的official_args完全一致。固定评估资产与两类参考完全一致，资产文件SHA也通过。参数两context相同；共同B配置相对A6新增321、相对GTR新增881参数。绝对差、百分比与参数差均重新计算，精确等于JSON；Markdown舍入正确。

1.4【正式收益边界】inference相对train的MSE/MAE变化分别为：ETTh1 +0.08244%/+0.03069%；Solar +0.04887%/+0.14070%；Weather −0.06946%/−0.01054%；ETTm2 +0.01132%/+0.01036%。仅Weather出现很小双指标test改善，不能认定通用收益。Weather的inference MAE仍比GTR高约1.07%，Solar仍比GTR双差；ETTh1/ETTm2相对GTR的改善属于整个A/B方案，不能归因于本次推理上下文修改。

2.1【完整验证机制】合并报告8项原始字段与四份完整val JSON逐值一致；新增dynamic−g1与helpful−harmful等派生值重新核算正确。四份源报告与三版归档诊断script SHA匹配，四CPU诊断进程退出码均0，均覆盖全部固定val窗口，配对indices/assets一致，首4与全量eval重建误差均0。没有读取test标签、反向或优化器步骤。此项为现有结果审计，没有重新执行诊断。

2.2【机制结论】Weather actual val MAE由0.256841782升至0.257205916，报告正确保留该退化；虽然两个checkpoint内dynamic均优于自己的constant/g1且inference的MSE动态收益略增，不能称一致双指标改善。ETTh1/ETTm2校准误差下降，但相对constant收益与效用相关减弱，两个版本的actual val MSE均输自己的g1。Solar的actual val MSE均输constant/g1，gate−gain相关仍负。四pair校准误差下降不等于预测有效，结论文档已明确说明。

2.3【反事实限制】g0、g1、constant都是同joint checkpoint的事后诊断，g0不是独立原GTR；constant均值取自验证集，并非独立训练消融。oracle使用val标签，只用于诊断MSE空间，不能当可部署模型或test成绩。完整val聚合与官方shuffle/drop_last early-stopping聚合不同，不能替换正式协议；powerweighted诊断的global val mean power也不同于训练minibatch辅助损失。Solar仅输出head dropout=0，内部GTR dropout=0.1，不能称无dropout控制。报告均遵守这些边界。

2.4【当前判断】`step_6_4_conclusions.md`与完整4组机制报告未发现实质误述。不将B9 inference设为通用默认、不扩本版本多seed的判断有当前证据支持；A/B/C与旧版本开关保留。所有结果仍是单seed探索，不能声称统计稳定、因果归因或投稿等级。时间受共享GPU、并发和早停影响，记录可复核，但不能作严格性能优劣比较。

3.1【核验快照】本次读取的证据SHA256：

- `step_6_4_paired_results.json`：`b9d8b7c950094c71037acdd13c2c77453083d61da1062e32d5c1bb3cd4e9f3d4`
- `step_6_4_gate_diagnostics.json`：`46bf320ce35be4a98149413cd8782c8e984dd1481c56345a7df4bc92e693bbd4`
- `step_6_4_conclusions.md`：`6cb2f31bcb513752b55cb541f697c8ed1e3668e4c546ed9c9c704910db3508ea`

3.2【核验过程说明】第一次合并报告检查尝试比较整个row字典，因合并报告合法新增派生字段而触发核验断言。随后改为逐项检查所有原始字段必须精确保留，同时独立计算新增派生字段；全部通过。这是审计比较范围修正，未改任何诊断结果或实验指标。
