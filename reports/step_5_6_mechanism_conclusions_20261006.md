# 5.6 机制判断与保留建议

1. **验收通过。** 对两个数据集的 A6、A8 归档科学源码和 best checkpoint 做 CPU 2 threads、batch16、完整固定验证窗口的流式分析：ETTh1 每模型2785个窗口、Solar每模型5161个窗口。首4窗口 model forward 重建和全量路由重建最大误差均为0；严格 checkpoint 加载、源码sha256、train-only历史库sha256、train/val固定资产与归一化统计全部一致。命令退出0，约64秒完成，无GPU、test数据集、反向、optimizer更新或checkpoint写入。

2. **ETTh1 支持继续保留 A8 候选。** 完整验证集 A6 MSE/MAE 为0.670047847/0.538180456，A8为0.668913047/0.537892757；A8相对A6改善MSE约0.169%、MAE约0.053%。在同一A8 checkpoint上切回A6的prior-only per_bank路由，MSE/MAE变为0.670011884/0.538583857。此处证据支持该checkpoint的候选证据路由有用；不能由此认定多seed稳定、通用有效或C已独立有效。

3. **Solar 不支持把 A8 定为通用默认。** 完整验证集 A6 MSE/MAE 为0.141778822/0.211880218，A8为0.145034156/0.224738283；A8相对A6退化约2.296%/6.069%。同一A8 checkpoint切回prior-only per_bank改善至0.144620559/0.224509283；同一A6 checkpoint强制用evidence路由则变差至0.142315260/0.212191975。因此，纯粹增加逆距离证据并不必然改善Solar的未来预测。

4. **“描述符更相似 ⇒ 未来更有用”的假设没有在Solar得到支持。** A8每库evidence与该库校正后的未来MSE增益 Pearson为−0.01822，选中历史candidate的distance与校正增益为+0.10810。长期年龄库的平均最近距离0.12108，比短/中库0.23564/0.24431更小，但该库单独校正的平均增益为−0.00072668；短/中库增益分别+0.00265380/+0.00231193。以上为实测相关与同checkpoint反事实，不是因果证明或可部署未来标签监督。

5. **近恒定输入确有问题，但不能独自解释全局退化。** 事先按输入各通道std≤0.1（记忆描述符尺度下限）分层，不使用未来标签定义分组。Solar这类只占0.7953%的通道窗口；A8的最大bank权重均值0.97580，其actual MSE0.342540高于同checkpoint branch0.341461，且切回prior-only改善至0.337145。不过，A6→A8全量MSE增量只有2.626%来自此类，97.374%来自非近恒定输入。不能把“Solar夜间近零”当作已证明的全部根因，std≤0.1也不等同于时间戳意义上的夜间。

6. **两版校正都改善自身分支，退化同时涉及联合训练后的GTR分支。** Solar A6为branch0.144272793→actual0.141778822（校正增益0.002493972），A8为branch0.147462240→actual0.145034156（校正增益0.002428084）。两模型actual差0.003255333843=branch差0.003189446395+校正增益差0.000065887448；前一数值为总差的97.976%，但只是代数误差分解，绝非因果比例，不能断言全部由route或某个具体优化原因造成，也不能当成 independently trained original GTR ablation。A8中41.13%的通道窗口校正有害，与其整体平均校正有益可以同时成立。best epoch差异只能作为待诊断线索，不能直接认定退化来自早停。

7. **保留与调整。** A8保留为ETTh1上的有效候选；Solar保留A6对照及A8负结果，A6也尚未超过原Solar test基线双指标。暂不把A8作为四数据集统一默认或提前扩多seed。新增Weather、ETTm2先进行同协议GTR/A6单因素对照。后续沿6.2检查主/辅损失和训练/推理差异，验证可由观测输入得到的历史未来效用校准，而非只强化距离接近性。A、B、C创新方案完整保留。

8. **口径与证据。** 本报告为完整验证窗口流式元素平均，和官方shuffle/drop_last早停val聚合不同；不与test数值混用。强制单库、A6↔A8路由切换均为同checkpoint推理反事实，不是重训消融。银行规则仍为“相同基础相位+三档相对历史年龄”，不能宣称已实现三个独立周期取模。窗口、资产、源代码、checkpoint哈希、prior/posterior/effective、熵/集中度、逐库效用、距离相关、分层MSE/MAE与资源消耗均保存在 `step_5_6_mechanism_full_val_20261006.json`；逐行摘要见同名`.md`。

9. **执行记录。** 使用 `.conda/bin/python scripts/diagnose_step56_mechanism.py --runs trustgtr_a6__solar__forecast__t96_h96__seed2024__20261005_162347 trustgtr_a8__solar__forecast__t96_h96__seed2024__20261005_221654 trustgtr_a6__etth1__forecast__t96_h96__seed2024__20261005_160105 trustgtr_a8__etth1__forecast__t96_h96__seed2024__20261005_221654 --sample-count 0 --batch-size 16 --output reports/step_5_6_mechanism_full_val_20261006.json`；stdout/stderr见 `logs/step_5_6_mechanism_full_val_20261006.log`，退出码0。进程峰值RSS约1.86GiB，无GPU内存占用。
