# 下一轮最短路径

5.6【已完成】ETTh1/Solar归档A6/A8完整val诊断；保留ETTh1 A8候选，Solar不做通用默认。Solar退化主要体现于联合训练GTR分支，误差分解不是因果；近恒定输入不是主解释。报告step_5_6_mechanism_conclusions_20261006.md。

6.2【已完成】按用户明确授权新增Weather/ETTm2，公开原样资产及官方协议已核验；覆盖旧1–2范围限制，后续共四数据集。原PDF保留历史快照。

6.3【已完成】新数据集GTR/A6公平对照8/8成功，无OOM；ETTm2双指标改善，Weather仅MSE小降而MAE退化，报告step_6_7_expanded_results.*。单seed2024、T96/H96、官方预算；不把A8当通用默认、不删A/B/C。

6.4【已完成】B推理一致校准单因素验证：17/17任务成功、2次Solar smoke OOM退避恢复、4完整val pair退出0。推理校准比train仅Weather test微改善，其他三pair双指标微差；4pair校准误差全降却无一致预测收益。ETTh1/ETTm2动态收益减弱，Solar动态MSE输constant/g1；Weather动态MSE有益但val MAE退化。保留开关，不作通用默认、不扩多seed。完整报告step_6_4_conclusions.md；g0仍属联合checkpoint，非独立原GTR。

6.5【已完成】C实际检索困难负例单因素配对：9smoke/8正式全部SUCCESS/退出0、0OOM，4完整val pair退出0/重建0。hard相对本轮extremes，ETTh1 MSE/MAE +0.1057%/+0.0713%，Solar −2.9889%/−5.9475%，Weather +0.1535%/+0.1314%，ETTm2 −0.0164%/−0.1872%。保留新旧C候选开关，不作通用默认、不扩多seed。Solar仍落后独立原GTR且实际14epoch vs 对照8epoch；共同前8epoch验证收益仅−0.1655%，不能称等实际计算量或纯采样因果。future标签不进入预测，B关闭以隔离C。完整报告step_6_5_conclusions.md；A/B/C研究范围不收缩。

6.6【下一步，待执行】当前历史库是同基础相位加短/中/长三档相对年龄，并非三个独立周期取模。保持完整短/中/长周期范围，独立实现显式candidate-bank周期规则；先固定train/val核验因果、候选覆盖与收益空间，旧模式回归通过后，再smoke和四数据集单因素公平对照。保留A/B/C及旧版本；尚未启动新训练。

6.7【待验证】有价值版本的A/B/C组合，后续覆盖ETTh1/Solar/Weather/ETTm2，保持资产与预算一致。

7.1–7.5【条件阶段】证据充分后冻结方法、补多seed/更多预测长度/强baseline/消融/论文；当前不扩这些部分，不声称投稿等级。
