# GTR 最快复现与 TrustGTR 迭代

用户最新目标：独立项目与Conda环境，先最快完成官方单seed关键数据集复现，紧接改进；保留A/B/C完整范围。2026-10-06用户明确授权5.6及后续新增两个数据集，覆盖旧1–2范围限制；仍单seed公平对照，不扩多seed。

固定步骤索引见 [docs/步骤与进度.md](docs/步骤与进度.md)。**6.5 complete，6.6 pending**：困难负例单因素四数据集配对已完成，Solar/ETTm2双指标改善、ETTh1/Weather微退化。保留新旧C开关，不作通用默认、不扩多seed；下一步独立多周期候选规则，A/B/C范围完整保留。

| 阶段 | 状态 | 完成判据 |
|---|---|---|
| 初始化与官方协议核验 | complete | 官方commit、独立环境、数据、配置与用户规范落盘 |
| 实验入口与非阻塞队列 | complete | 正式参数保留、元数据/恢复、10项假任务测试通过 |
| 基线 smoke 与单seed复现 | complete | 两个smoke和正式基线成功；MSE/MAE与论文五seed参考差异落盘 |
| TrustGTR创新逐项实现与验证 | in_progress | A/B/C保留，B先同协议对照，随后A与状态对比学习 |
| 结果判断与下一轮 | in_progress | 自动汇总；有提升再扩，无提升继续诊断迭代 |
| 项目完整计划PDF | complete | 18页PDF、可编辑源、指标与排版校验完成 |
| v5贡献归因与Solar新版 | complete | ETTh1 AC5及Solar A5同协议正式结果与判断落盘 |

不在本阶段扩展10个外部baseline，不使用多seed，不把smoke结果当复现结果。

| v6机制迭代 | complete | 路由/B描述符/B校准/C度量分开验证，先smoke再正式同协议对照 |

| v7组合归因与风险校准 | complete | Solar路由、ETTh1 A/C闭合、B风险校准同协议单因素验证 |

| v8候选感知路由 | complete | per_bank_evidence单因素；旧模式/基线回归、独立smoke、两正式结果与完整val机制记录完成 |
| 新增Weather/ETTm2 | complete | 原样公开资产、官方协议、4smoke与4正式对照成功，指标及成本完整落盘 |
| 6.4推理一致校准 | complete | 独立开关、46归档组精确兼容、9smoke/8正式/4完整val pair均成功，成本/审计/结果表已落盘 |

| 6.5实际检索困难负例 | complete | 因果采样/旧行为回归及train64通过；9smoke/8正式SUCCESS/退出0、0OOM；4完整val pair/成本/共同epoch/审计/结论与总表落盘 |
| 6.6完整多周期候选规则 | pending | 独立开关、真正短/中/长周期规则、因果/覆盖/旧模式兼容、smoke及同协议单因素对照 |

当前子步骤：6.5.1–6.5.5 complete；6.6尚未启动。正式与完整val方向一致，2/4数据集有收益但并非通用。Solar hard仍落后独立原GTR，实际14epoch vs 对照8epoch；共同前8epoch验证日志仅有小收益，不把最终差异声称纯采样因果。membership100%或跨checkpoint学得key距离下降不能证明检索质量。结论与证据见 [6.5完整报告](reports/step_6_5_conclusions.md)。
