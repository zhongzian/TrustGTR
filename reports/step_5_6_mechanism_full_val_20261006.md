# 5.6 固定验证集机制诊断

1. 仅归档 best checkpoint、完整固定验证窗口、CPU 2 threads、流式 batch16；无 test、反向、优化器或 checkpoint 写入。
2. 同 checkpoint GTR branch 是联合训练后的分支，并非独立原 GTR；强制单库和路由切换是推理反事实，不是重训消融。
3. 输入近恒定 strata 事先定义为各通道输入 std≤0.1，来自记忆描述符尺度下限；效用定义为本 checkpoint GTR branch MSE 减去对应预测 MSE，正值表示帮助。
4. 全窗口流式聚合不同于官方 shuffle/drop_last 的 early-stopping val 口径，不能混用。

| dataset | mode | windows | actual MSE | branch MSE | retrieved MSE | switched MSE | strength | harmful correction | bank max weight |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| solar | per_bank | 5161 | 0.141778822 | 0.144272793 | 0.171248395 | 0.142315260 | 0.195334 | 46.95% | 0.647186 |
| solar | per_bank_evidence | 5161 | 0.145034156 | 0.147462240 | 0.177193073 | 0.144620559 | 0.184863 | 41.13% | 0.700495 |
| etth1 | per_bank | 2785 | 0.670047847 | 0.673041443 | 0.825826849 | 0.669709610 | 0.074740 | 39.62% | 0.623454 |
| etth1 | per_bank_evidence | 2785 | 0.668913047 | 0.673691115 | 0.763900966 | 0.670011884 | 0.072544 | 38.47% | 0.608148 |

## trustgtr_a6__solar__forecast__t96_h96__seed2024__20261005_162347

5. prior / posterior / effective 各库均值：{"prior": [0.6262025867216826, 0.342522972827181, 0.03127445217143966], "posterior": [0.6262025867216826, 0.342522972827181, 0.03127445217143966], "effective": [0.6262025867065174, 0.342522972802198, 0.03127445217785203]}
6. 匹配质量与验证未来效用相关：{"bank_evidence_vs_correction_gain": {"count": 2121171, "pearson": -0.039065920203077434}, "bank_nearest_distance_vs_correction_gain": {"count": 2121171, "pearson": 0.17541025176841046}, "bank_evidence_vs_retrieval_gain": {"count": 2121171, "pearson": -2.8264550640873297e-05}, "bank_posterior_vs_correction_gain": {"count": 2121171, "pearson": 0.03226236017468433}, "selected_bank_evidence_vs_gain": {"count": 707057, "pearson": -0.04519515451792309}, "candidate_distance_vs_correction_gain": {"count": 6363513, "pearson": 0.13423793742887247}, "candidate_distance_vs_retrieval_gain": {"count": 6363513, "pearson": 0.0433915531693492}}
7. 分层 MSE / MAE：

| stratum | elements | actual MSE | actual MAE | branch MSE | switched MSE |
|---|---:|---:|---:|---:|---:|
| near_flat | 539808 | 0.331789145 | 0.354519872 | 0.340002220 | 0.340709884 |
| non_flat | 67337664 | 0.140255616 | 0.210736757 | 0.142703741 | 0.140724842 |

8. 每库诊断：

| bank | nearest distance | evidence | correction gain | forced-bank MSE | forced-bank MAE |
|---|---:|---:|---:|---:|---:|
| 0 | 0.160249 | 63.266893 | 0.002023308 | 0.142249486 | 0.212803246 |
| 1 | 0.172752 | 45.798628 | 0.001598848 | 0.142673945 | 0.211897963 |
| 2 | 0.091847 | 52.170485 | -0.002689768 | 0.146962561 | 0.215502634 |

9. 首4窗口 forward / route 重建最大误差：0.0 / 0.0；全量 route 重建最大误差：0.0。
10. 耗时 28.3s；详细数据、sha256、固定资产核验和窗口索引见同名 JSON。

## trustgtr_a8__solar__forecast__t96_h96__seed2024__20261005_221654

5. prior / posterior / effective 各库均值：{"prior": [0.4105291186048104, 0.5091325441564042, 0.08033834802362724], "posterior": [0.4405975157302777, 0.46897983816691036, 0.09042264614578976], "effective": [0.4405975156862818, 0.46897983822349726, 0.09042264613434814]}
6. 匹配质量与验证未来效用相关：{"bank_evidence_vs_correction_gain": {"count": 2121171, "pearson": -0.018221728368345447}, "bank_nearest_distance_vs_correction_gain": {"count": 2121171, "pearson": 0.13909971148868064}, "bank_evidence_vs_retrieval_gain": {"count": 2121171, "pearson": -0.07217720110135667}, "bank_posterior_vs_correction_gain": {"count": 2121171, "pearson": 0.002172836069265034}, "selected_bank_evidence_vs_gain": {"count": 707057, "pearson": -0.038146636954826764}, "candidate_distance_vs_correction_gain": {"count": 6363513, "pearson": 0.10810066333017986}, "candidate_distance_vs_retrieval_gain": {"count": 6363513, "pearson": 0.054670589343650634}}
7. 分层 MSE / MAE：

| stratum | elements | actual MSE | actual MAE | branch MSE | switched MSE |
|---|---:|---:|---:|---:|---:|
| near_flat | 539808 | 0.342540118 | 0.369921500 | 0.341460627 | 0.337144512 |
| non_flat | 67337664 | 0.143450862 | 0.223574431 | 0.145907064 | 0.143077203 |

8. 每库诊断：

| bank | nearest distance | evidence | correction gain | forced-bank MSE | forced-bank MAE |
|---|---:|---:|---:|---:|---:|
| 0 | 0.235639 | 59.008239 | 0.002653801 | 0.144808439 | 0.225847679 |
| 1 | 0.244314 | 45.031075 | 0.002311930 | 0.145150309 | 0.225032604 |
| 2 | 0.121078 | 66.706148 | -0.000726685 | 0.148188924 | 0.226191346 |

9. 首4窗口 forward / route 重建最大误差：0.0 / 0.0；全量 route 重建最大误差：0.0。
10. 耗时 32.7s；详细数据、sha256、固定资产核验和窗口索引见同名 JSON。

## trustgtr_a6__etth1__forecast__t96_h96__seed2024__20261005_160105

5. prior / posterior / effective 各库均值：{"prior": [0.5899760828708652, 0.18377819705215168, 0.22624573006367998], "posterior": [0.5899760828708652, 0.18377819705215168, 0.22624573006367998], "effective": [0.5899760826376882, 0.18377819690866734, 0.2262457301191915]}
6. 匹配质量与验证未来效用相关：{"bank_evidence_vs_correction_gain": {"count": 58485, "pearson": 0.0067574074354720655}, "bank_nearest_distance_vs_correction_gain": {"count": 58485, "pearson": -0.08387477396009073}, "bank_evidence_vs_retrieval_gain": {"count": 58485, "pearson": 0.15799902299907984}, "bank_posterior_vs_correction_gain": {"count": 58485, "pearson": -0.03375878869916509}, "selected_bank_evidence_vs_gain": {"count": 19495, "pearson": 0.08671229119380472}, "candidate_distance_vs_correction_gain": {"count": 175455, "pearson": -0.06430604784849873}, "candidate_distance_vs_retrieval_gain": {"count": 175455, "pearson": -0.17175485710721408}}
7. 分层 MSE / MAE：

| stratum | elements | actual MSE | actual MAE | branch MSE | switched MSE |
|---|---:|---:|---:|---:|---:|
| near_flat | 10560 | 4.785733036 | 1.706483291 | 4.793584110 | 4.783930169 |
| non_flat | 1860960 | 0.646693430 | 0.531550932 | 0.649659463 | 0.646363504 |

8. 每库诊断：

| bank | nearest distance | evidence | correction gain | forced-bank MSE | forced-bank MAE |
|---|---:|---:|---:|---:|---:|
| 0 | 1.454878 | 2.311212 | 0.001983955 | 0.671057489 | 0.538270440 |
| 1 | 1.551933 | 1.452406 | 0.003208668 | 0.669832776 | 0.538803807 |
| 2 | 1.557855 | 0.871779 | 0.004958561 | 0.668082882 | 0.538870947 |

9. 首4窗口 forward / route 重建最大误差：0.0 / 0.0；全量 route 重建最大误差：0.0。
10. 耗时 1.3s；详细数据、sha256、固定资产核验和窗口索引见同名 JSON。

## trustgtr_a8__etth1__forecast__t96_h96__seed2024__20261005_221654

5. prior / posterior / effective 各库均值：{"prior": [0.41610428858839815, 0.2956533408070962, 0.2882423820154167], "posterior": [0.5371229663462586, 0.26812190555588317, 0.1947551277520579], "effective": [0.5371229663690937, 0.2681219055791961, 0.19475512793001]}
6. 匹配质量与验证未来效用相关：{"bank_evidence_vs_correction_gain": {"count": 58485, "pearson": 0.017585023288732776}, "bank_nearest_distance_vs_correction_gain": {"count": 58485, "pearson": -0.08223184242227674}, "bank_evidence_vs_retrieval_gain": {"count": 58485, "pearson": 0.15568548824869805}, "bank_posterior_vs_correction_gain": {"count": 58485, "pearson": 0.03349091849875824}, "selected_bank_evidence_vs_gain": {"count": 19495, "pearson": 0.002313303617690517}, "candidate_distance_vs_correction_gain": {"count": 175455, "pearson": -0.05931178799515153}, "candidate_distance_vs_retrieval_gain": {"count": 175455, "pearson": -0.1692375433918093}}
7. 分层 MSE / MAE：

| stratum | elements | actual MSE | actual MAE | branch MSE | switched MSE |
|---|---:|---:|---:|---:|---:|
| near_flat | 10560 | 4.781665439 | 1.705543731 | 4.793576286 | 4.784102310 |
| non_flat | 1860960 | 0.645575272 | 0.531266933 | 0.650312866 | 0.646666517 |

8. 每库诊断：

| bank | nearest distance | evidence | correction gain | forced-bank MSE | forced-bank MAE |
|---|---:|---:|---:|---:|---:|
| 0 | 1.137326 | 2.462129 | 0.002025014 | 0.671666100 | 0.538641237 |
| 1 | 1.322888 | 1.526536 | 0.003318931 | 0.670372185 | 0.539071396 |
| 2 | 1.614100 | 0.854252 | 0.003818148 | 0.669872968 | 0.539706089 |

9. 首4窗口 forward / route 重建最大误差：0.0 / 0.0；全量 route 重建最大误差：0.0。
10. 耗时 1.3s；详细数据、sha256、固定资产核验和窗口索引见同名 JSON。
