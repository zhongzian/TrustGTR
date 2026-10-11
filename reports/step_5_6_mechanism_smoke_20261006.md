# 5.6 固定验证集机制诊断

1. 仅归档 best checkpoint、完整固定验证窗口、CPU 2 threads、流式 batch16；无 test、反向、优化器或 checkpoint 写入。
2. 同 checkpoint GTR branch 是联合训练后的分支，并非独立原 GTR；强制单库和路由切换是推理反事实，不是重训消融。
3. 输入近恒定 strata 事先定义为各通道输入 std≤0.1，来自记忆描述符尺度下限；效用定义为本 checkpoint GTR branch MSE 减去对应预测 MSE，正值表示帮助。
4. 全窗口流式聚合不同于官方 shuffle/drop_last 的 early-stopping val 口径，不能混用。

| dataset | mode | windows | actual MSE | branch MSE | retrieved MSE | switched MSE | strength | harmful correction | bank max weight |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| etth1 | per_bank | 4 | 0.401185375 | 0.401504688 | 0.471920776 | 0.401509868 | 0.074740 | 50.00% | 0.584201 |
| etth1 | per_bank_evidence | 4 | 0.402236795 | 0.403083925 | 0.464307439 | 0.402548015 | 0.072544 | 57.14% | 0.611597 |
| solar | per_bank | 4 | 0.116416184 | 0.123752800 | 0.115825566 | 0.119261499 | 0.195334 | 29.38% | 0.498907 |
| solar | per_bank_evidence | 4 | 0.151756135 | 0.155924494 | 0.192533078 | 0.149084762 | 0.184863 | 30.66% | 0.803395 |

## trustgtr_a6__etth1__forecast__t96_h96__seed2024__20261005_160105

5. prior / posterior / effective 各库均值：{"prior": [0.5413521621376276, 0.21003700606524944, 0.24861084084425653], "posterior": [0.5413521621376276, 0.21003700606524944, 0.24861084084425653], "effective": [0.5413521655968258, 0.21003700074340617, 0.24861083977988788]}
6. 匹配质量与验证未来效用相关：{"bank_evidence_vs_correction_gain": {"count": 84, "pearson": -0.17964817803567515}, "bank_nearest_distance_vs_correction_gain": {"count": 84, "pearson": 0.10970092001496724}, "bank_evidence_vs_retrieval_gain": {"count": 84, "pearson": 0.0577514587603057}, "bank_posterior_vs_correction_gain": {"count": 84, "pearson": 0.07315569383504483}, "selected_bank_evidence_vs_gain": {"count": 28, "pearson": -0.5641652076290927}}
7. 分层 MSE / MAE：

| stratum | elements | actual MSE | actual MAE | branch MSE | switched MSE |
|---|---:|---:|---:|---:|---:|
| near_flat | 0 | NA | NA | NA | NA |
| non_flat | 2688 | 0.401185375 | 0.421809965 | 0.401504688 | 0.401509868 |

8. 每库诊断：

| bank | nearest distance | evidence | correction gain | forced-bank MSE | forced-bank MAE |
|---|---:|---:|---:|---:|---:|
| 0 | 0.795465 | 2.892461 | 0.002510765 | 0.398993926 | 0.420607526 |
| 1 | 1.064741 | 1.516700 | 0.001145550 | 0.400359147 | 0.422975513 |
| 2 | 1.216425 | 0.887141 | -0.006827165 | 0.408331859 | 0.424572856 |

9. 首4窗口 forward / route 重建最大误差：0.0 / 0.0；全量 route 重建最大误差：0.0。
10. 耗时 0.3s；详细数据、sha256、固定资产核验和窗口索引见同名 JSON。

## trustgtr_a8__etth1__forecast__t96_h96__seed2024__20261005_221654

5. prior / posterior / effective 各库均值：{"prior": [0.38415902560310705, 0.3051049698676382, 0.31073602342179846], "posterior": [0.5042178846363511, 0.2539590752816626, 0.24182303273119032], "effective": [0.5042178920869317, 0.2539590754147087, 0.24182303479340458]}
6. 匹配质量与验证未来效用相关：{"bank_evidence_vs_correction_gain": {"count": 84, "pearson": -0.14869558218279197}, "bank_nearest_distance_vs_correction_gain": {"count": 84, "pearson": 0.06993583752775642}, "bank_evidence_vs_retrieval_gain": {"count": 84, "pearson": 0.07867398918646507}, "bank_posterior_vs_correction_gain": {"count": 84, "pearson": -0.006088629493081742}, "selected_bank_evidence_vs_gain": {"count": 28, "pearson": -0.26984734115522013}}
7. 分层 MSE / MAE：

| stratum | elements | actual MSE | actual MAE | branch MSE | switched MSE |
|---|---:|---:|---:|---:|---:|
| near_flat | 0 | NA | NA | NA | NA |
| non_flat | 2688 | 0.402236795 | 0.422203146 | 0.403083925 | 0.402548015 |

8. 每库诊断：

| bank | nearest distance | evidence | correction gain | forced-bank MSE | forced-bank MAE |
|---|---:|---:|---:|---:|---:|
| 0 | 0.686883 | 2.996186 | 0.003698647 | 0.399385282 | 0.420698416 |
| 1 | 0.991321 | 1.586545 | 0.002130567 | 0.400953361 | 0.422416814 |
| 2 | 1.198214 | 0.940577 | -0.004647926 | 0.407731857 | 0.424885153 |

9. 首4窗口 forward / route 重建最大误差：0.0 / 4.76837158203125e-07；全量 route 重建最大误差：4.76837158203125e-07。
10. 耗时 0.1s；详细数据、sha256、固定资产核验和窗口索引见同名 JSON。

## trustgtr_a6__solar__forecast__t96_h96__seed2024__20261005_162347

5. prior / posterior / effective 各库均值：{"prior": [0.4396624782115874, 0.44222896312275073, 0.11810856777154942], "posterior": [0.4396624782115874, 0.44222896312275073, 0.11810856777154942], "effective": [0.43966247883700105, 0.4422289648630323, 0.11810856799928159]}
6. 匹配质量与验证未来效用相关：{"bank_evidence_vs_correction_gain": {"count": 1644, "pearson": -0.028706170107212683}, "bank_nearest_distance_vs_correction_gain": {"count": 1644, "pearson": 0.14216266235082045}, "bank_evidence_vs_retrieval_gain": {"count": 1644, "pearson": 0.04442446832850705}, "bank_posterior_vs_correction_gain": {"count": 1644, "pearson": 0.09932539076068221}, "selected_bank_evidence_vs_gain": {"count": 548, "pearson": -0.10727741078208969}}
7. 分层 MSE / MAE：

| stratum | elements | actual MSE | actual MAE | branch MSE | switched MSE |
|---|---:|---:|---:|---:|---:|
| near_flat | 4320 | 0.251487203 | 0.332178890 | 0.270071288 | 0.267116134 |
| non_flat | 48288 | 0.104332296 | 0.177200473 | 0.110662677 | 0.106033947 |

8. 每库诊断：

| bank | nearest distance | evidence | correction gain | forced-bank MSE | forced-bank MAE |
|---|---:|---:|---:|---:|---:|
| 0 | 0.309731 | 122.475453 | 0.004518658 | 0.119234142 | 0.194424326 |
| 1 | 0.280651 | 91.673774 | 0.010202817 | 0.113549984 | 0.187166258 |
| 2 | 0.081052 | 53.653306 | -0.002882605 | 0.126635406 | 0.190828555 |

9. 首4窗口 forward / route 重建最大误差：0.0 / 0.0；全量 route 重建最大误差：0.0。
10. 耗时 7.7s；详细数据、sha256、固定资产核验和窗口索引见同名 JSON。

## trustgtr_a8__solar__forecast__t96_h96__seed2024__20261005_221654

5. prior / posterior / effective 各库均值：{"prior": [0.007204114414033388, 0.6735670080231707, 0.319228888196993], "posterior": [0.014691585078313989, 0.623005599760277, 0.362302815142244], "effective": [0.014691585053625689, 0.623005601168945, 0.36230281497059513]}
6. 匹配质量与验证未来效用相关：{"bank_evidence_vs_correction_gain": {"count": 1644, "pearson": -0.1272394953019531}, "bank_nearest_distance_vs_correction_gain": {"count": 1644, "pearson": 0.30490865055128374}, "bank_evidence_vs_retrieval_gain": {"count": 1644, "pearson": -0.052820420545517256}, "bank_posterior_vs_correction_gain": {"count": 1644, "pearson": -0.12513013922641722}, "selected_bank_evidence_vs_gain": {"count": 548, "pearson": 0.023140719907660003}}
7. 分层 MSE / MAE：

| stratum | elements | actual MSE | actual MAE | branch MSE | switched MSE |
|---|---:|---:|---:|---:|---:|
| near_flat | 4320 | 0.391873610 | 0.423415775 | 0.415855022 | 0.382667595 |
| non_flat | 48288 | 0.130274453 | 0.193780809 | 0.132670272 | 0.128187689 |

8. 每库诊断：

| bank | nearest distance | evidence | correction gain | forced-bank MSE | forced-bank MAE |
|---|---:|---:|---:|---:|---:|
| 0 | 0.412867 | 84.106954 | 0.010830724 | 0.145093772 | 0.215308527 |
| 1 | 0.425511 | 55.797598 | 0.016879234 | 0.139045260 | 0.207653847 |
| 2 | 0.113455 | 31.750202 | -0.001029534 | 0.156954029 | 0.213196470 |

9. 首4窗口 forward / route 重建最大误差：0.0 / 7.152557373046875e-07；全量 route 重建最大误差：7.152557373046875e-07。
10. 耗时 7.6s；详细数据、sha256、固定资产核验和窗口索引见同名 JSON。
