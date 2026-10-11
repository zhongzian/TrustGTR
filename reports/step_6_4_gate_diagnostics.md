# 6.4.5 四数据集完整固定验证诊断

1. 四CPU诊断进程退出码均0；每dataset两SUCCESS best checkpoint，全部固定val，首4与全量重建误差均0。
2. 无test、backward、optimizer；动态门、g0、g1、constant是同joint checkpoint反事实，g0并非独立原GTR，oracle仅验证标签MSE上限。
3. GTR内部Dropout.p四dataset均0.1；输出head ETTh1/Weather/ETTm2=0.5、Solar=0.0，不能将Solar当无dropout控制。
4. 流式float64误差聚合不同于official shuffle/drop_last early-stop val；powerweighted误差用global val mean power。

| dataset | context | windows | actual MSE | actual MAE | dynamic−constant MSE收益 | dynamic−g1 MSE收益 | cal MSE | gate−gain corr | helpful−harmful gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| etth1 | train | 2785 | 0.671870843 | 0.539106789 | 0.000266154 | -0.000364118 | 0.243593 | 0.198403 | 0.001596 |
| etth1 | inference | 2785 | 0.672509798 | 0.539435009 | 0.000092237 | -0.000717391 | 0.239292 | 0.098598 | -0.000970 |
| solar | train | 5161 | 0.144903252 | 0.225137436 | -0.000349770 | -0.000832520 | 0.254646 | -0.091113 | 0.023202 |
| solar | inference | 5161 | 0.145005516 | 0.225526840 | -0.000336840 | -0.000849986 | 0.251822 | -0.088742 | 0.024583 |
| weather | train | 5175 | 0.379155312 | 0.256841782 | 0.000841672 | 0.000998110 | 0.273860 | 0.042373 | 0.034856 |
| weather | inference | 5175 | 0.378898771 | 0.257205916 | 0.000979099 | 0.001132272 | 0.264366 | 0.051479 | 0.030493 |
| ettm2 | train | 11425 | 0.119043064 | 0.236300893 | 0.000114557 | -0.000157000 | 0.242001 | 0.074282 | 0.008301 |
| ettm2 | inference | 11425 | 0.119064806 | 0.236318995 | 0.000094251 | -0.000219794 | 0.237378 | 0.061708 | 0.004463 |

5. 收益列为reference MSE−actual MSE，正数表示动态门更好；仅与自己的joint checkpoint比较。

6. ETTh1: inference-context降低校准误差，但完整val两指标退化，动态相对constant收益与效用排序变弱。
7. ETTm2: 校准误差下降，但完整val两指标小退化，动态收益和gate-gain/oracle相关下降。
8. Solar: 完整val两指标退化；两context动态MSE均输constant和g1；gate-gain相关仍负，helpful/harmful分类门差小幅改善不足以成立。
9. Weather: 完整val MSE下降、MAE上升；两context动态均优于constant/g1且inference的MSE收益略增，但排序指标混合，不能称四数据一致可靠性收益。
10. 四context pair校准MSE均下降，只有Weather fullval MSE下降且MAE仍升；不应据辅助校准误差单独定型。
11. 四dataset GTR内部Dropout.p都为0.1；Solar仅输出head.p=0，不能称无dropout控制。
12. 只用single-seed筛选证据；没有任意gate有效阈值；反事实不是重训消融，oracle只有validation MSE上限意义。

13. 四原始JSON/MD及对应日志不覆盖；三版诊断script均按sha256归档于reports/step_6_4_diagnostic_sources/，合并JSON保存source_reports哈希、dropout audit、配对变化和全部细节。
