# 6.5 C实际检索困难负例诊断

CPU 2 threads；没有test、optimizer step、参数更新或checkpoint写入。本文件仅做固定validation；无反向或梯度计算。预测不接收future target。
两个采样版本共同使用euclidean/T=3/λ=0.002。旧AC7使用cosine/T=0.1，因此相对旧AC7的变化不能仅归因采样。

| dataset/reference | sampler | train windows | usable | negative实际检索覆盖 | aux/main encoder norm均值 | encoder gradient cosine均值 |
|---|---|---:|---:|---:|---:|---:|

trustgtr_ac10_extremes__ettm2__forecast__t96_h96__seed2024__20261006_214646: 固定val 11425窗口，actual MSE/MAE 0.1185586/0.2348504，同checkpoint GTR分支 0.1209087/0.2370082，retrieval 0.1720683/0.2839175，strength=0.1111969，重建误差=0。

trustgtr_ac10_hard__ettm2__forecast__t96_h96__seed2024__20261006_214646: 固定val 11425窗口，actual MSE/MAE 0.1184534/0.2345036，同checkpoint GTR分支 0.1210038/0.2371252，retrieval 0.1602191/0.2714286，strength=0.1093032，重建误差=0。

train-only静态库、raw/scaler/split/phase/数组/源代码/配置/checkpoint哈希与本次诊断明细见JSON。初始strength=0时主预测对key encoder的梯度为0；之前的训练64预诊断使用A6已训练权重测梯度冲突。
full-val分支/检索分解均为同一checkpoint事后诊断，不是分别训练的消融；val regime距离只在完成预测后计算，未用于选样、前向、重选checkpoint或调参。
