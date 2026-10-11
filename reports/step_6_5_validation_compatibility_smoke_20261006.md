# 6.5 C实际检索困难负例诊断

CPU 2 threads；没有test、optimizer step、参数更新或checkpoint写入。train梯度由8窗口微批获取，不能冒充官方训练batch；预测不接收future target。
两个采样版本共同使用euclidean/T=3/λ=0.002。旧AC7使用cosine/T=0.1，因此相对旧AC7的变化不能仅归因采样。

| dataset/reference | sampler | train windows | usable | negative实际检索覆盖 | aux/main encoder norm均值 | encoder gradient cosine均值 |
|---|---|---:|---:|---:|---:|---:|

trustgtr_ac6__etth1__forecast__t96_h96__seed2024__20261005_160731: 固定val 4窗口，actual MSE/MAE 0.4000656/0.421022，同checkpoint GTR分支 0.4045333/0.4231855，retrieval 0.4836371/0.4893513，strength=0.06941722，重建误差=0。

train-only静态库、raw/scaler/split/phase/数组/源代码/配置/checkpoint哈希、计数/距离/梯度明细见JSON。初始strength=0时主预测对key encoder的梯度为0；本预诊断使用A6已训练权重才能测有意义的梯度冲突。
full-val分支/检索分解均为同一checkpoint事后诊断，不是分别训练的消融；val regime距离只在完成预测后计算，未用于选样、前向、重选checkpoint或调参。
