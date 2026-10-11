# 6.5 C实际检索困难负例诊断

CPU 2 threads；没有test、optimizer step、参数更新或checkpoint写入。train梯度由8窗口微批获取，不能冒充官方训练batch；预测不接收future target。
两个采样版本共同使用euclidean/T=3/λ=0.002。旧AC7使用cosine/T=0.1，因此相对旧AC7的变化不能仅归因采样。

| dataset/reference | sampler | train windows | usable | negative实际检索覆盖 | aux/main encoder norm均值 | encoder gradient cosine均值 |
|---|---|---:|---:|---:|---:|---:|
| etth1 | regime_extremes | 64 | 0.9285714 | 0.1033654 | 0.4925479 | -0.0625567 |
| etth1 | retrieval_hard | 64 | 0.9285714 | 1 | 1.025226 | -0.1137366 |
| solar | regime_extremes | 64 | 0.9605383 | 0.05521254 | 0.4104135 | -0.0315505 |
| solar | retrieval_hard | 64 | 0.9077327 | 1 | 0.1973015 | 0.09181747 |
| weather | regime_extremes | 64 | 0.9241071 | 0.1022544 | 0.2394813 | 0.06251447 |
| weather | retrieval_hard | 64 | 0.9166667 | 1 | 0.41224 | 0.08974728 |
| ettm2 | regime_extremes | 64 | 0.9553571 | 0.1214953 | 0.3462269 | 0.1267925 |
| ettm2 | retrieval_hard | 64 | 0.9486607 | 1 | 0.9094475 | 0.2095076 |

train-only静态库、raw/scaler/split/phase/数组/源代码/配置/checkpoint哈希、计数/距离/梯度明细见JSON。初始strength=0时主预测对key encoder的梯度为0；本预诊断使用A6已训练权重才能测有意义的梯度冲突。
full-val分支/检索分解均为同一checkpoint事后诊断，不是分别训练的消融；val regime距离只在完成预测后计算，未用于选样、前向、重选checkpoint或调参。
