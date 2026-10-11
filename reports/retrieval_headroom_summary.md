# 历史检索内容诊断（未训练，仅固定验证集）

两数据集均使用已完成的 seed2024 官方 GTR 最佳 checkpoint，训练集历史内容独立冻结；候选未来 exclusive end <= 当前历史 exclusive end。固定检索规则和 blend 在计算前确定；没有构造测试 split 或用测试目标选择参数。oracle 含验证目标，只表示可达空间，不是模型性能。

|数据集/范围|内容|Baseline MSE/MAE|固定 blend 0.1 MSE/MAE|固定 blend 0.2 MSE/MAE|
|---|---|---|---|---|
|etth1 全val 2785|same_phase/跨年龄候选|0.675532699/0.540661812|0.669114232/0.538263738|0.667403519/0.538376808|
|etth1 全val 2785|same_phase/short_relative_age|0.675532699/0.540661812|0.674114764/0.539555311|0.677901387/0.540970802|
|etth1 全val 2785|same_phase/middle_relative_age|0.675532699/0.540661812|0.671394825/0.539654970|0.672991455/0.541950405|
|etth1 全val 2785|same_phase/long_relative_age|0.675532699/0.540661812|0.672518134/0.542046010|0.680338860/0.550305903|
|etth1 全val 2785|without_phase/跨年龄候选|0.675532699/0.540661812|0.672996700/0.541492105|0.676394761/0.545512259|
|etth1 全val 2785|without_phase/short_relative_age|0.675532699/0.540661812|0.677688718/0.542557299|0.685827792/0.547716916|
|etth1 全val 2785|without_phase/middle_relative_age|0.675532699/0.540661812|0.674549401/0.542551160|0.680436850/0.548246503|
|etth1 全val 2785|without_phase/long_relative_age|0.675532699/0.540661812|0.673927605/0.542366266|0.681394160/0.549847126|

来源：`reports/retrieval_headroom_etth1_fullval_banks.json`；同 phase 实际权重短/中/长：61.482%, 22.747%, 15.771%。

|solar 全val 5161|same_phase/跨年龄候选|0.143147901/0.214399382|0.143529564/0.213798687|0.145363972/0.213986263|
|solar 全val 5161|same_phase/short_relative_age|0.143147901/0.214399382|0.141645715/0.212455541|0.141478449/0.211475924|
|solar 全val 5161|same_phase/middle_relative_age|0.143147901/0.214399382|0.141818732/0.212009504|0.142160431/0.210821986|
|solar 全val 5161|same_phase/long_relative_age|0.143147901/0.214399382|0.145276085/0.215344593|0.149502218/0.217560887|
|solar 全val 5161|without_phase/跨年龄候选|0.143147901/0.214399382|0.142250940/0.213104516|0.144023523/0.213958517|
|solar 全val 5161|without_phase/short_relative_age|0.143147901/0.214399382|0.142793775/0.215335056|0.145486608/0.219004348|
|solar 全val 5161|without_phase/middle_relative_age|0.143147901/0.214399382|0.141444951/0.213654101|0.143678769/0.216366798|
|solar 全val 5161|without_phase/long_relative_age|0.143147901/0.214399382|0.143793270/0.214485273|0.147416219/0.217176452|

来源：`reports/retrieval_headroom_solar_fullval_banks.json`；同 phase 实际权重短/中/长：46.483%, 18.690%, 34.827%。

内容结论：ETTh1 三年龄候选组合在全验证集的固定小 blend 同时降低两指标；Solar 同相位短/中年龄内容有用，但长年龄内容有害，单纯输入相似度仍给长bank较高权重。支持尝试真实历史内容检索和可靠性选择，不支持直接替换基线或声称正式测试提升。

实现：`scripts/diagnose_historical_retrieval.py`；`history_bank`、`window_keys`、`age_coverage_candidates` 可复用。标准化 future 基于候选输入 mean/std，反变换使用当前输入 mean/std，std floor=0.1（数据集标准化单位）。候选为每个年龄三分位确定性均匀最多64个，单模式最多192，per-channel k=3；15维key由3个训练统计标准化特征及12-bin window shape构成。

验证范围明确包含 all val windows（不shuffle、不drop_last）；与训练早停的随机val批均值略有协议差异，诊断内部比较完全共用相同窗口和指标。
