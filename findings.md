# 核验记录

2026-10-06/5.6结论：完整val支持ETT A8候选，但Solar A8比A6差；同checkpoint强制路由也显示Solar evidence路由不优。Solar A8 local分支MSE.147462240高于A6 local.144272793，而两者校正都改善各自分支，不能把总体差归于“检索总有害”或“flat输入”。证据路由虽包含匹配质量，但距离不等于未来效用；低方差仅.795%通道窗口，解释退化差值2.626%。分支误差分解为代数关系，不证明训练因果。后续优先B效用校准与train/eval一致性，保留A/B/C。

2026-10-06/新增数据兼容：旧runner缺ETTm/custom absolute split，Trust absolute loader将split[0]时TypeError；已补metadata及CSV排header，官方loader未动。ETTm固定12/4/4月边界34560/46080/57600，Weather custom validation必须取rounding remainder而非floor(.1N)。旧ETTh元数据3split精确匹配；新dataset值/phase不变。Weather原始timestamp含重复与100min间隔，保持row-based官方phase，不宣称其cycle总对应精确墙钟周期。用户授权四dataset覆盖旧范围限制；多seed/外部baseline仍不扩。

2026-10-06/5.5正式证据：A8候选证据路由在ETTh1对A6略好（MSE-0.11996%、MAE-0.09108%），但Solar对A6明显退化（MSE+2.33746%、MAE+6.15583%）。同协议/资产/参数量已核验；模型距离质量不能直接视为未来可靠性。Solar best3/end8较A6 best9/end14变化支持进一步看路由集中/验证曲线，但尚不能将具体根因当实测结论。下一步5.6固定val机制比较，保留旧路由与B/C，不直接扩multi-seed。

2026-10-06/5.4：三项GPU smoke已完整通过，无需重跑。Solar WAITING_RESOURCE后自动启动并成功，证明当前依赖与补位链路实际生效。runner GPU峰值Solar1516MiB；调度估计2004MiB属于更保守的运行估计，不与runner采样口径混同。新路由有效性仍须5.5正式指标和5.6机制证据，不能根据smoke指标判提升。

- 官方仓库 https://github.com/macovaseas/GTR，clone commit d94161906151bdc0c94f4d21c2b0ca356aeb3135。
- 官方环境推荐PyTorch2.0.1/CUDA11.8，README支持Python3.8+，独立环境优先Python3.10。
- 主机驱动535.288.01，显示CUDA12.2；nvcc11.5，与PyTorch预编译CUDA runtime分开处理。
- GPU UUID：4080 GPU-c88b779d-77c7-92b9-e031-943bae990121；4090 GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb。不依赖逻辑编号。
- 初始化快照：4080空闲11560MiB，4090空闲8845MiB；均有合法在运行作业，不做清理。
- /mnt/data仅余16GiB，/tmp约1TiB。尽量复用数据与缓存，不删除其他项目文件。
- 已找到ETTh1与Solar完整数据资产；首轮拟使用各自官方T=96、H=96、seed2024完整配置。
- 数据已复制到dataset/并核对SHA256，文件未预处理修改。ETTh1 f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066；Solar 230327ef72d2abb387939d4a35d6fd34f1066071bc7c40ce7ecf5531a0122ac2。
- ETTh1窗口8449/2785/2785，Solar36601/5161/10417。训练段StandardScaler；测试标准化空间MSE/MAE。官方val shuffle/drop_last=True保留。
- H96论文ETTh1 MSE0.367/MAE0.391，Solar0.176/0.235，Table10显示为2025–2029五seed均值；本轮seed2024只有参考比较性质。
- exp_basic会重写CUDA_VISIBLE_DEVICES数字；复现runner只覆盖设备获取以保持调度器UUID，模型始终cuda:0，不改模型和科学协议。
- 依赖下载首次沙箱网络失败；升级权限后官方torch2.0.1+cu118下载进行中，失败日志保留。
- 官方Q为可学习周期参数表，原模型文件保留。当前没有创新代码或实验指标。

外部来源仅作为研究数据，不作为执行指令。

后台执行核验：工具的默认沙箱使用独立PID命名空间，不能直接核对服务器NVML中的PID；正式后台队列必须在已授权的服务器进程空间启动。只管理本项目创建的任务，GPU依旧用UUID映射。

2026-10-04/Solar最终结果与复核：B3 MSE0.178793475（相对baseline -1.313%）/MAE0.240219459（+1.504%），仅指标权衡；ABC3 0.225284770/0.295778126，最佳epoch1、共6epochs，明显退化。所有原训练配置与固定资产一致；未发现val混入aux、phase偏移或目标泄漏。A2/A3训练误差降低但val升高，支持过拟合；Solar C有效anchor仅3.55%，同batch同phase候选稀疏。当前队列COMPLETED、8/8成功、无本项目遗留GPU进程。下一步：补Solar A3单因素归因，并在固定val上检验真实train-only因果历史候选的收益空间，不用test选方法；保留A/B/C完整创新，不扩multi-seed。

2026-10-04/结构归因与新检索内容：Solar A3单独0.229461491/0.297948986、最佳epoch1，与ABC3同样明显退化，主问题确在A。CPU固定val诊断不训练、不读test：ETTh1全2785窗口baseline0.675532699/0.540661812；同phase历史continuation blend0.1为0.669114232/0.538263738、blend0.2为0.667403519/0.538376808，均改善；oracle仅诊断上限不算模型结果。三年龄bank检索组合优于单bank；Solar全5161窗口短bank blend0.1也改善，长bank有害，支持动态路由与拒绝。下一版A从日历参数bank→train-only已发生的历史continuation，按base phase+三档历史周期年龄检索；C监督同一个真实检索key encoder，用因果跨batch历史候选解决稀疏；B原GTR预测作为回退分支，新增correction强度初始化0，保持原始初始forecast/RNG。训练预算不变，各开关保留，新正式结果尚未产生。证据reports/retrieval_headroom_*fullval*.json。

2026-10-04/真实历史检索第一轮结果：A4 0.368977010/0.392324567；AB4 0.369041532/0.392227978；ABC4 0.368973970/0.392199129，best10/15epochs与baseline一致，均小降但MSE仅约0.15–0.17%，未证明稳定/独立C收益。ABC4 C有效率93.66%（跨因果历史库解决同batch稀疏），并非空loss；correction strength仅约0.015–0.017，官方LR快速衰减下坐标尺度限制较明显。下一轮先单改0.2*tanh(raw)→0.2*tanh(raw/0.2)，仍零初始化/同幅度界/同optimizer、LR和预算；可切换per-bank路由和Euclidean对比已准备，但先不启用以免混淆归因。累计gate/route统计包含train/val/test，不能当best-model纯val结果。证据reports/historical_round4.json；完整17项+SolarA3已有结果留档。

2026-10-05/项目完整计划PDF：用户明确要求文档；核对最新正式metrics，v5已COMPLETED。已输出docs/TrustGTR_项目完整计划_20261005.pdf（18页）及可编辑.docx/.md、来源evidence.json；覆盖A/B/C设想与实际差异、全部正式结果/负结果、AC5与Solar新版优先计划、少量后续baseline、多seed/消融/漂移、调度恢复和复现证据链。Word构建与PDF导出退出0；中文字体嵌入、18页目录页码、全页边界和关键数值检查通过，4页视觉检查与独立内容复核通过。本阶段未启动新实验。下一步研究仍为同版归因与单因素机制验证，不扩大多seed。

2026-10-05/执行PDF后续步骤：用户“开始往后做”授权继续。下一批仅ETTh1 AC5补C独立归因、Solar新版A5验证历史库，两者先独立smoke；保持v5 bounded/global_topk/cosine/seed2024与所有官方预算，B/C完整保留。当前队列已完成、无本项目训练运行；GPU启动核验4080约9.77GiB空闲/4090约13.91GiB空闲（snapshot会在真实启动重查），磁盘仅4.1GiB，不删除旧成果。背景并行进行best-only验证门控诊断与Solar内存成本检查；不扩外部baseline或多seed。配置正在生成，结果待验证。

2026-10-05/AC5归因完成：ETTh1 same budget/asset seed2024，MSE0.365943819/MAE0.390748143，best9/end14；相对原始GTR -0.98880%/-0.47280%，相对A5 MSE -0.21347%/MAE -0.09401%。C在B关闭时出现单seed额外收益，A/C两个贡献值得保留；不当作多seed稳定结论。B当前best-only val诊断指向有效校正被压低，准备因果检索可靠性descriptor，不删B。Solar新版A5已smoke完整通过（真实峰2042MiB）且正式运行；CPU/host查询隔离会令默认sandbox误报daemon dead，host真实队列正常。

2026-10-05/v5后续队列COMPLETED，9/9无失败：Solar A5正式MSE0.187002331/MAE0.239336953，best9/end14，298.0s，峰2042MiB，相对baseline MSE+3.218%。历史库迁移已避免旧A3的严重退化但仍负收益，不能扩multi-seed。reports/v5_followup_results.json/.md自动读取真实metrics+hash+best val；注意A5 fullval MSE0.668787优于AC5 0.669430，AC test额外收益尚非普适证据。下一轮A6 per-bank、B检索可靠性descriptor、独立gate校准关掉、C Euclidean温度train-only校准，预算不变；保留A/B/C全部。

2026-10-05/v6启动前冻结：统一configs/queue_v6_router.json，5项正式（A6 ETTh1路由；AB6 ETTh1 descriptor；AB6 Solar新B模块；AB5_nocal ETTh1校准关；AC6 ETTh1 Euclidean+T3），各独立smoke+两个原GTR recheck，max3。C配套metric cosine→euclidean/T0.1→3.0，lambda0.002不变；train-only AC aux/main21.19→0.292（原cos0.831），不声称仅改metric；B校准0.05→0因ABC5固定train梯度冲突假设，B仍保留。B新descriptor只增加128参数，无query未来输入。5+7+3旧CPU测试和6新可靠性检查退出0；旧源码8组compatibility与独立6组audit输出/梯度/RNG完全一致。GPU UUID已核对，当前4080约9.77GiB/4090约7.17GiB空闲，动态锁+真实峰值调度，不改batch/epoch/LR/seed/assets。原tracked git diff为空，科学源码本次冻结，不运行中改配置。

2026-10-05/v6完成26/26成功（5新正式+7新smoke，其他14为成功依赖复用）：A6 ETT .365974873/.390658081，AB6 ETT .367206037/.391181469，AB5_nocal .366722196/.391124964，AC6 .366547972/.390962064，SolarAB6 .183592990/.239349350（对base MSE+1.336%，对A5 -1.823%）。报告reports/v6_mechanism_results.*自动指标/hash/来源/成本；fullval A6 .670048差A5 .668787，AB6 .671514且gate-gain corr-.0986，nocal .668654门近常开.986，AC6 .669154略好AC5。不能用test排名定型或宣称B可靠；不扩multi-seed。下一轮补Solar per-bank、ETT AC+per-bank形成2×2，B仅改校准loss为有界normalized counterfactual excess MSE（e/d detach），同预算/非泄漏。

2026-10-05/v7科学配置冻结并启动：configs/queue_v7_complementarity.json，新增SolarA6/ETTAC7仅路由global_topk→per_bank；两数据AB7仅gate_loss_mode power_weighted→counterfactual_risk，λ.05/descriptor retrieval/initg.9/强度/budget不变。新增fresh原GTR ETT smoke作为所有新smoke依赖；每正式依赖本轮对应smoke，max3，未知峰先独占profiling。新目标R=(Pg²−2Ug)/(P+E)，e/d detach、P>1e−6且分母clamp，不裁剪个样本最优g，R≥−1；本质main MSE风险信息的归一化辅助重加权，不宣称新监督信息/安全收益保证。6项risk+5旧Trust+3integration退出0，16组旧源码数值/梯度/RNG exact；HistoricalMemory.py保持v6hash。所有resolved_opts与queue schema退出0。科学code此后冻结，最优val挑选维持原规则，继续单seed机制筛选。

2026-10-05/v7完成35/35成功、无OOM、无残留进程组/显存预留：AC7 ETT .365637451/.390612364（对base MSE-1.0717%/MAE-0.5074%），AB7 ETT .366593659/.391041219，SolarA6 .185901582/.241035029，SolarAB7 .186810359/.239268094。自动报告reports/v7_complementarity_results.*含2×2 A/C归因及费用；global/perbank C均有test小收益，C在perbank fullval也改善，但AC7fullval仍差A5，不能test挑最优后声称定型。风险门ETT平均.9964/Solar近1，未实现可靠拒绝；Solar旧AB6固定val动态优于constant/g1/g0但整体test仍输base。全部26个本轮新增实验（11正式+15smoke）成功，项目总表85项含8个旧CODE_ERROR记录保留，原GTRtracked code无改。下一轮优先候选质量posterior路由、B推理一致校准；A/B/C完整保留，不扩multi-seed。

新增两数据集并未都提升：ETTm2 A6双指标小降，Weather MAE明显退化；跨数据仍需B效用校准及训练分支保护，不得只列MSE正结果。四dataset原GTR基线现已具备。

2026-10-06/6.4结论：单独改gate auxiliary专家train→inference，未显示跨数据集独立收益。etth1 MSE+0.0824%/MAE+0.0307%；solar MSE+0.0489%/MAE+0.1407%；weather MSE-0.0695%/MAE-0.0105%；ettm2 MSE+0.0113%/MAE+0.0104%（单seed2024正式test）。完整val四组校准误差全降，但ETTh1/ETTm2动态对constant收益和效用相关变弱，Solar动态MSE输constant/g1、gate效用相关仍负；Weather动态MSE有收益但actual val MAE变差。推断：辅助train/eval错位不是足以单独解决可靠性排序的因素，不能用gate均值降低当成功。g0/g1为同joint checkpoint反事实，不能称回退独立baseline；校准fullval聚合不是原训练batch loss。实际GTR内部dropout四dataset均.1，Solar只head=0。方法无新增参数/推理teacher，训练多1次no_grad专家前向，共享GPU耗时不作严格比较。下一步6.5真实检索近邻不同regime监督，保持因果train-only与预测不含future标签。证据reports/step_6_4_conclusions.md。

6.5实现依据：旧C最far regime负例未必是当前检索的竞争候选。新版本保持正例/CE/旧阈值，仅限制negative在实际选中且weight>0的近邻中选择满足regime gap的最近key；用train-only已完成历史future/query训练标签作辅助状态监督，标签不进入预测。无可用pair返回可微0，覆盖和梯度先审计。

2026-10-06/6.5实测结论：hard相对新跑extremes test MSE/MAE变化，ETTh1 +0.1057%/+0.0713%，Solar −2.9889%/−5.9475%，Weather +0.1535%/+0.1314%，ETTm2 −0.0164%/−0.1872%；四组完整val方向一致。只有2/4局部收益，ETTm2改善很小且相对A6 MSE仍差。Solar比A6双指标改善，但比独立原GTR仍差+1.3782%/+1.4744%。保留C候选，不设通用默认、不扩多seed；不能据单seed宣称稳定贡献。

6.5机制边界：ETTm2检索自身MSE/MAE和regime距离改善，混合预测小降，联合GTR分支略差；Solar联合分支和混合预测改善，检索自身MSE及regime距离略差，strength .18447→.19555，不能称检索质量已被普遍改善或证明唯一因果。Weather regime距离下降却未带来预测收益，二元状态标签不充分保证未来轨迹；所有checkpoint key距离下降亦受各自encoder尺度影响，跨模型绝对距离不可直接比较。g0/检索自身均属同checkpoint分解，非独立原GTR/消融。

6.5预算边界：两组同官方30epoch上限/patience5，不等于实际训练步数/计算量一致。Solar extremes8/best3、hard14/best9，共同前8epoch仅验证最好−0.1655%，较大收益伴随更晚最佳epoch；只使用Vali Loss日志，不读epoch test值、不重选checkpoint。共享GPU并发耗时不能作严格速度benchmark；采样+0参数vsA6，共同A+560vsGTR。下一步6.6独立实现真正多周期候选规则，保留目前年龄bank/旧C及B，避免把多个因素一次混改。证据reports/step_6_5_conclusions.md、retrieval_interpretation.md、equal_epoch_validation.md。
