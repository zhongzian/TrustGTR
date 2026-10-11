# 进度

2026-10-06/6.4.1–6.4.3：新gate_calibration_context train(default)/inference实现冻结models/TrustGTR.py hash59fef362…，仅lazy辅助专家/descriptor no_grad eval，gate辅助grad保留；main forecast/dropout/RNG/gradient路径精确不变，无新参数。新12测试通过、46归档组与18legacy组合兼容；旧baseline5/history3/router6/risk6/reliability6均成功。首轮12子例因重建乘法结合顺序ulp差异，修夹具按正式顺序、未放宽容差，日志失败摘要诚实标注非rawstdout。queue准备alloff继承historical会验证失败的问题在启动前审查发现，改独立mode并记录，17resolved全部通过后才启动。source/config/DAG/资产/预算hash审计reports/step_6_4_protocol_audit.*完成；统一队列configs/queue_step_6_4_ab9.json仅17新ID，9smoke8formal，配对唯一context因子、共同A6 perbank/retrievalgate/powerweightedλ.05/init.9/Coff，原预算不变。启动bash scripts/queue.sh start退出0，PID1792390，日志runtime/step_6_4_ab9/queue.log；GPU4080空闲16078MiB/4090空闲1620MiB，UUID选择实时检查。

2026-10-06/5.6 complete、6.6 complete、6.7启动：完整val机制诊断4归档模型CPU2threads/batch16，约64秒退出0、RSS1.86GiB、GPU0；smoke与全量重建误差0，科学源/资产/历史库/checkpoint严格匹配。ETT A6 .670047847/.538180456→A8 .668913047/.537892757；Solar A6 .141778822/.211880218→A8 .145034156/.224738283，误差增量主要体现于已joint训练GTR branch（只是代数分解非因果）。近恒定输入仅.795%通道窗口、贡献差值2.626%，不是整体退化解释；距离证据与Solar单库效用弱负相关，长期库近而平均有害。结论保留ETT A8候选，新增数据先GTR/A6，不通用化A8、不删B/C。报告step_5_6_mechanism_*。

新增Weather52696×21与ETTm269680×7只读副本16.13MiB，来源zip/CRC/SHA与官方loader6split检查通过；原native天气时间重复/gap不修正以保协议。修runner仅补ETTm/custom绝对split记录和CSV排header行数，原ETTh/Solar分支保留；summarizer从指定SUCCESS baseline metrics动态读缺失对照值，不猜指标。新metadata测试首轮fixture误用label48，与官方0不符；修测试夹具为0后3/3退出0，原ETTh资产完全一致、两新dataset输入/目标/绝对phase核验正确，失败日志保留，正式参数从未改48。队列configs/queue_four_datasets_v1.json仅8新ID（4smoke/4formal）、首次smoke未知峰、formal实测估计、max3/依赖/重复准备/源码hash/原预算审计通过，启动bash scripts/queue.sh start退出0。磁盘启动前1.4GiB；只读数据小、新结果预计约0.5GiB，不删旧成果。GPU UUID核对：4080空闲11479MiB，4090空闲1620MiB，按实时资源补位。

2026-10-06/5.6及新增范围开始：用户“开始5.6，后续实验再多加两个数据集”明确授权，旧1–2范围由四数据集覆盖，仍单seed/先smoke/预算不变。并行：归档A6/A8 CPU固定完整val机制（无test/no optimizer），Weather+ETTm2资产及官方脚本核验，最小新增GTR/A对照队列准备。科学代码冻结不改；磁盘仅1.8GiB，不删除旧成果，新增数据/结果体积先核算，必要复用公开资产。

2026-10-06/5.5完成：用户询问是否跑完；核对host真实daemon与两个正式status/metrics，均SUCCESS/退出0、OOM0；队列22/22 SUCCESS、COMPLETED，running/pending/retrying0。ETTh1 A8 MSE.365535855/MAE.390302271，对原GTR -1.09918%/-0.58638%，对A6 -0.11996%/-0.09108%；14epochs/best9，40.90s，峰632MiB。Solar A8 .190246955/.255872726，对原GTR +5.00882%/+8.11863%，对A6 +2.33746%/+6.15583%；8epochs/best3，263.01s，峰2004MiB。Solar较早结束由原patience5触发，非缩预算。核验同资产、正式30epoch上限与关键配置一致、相对A6无新增参数；checkpoint与metrics哈希落盘reports/step_5_5_formal_results.json/.md，自动总表90条更新。主线转到5.6待执行，不扩多seed、不按ETT test最低值定型。

2026-10-06/5.4完成验收：用户要求“继续完成5.4”。真实后台核验发现显存释放后Solar A8 smoke已自动SUCCESS（11:14:45–11:15:00，4090 UUID），无重复启动。3项smoke均1epoch/3optimizer steps，7项链路检查全true，状态与resource exit0，checkpoint/best+last存在并保存SHA256，科学hash与冻结匹配，OOM次数0；reports/step_5_4_smoke_results.json落盘。runner峰值alloff444MiB/ETT628MiB/Solar1516MiB，耗时11.65/5.73/14.73秒；这些小配置指标不用于正式结论。自动总表90条已更新，原tracked diff无改。当前5.5 ETT正式SUCCESS、Solar正式RUNNING，保持既有队列继续执行。

2026-10-06/5.2完成、5.4启动：6项新路由+5项基线+3项历史集成检查全部退出0，独立只读审查无阻塞（reports/v8_router_review.md）。科学文件hash冻结reports/v8_source_freeze.json；原tracked git diff为空。启动命令 bash scripts/queue.sh start configs/queue_v8_evidence.json 退出0，daemon PID1477417，关闭模块回归smoke真实RUNNING；新A8各smoke与正式均按依赖排队，未称为正式结果。启动GPU快照：4080 UUID-c88b…空闲2161MiB；4090 UUID-a3af…空闲1030MiB，调度器按实时资源动态补位，不改batch。编号索引更新当前5.4，5.5已排队。

2026-10-06/步骤编号恢复：按用户要求新增 docs/步骤与进度.md，1.1–7.5固定编号，AGENTS与task_plan同步。中断后核验：A8新5任务run_dir均不存在、状态NOT_LAUNCHED；runtime COMPLETED/35成功属于旧v7，不能称v8已完成。当前5.2代码与CPU验收；5.3配置审计通过，5.4/5.5尚未开始。CPU验证脚本4窗口smoke已落盘成功；B固定train64归档源码梯度诊断完成，辅助与main gate梯度近同向（ETT cosine.9861/Solar.9985），不称B新方法有效。

2026-10-05/v8开始：用户“开始做下一步”授权继续，要求以后所有步骤编号。先仅改变A6路由per_bank→per_bank_evidence：库后验=查询prior×库内topk逆距离均值归一化；保留A/B/C与原GTR，不新增参数/改训练协议。并行实现与CPU边界检查、最小统一队列准备、B训练/推理校准只读审计。GPU真实UUID核验：4080空闲1263MiB、4090空闲9820MiB，无项目残留训练；磁盘可用6.2GiB。科学源码与smoke通过前不启动正式训练。

2026-10-04：收到实际执行授权，停止铺设外部baseline。完成硬件、磁盘、Conda、编译器与调度工具检查，未发现sbatch/qsub。官方仓库clone成功，初始git status为空，commit已核验。项目规范与计划建立。独立环境、复现入口与队列构建中。

独立Conda Python3.10.13创建成功；ETTh1与Solar数据已复制并SHA256核验，正式与smoke独立ID配置已生成。PyTorch官方wheel在升级网络权限后下载中，首次沙箱失败日志保留。队列6项CPU模拟测试通过，覆盖补位/OOM/失败/恢复/停止，runtime/queue_tests_20261004_210247/summary.json。用户“继续完成”确认继续当前目标。

复现入口与队列核验：官方源文件未修改，编译检查通过。修正OOM空闲显存参照点为启动时值、峰值字段对接、失败smoke依赖停止；代码快照包含未跟踪新文件，恢复指纹限于科学行为相关文件，避免无关队列更新阻止恢复。新增检查首次测试夹具漏刷新模拟GPU（IndexError），修复测试后8项全部通过：runtime/queue_tests_20261004_211638/summary.json。正式配置仍为官方T96/H96/seed2024/30epochs，smoke独立1epoch少量batch。

21:34：官方torch wheel分段断点下载完成，完整SHA256 a7a49d...ebf7b匹配官方index；从已核验本地wheel安装到独立Conda环境，科学依赖版本不变。原pip中断退出2的完整日志保留。

21:38：环境安装退出0、pip check通过；torch2.0.1+cu118/CUDA11.8可用，torch逻辑0=4090、逻辑1=4080确认反向映射。首次两个smoke因调度脚本遮蔽stdlib queue在import sklearn时退出1；已重命名为experiment_queue.py，保留旧失败记录，v2新ID重新smoke，正式任务尚未训练。

2026-10-04/首个正式基线成功：ETTh1 T96/H96 seed2024，MSE=0.369598418、MAE=0.392604440；官方30epoch上限、patience5触发第15epoch早停，最佳epoch10。论文H96五seed均值0.367/0.391，单seed差异分别+0.002598418/+0.001604440，不能当统计偏差结论。两个smoke全部检查通过；Solar正式运行中。证据：runs/gtr__etth1__forecast__t96_h96__seed2024__20261004_214520/metrics.json。下一步：B可靠性收益监督单因素对照，A/C完整模块保留。

创新模块与独立入口完成：models/TrustGTR.py、scripts/run_trust_experiment.py；正确性检查3/3通过（logs/trustgtr_validation.log）。生成configs/queue_trust_screen.json，同一总队列包含已完成基线（不重复启动）、baseline重检、B/A/AC/ABC独立smoke及同协议正式任务，最多3。前一未启动配置归档；当前ID时间戳20261004_220443。

2026-10-04/第一轮结果与诊断：B MSE0.367498815(-0.5681%)、MAE0.392714351(+0.0280%)；A0.378975898/0.399294019、AC0.380971253/0.397570848、ABC0.380604565/0.393894643均未提升。A验证平均路由[0.579,0.167,0.254]，长bank幅度高于基础bank；C活动anchor约87%并非空loss。固定256训练窗口诊断（不做optimizer step）：main编码器梯度norm0.004299，对比norm0.904512，乘0.01后为main2.1倍，夹角cos0.023（近正交，不能称负向冲突）。第二轮不删模块：A改base+按base相位零均值的长周期残差；新增参数初始化fork_rng保留原始训练随机流（默认旧行为False，v2显式True）；C权重0.01→0.002；所有变化已写新ID配置。新4项正确性检查通过，含初始预测与RNG精确恢复，预算/seed/资产不变。证据reports/round1_diagnostics.json、round1_gradient_diagnostic.json、logs/trustgtr_validation_v2.log。

2026-10-04/第二轮负结果→第三轮：B2 0.369723111/0.393384635（MSE+0.034%、MAE+0.199%）；A2 0.383215576/0.396112263、AC2 0.383070201/0.395521998、ABC2 0.383095205/0.395403773，均未改善。A2 train loss约0.328而val约0.700，密集长bank仍过拟合，基础相位残差约束不够。第三轮A采用同样三周期、按周期位置检索水平/趋势/幅度系数，与基础相位模板共享；长记忆参数6216→777，原512维主模型不裁剪，原因是泛化而非OOM。B改forecast agreement、descriptor detach、按delta能量加权oracle误差，弱贡献样本不应与高贡献样本等权；lambda_gate仍0.05。C完整保留、权重0.002、输入状态编码器仍训练真实路由。5项正确性检查通过；每项新ID先smoke，正式预算和固定资产不变；无多seed与外部baseline扩展。

2026-10-04/第三轮结果：B3 ETTh1 0.370674849/0.393580794；A3 0.385246754/0.397886038；AC3 0.385277301/0.397996336；ABC3 0.385066092/0.397727191，均未改善，降参数不能解决适配问题。停止重复扩大ETTh1搜索；将第三版B/ABC原样检测在最初已选且已有正式基线的Solar（T96<日周期144），仅检查本来1–2数据集范围中的检索适用性，不增加seed/预测长度/训练预算或调整方法超参。新ID20261004_224334，配置queue_solar_screen.json；若仍无效，结论是当前设计证据不足，需要架构重设计而非多seed掩盖负结果。

2026-10-04/Solar最终结果与复核：B3 MSE0.178793475（相对baseline -1.313%）/MAE0.240219459（+1.504%），仅指标权衡；ABC3 0.225284770/0.295778126，最佳epoch1、共6epochs，明显退化。所有原训练配置与固定资产一致；未发现val混入aux、phase偏移或目标泄漏。A2/A3训练误差降低但val升高，支持过拟合；Solar C有效anchor仅3.55%，同batch同phase候选稀疏。当前队列COMPLETED、8/8成功、无本项目遗留GPU进程。下一步：补Solar A3单因素归因，并在固定val上检验真实train-only因果历史候选的收益空间，不用test选方法；保留A/B/C完整创新，不扩multi-seed。

2026-10-04/结构归因与新检索内容：Solar A3单独0.229461491/0.297948986、最佳epoch1，与ABC3同样明显退化，主问题确在A。CPU固定val诊断不训练、不读test：ETTh1全2785窗口baseline0.675532699/0.540661812；同phase历史continuation blend0.1为0.669114232/0.538263738、blend0.2为0.667403519/0.538376808，均改善；oracle仅诊断上限不算模型结果。三年龄bank检索组合优于单bank；Solar全5161窗口短bank blend0.1也改善，长bank有害，支持动态路由与拒绝。下一版A从日历参数bank→train-only已发生的历史continuation，按base phase+三档历史周期年龄检索；C监督同一个真实检索key encoder，用因果跨batch历史候选解决稀疏；B原GTR预测作为回退分支，新增correction强度初始化0，保持原始初始forecast/RNG。训练预算不变，各开关保留，新正式结果尚未产生。证据reports/retrieval_headroom_*fullval*.json。

2026-10-04/真实历史检索第一轮结果：A4 0.368977010/0.392324567；AB4 0.369041532/0.392227978；ABC4 0.368973970/0.392199129，best10/15epochs与baseline一致，均小降但MSE仅约0.15–0.17%，未证明稳定/独立C收益。ABC4 C有效率93.66%（跨因果历史库解决同batch稀疏），并非空loss；correction strength仅约0.015–0.017，官方LR快速衰减下坐标尺度限制较明显。下一轮先单改0.2*tanh(raw)→0.2*tanh(raw/0.2)，仍零初始化/同幅度界/同optimizer、LR和预算；可切换per-bank路由和Euclidean对比已准备，但先不启用以免混淆归因。累计gate/route统计包含train/val/test，不能当best-model纯val结果。证据reports/historical_round4.json；完整17项+SolarA3已有结果留档。

2026-10-05/v5单因素优化：已通过旧5项Trust回归和3项历史集成检查（含raw/bounded两种坐标初始forecast及RNG精确恢复）。新队列configs/queue_historical_v5.json，ID时间戳20261005_110620；本轮仅启用strength_coordinates=bounded，global_topk与cosine保持v4。per-bank/router、Euclidean-C代码具备开关且已测试，但本轮未启用；避免多项同时改变导致无法归因。官方正式配置diff仅model/model_id；4080启动前仅55MiB空闲，4090约16GiB空闲，队列UUID选择可用设备并继续动态补位。

2026-10-05/项目完整计划PDF：用户明确要求文档；核对最新正式metrics，v5已COMPLETED。已输出docs/TrustGTR_项目完整计划_20261005.pdf（18页）及可编辑.docx/.md、来源evidence.json；覆盖A/B/C设想与实际差异、全部正式结果/负结果、AC5与Solar新版优先计划、少量后续baseline、多seed/消融/漂移、调度恢复和复现证据链。Word构建与PDF导出退出0；中文字体嵌入、18页目录页码、全页边界和关键数值检查通过，4页视觉检查与独立内容复核通过。本阶段未启动新实验。下一步研究仍为同版归因与单因素机制验证，不扩大多seed。

2026-10-05/执行PDF后续步骤：用户“开始往后做”授权继续。下一批仅ETTh1 AC5补C独立归因、Solar新版A5验证历史库，两者先独立smoke；保持v5 bounded/global_topk/cosine/seed2024与所有官方预算，B/C完整保留。当前队列已完成、无本项目训练运行；GPU启动核验4080约9.77GiB空闲/4090约13.91GiB空闲（snapshot会在真实启动重查），磁盘仅4.1GiB，不删除旧成果。背景并行进行best-only验证门控诊断与Solar内存成本检查；不扩外部baseline或多seed。配置正在生成，结果待验证。

2026-10-05/v5后续启动：configs/queue_v5_followup.json，新ID20261005_155314；ETTh1 AC5仅关B，Solar A5保留原Solar协议，2项formal相对对应baseline official_args除model/model_id无差异。3项历史集成+7项历史库CPU检查退出0，新manifest schema/uniqueID/dependencies验证退出0，启动nohup总队列退出0，PID1395116，先各配置smoke；formal复用本轮smoke峰值，最多3。源代码本轮未修改，预计Solar历史库/activation峰值约1.8–2.2GiB仅调度观察，真实smoke为准。

2026-10-05/AC5归因完成：ETTh1 same budget/asset seed2024，MSE0.365943819/MAE0.390748143，best9/end14；相对原始GTR -0.98880%/-0.47280%，相对A5 MSE -0.21347%/MAE -0.09401%。C在B关闭时出现单seed额外收益，A/C两个贡献值得保留；不当作多seed稳定结论。B当前best-only val诊断指向有效校正被压低，准备因果检索可靠性descriptor，不删B。Solar新版A5已smoke完整通过（真实峰2042MiB）且正式运行；CPU/host查询隔离会令默认sandbox误报daemon dead，host真实队列正常。

2026-10-05/v5后续队列COMPLETED，9/9无失败：Solar A5正式MSE0.187002331/MAE0.239336953，best9/end14，298.0s，峰2042MiB，相对baseline MSE+3.218%。历史库迁移已避免旧A3的严重退化但仍负收益，不能扩multi-seed。reports/v5_followup_results.json/.md自动读取真实metrics+hash+best val；注意A5 fullval MSE0.668787优于AC5 0.669430，AC test额外收益尚非普适证据。下一轮A6 per-bank、B检索可靠性descriptor、独立gate校准关掉、C Euclidean温度train-only校准，预算不变；保留A/B/C全部。

2026-10-05/v6启动前冻结：统一configs/queue_v6_router.json，5项正式（A6 ETTh1路由；AB6 ETTh1 descriptor；AB6 Solar新B模块；AB5_nocal ETTh1校准关；AC6 ETTh1 Euclidean+T3），各独立smoke+两个原GTR recheck，max3。C配套metric cosine→euclidean/T0.1→3.0，lambda0.002不变；train-only AC aux/main21.19→0.292（原cos0.831），不声称仅改metric；B校准0.05→0因ABC5固定train梯度冲突假设，B仍保留。B新descriptor只增加128参数，无query未来输入。5+7+3旧CPU测试和6新可靠性检查退出0；旧源码8组compatibility与独立6组audit输出/梯度/RNG完全一致。GPU UUID已核对，当前4080约9.77GiB/4090约7.17GiB空闲，动态锁+真实峰值调度，不改batch/epoch/LR/seed/assets。原tracked git diff为空，科学源码本次冻结，不运行中改配置。

2026-10-05/v6完成26/26成功（5新正式+7新smoke，其他14为成功依赖复用）：A6 ETT .365974873/.390658081，AB6 ETT .367206037/.391181469，AB5_nocal .366722196/.391124964，AC6 .366547972/.390962064，SolarAB6 .183592990/.239349350（对base MSE+1.336%，对A5 -1.823%）。报告reports/v6_mechanism_results.*自动指标/hash/来源/成本；fullval A6 .670048差A5 .668787，AB6 .671514且gate-gain corr-.0986，nocal .668654门近常开.986，AC6 .669154略好AC5。不能用test排名定型或宣称B可靠；不扩multi-seed。下一轮补Solar per-bank、ETT AC+per-bank形成2×2，B仅改校准loss为有界normalized counterfactual excess MSE（e/d detach），同预算/非泄漏。

2026-10-05/v7科学配置冻结并启动：configs/queue_v7_complementarity.json，新增SolarA6/ETTAC7仅路由global_topk→per_bank；两数据AB7仅gate_loss_mode power_weighted→counterfactual_risk，λ.05/descriptor retrieval/initg.9/强度/budget不变。新增fresh原GTR ETT smoke作为所有新smoke依赖；每正式依赖本轮对应smoke，max3，未知峰先独占profiling。新目标R=(Pg²−2Ug)/(P+E)，e/d detach、P>1e−6且分母clamp，不裁剪个样本最优g，R≥−1；本质main MSE风险信息的归一化辅助重加权，不宣称新监督信息/安全收益保证。6项risk+5旧Trust+3integration退出0，16组旧源码数值/梯度/RNG exact；HistoricalMemory.py保持v6hash。所有resolved_opts与queue schema退出0。科学code此后冻结，最优val挑选维持原规则，继续单seed机制筛选。

2026-10-05/v7完成35/35成功、无OOM、无残留进程组/显存预留：AC7 ETT .365637451/.390612364（对base MSE-1.0717%/MAE-0.5074%），AB7 ETT .366593659/.391041219，SolarA6 .185901582/.241035029，SolarAB7 .186810359/.239268094。自动报告reports/v7_complementarity_results.*含2×2 A/C归因及费用；global/perbank C均有test小收益，C在perbank fullval也改善，但AC7fullval仍差A5，不能test挑最优后声称定型。风险门ETT平均.9964/Solar近1，未实现可靠拒绝；Solar旧AB6固定val动态优于constant/g1/g0但整体test仍输base。全部26个本轮新增实验（11正式+15smoke）成功，项目总表85项含8个旧CODE_ERROR记录保留，原GTRtracked code无改。下一轮优先候选质量posterior路由、B推理一致校准；A/B/C完整保留，不扩multi-seed。

2026-10-06/编号状态更新：5.4关闭模块与ETTh1 A8 smoke已SUCCESS；5.5 ETTh1正式RUNNING(PID1481690)；Solar 5.4 WAITING_RESOURCE、正式PENDING依赖。资源不足保留原协议等待，未改配置，后续自动补位。

2026-10-06/6.7 smoke验收：新增Weather/ETTm2四项GTR/A6 smoke全部SUCCESS、28/28链路check通过，预算1epoch/3step、label_len0；固定slice与官方边界精确一致。峰值GTRWeather72/A6Weather262/GTRettm2132/A6ettm2816MiB，报告expanded_smoke_acceptance.json。当前正式训练后台进行，A6正式依赖对应baseline完成；源配置运行中不改，模型/预算/评估资产保持冻结。

2026-10-06/6.7 complete：host队列COMPLETED、8/8 SUCCESS/退出0、无OOM、running/pending/retrying0。Weather GTR .156770736/.201339573，A6 .156595722/.207328215（MSE-0.11164%、MAE+2.97440%），指标取舍；ETTm2 GTR .167993650/.251111001，A6 .166146219/.250517786（MSE-1.09971%、MAE-0.23624%），单seed双指标小改善。相同训练预算与资产逐项验收、正式无smoke缩水、原patience5早停；报告step_6_7_expanded_results.json/.md与总表98条已更新。下一研究步骤6.2，未启动新训练；四数据集范围保留，不扩多seed。

2026-10-06/步骤编号纠正：此前模块编号与实际执行顺序混用导致完成6.7后回6.2；用户指出后第6阶段一次性按执行顺序修正。映射旧6.1→新6.1、旧6.6→新6.2、旧6.7→新6.3、旧6.2→新6.4、旧6.3→新6.5、旧6.4→新6.6、旧6.5→新6.7。当前完成至新6.3，下一步新6.4。旧索引归档docs/步骤与进度_旧编号_20261006.md，历史ID/结果报告/log不改；所有实验状态和数值不变，未启动训练。

2026-10-06/6.4开始：用户明确授权。新增gate_calibration_context train(default)/inference独立开关，仅gate auxiliary专家/descriptor上下文改变；main dropout/main gate梯度/strength/optimizer/loss权重原样。共同固定A6 per_bank与retrieval descriptor/power_weightedλ.05/ginit.9/Coff，四数据集各control与inference配对（8formal+8smoke+freshalloffsmoke）。不用常开risk同时改目标，避免混因子。CPU回归、统一队列配置与fixedval gate诊断并行开发，未启动GPU。pre-sourcehash已落盘；磁盘1.2GiB，GPU4080空闲16078MiB/4090仅1620MiB，真实UUID核对、无项目残留。

2026-10-06/6.4.3 complete：9/9 smoke SUCCESS/退出0，63链路checks全true。Solar两项首次在共享4090仅1620MiB空闲OOM，清理各自进程/退避/资源改善后第二attempt均成功，累计OOM2不隐藏；无batch/协议变更、无代码/数据确定性错误。真实源码/17config哈希保持。报告step_6_4_smoke_acceptance.json及独立runtime_audit.*。6.4.4正式进行，6.4.5 ETT完整val显示inference aux校准小降但动态门排序和actual指标未优control，不能将gate均值下降算可靠拒绝成功。

2026-10-06/6.4.4–6.4.5 complete：8/8正式SUCCESS/退出0，4dataset完整val pair(ETTh1 2785/Solar5161/Weather5175/ETTm2 11425) CPU退出0、8模型重建误差0；inference相对train etth1 MSE+0.0824%/MAE+0.0307%；solar MSE+0.0489%/MAE+0.1407%；weather MSE-0.0695%/MAE-0.0105%；ettm2 MSE+0.0113%/MAE+0.0104%。队列17/17成功、COMPLETED，累计19attempt含2次smoke OOM退避恢复、max3；host最终核验无本项目残留进程/僵尸/显存预留。14科学源/17输入配置hash未变，所有run归档完整，resolved pair仅context不同、固定资产等GTR/A6。自动总表115条保留旧失败；report_step64.py与summarize_results.py退出0，正式/机制/成本/独立审计报告已落盘。4pair校准误差下降却无一致收益；Weather val MSE略降、MAE升，前三动态门都输自身g1，Solar还输constant。保留B context开关、不作通用默认、不扩多seed；下一步6.5待执行，A/B/C与四dataset范围保留。证据reports/step_6_4_conclusions.md、paired_results.*、gate_diagnostics.*、runtime_audit.*。

2026-10-06/6.5开始：用户授权“开始下一步”。原C按全库最远future regime选负例→新增contrast_sampling retrieval_hard，只从当前真实retrieval selected/weight>0且满足原regime gap的候选中选最近key负例；正例仍全因果非重叠候选regime最近，原两类CE/positive距离1/gap.25不变。对照regime_extremes精确保留旧行为；共同四dataset A6per_bank/bounded、Boff、C euclidean/T3/λ.002（明确共同协议），唯一paired因子sampling，不改变official seed/batch/预算/评估资产。17新任务拟按先smoke再正式后台max3，尚未启动GPU。风险：困难负例候选更少/有效率或下降，λ加权梯度必须审计；不为覆盖率静默放宽阈值。pre-code14源hash等6.4已保存reports/step_6_5_before_code.json；磁盘1.4GiB，4080UUID空闲7700MiB、4090空闲5304MiB，启动时重查。

6.5.1旧行为回归：baseline5/historical3/B9 inference12(含46归档配置)均退出0，logs/step_6_5_*_cpu.log。新C首11项检查通过并补cosine/空槽分支中；科学代码冻结前不排GPU。A6 best参数加载train64 CPU预诊断准备中，每chunk8仅计算梯度不优化、明确非官方batch梯度。共同euclidean/T3来自AC6，旧AC7实际cosine/.1；不能把与AC7的差异归因sampling。

2026-10-06/6.5.1–6.5.2 complete、6.5.3启动：新12项/旧5+3+12检查退出0、18冻结B9双源+46归档输出/各梯度/RNG exact；无新参数。fixedA6 train64 CPU预诊断退出0，每8微批非正式训练batch，hard有效率ETT92.86/Solar90.77/Weather91.67/ETTm294.87%，实际检索membership100%是设计验证非收益；phase/causal/nonoverlap违规0/aux非encoder梯度0/主预测exact。ETT weighted aux/main norm约1.025/cos−.114、Solar覆盖下降5.28个百分点为风险，λ/阈值不改。queue_step_6_5_ac10.json已冻结，timestamps20261006_214646/17ID/14science，重复prepare未覆盖/新增；共同euclidean/T3来自AC6设置、旧AC7cos/.1不作采样单因素参考。bash scripts/queue.sh start configs/queue_step_6_5_ac10.json退出0，launcher3429130、runtime独立、max3，先smoke后正式。科学hash Modelde07b0dd.../Memory60d848f2...冻结。原11项成功测试日志被追加12项运行覆盖→final原始stdout单独保存/首轮仅工具成功摘要注明非原stdout，两轮均无失败，未隐藏失败。诊断兼容旧AC6 4val窗口退出0/重建0；fullval预测后regime统计，不读test/不重选ckpt。证据reports/step_6_5_{protocol_audit,freeze_check,preflight_train64_20261006,sampler_implementation_audit}.*，logs/step_6_5_retrieval_hard_final_cpu.log。

6.5.3/6.5.4并行执行：每配置smoke通过即启动其正式，不等全部smoke，满足逐配置先smoke依赖。当前6smoke及ETT control正式SUCCESS，ETT hard/Solar control/Weather control正式RUNNING，3其余smoke WAITING_RESOURCE(未启动、非OOM)；OOM累计0。原3slots满，任务结束或显存改善自动补位，不修改科学配置。

6.5.3 complete：9/9 smoke、63checks通过，OOM0；3/8正式成功、其余按资源补位。6.5.5 ETTh1 fullval CPU退出0/两best重建0；hard比extremes正式双指标微差、完整val也微差，joint GTR branch与R误差均略升。不同已学encoder的key距离不直接可比，负例实际检索membership是设计验证，不是收益。四data配对继续，源/config/预算未改。

2026-10-06/6.5调度偏差：用户要求资源够即补位→生成器沿用6.4多加hard正式依赖extremes正式完成，实际无数据依赖（固定资产来自A6，ownsmoke已通过）。ETTm2hard结束后已观察第三slot空等≥76秒，Weatherhard DEPENDENCY阻塞而非显存不足；两卡空闲约5.1/5.3GiB。现有daemon仅启动时读取spec，无安全热reload；不中断现有训练条件下保留本轮队列，禁止手动重复同ID。未来生成器删除多余control依赖，只留真实资产/alloff/ownsmoke，旧生成器SHA快照归档，当前17config/14science/inputqueue保持。好：发现等待根因且未伪造为显存不足；坏：本轮造成排队延迟，阶段末记录实际时间范围。证据reports/step_6_5_scheduling_note.md。

2026-10-06/6.5.4–6.5.5 complete：8/8正式SUCCESS/退出0，4完整val pair/8best CPU退出0、覆盖2785/5161/5175/11425窗口、重建误差0。hard相对本轮extremes的test MSE/MAE：ETTh1 +0.1057%/+0.0713%，Solar −2.9889%/−5.9475%，Weather +0.1535%/+0.1314%，ETTm2 −0.0164%/−0.1872%；完整val方向一致。Solar hard .183669269/.240148470，仍比独立原GTR差+1.3782%/+1.4744%；ETTm2相对A6是MSE/MAE取舍，不能省略。新采样无新增参数，两组对A6均+0，对原GTR共同A新增560。官方最大30epoch/patience5/资产与关键参数一致；Solar实际hard14/best9 vs extremes8/best3，共同前8epoch验证最好仅−0.1655%，不称严格等实际计算量或纯采样因果。

6.5最终归档：17/17新任务SUCCESS（9smoke+8正式）、63/63smoke链路检查、17attempt、0OOM、最大并发3，daemon COMPLETED。2026-10-06T22:16:06+08:00宿主机只读验收原身份/进程组/僵尸/本队列GPU预留与compute均无残留。14科学源/17输入配置哈希保持，13类run归档及best/last完整性、恢复字段通过；总表132条=124SUCCESS+8历史CODE_ERROR保留。启动bash scripts/queue.sh start configs/queue_step_6_5_ac10.json、汇总.conda/bin/python scripts/report_step65.py及scripts/summarize_results.py均退出0。模型差异TrustGTR +51/−3、HistoricalMemory +21/−3见step_6_5_model_changes.diff；两源SHA de07b0dd728c20111592624eac9fda1200bafe44090a73a174ee7dc4b464311b / 60d848f2b087f7448e33006ab6f72e0dd21b935a86a2c5b94f57f82f114583bd。原tracked diff为空，新增未跟踪文件已按实验保存，未commit/push。

6.5调度偏差定量收尾：Weather hard额外等待261.15–261.49秒（约4分21秒），不是显存不足；未来prepare生成器已删无数据依赖的control正式前置，原SHA9578dde4…快照归档、修后SHA79506582…，当前队列与17配置未覆盖。证据step_6_5_scheduling_note.md。机制：Solar联合分支改善而R自身MSE略差，ETTm2 R及混合改善，Weather regime距离改善而预测微差；联合分支/检索分解不是独立消融、跨训练checkpoint key尺度不能直接比较。结论step_6_5_conclusions.md：保留新旧C候选，不作通用默认/扩多seed；docs/步骤与进度.md、task_plan.md、next_iteration.md同步6.5完成，6.6待执行。用户新全局科研规范已接收，后续继续固定编号与完整归档。

6.5.5最终独立复核通过：step_6_5_final_review.md核对五份JSON、八份原始metrics/hash与resolved配置、132条总表；未发现事实错误，四组方向、Solar原GTR差距、+0vsA6/+560vsGTR参数、14vs8epoch及机制边界均保留。仅新增复核报告，无重复训练/GPU/大型测试。根代理轻量复核两模型SHA仍冻结、future prepare SHA匹配修复、总表124SUCCESS+8旧CODE_ERROR，文档编号一致。6.5已完成，6.6仍待执行。
