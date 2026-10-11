# 6.6 两个新增数据集资产与协议

用户明确授权后续扩到四个数据集。新增 Weather 与 ETTm2；只跑 seed=2024、T=H=96，先独立 smoke 再正式对照。选择依据为数据可用性和领域/采样间隔互补，没有依据新数据测试结果挑选。

| 数据集 | 原始行数 / 变量 | 采样 | cycle | batch | LR | dropout | RevIN | 正式预算 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Weather | 52696 / 21 | 10 min | 144 | 64 | 0.001 | 0.5 | 1 | 最多30 epoch，patience=5 |
| ETTm2 | 69680 / 7 | 15 min | 96 | 256 | 0.001 | 0.5 | 1 | 最多30 epoch，patience=5 |

协议直接来自保留的官方 scripts/GTR/weather.sh、ettm2.sh、run.py 默认值和原 data_loader；未改训练参数。官方默认 freq=h 原样保留，GTR 不读取时间特征。

公开来源：本地归档 TimeBridge 官方 README 的 Google Drive 数据链接（https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link），复用已下载 ZIP。全包 SHA256 与既有 manifest 完全一致，两个 ZIP member 的 CRC、CSV SHA256、行列数、有限值与时间递增检查通过；ETTm2 采样严格规则，Weather 原生异常详见下文。无需下载、无预处理、原资产只读。

在项目 dataset/ 新增独立只读副本共 16.13 MiB；没有删除任何文件，没有对其他项目文件 chmod，也不以硬链接绑定外部可变数据。来源、脚本和数据哈希详见同名 JSON；原 artifacts/data_assets.json 保留。

| 数据集 | train切片 [start,end) | val切片 | test切片 | train/val/test 窗口数 |
|---|---|---|---|---|
| weather | [0, 36887] | [36791, 42157] | [42061, 52696] | 36696/5175/10444 |
| ettm2 | [0, 34560] | [34464, 46080] | [45984, 57600] | 34369/11425/11425 |

ETTm2 按官方12/4/4 nominal 30-day月划分，使用前57600行；后12080行保留原文件但不进入官方训练/评估。Weather按70%/剩余10%/20%划分。val/test回看重叠只是输入历史，目标不跨入前一split。归一化只拟合训练集。

Weather 的官方 CSV 原生含1次重复时间（0-based row19044，2020-05-12 06:00）与1次100min间隔（row21514，2020-05-29 09:30→11:10），不是本项目预处理造成。保持原始行顺序和官方行序相位，不去重、不补点；cycle144并不始终对应精确墙钟日相位。这一限制保留供机制分析。

现成 loader 已支持 ETTm2/custom；runner 需要为其 evaluation_assets.json 补充上述 absolute_data_slice，供历史记忆读取正确的绝对预测时间。该修改只补资产元数据，不能改原 loader。

现成官方 loader 的 CPU 检查通过：两个数据集各 train/val/test 共6个 split，窗口数、前向输入/目标维度、有限值与三个split共用训练拟合的归一化均符合预期；见 two_dataset_loader_check.json。这不是 GPU smoke，训练、保存、加载仍须独立 smoke 验收。

本步骤无 GPU 训练，没有新增性能结果。下一步由统一调度队列生成正式GTR与固定A6（per_bank，不含B/C）配置；所有有效参数、代码与评估资产独立归档。原论文这两个数据集数值尚未重新核对，暂留 NA。
