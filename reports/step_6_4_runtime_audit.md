# 6.4 运行验收

更新时间：2026-10-06T12:47:23.237742+08:00。daemon：COMPLETED；{'failed': 0, 'pending': 0, 'retrying': 0, 'running': 0, 'stopped': 0, 'succeeded': 17}。

科学源码 14 项、输入配置 17 项哈希检查；异常 0 项。成功 smoke 9/9，共 63 项验收检查；累计 OOM 尝试 2 次，保留重试记录。

|任务|状态|attempt|OOM次数|峰值MiB|上次OOM启动空闲MiB|估计MiB|下次资格时间|
|---|---|---:|---:|---:|---:|---:|---|
|trustgtr_alloff__etth1__smoke__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|444.0|None|444.0|None|
|trustgtr_ab9_train__etth1__smoke__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|644.0|None|644.0|None|
|trustgtr_ab9_train__etth1__forecast__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|652.0|None|652.0|None|
|trustgtr_ab9_inference__etth1__smoke__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|544.0|None|544.0|None|
|trustgtr_ab9_inference__etth1__forecast__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|656.0|None|656.0|None|
|trustgtr_ab9_train__solar__smoke__t96_h96__seed2024__20261006_122231|SUCCESS|2|1|1860.0|1620.0|1860.0|1791260694.0482457|
|trustgtr_ab9_train__solar__forecast__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|1966.0|None|1966.0|None|
|trustgtr_ab9_inference__solar__smoke__t96_h96__seed2024__20261006_122231|SUCCESS|2|1|1860.0|1620.0|1860.0|1791260718.154028|
|trustgtr_ab9_inference__solar__forecast__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|1966.0|None|1966.0|None|
|trustgtr_ab9_train__weather__smoke__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|752.0|None|752.0|None|
|trustgtr_ab9_train__weather__forecast__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|580.0|None|752.0|None|
|trustgtr_ab9_inference__weather__smoke__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|752.0|None|752.0|None|
|trustgtr_ab9_inference__weather__forecast__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|582.0|None|752.0|None|
|trustgtr_ab9_train__ettm2__smoke__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|838.0|None|838.0|None|
|trustgtr_ab9_train__ettm2__forecast__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|668.0|None|838.0|None|
|trustgtr_ab9_inference__ettm2__smoke__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|840.0|None|840.0|None|
|trustgtr_ab9_inference__ettm2__forecast__t96_h96__seed2024__20261006_122231|SUCCESS|1|0|670.0|None|840.0|None|

四数据集正式配对实际 resolved_options、官方训练参数和固定评估资产逐项核验，详细见同名JSON。
只读审计，未启动额外实验、未改配置、未杀进程、未删除数据。

6.4 最终验收：17/17任务成功（9 smoke、8正式），63/63 smoke检查通过；daemon COMPLETED，pending/running/retrying/failed/stopped均0，汇总退出码0。

累计19次尝试，其中2次Solar smoke OOM经超过60秒冷却、显存改善后成功；首轮日志、退出文件、OOM前峰值/启动空闲/显存估计/下次资格时间及重试记录均保留。最大观察并发3。

四数据集actual resolved配对仅gate_calibration_context不同，固定资产与GTR/A6一致，新增参数均321；所有成功run的13类必需记录、best/last checkpoint和科学代码快照哈希均通过。

宿主机只读验收：时间2026-10-06T12:47:57.912374+08:00，19个attempt身份与daemon进程组无残留/僵尸，本队列与本用户全局GPU预留均0，项目GPU compute进程0，GPU查询退出码0。未启动额外实验、未改配置/科学代码/调度器、未杀进程、未删除文件。
