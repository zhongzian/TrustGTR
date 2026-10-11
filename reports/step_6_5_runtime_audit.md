# 6.5 运行验收

更新时间：2026-10-06T22:15:20.152163+08:00；daemon COMPLETED；{'failed': 0, 'pending': 0, 'retrying': 0, 'running': 0, 'stopped': 0, 'succeeded': 17}。

科学源码14项、冻结配置17项哈希核验；异常0项。smoke 9/9成功，63/63检查通过；最大观察并发3，累计OOM0次。

| 任务 | 状态 | attempt | OOM | 峰值MiB | 新增参数/A6 |
|---|---|---:|---:|---:|---:|
| trustgtr_alloff__etth1__smoke__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 444.0 | NA |
| trustgtr_ac10_extremes__etth1__smoke__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 646.0 | NA |
| trustgtr_ac10_extremes__etth1__forecast__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 650.0 | 0 |
| trustgtr_ac10_hard__etth1__smoke__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 408.0 | NA |
| trustgtr_ac10_hard__etth1__forecast__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 650.0 | 0 |
| trustgtr_ac10_extremes__solar__smoke__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 926.0 | NA |
| trustgtr_ac10_extremes__solar__forecast__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 1928.0 | 0 |
| trustgtr_ac10_hard__solar__smoke__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 2102.0 | NA |
| trustgtr_ac10_hard__solar__forecast__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 1930.0 | 0 |
| trustgtr_ac10_extremes__weather__smoke__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 454.0 | NA |
| trustgtr_ac10_extremes__weather__forecast__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 612.0 | 0 |
| trustgtr_ac10_hard__weather__smoke__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 780.0 | NA |
| trustgtr_ac10_hard__weather__forecast__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 610.0 | 0 |
| trustgtr_ac10_extremes__ettm2__smoke__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 836.0 | NA |
| trustgtr_ac10_extremes__ettm2__forecast__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 840.0 | 0 |
| trustgtr_ac10_hard__ettm2__smoke__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 836.0 | NA |
| trustgtr_ac10_hard__ettm2__forecast__t96_h96__seed2024__20261006_214646 | SUCCESS | 1 | 0 | 840.0 | 0 |

已完成实际resolved配对核验4/4组；所有成功任务检查13类归档、科学代码快照与best/last ZIP CRC。固定资产=GTR/A6，正式训练预算=GTR/A6，C相对A6新增参数0。

只读验收；未启动、重启、修改科学源码/配置/调度器、清理进程或删除成果。

宿主机只读最终验收：2026-10-06T22:16:06.252987+08:00；17次attempt身份与daemon核验，原身份残留0、进程组成员（含僵尸）0、本队列GPU预留0、本队列compute进程0；GPU/compute查询退出码0/0；异常0。

6.5 最终验收：17/17成功（9 smoke、8正式），63/63 smoke检查，17次尝试、0 OOM；daemon COMPLETED，汇总退出码0。14科学文件与17冻结配置保持哈希，四actual resolved配对仅contrast_sampling变化；正式新增参数/A6均0、原预算/固定资产一致。所有成功run均有13类归档，best/last ZIP CRC、恢复字段/RNG/optimizer/scheduler、best对应last.best_model_state及有限权重全部通过。保守调度依赖造成的连续空槽261.15–261.49秒已单独记录，未来生成器修复且原版本已归档。
