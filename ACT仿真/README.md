# ACT 仿真

[项目首页](../) · [真机阶段](../ACT真机/) · [八点同步回放](results/stage1_20261024/videos/v04_final8_overview.mp4)

在 MuJoCo 中复现 B601 抓起红块并放进灰盒。从公开权重迁移开始，依次验证单点适配、多位置训练与动作执行方式。

## 数据与验证

| 项目 | 结果 |
| --- | --- |
| 选定策略 | 5,000 步实验分支中的 3,000 步检查点，ensemble 1 |
| 数据 | 25 个合成训练位置，10,075 帧。4 个验证位置，1,612 帧。 |
| 固定验证 | 原位 **1/1**，4 个验证位置 **4/4** |
| 冻结后验证 | 预先保留 **8/8**，随后追加的点 **1/1**，分别记账 |
| 适用范围 | 固定物体、相机与动力学下的局部 XY 插值 |

[验证数据](docs/stage1_20261024/runtime_inputs/compact_results.json) · [双相机输入](results/stage1_20261024/videos/v05_final8_test00.mp4) · [追加位置](results/stage1_20261024/videos/v13_extra_p10_0.mp4)

## 复现索引

| 内容 | 入口 |
| --- | --- |
| 环境、数据生成、训练与评估 | [复现步骤](docs/stage1_20261024/训练复现步骤.md) · [最小依赖与输入](docs/stage1_20261024/public_reproduction.md) |
| 场景、教师、执行和物理评分 | [仿真源码](simulation/) |
| 超参数与 5,000 步指标 | [训练记录](results/stage1_20261024/training_record.json) · [逐步 loss](results/stage1_20261024/training_metrics.csv) |
| 选模、失败迭代与发布凭证 | [在线复盘](docs/stage1_20261024/ACT仿真训练复现与调参复盘.md) · [PDF](reports/ACT仿真训练复现与调参复盘.pdf) · [Word](reports/ACT仿真训练复现与调参复盘_修订版.docx) · [SHA-256 清单](results/stage1_20261024/release_manifest.json) |
| 成功与失败录像 | [署名视频目录](results/stage1_20261024/videos/) |

从 `ACT仿真` 目录执行复现命令。约 20 GiB 的 RGB 缓存和大模型按复现步骤准备。 [作者与上游来源](../AUTHORS.md)
