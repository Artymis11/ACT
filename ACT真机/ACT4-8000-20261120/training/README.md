# 4. ACT 训练与权重

ACT4 从新建的 ACT 策略开始训练，ResNet-18 视觉骨干用 ImageNet 权重初始化。30 Hz 双路 RGB224、七轴当前状态、100 步训练动作块，固定 54/6 划分。训练基线的 batch 32、种子 42，目标由动作 L1、10×KL 和 0.05×有效动作二阶差分 L1 组成。

## 环境与训练入口

AutoDL 的实测训练环境为 Linux、单卡 CUDA、LeRobot 0.6.1、PyTorch 2.10.0+cu128、Pillow 12.3.0。 [版本锁定文件](../autodl/runs/act4_224_scratch_20261109/environment.cloud.lock.txt)

1. 按[数据处理](../dataset/)生成原标签和三帧均值标签版本。
2. 通过 [prepare_video_cache.py](../autodl/prepare_video_cache.py) 准备双摄解码缓存。`--dataset-root` 指向平滑数据集，`--output-dir` 存放缓存。设置 `ACT_VIDEO_CACHE_MANIFEST` 为生成的 `manifest.json`。
3. 将 [train_config.example.json](train_config.example.json) 的数据和输出目录换成本机路径，预缓存 TorchVision 的 ImageNet 骨干。在版本目录运行：

```bash
python autodl/runs/act4_224_scratch_20261109/train_act4_scratch.py \
  --split-manifest local/data/act4_224_action_smoothed_v1/split_manifest.json \
  --vision-init imagenet \
  --smoothness-weight 0.05 \
  --config_path=training/train_config.example.json
```

原工程保存了 7,500 步作为分支源点。低学习率分支将主网络从 `1e-4` 调到 `1e-5`，视觉骨干保持 `1e-5`，在累计第 8,000 步保存了首个分支检查点。历史运行的总目标是 12,500 步，每 500 步保存和验证一次。8,000 步为本发布选择的模型位置。

[train_act4_lr_branch.py](../autodl/runs/act4_224_scratch_20261109/train_act4_lr_branch.py) 保留续训入口。先通过 `--check-only` 核对完整的 7,500 步源点、优化器及 RNG、原路径对应的数据与缓存、原编译对照和哈希。主要参数为 `--checkpoint`、`--source-audit`、`--compiled-probe`、`--split-manifest`、`--output-dir`、`--main-lr 1e-5` 与 `--target-step 12500`。精确恢复原训练分支需要这批归档依赖。推理包包含 8,000 步权重和处理器，可用于离线与推理复现。

## 选 8,000 步的依据

| 同六条留出集、原始动作标签 | 8,000 步 | 30,500 步 | 80,000 步 |
| --- | ---: | ---: | ---: |
| 前 10 步，全轴归一化 L1 | 0.11020 | 0.11464 | 0.11466 |
| 前 50 步，全轴归一化 L1 | **0.18280** | 0.18812 | 0.18947 |
| 前 50 步，J7 平均误差 | 18.17° | 18.18° | **17.41°** |
| 首次持续闭合附近，J7 平均误差 | **9.16°** | 10.22° | 10.17° |

8,000 步在六条验证中的五条取得更低的前 50 步 L1。夹爪单轴误差和抓取成功也单独核验。权重与处理器的完整文件身份见[模型清单](../inference/model-manifest.json)，现场结果见[评测](../evaluation/)。
