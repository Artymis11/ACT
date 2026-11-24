# 3. 数据集与预处理

ACT4 示教为 **60 集、32,283 帧**，其中 54 集 29,094 帧训练，6 集 3,189 帧留出。每个连续 10 集的采集区段抽一集留出，归一化只聚合训练集。 [数据规格](dataset-spec.json) · [划分清单](split_manifest.json)

## 示例

[第 0 集](example/)包含 491 帧的顶部和腕部 MP4，以及逐帧七轴实测状态与动作标签 CSV。两路视频是供观看的 H.264 RGB224 预览。验证模型数值时使用[两张无损首帧和原始 RPC 响应](../evaluation/rpc_fixture/)。

## 从 SenseCraft 导出重建

在版本目录执行下列命令。`local/raw-act4` 指向原始 LeRobot 导出，`local/recording` 指向同一次 SenseCraft 录制的元数据目录，内有 `state.json` 和录制日志。运行时依赖见[训练环境](../training/)。

```bash
python autodl/runs/act4_224_scratch_20261109/prepare_act4_data.py \
  --source local/raw-act4 --recording local/recording --output-root local/data
python autodl/runs/act4_224_scratch_20261109/verify_act4_data.py --data-root local/data
```

原始 640×480 RGB 按双线性插值等比缩到 224×168，上下各补 28 像素黑边。图像处理由[同一实现](../autodl/image_preprocessing.py)供训练和推理共用。动作标签以集为界做三帧中心均值，靠近边界时只使用本集可用帧。七轴实测状态与逐帧时间单独保存。处理器保存训练统计，训练时用平滑标签，留出评估时也对照原标签。

完整训练数据保存在原采集归档。示例保留真实的一集与数据字段，重训需要提供完整数据集。
