# 第三方来源与署名

- ACT 的原论文为 [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)。训练和推理接口来自 [Hugging Face LeRobot](https://github.com/huggingface/lerobot)。云端为 0.6.1，封存的 Windows 原厂运行时为 0.6.0。
- 公开基线来自 [nikodembartnik/act_rebot_cube](https://huggingface.co/nikodembartnik/act_rebot_cube/tree/409b639946cc4140b3934a6e3d2c255f4335ee9d)，实采 ACT4 策略训练从新策略和 ImageNet ResNet-18 骨干开始。初始化权重通过 TorchVision 获取。模型和数据分别保留版本来源。
- SenseCraft Robotics、Seeed 插件、MotorBridge、RealSense 与 PCAN 通过各自发布方安装。采集插件的固定提交见 [hardware/sources.json](../hardware/sources.json)。原厂推理接口校验见 [vendor-source-hashes.json](../inference/vendor-source-hashes.json)。
- 这个版本目录发布项目的适配代码、说明与验证资产。厂商程序、驱动、标定、私有设备信息和完整会话归档保存在原环境或各自上游。
