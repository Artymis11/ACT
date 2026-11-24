# 1. 硬件与标定

| 角色 | 硬件与接口 |
| --- | --- |
| 示教主臂 | ReBot Arm 102-LD，经 Windows 串口接入 SenseCraft |
| 执行从臂 | Seeed ReBot B601-RS，经 PEAK PCAN 接入，封存入口映射 can0 → PCAN_USBBUS1 |
| 顶部相机 | Intel RealSense D435i，SenseCraft `front`，数据键 `observation.images.front` |
| 腕部相机 | Intel RealSense D405，SenseCraft `wrist`，夹爪角色 `gripper` |
| 采集输入 | 双路 640×480 RGB，30 FPS，按设备序列号绑定 |

数据的七轴顺序为 `shoulder_pan`、`shoulder_lift`、`elbow_flex`、`wrist_flex`、`wrist_yaw`、`wrist_roll`、`gripper`。模型策略坐标在原厂执行层转换为电机侧坐标，封存适配器的比例为 `[1, 1, -1, -1, -1, 1, 6]`。核对各轴正方向、角度与弧度的转换、夹爪闭合方向和电机侧关节限位。

## 现场验收顺序

1. 在 SenseCraft 里确认串口、PCAN、主从臂设备类型与当前标定文件。连线变动后重新枚举设备端口。
2. 分别核对顶部和腕部取景。顶部画面包含方块起点和盒沿，腕部能看清夹爪与方块接触。固定相机支架并记录光线、纸垫位置和物体摆位。
3. 在操作员现场监督下逐轴核方向和中性位，再检查跟随、夹爪和关闭后的释放回执。
4. 保存本机标定与已验位的起始姿态。真机入口检查零位文件的设备身份及标定 SHA，换设备后需要重新验位。

采集调试环境的包版本与上游提交见 [sources.json](sources.json)。封存真机使用独立的 [SenseCraft 原厂运行时](../inference/runtime-versions.json)。
