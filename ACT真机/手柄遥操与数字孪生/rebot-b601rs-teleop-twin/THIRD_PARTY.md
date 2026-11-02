# 第三方来源

本项目借鉴 [Tony 的手柄遥操设计与源码](https://github.com/TonyWeyyy/reBot_B601_rs_xbox_teleop)。固定参考提交为 `f402f80ccab68d457178b5a8c688642147e069a4`。

| 资源 | 来源 | 处理方式 |
| --- | --- | --- |
| Tony 输入参考源码 | 上述仓库 | 首次准备时按固定提交获取到本地忽略目录；哈希见 `teleop/tony_source_manifest.json`；不随公开源码压缩包重新分发 |
| RS 模型、Web 渲染参考资源 | [Seeed 文档收录的数字孪生项目](https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_RS) | 固定提交 `aa807cf2affe85b13b86221159871aae7742c01b`；按 `digital_twin/vendor_manifest.json` 获取 |
| 原始机械臂设计与模型来源 | [Seeed-Projects/reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) | 原项目 CERN-OHL-W-2.0 文本保留于 `LICENSES/` |
| Three.js r128、OrbitControls、STLLoader | [Three.js](https://github.com/mrdoob/three.js/tree/r128) | MIT，文本保留于 `LICENSES/` |
| URDFLoader | [urdf-loaders](https://github.com/gkjohnson/urdf-loaders) | Apache-2.0，文本保留于 `LICENSES/` |
| PCANBasic | [PEAK-System](https://www.peak-system.com/products/software/development-packages/pcan-basic/?L=1) | 用户自行安装官方驱动/DLL；不随本包分发 |
| RobStride 线协议参考 | [MotorBridge](https://github.com/motorbridge/motorbridge)、[Seeed RobStride 文档](https://wiki.seeedstudio.com/robstride_control/) | 静态协议参考；默认控制入口不导入 MotorBridge SDK 或启动其 Gateway |

根目录 `LICENSE` 只适用于本项目新增且有权许可的代码与文档；第三方代码、模型和驱动遵循各自许可。参考源码缓存、模型下载目录、DLL均已在 `.gitignore` 中排除。资源获取只是下载和校验，不会执行参考项目的安装、使能、校零或硬件控制入口。
