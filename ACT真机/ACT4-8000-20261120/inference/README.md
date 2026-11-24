# 5. 云端推理与真机执行

## 模型与输入输出

[下载 8,000 步推理包（183 MiB）](https://github.com/Artymis11/ACT/releases/download/act4-8000-20261120/act4-low8000-pretrained-model.zip)

[模型清单](model-manifest.json)固定封存推理包的 11 个文件、长度及 SHA-256，包含原始配置、权重、预处理与后处理。解包后保留 `pretrained_model/` 结构。核验命令为 `python inference/verify_model.py local/model/pretrained_model`。

输入为按双线性插值等比缩放、补边的 RGB224 和当前七轴状态。每个 ACT 预测输出 100 步，RPC 取前 50 步并通过保存的处理器反归一化。本机的策略时基为 30 Hz，派发为 60 Hz。J1–J6 使用限速插值和时间集成，J7 沿用原执行链。

## 云端服务与 SSH

安装[固定云端环境](../training/)，在版本目录执行：

```bash
python autodl/prepare_serving_model.py \
  --model local/model/pretrained_model --output-root local/serving
python autodl/image_policy_server.py \
  --model_path=/absolute/path/printed/by/prepare_serving_model \
  --host=127.0.0.1 --port=8164 --fps=30 \
  --inference_latency=0.033 --obs_queue_timeout=1
```

本机建立双向无损 SSH 转发，云端服务监听回环。下列用户名和主机名为示例。

```powershell
ssh -C -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8165:127.0.0.1:8164 user@your-host
```

也可用 [cloud_forward.py](../autodl/cloud_forward.py) 的 `--compress`，通过可信主机校验并将本机端口映射至远端回环。SSH 压缩对象是隧道中的传输字节。224 图像在进入隧道前已于本机生成。

## Windows 真机预检

1. 将 [site.example.json](site.example.json) 复制为 `local/site.json`。填写自己的 SenseCraft Python、后端、数据目录、已完成配置的项目与录制 ID，以及云端输出的 serving 路径。若原厂 Python 内置 Pillow 与封存的 12.3.0 不同，将 12.3.0 安装到匹配该 Python ABI 的独立目录，并在 `pillow_root` 填入路径。
2. 将同一设备、标定和验证过的本机七轴零位保存为 `local/zero_state.json`。记录位姿需与当前物理装置相符。现场绑定凭据、标定和位姿均置于本机。
3. 用 `site.json` 中的 SenseCraft Python 运行以下命令。程序读取五个[厂商源码哈希](vendor-source-hashes.json)、模型、标定身份及保存零位。

```powershell
& 'C:\path\to\SenseCraft\runtime\python.exe' autodl/run_act4_native_sensecraft.py --site local/site.json --low8000 --arm165-te001-manual
```

4. 预检通过并完成现场确认后，追加 `--start` 进入待键状态。首轮由操作员亲按 Ctrl+C。运行中 Ctrl+C 停止当前轮并回保存位。验位后下一次 Ctrl+C 开新轮。Ctrl+V 回位、验位并失能。回位失败时查看控制器状态和本机实际姿态，保持现场操作员掌握释放。

[原版配置](trials_profile_arm165_te001_manual.json)固定了 165°/秒、0.01 六轴时间集成、回零误差 2°、45 秒超时及手动控制规则。已记录的四轮是该配置的真机证据。
