# GitHub 发布说明

本项目现集成于 [Artymis11/act](https://github.com/Artymis11/act)，位置为：

```text
ACT真机/手柄遥操与数字孪生/rebot-b601rs-teleop-twin/
```

展示视频和海报放在相邻的 `ACT真机/手柄遥操与数字孪生/media/`。本次发布范围是手柄遥操、实测反馈三维监看、说明文档与展示媒体；不包含 ACT 模型权重、采集数据或旧部署包。

## 维护与提交

1. 在现有 `act` 仓库中维护此子目录，不要在子目录再次执行 `git init`，也不要为发布本项目上传整个本地 ACT 工作区。
2. 保留 README、docs、源码、测试、启动脚本、LICENSE、THIRD_PARTY 及来源哈希。
3. 确认忽略规则生效。不要强制添加 vendor 缓存、DLL、runtime、reports、个人日志、`.env`、模型权重或其他项目。
4. 在本子项目目录执行 `python tools/bootstrap.py --teleop-only` 与 `python tools/check.py` 复现离线检查；它们不连接 CAN。提交前逐项核对暂存文件，仅提交本次改动。
5. 源码清单随实际公开文件更新；不要把已有清单中的哈希当成修改后文件仍一致的证据。

## 发布时保留的事实与来源

- 作者署名为 **artymis**，GitHub 账号为 [Artymis11](https://github.com/Artymis11)。
- Tony 致谢保持为简短说明和原源码链接；固定提交与第三方许可范围见 [THIRD_PARTY.md](THIRD_PARTY.md)，不将第三方内容重新授予 MIT 许可。
- J2/J3 为 0°～126°的软件手动目标区间；70%表示保留70%参考工作跨度，不是取消硬件保护。
- X使能／恢复；A长按3秒回本次实测起点后失能；B缓慢回全零后失能；Y只回本次起点并保持使能。
- 数字孪生是实测反馈姿态监看，不宣称完整物理仿真、ACT自主抓取或完整数据集采集系统。
- 软件测试通过不能改写为实机全范围、全负载或功能安全认证。正常返回与热故障紧急失能也应区分。
- 参考资源按固定清单首次获取，不将本机缓存的第三方源码、模型或驱动强行加入公开提交。

## GitHub Actions 的位置

GitHub 仅识别仓库根目录的 `.github/workflows/`。本子目录内的 `.github/workflows/offline-checks.yml` 是独立仓库使用模板，放在当前层级不会自动运行。

当前 `act` 仓库由根目录配置同等离线检查，并将工作目录指向本子项目。检查只准备固定输入参考并运行软件回归，不运行真实控制入口；完整三维资源在使用时通过完整 bootstrap 获取。
