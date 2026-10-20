# Changelog

## v0.1.0 — Windows Desktop（候选，2026-10-20）

### Added

- Electron Windows 桌面壳：无地址栏窗口、启动画面、单实例、后端生命周期管理。
- PyInstaller onedir 后端打包：FastAPI + SQLite + 前端静态资源，用户无需 Python/Node。
- 完全内置模型服务配置（URL + Token），启动即连接；支持 `--demo` 离线演示。
- 模型服务启动脚本支持 `DGB_API_TOKEN` 固定 Token，便于桌面安装包内置同一凭证。
- 桌面桥接：本地随机管理 Token 注入、打开数据目录/日志；模型服务 Token 不进入渲染进程。
- GitHub Actions `windows-latest` 构建 NSIS 安装器与 portable exe。
- 构建后校验 `win-unpacked/DriveGuard AI.exe` 与后端可执行文件，缺失或主程序小于 100MB 即失败；额外产出 `win-unpacked.zip` 供绕过安装器诊断。
- 内置配置生成拒绝 `example.com` 等示例域名，避免把占位地址打进安装包。
- 桌面版运行手册、发布说明与第三方许可清单。

### Changed

- 导航从 5 页签收敛为 4 页签：实时监控 / 离线分析 / 事件历史 / 系统设置。
- 系统设置只保留真正生效的基础配置；隐藏告警/风险/时序阈值与模型路径。
- 事件列表移除未实现的片段列；分析页不再展示服务器绝对素材路径。
- FastAPI 描述改为 demo/live 双模式，不再声明“模型未实现”。
- live 客户端改为 HTTP/1.1 持久连接 + TCP_NODELAY，避免逐帧 TCP/TLS 握手；系统状态与前端新增模型预热进度（`warmup_slots` / `warmup_total`）。
- 桌面版公网默认改为 640×480 / JPEG q60（可用 `DGB_MODEL_IMAGE_SIZE`、`DGB_MODEL_CAMERA_*` 覆盖）；局域网/高带宽可改回 960×540 或 1280×720。

### Not included

- 统计分析页、多用户/RBAC、多会话/多摄像头、事件片段、自动更新、代码签名。
- 模型权重/NPU 运行时（由外部模型服务提供）。
