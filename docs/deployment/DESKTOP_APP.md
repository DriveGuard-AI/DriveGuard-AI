# DriveGuard AI Desktop（Windows）开发与打包

状态：工程实现，待 Windows CI 构建与实机验收（2026-10-20）。

## 1. 目标形态

交付一个 Windows x64 安装包/便携 exe。用户不需要安装 Python、Node、SQLite，
也不需要配置环境变量；模型服务地址与 Token 在构建时完全内置。

```text
用户双击 DriveGuardAI.exe
        │
        ▼
Electron 主进程
  ├─ 读取 resources/config.json（内置模型服务 URL + Token）
  ├─ 生成随机本地管理 Token
  ├─ 启动 resources/backend/driveguard-backend.exe
  ├─ 等待 http://127.0.0.1:<port>/api/v1/health
  ├─ 打开无地址栏主窗口
  └─ 退出时关闭后端
        │
        ▼
现有 React Dashboard（frontend/dist）
```

`demo` 模式（启动参数 `--demo`）使用脚本化回放，不需要模型服务；发布包默认
`live` 模式，连接内置的模型服务。启动时会先探测 `/api/live/health`：不可达时
弹出明确提示，用户可选择仍然打开（后端自动重试）或退出。

## 2. 目录

| 路径 | 作用 |
| --- | --- |
| `desktop/electron/main.cjs` | 主进程：配置、端口、后端生命周期、窗口、日志 |
| `desktop/electron/preload.cjs` | 只暴露本地 Token、版本、打开数据目录/日志 |
| `desktop/electron/splash.html` | 后端启动期间的启动画面 |
| `desktop/backend_entry.py` | PyInstaller 后端入口（uvicorn + 静态资源兜底） |
| `desktop/packaging/backend.spec` | 后端 onedir 打包配置 |
| `scripts/desktop/write_config.py` | 从环境变量生成内置 `config.json` |
| `scripts/desktop/build_windows.ps1` | Windows 一键构建 |
| `.github/workflows/windows-desktop.yml` | GitHub Actions Windows 构建 |

## 3. 本地开发（Linux/Windows）

先准备后端环境：

```bash
cd backend
./run.sh          # 首次创建 .venv 并安装依赖；或按 README 手动创建
```

生成开发配置（二选一）：

```bash
# 完全内置 live：需要可访问的模型服务
DGB_MODEL_SERVICE_URL=https://dms.example.com \
DGB_MODEL_SERVICE_TOKEN='<at least 24 chars>' \
python scripts/desktop/write_config.py --output desktop/config.json

# 或离线 demo（不需要 config.json）
```

启动 Electron：

```bash
cd desktop
npm install
npm start                 # live：读取 desktop/config.json
npm start -- --demo       # demo：脚本化回放
```

开发模式下，主进程使用 `backend/.venv` 里的 Python 启动 `uvicorn app.main:app`；
`DGB_DESKTOP_PYTHON` 可覆盖解释器路径。

## 4. Windows 构建

前置：

- Windows 10/11 x64、Python 3.12、Node 22+；
- 模型服务地址与专用 Token；
- 建议在 GitHub Actions `windows-latest` 上构建，避免本机环境差异。

```powershell
$env:DGB_MODEL_SERVICE_URL = "https://dms.example.com"
$env:DGB_MODEL_SERVICE_TOKEN = "<at least 24 chars>"
.\scripts\desktop\build_windows.ps1
```

产物：

```text
desktop/release/
  DriveGuardAI-Setup-0.1.0.exe
  DriveGuardAI-0.1.0-portable.exe
  DriveGuardAI-0.1.0-win-unpacked.zip   # 诊断用：解压直接运行，绕过安装器
  SHA256SUMS.txt
```

构建脚本会检查 `win-unpacked/DriveGuard AI.exe` 与
`win-unpacked/resources/backend/driveguard-backend.exe` 是否存在，并拒绝小于 100MB
的主程序；缺失或截断则构建失败，避免发布没有主程序的安装包。`win-unpacked.zip`
用于区分“安装器问题”和“运行环境拦截”。

GitHub Actions 使用仓库 Secrets：`DGB_MODEL_SERVICE_URL`、`DGB_MODEL_SERVICE_TOKEN`；
可用 Variables 覆盖 `DGB_MODEL_VIEW`、`DGB_MODEL_IMAGE_SIZE`、`DGB_MODEL_ROI`。
构建产物在 Actions 的 `driveguard-windows` artifact 中；推送 `v*` tag 时还会创建
draft GitHub Release（不会自动公开发布）。

## 5. 完全内置配置

`desktop/config.json`（构建时生成，不提交）：

```json
{
  "schemaVersion": 1,
  "modelServiceUrl": "https://dms.example.com",
  "modelServiceToken": "<token>",
  "view": "body",
  "imageSize": "1280x720",
  "driverRoi": "0.15,0.0,0.75,1.0",
  "cameraIndex": 0,
  "cameraWidth": 1280,
  "cameraHeight": 720,
  "cameraQuality": 85,
  "cameraBackend": "",
  "inferenceHz": 5,
  "previewMaxFps": 10
}
```

安全代价：Token 会随安装包分发，必须使用专用、可轮换、权限受限的服务端 Token；
不要把管理员 Token 放进公开安装包。更安全的首次配对方案属于 v0.2。

模型服务端必须使用同一个 Token 启动（否则桌面版会收到 401）：

```bash
DGB_API_TOKEN="<与构建时相同的 Token>" \
  SESSION_DIR=runs/live-composite-session-desktop \
  bash scripts/server/start_live_composite.sh
```

## 6. 运行时数据

| 内容 | 位置 |
| --- | --- |
| SQLite | `%APPDATA%\DriveGuardAI\driveguard.db` |
| 上传媒体/帧 | `%APPDATA%\DriveGuardAI\media` |
| 后端日志 | `%APPDATA%\DriveGuardAI\logs\backend.log`（>5MB 轮转一次） |
| 模型服务 Token 副本 | `%APPDATA%\DriveGuardAI\model-service.token`（仅本机后端读取） |
| 本地管理 Token | 每次启动随机生成，仅通过 preload 传给渲染进程，不落盘 |

卸载默认保留用户数据；`nsis.deleteAppDataOnUninstall=false`。

## 7. 安全与隐私

- 后端只监听 `127.0.0.1`，不开放入站端口。
- 渲染进程 `sandbox: true`、`contextIsolation: true`、`nodeIntegration: false`。
- 模型服务 Token 不进入渲染进程，只写用户目录文件供后端读取。
- 默认不保存原始视频；离线分析上传的素材按保留期清理（NFR-003）。
- 预览默认关闭；开启后仍只读内存最新帧。
- 未签名安装包会触发 SmartScreen；公开分发前应做代码签名与杀毒扫描。

## 8. 已知限制

- 模型服务是外部依赖；安装包不含权重、CANN、NPU 运行时。
- 模型服务当前单活动会话（KL-06），多用户需部署多实例或做会话路由。
- 摄像头采集发生在运行软件的机器上；服务器无摄像头时使用上传视频。
- 公网摄像头串流带宽约 10–16Mbps/用户（720p/5–8fps）；广域网建议降分辨率。
- v0.1 不包含统计页、多用户/RBAC、事件片段、自动更新、代码签名。
- 当前 Linux 环境无法构建 Windows exe；Windows 产物需由 Actions/Windows 机器验证。

## 9. 验收清单

- [ ] 干净 Windows 10/11 x64（无 Python/Node）安装并启动。
- [ ] 主窗口无地址栏，桌面/开始菜单快捷方式正常。
- [ ] 内置模型服务地址可连接，四任务/六参数/预览正常。
- [ ] 摄像头权限、上传分析、事件持久化、重启恢复。
- [ ] 无控制台窗口、日志落在 `%APPDATA%`。
- [ ] 杀毒扫描、SmartScreen 提示记录、卸载行为。
- [ ] 安装包与 SHA256 与发布说明一致。
