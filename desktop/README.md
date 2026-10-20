# DriveGuard AI Desktop（开发说明）

Windows 桌面版使用 Electron + PyInstaller，把现有 FastAPI 后端与 React 前端打成一个
安装包。用户不需要安装 Python/Node，也不需要配置环境变量。

完整设计、构建、安全与验收说明见
[docs/deployment/DESKTOP_APP.md](../docs/deployment/DESKTOP_APP.md)。

## 快速开发

```bash
# 1. 准备后端环境（首次）
cd backend && ./run.sh

# 2. 生成内置配置（live），或直接使用 --demo
DGB_MODEL_SERVICE_URL=https://dms.yourdomain.com \
DGB_MODEL_SERVICE_TOKEN='<at least 24 chars>' \
python scripts/desktop/write_config.py --output desktop/config.json

# 3. 启动 Electron
cd desktop
npm install
npm start            # live
npm start -- --demo  # 离线演示
```

## Windows 打包

```powershell
$env:DGB_MODEL_SERVICE_URL = "https://dms.yourdomain.com"
$env:DGB_MODEL_SERVICE_TOKEN = "<at least 24 chars>"
.\scripts\desktop\build_windows.ps1
```

产物在 `desktop/release/`。推荐通过 GitHub Actions `Windows Desktop Release`
工作流构建，Secrets 为 `DGB_MODEL_SERVICE_URL` / `DGB_MODEL_SERVICE_TOKEN`。

## 注意

- `desktop/config.json` 含 Token，已被 `.gitignore` 忽略，禁止提交。
- 安装包中的 Token 可被提取；只使用专用、可轮换、权限受限的 Token。
- 模型服务是外部依赖；安装包不包含权重、CANN 或 NPU 运行时。
