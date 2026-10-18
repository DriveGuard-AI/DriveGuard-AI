# DriveGuard AI 后端

FastAPI + SQLite 的驾驶风险监测后端，实现 `docs/requirements/REQUIREMENTS.md`、
`docs/api/API_DESIGN.md` 与 `docs/database/DATABASE_DESIGN.md` 定义的应用层与数据层契约。

> ⚠️ **模型识别与判断尚未实现**：推理由 `app/inference/fake.py` 的脚本化回放提供，
> 快照会携带 `inference_mode = "fake"`，`/api/v1/system/status` 也会标记 `degraded = true`。

## 1. 环境要求

| 项目 | 要求 |
| --- | --- |
| Python | **锁定 3.11**（见 `.python-version`），`run.ps1` 会校验实际解释器版本 |
| pip | 建议 `python -m pip install --upgrade pip`（旧版 pip 读取 requirements 会报编码错误） |
| 数据库 | SQLite（标准库自带，无需额外安装） |

## 2. 一键启动（推荐）

脚本会自动创建 `.venv`、安装依赖并启动服务：

```powershell
.\run.ps1                 # 默认端口 8000，Token = dev-token
.\run.ps1 -Port 8123      # 自定义端口
.\run.ps1 -AdminToken ""  # 该次运行不启用鉴权（仅限本地演示）
```

macOS / Linux：

```bash
chmod +x run.sh && ./run.sh
```

停止（后端与 Vite 开发服务器）：

```powershell
.\stop.ps1                 # 停止 8000 与 5173 上的进程
.\stop.ps1 -Ports 8000     # 只停后端
.\stop.ps1 -WhatIf         # 只报告，不实际停止
```

也可以用最常规的方式：在运行服务的终端里按 **Ctrl+C**。

## 3. 手动安装与启动

Windows PowerShell：

```powershell
cd backend
py -3.11 -m venv .venv                       # 版本见 .python-version
.\.venv\Scripts\Activate.ps1                 # 注意必须带路径，Activate.ps1 不在 PATH 中
python -m pip install --upgrade pip
pip install -r requirements-dev.lock.txt

$env:DGB_ADMIN_TOKEN = "dev-token"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

macOS / Linux：

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.lock.txt     # 与 Windows 使用同一份锁文件

export DGB_ADMIN_TOKEN=dev-token
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

> 不想激活虚拟环境时，可直接用绝对/相对路径调用 venv 解释器，最不容易出错：
> `.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000`

- Dashboard：http://127.0.0.1:8000/ （需先执行 `cd ../frontend && npm run build`）
- OpenAPI 文档：http://127.0.0.1:8000/docs

## 4. 测试与迁移

```bash
python -m pytest -q        # 契约 / 集成 / WebSocket / 隐私 / 事件生命周期
alembic upgrade head       # 版本化迁移（也可用 DGB_AUTO_CREATE_SCHEMA=true 自动建表）
```

## 5. 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DGB_DATABASE_URL` | `sqlite:///backend/var/driveguard.db` | SQLAlchemy 连接串 |
| `DGB_ADMIN_TOKEN` | 空 | 管理 Token；配置后除 `/health` 外全部要求鉴权 |
| `DGB_INFERENCE_HZ` | `5` | 快照产生频率 |
| `DGB_INFERENCE_SCENARIO` | `default` | 脚本化场景：`default` / `safe` / `fatigue` / `degraded` |
| `DGB_WS_MAX_HZ` | `10` | WebSocket 快照推送上限 |
| `DGB_HEARTBEAT_SECONDS` | `15` | 心跳间隔 |
| `DGB_AUTO_CREATE_SCHEMA` | `true` | 启动时自动建表（生产建议关闭并用 Alembic） |
| `DGB_STATIC_DIR` | `frontend/dist` | 前端构建产物目录 |
| `DGB_MEDIA_DIR` | `backend/var/media` | 允许保存事件片段的位置（默认不使用） |

## 6. 常见问题

**`UnicodeDecodeError: 'gbk' codec can't decode ...`（pip install 时）**
requirements 文件历史上含中文注释，旧版 pip 按系统编码读取导致失败。现已改为纯 ASCII；
若仍报错，请升级 pip：`python -m pip install --upgrade pip`。

**`无法将"DGB_ADMIN_TOKEN=dev-token"项识别为 cmdlet...`**
`VAR=value command` 是 Bash 语法。PowerShell 请用 `$env:DGB_ADMIN_TOKEN = "dev-token"`，
且**必须与启动命令分开两行或使用分号**。

**`ERROR: Could not find a version that satisfies the requirement pydantic>=2.7`**
当前解释器过旧。先确认 **`python --version`**：若显示 2.7 或 3.7，请用 `py -3.11 -m venv .venv`
（版本以 `.python-version` 为准）创建新虚拟环境，并在激活后使用该环境里的 `python`。

**Dashboard 打不开（`GET /` 返回 `dashboard: not-built`）**
先构建前端：`cd ../frontend && npm install && npm run build`，然后重启后端。

**`409 MONITORING_NOT_ACTIVE`**
当前无活动监测会话，通常表示推理引擎未启动（检查启动日志）。

**`Activate.ps1 : 无法将“Activate.ps1”项识别为 cmdlet...`**
`Activate.ps1` 不在 `PATH` 中，必须带路径：`.\.venv\Scripts\Activate.ps1`。
或干脆不激活，直接调用 `.\.venv\Scripts\python.exe`。

**自行编写的 `.ps1` 脚本报语法错误或中文乱码**
Windows PowerShell 5.1 按**系统 ANSI 编码（中文系统为 GBK）**读取 `.ps1` 文件。
仓库内的 `run.ps1` 因此刻意保持**纯 ASCII**；若你要新增含中文的脚本，请保存为
**UTF-8 with BOM**，或改用 PowerShell 7（`pwsh`，默认 UTF-8）。
