# 最新组合模型：实时摄像头联调

这版服务运行真实权重，不读取历史预测 CSV，也不需要在运行时访问 Dataset
或训练缓存。浏览器读取本机摄像头，上传未镜像 JPEG；服务器在现有感知环境与
NPU 环境分别执行感知与模型。当前是单摄像头开发联调服务，不是业务告警平台。

## 固定模型组合

| 输出 | 模块 |
|---|---|
| body 疲劳 | ActionModel yawn + B_eye_native seed42 best + dense_track + 1000ms 闭眼事件 |
| body 分心 | C_fusion seed43 best，沿用已有 validation_fpr03 阈值；RGB 不可用回退 gap80，阈值 0 |
| body 手机、饮水；face 四任务；注视方向 | coverage25，各自原有效性条件 |

body 注视输出标记 `unvalidated_body_view`；有输出不表示该视角有验证成绩。
疲劳分支不可用时不悄悄切换到另一套疲劳判定。眼部缺失只重置闭眼状态，仍可
使用有效打哈欠证据。分数是未校准的 sigmoid，不是事故概率。不输出 SAFE/WARNING/DANGER。

## 1. 在训练服务器打包

在仓库根目录执行：

```bash
bash scripts/server/export_live_composite.sh
```

默认读取：

- `runs/coverage25-composite-001/all-tasks/report.json`
- `runs/body-fatigue-eye-recovery-001/comparison/comparison.json`
- `models/local/resnet18-f37072fd.pth`
- `runs/dmd-full-yolo11.local.runtime.json`（原 YOLO11 提取配置）

目录不一致时用环境变量：`ALL_TASKS_REPORT`、`RECOVERY_REPORT`、`RGB_WEIGHTS`、
`RUNTIME_CONFIG`、`PYTHON_BIN`。可从原批处理 JSON 的 `runtime_config` 找到真实配置。
不要换成旧 YOLOX 示例配置。导出阶段需要原 head/RGB 缓存的 manifest 以核对裁剪和编码器；
打包完成后运行不再依赖这些缓存。导出也核对原特征提取的检测器设置和特征协议，
防止使用了另一个 YOLO/阈值组合而没有发现。

产物 `models/releases/live-composite-v1/` 包含：

- `bundle.json`：固定分支、阈值、裁剪、跟踪与事件规则、源报告哈希。
- `weights/`：coverage25、gap80、C_fusion、ActionModel、EyeModel、ResNet18 及校验文件。
- `assets/`：YOLO11、MediaPipe face/hand 资产。
- `runtime.json`：感知配置；不打包令牌内容。

这是一套完整的推理工件集合，不是重新训练的单一网络。导出不覆盖已有目录，
不下载、不发布、不调整阈值。转移模型包时还需相同版本仓库代码与对应运行环境。
包内记录推理和模型代码校验值；代码变更后需重新导出一个版本，不能沿用旧包假装配置未变。

## 2. 启动服务

```bash
nohup bash scripts/server/start_live_composite.sh \
  > live-composite-launch.log 2>&1 < /dev/null &
echo $!
```

默认解释器：

- `PYTHON_BIN=.venv-visual-npu/bin/python`
- `DETECTOR_PYTHON=.venv-yolo-npu/bin/python`
- `PERCEPTION_PYTHON=.venv-perception-arm64/bin/python`

按现有服务器环境设置 `DETECTOR_PYTHON`；不会自动 pip 安装或修改环境。
启动脚本先启动已有 YOLO11 服务，再加载全部组合模型。默认 Web 端口 8765、
检测器端口 8766，都只监听回环地址。端口占用会报错，不停止其他进程。

日志与令牌在 `runs/live-composite-session-001/`：

- `detector.log`：物体检测器启动和执行。
- `model.log`：模型装配、感知子进程错误、服务 `status=ready`。
- `open-url.txt`：包含临时访问令牌的本机页面 URL。

重启使用新的会话目录，例如：

```bash
SESSION_DIR=runs/live-composite-session-002 bash scripts/server/start_live_composite.sh
```

停止时对启动脚本 PID 发送 `kill <launcher_pid>`，脚本负责清理它启动的两个服务。
页面停止按钮仅关闭当前摄像头会话，不关闭模型服务。
页面意外退出后，超过 60 秒无帧的会话会自动释放。推理中点击停止会等待当前请求结束再清理。

## 3. 本机摄像头访问远程模型

在电脑终端建立 SSH 转发（替换实际 SSH 用户、地址和端口）：

```bash
ssh -N -L 8765:127.0.0.1:8765 用户@服务器地址
```

在服务器读取 `runs/live-composite-session-001/open-url.txt`，将其中的
`http://localhost:8765/#token=...` 粘贴到电脑浏览器。摄像头权限属于电脑浏览器，
不需要服务器连接 USB 摄像头。localhost 满足浏览器摄像头安全上下文要求。
没有 SSH 时需要配置 HTTPS 反向代理后再扩展服务的 Host/Origin 策略，当前版本
不会直接开放公网 HTTP。

打开摄像头，选择 body 或 face，核对驾驶员框和 body 紫色头部框，并明确勾选
驾驶员区域归属后开始推理。未确认归属时，body RGB/头部专用输入保持不可用。
需要调整驾驶员区域时，可直接在预览画面上按住鼠标拖拽框选：拖拽过程中显示黄色
虚线草稿框，松开后把归一化坐标写入输入框（过小或退化的框会被忽略），点击
「开始推理」时才真正生效。改变视角、分辨率或区域必须停止并新建会话。页面隐藏时停止摄像头。

## 时间与性能

- 25 Hz 因果特征网格，100 槽窗口，25 槽输出步长。
- RGB 分心采样槽每秒 `[12,24]`；头/眼 `[2,5,8,12,15,18,21,24]`。
- 仅使用已采集的帧，不用未来帧回填历史。超过 80ms 的源帧不可作为新像素证据。
- 跟踪最长 250ms，年龄从最后一次确认检测起算；闭眼至少连续 1000ms。
- 浏览器同时只提交一个请求，避免延迟不断累积；实际传输/推理速度不足会产生
  丢帧、不可用窗口或过期状态。这是需要联调测量的性能限制，不能靠放宽有效性伪装实时。
- 结果超过 1.5 秒失效，页面清空旧任务结果。四秒只是最短预热，不能保证任意来帧
  速度下所有分支均可用。上游 JPEG 压缩和新摄像头视角可能改变预测。

## 接口（开发契约 live v1）

除静态页面和 health 外都使用 `Authorization: Bearer <api.token>`。同源访问，
没有通配 CORS。没有实现旧 `docs/api/openapi.yaml` 中的业务告警和数据库端点。

| 方法和路径 | 用途 |
|---|---|
| GET `/api/live/health` | 模型已加载的服务存活状态 |
| GET `/api/live/status` | 模型校验值、当前会话、未过期结果和错误 |
| POST `/api/live/sessions` | 创建会话：view、image_size、归一化 driver_roi、显式 driver_region_confirmed_exclusive |
| POST `/api/live/sessions/{id}/frames` | `{timestamp_ms, jpeg}`；时间从 0 开始严格递增，JPEG 为 base64 |
| DELETE `/api/live/sessions/{id}` | 停止并清除感知/时序状态 |

帧响应 `status` 为 warming/ready/stale。`result.tasks` 的每项包括 available、
logit、score、threshold_logit、detected、source。不可用的 score/detected 为 null。
`result.gaze` 包含类别分数和视角验证状态。错误返回 error 和 result:null；推理错误
会关闭会话，不能继续复用已部分推进的状态。忙碌返回 409，不排队摄像头帧。
停止请求最多等待 35 秒；页面也会处理权限弹窗或会话创建期间的取消操作。

## 验收边界

本地测试覆盖采样、时序、真实 PyTorch 子结构的离线输入对齐、路由与 HTTP 生命周期。
需要在持有真实模型和 NPU 的服务器验证：完整包导出/加载、浏览器摄像头、源帧率、
端到端延迟、断流重连以及四任务/注视输出。此前验证集成绩不会自动成为新摄像头成绩。
当前未重新训练，也未宣称完成真实 NPU/摄像头验收。
