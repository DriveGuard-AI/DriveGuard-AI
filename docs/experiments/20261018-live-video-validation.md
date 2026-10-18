# Live 视频端到端验证（本机 CPU）

2026-10-18 · GitHub Copilot · `model` 分支。范围：真实视频 → live 服务 → 四任务/注视输出。
不涉及训练、阈值调整、NPU 执行或应用层接入。

## 环境与素材

| 项 | 值 |
|---|---|
| 素材 | `D:\data\Dataset\Video Data\Electronic Media\electr1.mp4` |
| 原始规格 | 1920×1080 · 28.974 fps · 1114 帧 · 38.4 秒 · 47.1 MB |
| 内容 | 驾驶员视角，模拟方向盘，人物全程手持手机贴耳 |
| 会话参数 | `view=body`、`image_size=1280×720`、`driver_roi=[0.15,0,0.75,1.0]`、`driver_region_confirmed_exclusive=true` |
| 喂帧方式 | 时间戳 `0,40,80…`（25 Hz 网格），与实时墙钟解耦 |
| 环境 | `.venv`（主服务）/ `.venv-perception`（感知）/ `.venv-detector`（YOLO11s），全部 CPU |

命令：

```powershell
.\.venv\Scripts\python.exe scripts\local\smoke_live_video.py `
  --video "D:\data\Dataset\Video Data\Electronic Media\electr1.mp4" `
  --output runs\live-local\electr1-full.jsonl --retries 2
```

## 第一次运行：中断

第 385 帧（15.4 秒时间线）失败，live 服务返回
`detector service request failed: [WinError 10060]`，并按设计关闭会话，整段视频作废。

定位过程：

1. 检测器进程存活、端口 LISTENING、`/health` 18ms 返回、CPU 空闲 → 不是崩溃；
2. 端口上 224 个 TIME_WAIT、0 个 ESTABLISHED → 连接被正常关闭，不是句柄泄漏；
3. 单独压测检测器 400 次（1280×720 帧）：全部成功，p50 122ms、max 210ms、零卡顿。

结论：检测器是单线程 `HTTPServer`，只有在 live 服务（6 个 torch 模型）+ 感知子进程
（MediaPipe）+ 检测器同时争抢 8 个核时，一个慢请求才会让 accept 循环长期停滞，
backlog 排满后新连接超时。

修复见 `docs/testing/KNOWN_LIMITATIONS.md` 的"已解决"表（`44c373b`）：连接改为线程化
接收，推理仍由一把锁严格串行。

## 第二次运行：完整通过

| 指标 | 值 |
|---|---|
| 帧数 / 中断 | 1114 / **0**（`segments=1`） |
| 结果数 | 1014（每 25 槽一个） |
| 时间窗口 | 0 → 43960 ms |
| 处理耗时 | min 172 / **p50 250** / max 421 ms |
| 墙钟 | 322.5 秒（0.14× 实时，约 7 倍慢） |

任务输出：

| 任务 | `available` | `detected` |
|---|---|---|
| fatigue | 1014/1014 | 25/1014 |
| phone_use | 1014/1014 | 989/1014 |
| drinking | 1014/1014 | **1014/1014** |
| distraction | 1014/1014 | 664/1014 |

状态翻转点（秒，`(fatigue, phone, drink, distraction)`）：

```
4.0  (F,T,T,F)   手机举起，phone 开始
7.0  (F,T,T,T)
8.0  (F,T,T,F)
14.0 (F,T,T,T)
19.0 (F,T,T,T)
29.0 (T,T,T,T)   fatigue 单点
30.0 (F,T,T,F)
32.0 (F,T,T,T)
43.0 (F,F,T,F)   手机放下
44.0 (F,T,T,T)   手机又抬起
```

注视：`steering_wheel` 589、`infotainment` 50、不可用 375；全部标记
`validation_status=unvalidated_body_view`。

## 结论

1. 链路本身可用：一次运行 1014 个结果、四任务全程 `available`、无中断。
2. `phone_use` 与素材事实一致（人物全程贴耳打电话）。
3. `drinking` 在该素材上恒为 `detected`，属**误报**——手部靠近脸/嘴的几何判据被手机
   触发。记录为 KL-01，需要产品决策：降级展示、重新训练，或暂不映射该分支。
4. `fatigue` 在 29 秒处单点触发，单次样本不足以判定真伪，记录为 KL-02。
5. 本机吞吐是不实时（0.14×），实时性结论仅适用于 CPU 环境，见 KL-04 与
   `docs/deployment/LOCAL_CPU_DEV.md`。

## 未验证

- 同一素材在 NPU/GPU 上的帧率与端到端延迟；
- 其他视角、遮挡、切镜头素材下的 `available` 比例；
- 峰值内存与长时间（>30 分钟）稳定性；
- 应用层消费这些输出后的风险等级与事件聚合行为（尚未接入）。
