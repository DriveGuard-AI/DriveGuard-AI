# YOLO11服务器并行提取：工程验证

2026-09-21 · shadow。分支model。用户要求只准备服务器配置与代码，本机不安装环境、不下载模型。使用现有 `.venv-training` 验证；没有连接用户NPU服务器。

## 本轮实现

- YOLO11s 640 FP32单例NPU检测服务；原MediaPipe CPU Face/Hand/gaze保留。
- CPU/NPU环境分离、固定依赖/权重来源、无隐式设备回退；二进制帧通过鉴权回环HTTP传递，不传pickle。
- spawn视频调度：默认6 dist＋3 drows＋3 gaze；每路独立有状态感知实例；有界提交与逐任务日志。
- 非覆盖配置迁移保留ROI、preparation hash和原始文件；感知版本进入缓存溯源。
- nohup监督、就绪检查、自动发布、合作停止及记录级恢复。

## 实际验证证据

```text
.venv-training/Scripts/python.exe -m unittest discover -s tests -q
Ran 288 tests in 65.201s
OK (skipped=2)
```

即286项通过、2项跳过。包含35项新增检测服务、调度和服务器集成回归；独立质量审核重新运行这35项，也全部通过。

四个 `scripts/server/*.sh` 分别经已有Git Bash的 `bash -n` 语法检查通过；server_setup、server_runner及gaze_batch run的 `python -S ... --help` 通过；`git diff --check` 通过。测试中Git报告沙箱不能读取用户级ignore配置，但仓库测试与命令退出成功；这不代表服务器Git配置故障。

红绿回归覆盖：原配置不支持远程变体、缺少并行入口、服务指纹不匹配未阻止记录发布、服务异常误报停止、发布停止被忽略、强制终止发布器遗留锁，以及端口在模型构造后才绑定等问题。最终规格和独立质量审核均PASS。

## 未验证或不应推断的内容

没有使用真实YOLO11权重在本机推理，没有验证Ascend910B3算子、Linux完整nohup生命周期、服务器依赖安装、全量提取时长、DMD召回/精度，也没有完成新版正式训练。socket协议、spawn调度及边界替身测试不等于这些真实运行已经成功。

服务启动会做真实NPU前向预热并失败即停；全量运行后仍需看缓存发布覆盖报告和真实评估。NPU检测当前batch=1，12路指视频CPU任务，不是12路NPU并发或微批推理。既有240GiB可见内存不保证容器配额或性能。

操作入口见[服务器全量提取](../training/SERVER_EXTRACTION.md)。本轮不删除旧缓存、不发布权重、不自动提交代码。
