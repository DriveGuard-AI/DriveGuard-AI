# 本机感知环境与工件准备

2026-09-19 · shadow · model分支。已完成用户指定的第1步：三个真实工件、独立CPU环境、本地模型配置及实际加载检查。**没有开始第2步真实视频验证，没有恢复DMD处理或训练，也没有配置远程昇腾服务器。**

## 已准备的位置

| 项目 | 当前Windows工作区位置 |
| --- | --- |
| 独立解释器 | `D:\DriveGuard-AI\.venv-perception\Scripts\python.exe` |
| YOLOX/Face/Hand模型 | `D:\DriveGuard-AI\models\local\` |
| 已填工件的本地配置 | `D:\DriveGuard-AI\configs\perception\runtime.local.json` |
| 安装日志、下载receipt、加载报告 | `D:\DriveGuard-AI\runs\perception-setup-20260919\` |

上述环境、模型副本、本地配置和运行产物均被Git忽略。可版本管理的[工件清单](../../models/PRETRAINED_ASSETS.md)记录来源、许可、字节数和SHA-256；[Windows CPU版本快照](../../requirements-perception-win-cpu.lock.txt)记录已安装依赖。不是把虚拟环境提交到仓库。

原 `.venv-training` 保持 torch2.14.0+cpu / NumPy2.3.5；requirements.txt及requirements-npu.txt本轮未修改。本机与远程服务器并不同步，不能认为服务器已经有这些文件。

## 实际加载结果

Python3.12.14（Windows x64）：torch2.7.1+cpu、torchvision0.22.1+cpu、YOLOX0.3.0、MediaPipe0.10.21、OpenCV4.11.0、PyAV12.3.0、NumPy1.26.4。

三个工件均通过本地SHA-256核对及真实模型构造；YOLOX严格 `weights_only=True` 加载成功。两张640×480空白BGR帧（0/40ms）实际经过三个模型和特征桥接：三个专家status均为ok，均无目标/脸/手；每帧19个有限特征，仅两个物体置信度零值有效，其余缺失mask为0。整个检查包括创建和关闭模型约0.625秒，**不是性能基准或真实视频FPS**。

原始证据为忽略目录中的 `model-load-report.json`，复验脚本为同目录 `verify_models.py`。测试帧没有人，因此尚未证明关键点坐标和驾驶场景检测质量；PyAV只验证导入，尚未在真实视频中验证解码/容器限制。

脚本检查帧数、时间戳、专家状态、特征维度/有限值及空白帧预期；不满足则记录FAIL并非零退出。再次执行须指定尚不存在的报告路径，保留原始证据，例如：

```powershell
.\.venv-perception\Scripts\python.exe -B runs/perception-setup-20260919/verify_models.py --output runs/perception-setup-20260919/model-load-recheck.json
```

复验脚本和报告仅保留于当前机器的Git忽略目录，不随仓库分发；上述命令不是克隆仓库后的通用安装入口。

MediaPipe输出XNNPACK、feedback tensor和NORM_RECT/IMAGE_DIMENSIONS警告；本次未抛异常，已如实保留。尤其坐标投影警告不能用空白帧排除影响，下一步须用真实非正方形视频核对关键点位置。

全量105项unittest在新环境（torch2.7.1+cpu / NumPy1.26.4）通过，15.863秒；原训练环境安装前基线同样105项通过。测试本身仍采用外部API替身，实际工件前向为单独检查，两类证据不混用。

## 已知依赖边界：不是全部pip检查通过

`pip check`返回非零，并报告上游YOLOX的4项声明依赖未满足：

```text
yolox 0.3.0 requires onnx, which is not installed.
yolox 0.3.0 requires onnx-simplifier, which is not installed.
yolox 0.3.0 requires onnxruntime, which is not installed.
yolox 0.3.0 requires opencv-python, which is not installed.
```

原因与处理范围：

- 上游0.3.0把旧ONNX导出工具列为必需依赖，本次仅启用PyTorch推理，不启用其ONNX导出；没有为了消除警告安装过时导出包。
- 使用单一 `opencv-contrib-python` 提供真实可导入的cv2，满足MediaPipe；不再安装 `opencv-python` 覆盖同名cv2。pip的包名检查无法把两个distribution视为替代。
- YOLOX由[固定官方提交](https://github.com/Megvii-BaseDetection/YOLOX/commit/419778480ab6ec0590e5d3831b3afb3b46ab2aa3)源码以 `--no-deps --no-build-isolation` 安装，模型源码未改，Windows不支持的演示文档符号链接在解压时排除。
- 本次验收为**受限CPU感知运行时可加载/前向**，不是完整YOLOX训练、导出或全依赖一致环境。未来启用训练/ONNX时另建匹配环境，不能忽略新出现的依赖错误。

## 后续使用

无需激活环境或修改PowerShell执行策略，可以直接调用：

```powershell
Set-Location D:\DriveGuard-AI
.\.venv-perception\Scripts\python.exe -B -m inference.perception.video --config configs/perception/runtime.local.json --check-config
```

本地配置的**模型部分已填好**；但录制ID是 `setup-smoke-not-a-real-recording`、ROI是640×480测试范围、驾驶员归属为false。真实视频运行前须替换录制ID/视角/原图ROI；未经逐录制确认不能把归属设true。不要直接把这份测试ROI套到DMD视频上。

重建同类Windows CPU环境时，先安装Python3.12并创建被忽略的独立venv；安装[版本快照](../../requirements-perception-win-cpu.lock.txt)，再安装核对过的YOLOX固定源码：

```powershell
.\.venv-perception\Scripts\python.exe -m pip install -r requirements-perception-win-cpu.lock.txt
.\.venv-perception\Scripts\python.exe -m pip install --no-deps --no-build-isolation --use-pep517 .\models\local\yolox-0.3.0-src
```

快照只固定版本，不包含下载hash锁定或源码；必须先按[工件清单](../../models/PRETRAINED_ASSETS.md)准备源码与工件。不要将Windows CPU快照用于Linux/Ascend/CUDA。本机C盘空间很少，安装时应将进程的TEMP/TMP/PIP_CACHE_DIR指向D盘项目临时目录，避免默认缓存占满C盘。

下一步是用户提供/选定一段短视频后按[运行指南](EXPERT_RUNTIME.md)验证真实感知输出，不需要重新做NPU小型Linear检查。YOLOX权重再发布许可仍为pending_permission，不能作为本项目正式公开权重交付。
