# 昇腾服务器初始化：先同步文件，再安装与检查

2026-09-20 · shadow · 适用当前已有 CANN/PyTorch/torch_npu 的 Linux 服务器。

本指南不代表已经登录服务器或完成远程安装。本地核查发现 `tests/` 曾整体被忽略，测试源文件未进入 Git；`Dataset/` 也未被大小写敏感的 Linux 忽略规则覆盖。本次移除 `tests/` 规则并增加 `/Dataset/`。模型结构、训练入口、JSON 模板及 requirements 原本已跟踪。

本地验证记录：新增 3 项 Git 忽略回归检查先复现错误，修正后通过；在 Windows `.venv-training` 执行 `python -B -m unittest discover -s tests -t . -v`，131 项通过（35.006 秒）。期间沙箱读取用户全局 Git ignore 出现权限警告；另以只读授权核对实际 Git 状态与忽略文件。`npu_train --check-config` 返回 `config_valid`、`device_verified=false`。上述均不是服务器安装或 NPU 训练成功记录。

## 1. 先把本次本地改动同步到服务器

删除忽略规则不会自动把文件加入历史。需要将 `.gitignore`、`tests/` 的 Python 源文件、本指南和相关文档/requirements 改动一起提交，再通过现有方式同步到服务器。先检查差异，不能只提交 `.gitignore`，也不要 `git add -f` 数据或模型目录。

本地检查（Git 命令在 Windows/Linux 均可运行）：

```bash
git status --short --untracked-files=all
git ls-files --others --exclude-standard tests
git diff -- .gitignore requirements-npu.txt README.md docs/README.md docs/training/RELATION_RUNBOOK.md
```

提交/推送由操作者按现有流程进行；本次工具修改不等于已经提交、推送或部署。

服务器在已同步本次修改后，执行：

```bash
cd /mnt/workspace/DriveGuard-AI
git branch --show-current
git rev-parse --short HEAD
git status --short
git ls-files --error-unmatch requirements.txt requirements-npu.txt training/relation/model.py training/relation/smoke.py training/relation/npu_train.py configs/training/relation_gated.json configs/perception/runtime.example.json tests/test_repository_sync.py docs/training/SERVER_SETUP.md
```

最后一条必须列出全部文件且退出码为 0；缺失时先解决版本/提交同步，不要继续安装或运行测试。不要盲目 `reset --hard` 或覆盖服务器改动。

文件分类：

| 文件 | 是否通过 Git 同步 | 服务器处理方式 |
| --- | --- | --- |
| training、inference、tests 源码 | 是 | 跟随同一提交；tests 不影响普通训练导入，但用于复验 |
| requirements、普通配置和 example 模板 | 是 | 不使用本机虚拟环境替代依赖清单 |
| Dataset、data、datasets | 否 | 原始数据/特征留在服务器，路径单独配置 |
| .venv、.venv-*、缓存、runs | 否 | 不复制 Windows 环境，不上传运行产物 |
| models/local、上游 YOLOX 源码 | 否 | 真实视频阶段单独获取、核对来源与 hash |
| configs/*/*.local.json | 否 | 按服务器路径、真实录制信息和驾驶员区域填写 |
| models/releases 中选定权重 | 可跟踪 | 保留正式权重公开交付要求，需确认许可 |

`git check-ignore -v Dataset/example.mp4` 应命中 `/Dataset/`。忽略规则不会移除已跟踪数据；若 `git ls-files Dataset` 有输出，先核对后再处理，不在初始化过程中删除数据。

## 2. 确认 Python，再安装 NPU requirements

沿用之前已经通过 NPU Linear 前后向的解释器/容器，不重新安装 CANN，不创建无法访问预装 torch 的空环境。

```bash
python3 --version
python3 -m pip --version
python3 -c "import sys, platform; print('python:', sys.executable); print('machine:', platform.machine())"
python3 -c "import torch; print('torch:', torch.__version__); print('path:', torch.__file__)"
```

仓库记录的服务器基线为 Python 3.11.4、torch 2.7.1+cpu、torch_npu 2.7.1.post4、CANN 9.0.0。这是已有环境记录，不是新服务器自动适配承诺。若 torch 版本不同或不能导入，先提供输出；不要通过盲目升级来修复。

可以先检查 pip 将执行什么（不安装）：

```bash
python3 -m pip install --dry-run -r requirements-npu.txt
```

如果计划替换已工作的 torch/torch_npu，先停下核对解释器与版本。确认后安装：

```bash
python3 -m pip install -r requirements-npu.txt
```

注意：必须有 `-r`。此文件通过 `-r requirements.txt` 包含共享依赖，不必再单独安装 requirements.txt。不要加 `--upgrade`，不要把 requirements-perception-win-cpu.lock.txt 安装到昇腾环境。

`Requirement already satisfied` 是正常结果；`Defaulting to user installation` 本身不是报错，但用户目录包可能覆盖系统包，所以下一步需打印真实导入路径。

## 3. 检查环境与配置，不先跑全量训练

```bash
python3 -m pip check
python3 -c "import numpy, torch, torch_npu; print('numpy:', numpy.__version__, numpy.__file__); print('torch:', torch.__version__, torch.__file__); print('torch_npu:', torch_npu.__version__, torch_npu.__file__); print('npu_available:', torch.npu.is_available())"
python3 -S -m training.relation.npu_train --check-config
python3 -m unittest tests.test_repository_sync -v
```

逐条执行；任何命令异常，保存完整输出，不继续启动 NPU 训练。pip check 可能揭示镜像预装包已有的冲突，需要判断与本项目的关联，不能自动卸载/升级整个环境。

这里的 NPU 检查仅做导入与可用性查询，不重跑 Linear，也不证明完整关系模型算子已验收。配置检查应返回 `config_valid` 且 `device_verified=false`；后者是预期结果，不是假装做过硬件验证。仓库同步测试需要 Git，但不需要数据、torch、MediaPipe 或模型权重。

到此才可称“服务器训练依赖与配置预检通过”，并不等于真实训练或完整感知环境可用。

## 4. 最小训练闭环：CPU 合成数据

```bash
python3 -m training.relation.smoke --output-dir runs/relation-smoke-server-001
```

输出目录必须不存在，重复运行更换编号。它只生成少量合成特征，执行训练/验证/保存/续训/重载，不访问 Dataset，不使用 NPU，不下载模型。

检查输出报告中的 `resume_matches_continuous=true`、`reload_finite=true`、`parameter_count=13524`。所有结果仅用于工程验证。成功后按需运行完整测试：

```bash
python3 -m unittest discover -s tests -t . -v
```

现有感知单元测试使用外部 API 替身；完整测试通过也不等于真实 YOLOX/MediaPipe 在该服务器可运行。

## 5. 被忽略的感知文件：单独准备，不混入 NPU 环境

CPU 合成训练和读取已提取特征都不依赖以下文件；原始视频提取则必须具备：

- models/local/yolox_nano.pth
- models/local/face_landmarker.task
- models/local/hand_landmarker.task
- YOLOX 0.3.0 对应源码及独立感知依赖
- configs/perception/runtime.local.json

三个模型的固定官方 URL、SHA-256 和许可记录见[工件清单](../../models/PRETRAINED_ASSETS.md)。可从已有本机文件传输，也可从清单中的官方来源获取；传输后用 `sha256sum` 对比清单，不把这些文件强制提交到 Git，不复制 Windows 虚拟环境。

配置以[模板](../../configs/perception/runtime.example.json)为起点；其中路径/hash/来源以及录制 ID、ROI、驾驶员独占区域确认都是占位信息，需要填写，模板原样不能运行。不要照搬 Windows 空白帧测试用的 runtime.local.json。

独立感知环境的依赖方案需根据上面输出的 Linux CPU 架构和 Python 版本核对；当前仅记录过 Windows x64 的真实工件加载，不能声称 requirements-perception.txt 已在 Linux x86_64/aarch64 验证。先提供架构信息，再配置这部分；不向已工作的 NPU 环境直接安装 MediaPipe/torchvision。

实际视频调用和受限上游依赖说明见[感知运行指南](EXPERT_RUNTIME.md)。先做一小段单视角视频的特征检查，再进行全量提取。

## 6. 真实数据训练的剩余前置条件

还需 distraction/drowsiness/gaze 各一份真实标注 JSON，用于实现并核验原始标签映射；视频留在服务器。当前原始 DMD 标注解析器尚未实现。真实视频提取默认导出的标签为未知，不能把它们当作负例或直接开始四任务训练。

完成特征、有效性掩码、真实标签、按被试划分的 v2 manifest 后，再使用[关系门控训练指南](RELATION_RUNBOOK.md)的 NPU 入口。`--data-root` 指特征目录，不是 Dataset 原始视频目录。不要把合成 manifest 的标签或主体身份复制成真实数据元信息。
