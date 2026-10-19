# Body 分心改进实施计划

> **For agentic workers:** Use superpowers:subagent-driven-development for independent components, then review spec compliance and code quality. 用户已授权编码；沿用当前工作区，不提交、不推送。

**Goal:** 提供可直接用旧缓存训练的独立头姿补充模型，以及只读感知对照实验。

**Architecture:** 保留四组契约，以新model.kind增加仅供分心的H分支；任务可用性贯穿训练/审计/推理/诊断。感知实验独立入口，不修改生产提取默认参数。

**Tech Stack:** Python, NumPy, PyTorch, MediaPipe Tasks, PyAV, OpenCV；已有训练/感知虚拟环境。

- [x] 模型与任务可用性：tests/test_gaze_head.py先验证缺失能力；training/gaze/head_evidence.py纯NumPy契约和head_model.py torch实现；common/config.py和model_factory.py允许显式新kind。
- [x] 集成：gaze/train.py、data.py、checkpoint.py、predict.py、diagnostics.py适配各任务可用性，旧路径不变。使用gaze_fixture测试真实训练/续训/重载；验证原模型断点拒绝进入新模型。
- [x] 感知实验：独立代理实现training/dmd/face_experiments.py及相关独立模块、实验配置、tests/test_face_experiments.py；只复用face_diagnostics只读验证接口，不改共同训练文件。
- [x] 诊断与服务器指南：新增配置文件及docs/training/BODY_DIST_IMPROVEMENT.md；明确新模型必须从头训练、指标覆盖口径、实验数据不能直接发布特征。
- [x] 验证：运行新测试及gaze、relation、perception相关回归、stdlib --check-config、真实OpenCV测试及git diff --check。对两个组件独立审查规范与实现，修复后汇总实测边界。

关键验收例：`head_mask[:,5:8]=1`而其他面部mask=0时旧F失败、新H通过；只剩H时task_available应为[False,False,False,True]；`head_mask[-1,5]=0`后H必须失败。实验基线必须使用与原VIDEO重放完全一致的调用时序和原工件哈希。

## 验证记录

本机`.venv-training`：gaze系列43项（含独立H新测试10项）、relation系列23项、face diagnostics/replay与expert MediaPipe/video/runtime flow共33项、感知实验新测试10项全部通过。`.venv-perception`真实OpenCV截图/HTML测试2项通过。总计111项，无失败。NPU训练新配置和感知实验均通过stdlib配置预检查，感知环境CLI帮助与git diff --check通过。未运行真实NPU或DMD视频性能实验。

新H模型以dropout=.1验证连续两轮与一轮后续训的model_state、optimizer_state、history完全一致；旧→新及新→旧断点均拒绝。独立审查发现H不合格时其quality仍进入分心门控，已先用多原组可用、原F质量相同而H同时覆盖不同的测试复现，再对不合格H的质量向量归零，复审通过。

感知实验完成独立规范与代码审查，未发现重要问题；审查方另行复跑10项实验测试通过。改动保留在当前工作区，未提交或推送。
