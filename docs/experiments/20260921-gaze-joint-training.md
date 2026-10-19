# 四行为 + Gaze 联合训练工程验证

2026-09-21 · shadow · `model` 分支。范围：新版感知/标签/缓存/联合训练/续训/预测；不是DMD精度报告。

## 实现与审核

- 原四行为门控保留，九类gaze辅助头使用共享F时序表示；23特征拼接mask为46通道。使用现有依赖，无新增安装。
- 新版显式runtime-v2、interaction-v2、groups-v2、training-v3及record/manifest-v3；旧入口、缓存和checkpoint不自动升级。
- 独立规格审核与质量审核已完成。修复了配置根节点非对象时未受控的异常、完全无监督batch遗漏未知指标统计；均先测试复现，再验证修复。
- 另有回归覆盖未监督分类头不被Adam权重衰减更新、checkpoint映射校验、源代码指纹包含新模块、完整缺失传播和NumPy/Torch可用性一致性。

## 本机测试

最终命令：

```powershell
.venv-training/Scripts/python.exe -m unittest discover -s tests -q
git diff --check
```

最终回归：253项，251通过、2跳过，耗时104.429秒，退出码0。跳过项为Windows符号链接权限相关用例。Git读取用户全局ignore出现权限警告，但仓库同步断言通过；新源码、配置和测试的check-ignore检查未命中忽略规则。diff检查无空白错误，存在LF/CRLF提示。

新增26项测试覆盖几何、标签、批处理和联合训练。端到端测试在感知API边界使用模拟输出，之后运行真实标签转换、缓存/manifest发布、CPU训练；不宣称完成真实视频联合训练。

CPU续训与连续训练结果一致，checkpoint重载预测可用；未知/不可用gaze不输出伪类别。跳过无监督batch不改变模型、损失和随机状态。CPU/GPU/NPU的`-S --check-config`测试通过，只证明配置与薄入口可用，不验证设备。

## 用户真实样例的只读验证

1. 对用户提供的preparation运行新版`init`，生成105任务（Dist70+Drows24+Gaze face11），默认未确认ROI为0个已确认；本机输出位于被忽略的`runs/gaze-full-local-cli-review-20260921.local.json`，没有修改原preparation。它是本机检查工件，服务器应按指南重新生成并复制已确认ROI。
2. 读取`gA_1_s6_2019-03-08T09;15;15+01;00`真实gaze标注，按5340个声明帧解析得到：left_mirror269、left214、front222、center_mirror215、front_right218、right_mirror186、right204、infotainment169、steering_wheel435、unknown3208。这里只验证标注帧与类别映射，没有解码视频或确认PTS，不能当作训练窗口数量、可用率或全体覆盖。
3. 本机真实MediaPipe对用户提供的face截图执行新版接口：status=ok，478个关键点，头姿估计yaw约-16.07°、pitch约-21.29°、roll约2.34°，左右虹膜特征均为有限值。只证明实际API与特征接口可工作；不证明相机标定、眼动精度或服务器视频可用率。

## 未执行与后续

- 未连接服务器、未启动全量感知或新版NPU/CUDA训练；没有正式联合训练权重或真实指标。
- 原四任务NPU合成验证是用户此前回传的独立证据，不能替代本轮验证。
- 新版全量路径见[正式训练指南](../training/GAZE_FULL_RUNBOOK.md)。仍需确认剩余ROI、提取新版特征、检查可用标签覆盖，然后启动NPU训练；无需另跑旧八段/十二段pilot。
- Gaze批处理使用显式新feature/group profile，旧单视频CLI继续维护旧版用法。公共应用API、风险判定、数据库未在本轮扩展。
- 分支保留`model`；未自动提交、推送、合并或删除旧缓存。
