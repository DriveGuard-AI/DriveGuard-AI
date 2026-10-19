# coverage25 输出与现有 C_fusion 的组合对照

## 比较内容

保持原 C_fusion 的权重、gap80 上下文、视觉特征缓存及阈值不变，仅替换
其他任务的输出来源。评估同一验证集上的两个完整基础模型及六种组合：

| 非 body-dist 输出来源 | body-dist 输出 |
|---|---|
| gap80 | gap80（原完整模型） |
| coverage25 | coverage25（原完整模型） |
| gap80 / coverage25 各一组 | C42 best，logit阈值0 |
| gap80 / coverage25 各一组 | C43 best，logit阈值0 |
| gap80 / coverage25 各一组 | C43 best，原验证报告的低误报阈值 |

其他输出包括 body 疲劳/手机/饮水、face 四行为和所有 gaze 输出。每个
基础模型使用自己的训练配置及 normalizer。只有窗口、原始标签和源数据
身份必须一致，头姿可用性可以不同。body 错误分组仍按冻结的 gap80 H 状态。
RGB不可用时保留原 C 分支的 gap80 回退，不让候选输出基础模型改变该分支。

这次不做训练或阈值搜索；沿用原实验选定的验证工作点。结果不能当成独立
test成绩。三个随机种子之外，不能通过本次挑选进一步声称普遍最优。

## 一次运行全部评估与疲劳诊断

先同步本次 Python 文件和服务器脚本。进入服务器项目根目录后执行：

```bash
nohup bash scripts/server/evaluate_coverage25_composite.sh \
  > runs/coverage25-composite-001.log 2>&1 &
echo $!
```

脚本依次执行：全任务对照、gap80 疲劳证据审计、coverage25 疲劳证据审计。
使用一张NPU串行处理，避免两个进程争抢影响耗时比较。没有新增训练实验，
不会重新下载权重、解码视频或提取RGB特征。已有输出目录拒绝覆盖。

默认输入仍位于原服务器目录：

```text
runs/body-visual-comparison-001/prepared-001
runs/body-visual-stability-001/comparison.json
runs/relation-gaze-head-coverage25-npu-001/best.pt
```

gap80权重、manifest和数据根路径从 prepared.json 读取，两次疲劳诊断也使用
同一数据源并分别核验对应模型的fingerprint。需要两套权重及其SHA256 sidecar、
prepared/visual缓存，以及原始结构化特征缓存。基础模型路径变化时：

```bash
OUTPUT_BASE_CHECKPOINT=/absolute/path/to/coverage25/best.pt \
OUTPUT_DIR=runs/coverage25-composite-002 \
bash scripts/server/evaluate_coverage25_composite.sh
```

也可覆盖 `PYTHON_BIN`、`PREPARED_DIR`、`STABILITY_REPORT`、`BACKEND`、
`DEVICE_INDEX`、`BATCH_SIZE`。本机CPU测试只证明执行正确，真实效果需服务器报告。

默认结果位置：

```text
runs/coverage25-composite-001/all-tasks/report.json
runs/coverage25-composite-001/all-tasks/metrics.csv
runs/coverage25-composite-001/all-tasks/predictions.jsonl
runs/coverage25-composite-001/fatigue-gap80/summary.json
runs/coverage25-composite-001/fatigue-gap80/windows.csv
runs/coverage25-composite-001/fatigue-coverage25/summary.json
runs/coverage25-composite-001/fatigue-coverage25/windows.csv
```

优先回传全任务 `report.json` 和两份疲劳 `summary.json`。疲劳字段解读见
[疲劳证据审计](BODY_FATIGUE_DIAGNOSTICS.md)。如果后续阶段失败，已完成报告
仍可查看；排除原因后可单独用疲劳CLI和新的输出目录运行失败阶段。

## 成本口径

报告记录缓存特征评估阶段的墙钟耗时、模型张量字节数，以及加速器支持时
记录该阶段的张量分配峰值。耗时包含相应阶段的主机工作和缓存访问，明确
标记是否含加载权重；没有warmup，不等于部署延迟。C分支的gap80上下文
已预先计算，因此不能把probe推理阶段耗时当成整个组合的在线成本。
阶段顺序、冷启动与文件缓存会影响单次时长，不据此宣称固定的加速比。

新增组合在实际运行时仍需 coverage25、gap80 和一个 C_fusion 头，另有
共同依赖的RGB编码器。张量字节数不含激活、中间上下文、视频缓冲和运行库；
分阶段测得的显存峰值不代表这些模型同时常驻的峰值，CPU显存字段为空。

加速器计时前后做同步；张量分配峰值的接口语义参考
[PyTorch内存统计](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html)、
[Ascend内存实现](https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/memory.py)
及[Ascend同步接口](https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/utils.py)。

## 判断顺序

1. 数据与两来源一致性审计全部通过，C分支逐窗口输出不变。
2. 全任务和body/face指标：是否找回疲劳/手机/饮水，gaze与face分心付出多少代价。
3. 覆盖率与受试者分组，不能仅凭汇总F1宣布所有任务更优。
4. 对比新增模型依赖的成本，暂不将多个模型的组合当作最终部署方案。
5. 疲劳诊断区分无标签、无输出、面部证据缺失、低分及门控现象；仅有
   相关性不足以判断哪项是根因，也不因此自动改阈值。

## 本地验证

2026-09-24：33项相关测试通过（组合与成本17项、疲劳诊断9项、旧诊断7项）；
两入口CLI help与Git Bash语法检查通过。使用真实模型的合成CPU数据验证，
未运行服务器真实权重。复现命令：

```powershell
.venv-training/Scripts/python.exe -m unittest tests.test_evaluation_cost tests.test_visual_all_tasks tests.test_visual_output_base
.venv-training/Scripts/python.exe -m unittest discover -s tests -p test_fatigue_diagnostics.py
.venv-training/Scripts/python.exe -m unittest discover -s tests -p test_gaze_diagnostics.py
```
