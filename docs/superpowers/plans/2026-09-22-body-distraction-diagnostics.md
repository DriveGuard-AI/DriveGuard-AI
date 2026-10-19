# Body 分心诊断实施计划

范围已获用户确认。顺序执行，在当前 model 分支工作；不提交或启动服务器训练。

1. 新增 tests/test_gaze_diagnostics.py：构造实际联合模型 checkpoint 和 gaze_fixture 缓存。设计已知正例/负例、未知标签、全缺失、头姿有效而 F 比例不足、过期面部输入，断言窗口末槽标签和时间、指标与 run_epoch 一致、指纹和覆盖保护。运行 `.venv-training/Scripts/python.exe -m unittest discover -s tests -p test_gaze_diagnostics.py -v`，先确认工具不存在时失败。
2. 新增 training/gaze/diagnostics.py：load_model + FeatureCorpus + WindowDataset；配置从 checkpoint 读取，CLI 可覆盖运行路径；核对 data_summary；按筛选后的索引分批 eval 推理。导出 CSV 和分组汇总，未知真值/不可用概率为空；summary.json 最后原子写入。输出用 mkdir(exist_ok=False) 保护。
3. 添加逐窗口原始/最终 mask 统计、F 筛选条件、四组门控权重、训练口径混淆计数、覆盖率、全部已知标签正例检出比例，以及固定阈值对照表。以测试区分不同缺失类型，避免缺失被当作阴性。
4. 更新 docs/training/GAZE_FULL_RUNBOOK.md：本次 epoch15 checkpoint 的 CPU/NPU 命令、必要文件、CSV 列语义、视频时间口径、阈值分析限制和结果阅读步骤。既有缓存不重提取。
5. 运行新增测试和 gaze 相关回归，做独立代码审核，修复发现的问题；完成 git diff --check。最终交付代码与服务器命令，明确尚未运行真实服务器诊断。

## 执行证据

- 步骤1：新增4项测试在工具尚未存在时均因 `diagnostic tool missing` 失败，确认测试覆盖尚未实现的能力。
- 步骤2–4：实现离线诊断与操作说明，另补CLI实际调用、原文件哈希不变和近零logit边界回归，共7项诊断测试。
- 独立审核发现FP32 sigmoid将极小负logit舍入为0.5的统计边界问题；新增测试先复现TP/FN不一致，再统一改为logit边界判定。保留概率和原始logit以便核查。
- `.venv-training/Scripts/python.exe -m unittest discover -s tests -p 'test_gaze*.py' -q`：最终33项通过，17.364秒；包括诊断、联合训练/续训、感知、标签与批处理回归。
- CLI `--help` 返回0；`git diff --check` 返回0。Git有用户级ignore权限和LF/CRLF提示，不影响这些检查。
- 服务器NPU诊断尚未执行；不推断真实body特征缺失比例或根因。
