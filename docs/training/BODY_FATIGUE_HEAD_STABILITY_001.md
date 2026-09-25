# Body 疲劳：头部模型训练稳定性对照

上一轮头部 seed42 best 在固定阈值下检出 12/61、误报 3/215，其中 subject7 检出 4/35。
两个头部模型的最佳轮均为第 1 轮；末轮误报升至 47、44 个。本轮固定头部输入，
比较降低学习率和增强正则化是否保住早期收益。效果仍需真实服务器实验确认。

## 四组，两种子，共八次顺序训练

| 组 | 学习率 | dropout | weight decay |
| --- | --- | --- | --- |
| A_control | 原值（本次 0.001） | 原值（0.1） | 原值（0.0001） |
| B_slow | 原值 /10（0.0001） | 原值 | 原值 |
| C_regularized | 原值 | 0.4 | 0.01 |
| D_slow_regularized | 原值 /10 | 0.4 | 0.01 |

每组 seed42、43。使用普通 BCE；模型容量、窗口、头部区域和训练集归一化保持一致。
增强正则化组同时改变 dropout 与 weight decay，不能从这一轮区分两者的单独贡献。
复用旧特征，训练模型从各自种子初始化；不从验证集选出的 best 微调。
训练其余参数继承空间对照报告，通常最多 20 轮、耐心值 4、batch64。
best 仍按验证 BCE 保存，last 用于观察训练退化；不改为按这 61 个正例挑选轮次。

## 启动

将本次代码同步至服务器 `/mnt/workspace/DriveGuard-AI` 后，在仓库根目录执行：

```bash
mkdir -p runs
nohup bash scripts/server/train_body_fatigue_head_stability.sh \
  > runs/body-fatigue-head-stability-001.log 2>&1 &
echo $!
```

```bash
tail -f runs/body-fatigue-head-stability-001.log
```

默认读取 `runs/body-fatigue-spatial-001/comparison/comparison.json`。由此定位旧检查点、
准备数据、整体与头部 RGB 缓存；路径不同可设置 `SPATIAL_REPORT`。
默认解释器 `.venv-visual-npu/bin/python`、NPU 0，可用 `PYTHON_BIN`、`DEVICE_INDEX` 覆盖。
不重新提取视频特征，不下载权重。旧参考的哈希、来源、分数与指标复现失败时停止，
应检查输入和运行环境，不要删除校验或手工修改报告让实验继续。

输出默认 `runs/body-fatigue-head-stability-001`，可用 `OUTPUT_DIR` 指定新目录。
异常中断后确认原进程结束，再执行：

```bash
nohup bash scripts/server/train_body_fatigue_head_stability.sh --continue-run \
  >> runs/body-fatigue-head-stability-001.log 2>&1 &
echo $!
```

已完成训练复用，未完成训练使用已校验的 epoch 检查点恢复。输入或参数改变会拒绝续跑。
不要同时往同一输出目录启动两个进程。

## 回传结果

- `comparison.json`：两个旧头部 best 参考，以及八次训练的 best/last，共 18 行。
- `comparison.csv`：整体固定阈值 TP、FP、FN、F1、召回、误报率与 3% 误报条件是否满足。
- `by-subject.csv`：每个人各自的 TP、FP、FN、分母和各项指标。
- 各组 `train-001/history.json`：每轮训练/验证 BCE 和总体混淆矩阵。
- JSON 的 `paired_best_deltas`：各组相对同种子 control 的变化。
- JSON 的 `last_minus_best`：末轮相对最佳轮的变化；FP 为正表示误报增加。
- JSON 的 `aggregate_best`：各组和各受试者跨种子汇总。

优先回传上述三个比较文件；需要定位训练退化时再看 history.json。
固定阈值为 logit 0（概率 0.5）。FPR03 扫描仅作验证诊断，不能替代固定阈值结果。
先看每个种子的误报率是否不超过 3%，再看召回；不能用两种子平均误报率掩盖单次超标。
同时比较 subject7 是否保留/增加检出，subject14 是否退步，并检查 best/last 差距。
未知标签不作为负例，重叠窗口不等于独立疲劳事件。

本轮仅训练和评估 body 疲劳候选，不自动替换当前组合模型。候选确认后再做全任务回归评估，
验证凝视、分心、电话、饮水等输出。两位验证受试者上的改善不等于独立测试或部署效果。
