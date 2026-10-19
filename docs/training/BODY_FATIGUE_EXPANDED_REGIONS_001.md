# 扩大头部区域对照 001

上一轮两张 `fatigue_known_positive_outside_head_region` 预览确认：第一段低头
时真脸候选越过区域右侧及下侧，第二段下巴附近略微越过下侧。此次只扩大区域，
检测阈值、模型和三组规则均不变。扩大区域并不证明眼部关键点准确。

| 记录 | 原归一化 xyxy | 新归一化 xyxy |
| --- | --- | --- |
| 4b680f020a0947c1fdebfea8 | [0.23923445, 0.03086787, 0.43839712, 0.39450424] | [0.23923445, 0.03086787, 0.50, 0.52] |
| cc3234cf4419361ccc018468 | [0.20275118, 0.02448829, 0.38277511, 0.32113901] | [0.20275118, 0.02448829, 0.38277511, 0.40] |

表格仅为显示而舍入，JSON 保留未修改边界的原精度、视频 SHA256 和图像尺寸。
新边界针对已提供的误拒绝画面留出余量；没有完整原视频，尚不能证明覆盖全程。
旧 `runs/head-region-record-{1,2}.json` 保留，新文件带 `-expanded-001` 后缀。

已合并配置：`runs/body-fatigue-head-regions-expanded-001.json`。
服务器处于 model 分支时，在仓库根目录执行：

```bash
git pull --ff-only origin model
OUTPUT_DIR=runs/body-fatigue-head-regions-expanded-001 \
  nohup bash scripts/server/compare_fatigue_head_regions.sh \
    runs/body-fatigue-head-regions-expanded-001.json \
    > runs/body-fatigue-head-regions-expanded-001.log 2>&1 &
echo $!
```

这是两段录像 × baseline/head_gate/head_crop 三组的六次顺序感知回放，不是训练。
输出目录必须不存在。先确认 pull 成功，再启动回放。

查看新报告中的正例窗口恢复及眼部时序有效率，并检查原先 1.378 s / 175.975 s
真脸候选是否保留，同时复查手部假人脸是否被重新接受。比较同一方案的新旧区域，
不要仅凭更多数值有效槽就认定效果改善。此次不修改生产缓存、权重或有效性规则。
