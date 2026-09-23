# 单视频头部裁剪与连续性对照

用户同意在 H/coverage25 诊断后优先改善感知覆盖。当前不选择新 MediaPipe 模型，也不将裁剪角度写入训练缓存。

新增 `training.dmd.face_crop_preview`：验证 manifest/cache/video，按原 25 Hz 因果采样从一段 body 视频均匀导出 12 张完整画面与离线 HTML。用户在画面拖出或输入归一化框，所有样本同步显示该框；检查后下载仅包含 baseline/crop 两方案的 JSON。预览不猜测或默认选中裁剪框。抽样不保证穷尽头部活动，后续仍需完整视频对照与人工检查。

导出 JSON 携带可选 crop_scope：record_id、video_sha256、image_size。实验执行前验证 scope 与唯一选择的视频一致；任何裁剪实验仅允许单视频，既有单视频无 scope 手工配置保留兼容。新 UI 导出必须有 scope，防止下载文件误用于其他录制。坐标严格归一化，回映关键点沿用现有 adapter；头姿仍标记 crop-experimental，禁止解释为原相机精确角度。

实验报告新增缺失连续段统计及 100 槽 / 25 槽步幅头姿窗口代理资格：strict50、coverage25、gap80。每窗口按末槽标签分正/负/未知，分别统计 cached/candidate 可用、增益与损失；代理资格不使用 normalizer，不推理，不代表正确检出或真实训练可用。输出 head_windows.csv 支持逐窗口核查。跨录制不连接缺失段。

验收：视频绑定和 crop 参数错误拒绝；预览闭合 decoder、不加载模型、不改输入；JPEG/HTML 可离线查看并导出有效 JSON；统计窗口边界、全缺失、段尾、标签分母、prefix 空窗口及跨记录聚合正确；原 baseline 再现检查保持。
