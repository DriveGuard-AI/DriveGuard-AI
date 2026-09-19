# 本地感知运行时工程验证

2026-09-18 · shadow · model 分支 · 未提交。环境：Python3.12.14、torch2.14.0+cpu、NumPy2.3.5；没有安装或下载新的依赖、视频、数据集、模型。目标服务器 torch2.7.1/Ascend 尚未用于本轮测试。

## 结果

- `python -B -m unittest discover -s tests -q`：105项通过，28.151秒；原62项保持通过，新增43项。
- 新增覆盖：YOLOX20、工件2、MediaPipe7、跟踪4、视频6、贯穿运行时4。外部 YOLOX/OpenCV/MediaPipe/PyAV 使用API边界替身；实际运行torch安全加载/FP32、NumPy几何/NMS、适配器、特征提取与数组导出。
- 视频 CLI `--check-config`、原GPU/NPU配置预检均在 `python -S` 下通过；不访问权重、视频或设备。预检不是环境验证。
- 规格和质量独立审查完成。修复并保留回归：全局float64默认值影响YOLOX模型、多人脸时手部归属未撤销、运行时资源关闭晚于成功发布、播放列表引入未计入hash的其他素材。
- 文档本地链接/场景追溯/内存SQLite/OpenAPI静态引用检查通过；未做完整YAML schema验证或图形视觉验收。API/数据库结构未因本轮离线算法入口改变。

## 验证口径

YOLOX参考0.3.0内置Nano/Tiny，预训练工件仍须显式提供来源、SHA-256、许可声明。Face/Hand使用CPU Tasks VIDEO；头姿保持未知。时间来自PTS，25Hz只取最近过去帧；跟踪不是类别编号，也不是经实测认证的身份算法。解码打开接口限制自包含容器/file协议和外部轨道；**实际PyAV/FFmpeg对限制的执行仍未本机实测**。

贯穿测试使用100个合成帧与真实适配器代码导出19特征，标签全部未知，证明工程边界连通，不能证明模型检测能力。没有真实模型精度、训练跑分、FPS/延迟、GPU/NPU兼容性、自动身份或正式权重交付。

入口和后续验收见[运行指南](../training/EXPERT_RUNTIME.md)。候选 requirements-perception.txt 未实装锁定，与现有昇腾训练依赖分开；正式训练前仍需真实模型小视频验证、标注/主体划分、平台验证和权重许可/公开交付。
