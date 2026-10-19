# 感知原因诊断实施计划

已获用户执行授权；在当前model工作区增量实现，不提交、不连接服务器、不改训练行为。

1. 新增tests/test_face_diagnostics.py，以纯MediaPipe候选点测试每种失败及重放等价；先运行确认缺失能力导致失败。
2. 新增inference/experts/face_diagnostics.py纯原因解释；在mediapipe_adapter.py增加默认关闭的诊断属性，保持返回值与模型ID不变。
3. 新增training/dmd/face_diagnostics.py：验证原发布记录和来源，原参数连续重放，逐槽对照原mask，生成frames.csv/candidates.jsonl及汇总。图片输出和HTML展示放独立face_diagnostic_samples.py，保持原视频和缓存只读。
4. 新增tests/test_face_diagnostic_replay.py，使用现有runtime_fixture的模型/解码边界构造完整缓存，验证重放、异常、路径/哈希、部分运行及样例上限；新增真实图片写入检查。
5. 更新docs/training/GAZE_FULL_RUNBOOK.md，提供本次服务器准确路径和感知环境命令、输出语义。跑相关回归和独立代码审核，修复问题后交付。

验证命令：`.venv-training/Scripts/python.exe -m unittest discover -s tests -p 'test_face_diag*.py' -v`；既有expert/gaze/dmd回归；`git diff --check`。实际服务器原因必须由新报告确认。

执行记录：五项均已实现。新增原因／重放14项测试通过；在已有`.venv-perception`中真实OpenCV图片与HTML测试2项通过。既有MediaPipe、视频、runtime flow、gaze perception与DMD cache共38项（37通过、1因Windows符号链接权限跳过）；既有gaze diagnostics按discover入口运行7项通过。感知环境CLI帮助和`git diff --check`通过。独立审查发现“校验后、轮到重放前源视频被修改”可能沿用旧哈希，已增加校验前后及重放前后的文件身份检查与回归测试。未执行服务器真实视频重放，未提交或推送改动。
