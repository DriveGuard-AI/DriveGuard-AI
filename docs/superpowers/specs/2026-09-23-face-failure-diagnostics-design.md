# Body 感知失败原因诊断

用户已确认继续原因诊断。上一轮2796窗口证据：body的1396窗口仅34个F可用，245个FN中241个F不可用；本轮定位缓存写入前的失败，不调整模型、训练、阈值、ROI或F规则。

输入原manifest、特征根目录、原视频Dataset根目录。默认只重放val/body/distraction六段，允许明确选择face/all和record_id。原complete.json、input.json、features/record.json与数组哈希需通过验证；视频、ROI和模型工件来自原来源记录，不猜测视频路径。使用原MediaPipeAdapter的同一VIDEO参数和25Hz因果采样，从每段视频起点连续处理，复用帧时间时不重复调用。只运行面部/手部适配器，无YOLO/NPU服务、无新特征发布。

为适配器增加默认关闭的诊断选项，只附加独立last_face_diagnostics属性，不改变输出packet、模型ID或推理设置。纯诊断函数区分no_face_detected、invalid_landmarks（数量/非有限/越界细分）、outside_roi、ambiguous_faces、driver_unconfirmed、pose_matrices_missing/count_mismatch、pose_matrix_invalid和ok；虹膜成对有效性单独报告，避免把闭眼当整张脸检测失败。原逻辑中任一候选脸非法即整组失败的行为必须如实反映。

重放每个25Hz槽输出slot时间、原帧PTS、原因、候选脸计数/框、ROI、原缓存与重放头姿/虹膜有效性、标签。明确区分face处理成功但检测不到脸与stale_observation；原因统计按槽计数，唯一推理帧数另列。对原缓存差异按槽统计，版本/模型ID不一致及差异存在时不得宣称严格复现。

输出全新目录：summary.json、每段frames.csv和candidates.jsonl、每原因有限张带ROI/脸框标注JPEG、index.html本地画廊。默认每段每原因最多2张、至少间隔1秒，可设0禁用。不保存整段视频或改动原缓存。summary最后写入，失败无完成标记。可限制每段前N秒并明确标记局部诊断，默认完整重放。

测试：原因分支及副作用、适配器启停诊断输出等价、原缓存完整性/路径与原视频哈希拒绝、真实Python链路合成外部API重放的一致性、重复PTS复用、局部标记、图片上限/截图与HTML转义、异常清理。CPU本地验证不代表服务器实测。
