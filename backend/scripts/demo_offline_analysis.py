"""离线分析演示后端（脚本化推理，不依赖模型服务）。

用途：评审/验收时演示"上传一段视频 → 服务端抽帧 → 重跑分析 → 按会话看结果"。
它**不是**真实推理：风险等级由脚本按时序生成，只用来证明链路与界面工作正常。

    python scripts/demo_offline_analysis.py --port 8010

启动后：
- 页面：http://127.0.0.1:8010/
- Token：dev-token
- 样例视频：backend/var/demo-media/sample/clip.mp4（脚本自动生成，可直接上传）

真实模型联调请用 `live` 模式 + 模型仓库的 live 服务，不要用本脚本。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from datetime import datetime

BACKEND = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

DEFAULT_SIZE = (320, 240)
DYNAMIC_RATIO = (0.15, 0.60)  # 这一段区间输出 DANGER，两侧 SAFE


def ensure_sample_video(media_root: pathlib.Path, frames: int) -> pathlib.Path:
    """生成一段合成驾驶画面，供页面选择；已存在则复用。"""

    target = media_root / "sample" / "clip.mp4"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - 部署错误
        raise SystemExit(
            f"缺少解码依赖（{error}）。请先运行 .\\run.ps1 或安装 requirements-dev.lock.txt。"
        ) from None
    width, height = DEFAULT_SIZE
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height))
    if not writer.isOpened():  # pragma: no cover - 取决于本机编码器
        raise SystemExit("当前环境没有可用的 mp4 编码器，无法生成样例视频")
    try:
        for index in range(frames):
            frame = np.zeros((height, width, 3), dtype="uint8")
            frame[:, :, 0] = (index * 5) % 255
            cv2.circle(frame, (100 + (index % 60) * 2, 120), 30, (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return target


def make_port(frames: int):
    from app.schemas import RiskSnapshot

    class DemoInference:
        """按时序给出 SAFE → DANGER → SAFE 的确定性快照。"""

        mode = "fake"
        degraded = False

        def __init__(self, total: int) -> None:
            self.total = max(1, total)
            self.sent = 0
            self._start = max(5, int(self.total * DYNAMIC_RATIO[0]))
            # 事件最短 1.5 s；40 ms 栅格下至少要 38 帧，这里留够。
            self._end = max(self._start + 40, int(self.total * DYNAMIC_RATIO[1]))

        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

        def status(self) -> dict:
            state = "completed" if self.sent >= self.total else "running"
            return {"state": state, "frames_sent": self.sent}

        def next_snapshot(self, *, session_id: str, sequence: int, now: datetime) -> RiskSnapshot:
            risky = self._start <= self.sent < self._end
            self.sent += 1
            return RiskSnapshot(
                session_id=session_id,
                sequence=sequence,
                captured_at=now,
                risk_level="DANGER" if risky else "SAFE",
                primary_risk="fatigue" if risky else None,
                confidence=0.91 if risky else 0.2,
                quality="good",
                risk_scores={"fatigue": 0.91} if risky else {},
                reason_codes=["FATIGUE_DETECTED"] if risky else ["NO_BEHAVIOUR_DETECTED"],
                inference_mode="fake",
                source_timestamp_ms=sequence * 40,
            )

    return DemoInference(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--token", default="dev-token")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--frames", type=int, default=50, help="样例视频帧数（25 fps，默认 2 秒）")
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="保留上次演示的库；默认重建（表结构变了也能直接跑起来）",
    )
    args = parser.parse_args()

    media_root = BACKEND / "var" / "demo-media"
    database = BACKEND / "var" / "demo.db"
    # 演示必须和外部环境隔离：之前用 setdefault 继承了一个旧 DGB_DATABASE_URL，
    # 结果演示库与页面看到的库不是同一个。这里直接覆盖。
    os.environ["DGB_DATABASE_URL"] = f"sqlite:///{database.as_posix()}"
    os.environ["DGB_MEDIA_DIR"] = str(media_root)
    os.environ["DGB_ADMIN_TOKEN"] = args.token
    os.environ["DGB_INFERENCE_HZ"] = "20"
    os.environ["DGB_HEARTBEAT_SECONDS"] = "30"
    # 有构建产物就交给 FastAPI 托管；没有则只提供 API 与 /docs。
    dist = BACKEND.parent / "frontend" / "dist"
    if dist.is_dir():
        os.environ["DGB_STATIC_DIR"] = str(dist)

    from app.config import load_config
    from app.main import create_app

    if not args.keep_db:
        # `create_all` 不会改已存在的表；旧演示库缺 `step_ms` 会直接 500。
        for suffix in ("", "-shm", "-wal"):
            stale = database.with_name(database.name + suffix)
            if stale.exists():
                stale.unlink()

    config = load_config()
    app = create_app(config)

    def builder(source):
        total = len(list(pathlib.Path(source.frame_sequence).glob("*.jpg")))
        print(f"[demo] 新素材 {source.frame_sequence}，{total} 帧", flush=True)
        return make_port(total)

    # 真实运行时构建器会去连 live 服务；演示不需要模型，换成脚本端口。
    app.state.context.engine._builder = builder

    sample = ensure_sample_video(media_root, args.frames)
    base = f"http://{args.host}:{args.port}"
    print("=" * 68)
    print("离线分析演示（脚本化推理，不是真实模型结论）")
    print(f"  页面      {base}/")
    print(f"  API 文档  {base}/docs")
    print(f"  Token     {args.token}")
    print(f"  数据库    {database}")
    print(f"  样例视频  {sample}")
    print("-" * 68)
    print("  1) 页面右上角填 Token → 打开「离线分析」")
    print("  2) 选择样例视频，抽帧步长保持 40 ms，点「开始分析」")
    print("     （实时在跑时会先弹一句确认：切换会停掉实时监控，确认后才开始抽帧）")
    print("  3) 看进度从 0 → N/N、素材时间轴推进、状态变「已完成」")
    print("  4) 点「查看该会话事件」→ 事件表只剩这一轮（含一条 fatigue/DANGER）")
    print("  5) 点「查看该会话统计」→ 危险事件总数应为 1，累计时长大致等于危险段时长")
    print("  6) 回「实时监控」→ 顶部提示「当前离线回放，不是实时采集」")
    print("  7) 在任务卡下方看「回放」→ 拖动风险曲线或点事件「定位」跳到出问题的那一帧")
    print("-" * 68)
    print("  注：脚本生成的样例是 mp4v，浏览器一般解不了，回放会自动退回抽帧序列")
    print("      （帧号同样精确）。真实手机/行车记录仪的 H.264 视频才会用 <video> 播放。")
    print("=" * 68, flush=True)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
