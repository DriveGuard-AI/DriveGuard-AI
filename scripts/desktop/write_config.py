#!/usr/bin/env python3
"""生成桌面版完全内置的 `config.json`（模型服务地址 + Token）。

Usage:
    DGB_MODEL_SERVICE_URL=https://dms.yourdomain.com \
    DGB_MODEL_SERVICE_TOKEN=... \
    python scripts/desktop/write_config.py --output desktop/dist/config.json

Token 会写进安装包，属于“完全内置”方案的已知代价：只应使用专用、可轮换、
权限受限的服务端 Token，不要把管理员 Token 放进公开安装包。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

IMAGE_SIZE = re.compile(r"^\d{2,4}x\d{2,4}$")
ROI = re.compile(r"^0(?:\.\d+)?,\s*0(?:\.\d+)?,\s*(?:1(?:\.0+)?|0\.\d+),\s*(?:1(?:\.0+)?|0\.\d+)$")


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def env_int(name: str, default: int) -> int:
    raw = env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"error: {name} 必须是整数，收到 {raw!r}") from None


def env_float(name: str, default: float) -> float:
    raw = env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"error: {name} 必须是数字，收到 {raw!r}") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--demo", action="store_true", help="写一个离线演示配置（不需要模型服务）")
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.demo:
        payload = {"demo": True}
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote demo config: {output}")
        return 0

    url = env("DGB_MODEL_SERVICE_URL")
    token = env("DGB_MODEL_SERVICE_TOKEN")
    if not url:
        raise SystemExit("error: 未设置 DGB_MODEL_SERVICE_URL")
    if not token:
        raise SystemExit("error: 未设置 DGB_MODEL_SERVICE_TOKEN")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SystemExit(f"error: DGB_MODEL_SERVICE_URL 必须是 http(s) 地址，收到 {url!r}")
    hostname = (parsed.hostname or "").lower()
    if hostname == "example.com" or hostname.endswith(".example.com") or hostname in {
        "example.org",
        "example.net",
    }:
        raise SystemExit(
            "error: DGB_MODEL_SERVICE_URL 仍是示例域名，必须换成真实的模型服务地址"
        )
    if len(token) < 24 or any(ch.isspace() for ch in token):
        raise SystemExit("error: DGB_MODEL_SERVICE_TOKEN 至少 24 字符且不能包含空白")

    image_size = env("DGB_MODEL_IMAGE_SIZE", "640x480") or "640x480"
    if not IMAGE_SIZE.match(image_size):
        raise SystemExit(f"error: DGB_MODEL_IMAGE_SIZE 需形如 640x480，收到 {image_size!r}")
    roi = env("DGB_MODEL_ROI", "0.15,0.0,0.75,1.0") or "0.15,0.0,0.75,1.0"
    if not ROI.match(roi):
        raise SystemExit(f"error: DGB_MODEL_ROI 需形如 0.15,0.0,0.75,1.0，收到 {roi!r}")
    view = env("DGB_MODEL_VIEW", "body") or "body"
    if view not in {"body", "face"}:
        raise SystemExit(f"error: DGB_MODEL_VIEW 只支持 body/face，收到 {view!r}")
    camera_backend = env("DGB_MODEL_CAMERA_BACKEND", "") or ""
    if camera_backend and camera_backend not in {"any", "dshow", "msmf", "v4l2", "gstreamer"}:
        raise SystemExit(
            "error: DGB_MODEL_CAMERA_BACKEND 只支持 any/dshow/msmf/v4l2/gstreamer 或留空"
        )

    payload = {
        "schemaVersion": 1,
        "modelServiceUrl": url.rstrip("/"),
        "modelServiceToken": token,
        "view": view,
        "imageSize": image_size,
        "driverRoi": roi,
        "cameraIndex": env_int("DGB_MODEL_CAMERA_INDEX", 0),
        "cameraWidth": env_int("DGB_MODEL_CAMERA_WIDTH", 640),
        "cameraHeight": env_int("DGB_MODEL_CAMERA_HEIGHT", 480),
        "cameraQuality": env_int("DGB_MODEL_CAMERA_QUALITY", 60),
        "cameraBackend": camera_backend,
        "inferenceHz": env_float("DGB_MODEL_INFERENCE_HZ", 5.0),
        "previewMaxFps": env_float("DGB_MODEL_PREVIEW_MAX_FPS", 10.0),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        output.chmod(0o600)
    except OSError:
        pass
    # 不打印 Token。
    print(f"wrote config: {output} url={payload['modelServiceUrl']} view={payload['view']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
