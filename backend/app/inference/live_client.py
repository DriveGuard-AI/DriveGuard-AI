"""live 推理服务的窄客户端（仅标准库）。

应用层只需要三个调用：开会话、提交帧、关闭会话。令牌从文件读取，
不接受明文参数传入，避免出现在命令行与日志里。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
import urllib.error
import urllib.request

MIN_TOKEN_LENGTH = 24


class LiveError(RuntimeError):
    """live 服务不可达或返回非预期结果。"""


@dataclass(frozen=True)
class LiveCamera:
    """会话参数。`driver_roi` 为归一化坐标，`image_size` 必须与帧尺寸一致。"""

    view: str = "body"
    image_size: tuple[int, int] = (1280, 720)
    driver_roi: tuple[float, float, float, float] = (0.15, 0.0, 0.75, 1.0)
    driver_region_confirmed_exclusive: bool = True

    def as_payload(self) -> dict[str, object]:
        return {
            "view": self.view,
            "image_size": [self.image_size[0], self.image_size[1]],
            "driver_roi": list(self.driver_roi),
            "driver_region_confirmed_exclusive": self.driver_region_confirmed_exclusive,
        }


class LiveClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0) -> None:
        if not token:
            raise ValueError("live 令牌不能为空")
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = float(timeout)

    @classmethod
    def from_token_file(cls, base_url: str, token_file: str | Path, *, timeout: float = 30.0) -> "LiveClient":
        token = Path(token_file).read_text(encoding="utf-8").strip()
        if len(token) < MIN_TOKEN_LENGTH:
            raise ValueError(f"live 令牌至少需要 {MIN_TOKEN_LENGTH} 个字符: {token_file}")
        return cls(base_url, token, timeout=timeout)

    @property
    def base_url(self) -> str:
        return self._base_url

    def _request(self, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(self._base_url + path, data=data, method=method)
        request.add_header("Authorization", "Bearer " + self._token)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            try:
                detail = str(json.loads(detail).get("error") or detail)
            except ValueError:
                pass
            raise LiveError(f"{method} {path} 返回 {error.code}: {detail[:200]}") from None
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise LiveError(f"{method} {path} 不可达: {error}") from None

    def health(self) -> dict:
        _, body = self._request("GET", "/api/live/health")
        return body

    def status(self) -> dict:
        _, body = self._request("GET", "/api/live/status")
        return body

    def open_session(self, camera: LiveCamera) -> str:
        status, body = self._request("POST", "/api/live/sessions", camera.as_payload())
        session_id = body.get("session_id") if isinstance(body, dict) else None
        if status not in (200, 201) or not isinstance(session_id, str):
            raise LiveError(f"创建会话失败（{status}）: {body}")
        return session_id

    def send_frame(self, session_id: str, timestamp_ms: int, jpeg: bytes) -> dict:
        payload = {
            "timestamp_ms": int(timestamp_ms),
            "jpeg": base64.b64encode(jpeg).decode("ascii"),
        }
        status, body = self._request("POST", f"/api/live/sessions/{session_id}/frames", payload)
        if status != 200 or not isinstance(body, dict):
            raise LiveError(f"提交帧失败（{status}）: {body}")
        return body

    def close_session(self, session_id: str) -> None:
        try:
            self._request("DELETE", f"/api/live/sessions/{session_id}")
        except LiveError:
            # 会话可能已被服务端因推理错误关闭；关闭失败不影响本地状态清理。
            pass


__all__ = ["LiveCamera", "LiveClient", "LiveError", "MIN_TOKEN_LENGTH"]
