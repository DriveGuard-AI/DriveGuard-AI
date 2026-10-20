"""live 推理服务的窄客户端（仅标准库）。

应用层只需要三个调用：开会话、提交帧、关闭会话。令牌从文件读取，
不接受明文参数传入，避免出现在命令行与日志里。

**持久连接**：live 服务逐帧调用是最热的路径。每帧重新建 TCP/TLS 会在公网
路径上多花 100–200ms；这里复用同一个 HTTP(S) 连接，服务端对应使用
HTTP/1.1 keep-alive。连接被服务端关闭或出错时丢弃，下一次请求自动重连。
"""

from __future__ import annotations

import base64
import http.client
import json
import socket
import ssl
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

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
        base = base_url.rstrip("/")
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"live 地址必须是 http(s) URL: {base_url}")
        self._base_url = base
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        # 支持反向代理前缀（例如 https://host/dms），live API 仍以 /api/live 开头。
        self._prefix = parsed.path.rstrip("/")
        self._token = token
        self._timeout = float(timeout)
        self._connection: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def from_token_file(cls, base_url: str, token_file: str | Path, *, timeout: float = 30.0) -> "LiveClient":
        token = Path(token_file).read_text(encoding="utf-8").strip()
        if len(token) < MIN_TOKEN_LENGTH:
            raise ValueError(f"live 令牌至少需要 {MIN_TOKEN_LENGTH} 个字符: {token_file}")
        return cls(base_url, token, timeout=timeout)

    @property
    def base_url(self) -> str:
        return self._base_url

    def _connect(self) -> http.client.HTTPConnection:
        if self._scheme == "https":
            connection: http.client.HTTPConnection = http.client.HTTPSConnection(
                self._host,
                self._port,
                timeout=self._timeout,
                context=ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        connection.connect()
        # 小请求/小响应逐帧往返时，Nagle + 延迟 ACK 会稳定引入约 40ms。
        if connection.sock is not None:
            connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection

    def _drop_locked(self) -> None:
        """丢弃当前连接；调用方必须已持有 self._lock。"""

        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def close(self) -> None:
        """关闭底层连接；之后不应再使用本客户端。"""

        with self._lock:
            self._closed = True
            self._drop_locked()

    def _request(self, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        headers = {
            "Authorization": "Bearer " + self._token,
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        target = f"{self._prefix}{path}" or "/"

        with self._lock:
            if self._closed:
                raise LiveError("live 客户端已关闭")
            try:
                connection = self._connection
                if connection is None:
                    connection = self._connect()
                    self._connection = connection
                connection.request(method, target, body=data, headers=headers)
                response = connection.getresponse()
                raw = response.read()
                status = response.status
                # 服务端 HTTP/1.0、Connection: close 或异常时会要求关闭；下次请求重连。
                if response.will_close:
                    self._drop_locked()
                if status >= 400:
                    detail = raw.decode("utf-8", "replace")
                    try:
                        detail = str(json.loads(detail).get("error") or detail)
                    except (ValueError, TypeError):
                        pass
                    self._drop_locked()
                    raise LiveError(f"{method} {path} 返回 {status}: {detail[:200]}")
                try:
                    return status, json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeError):
                    self._drop_locked()
                    raise LiveError(f"{method} {path} 返回了非 JSON 响应") from None
            except LiveError:
                raise
            except (http.client.HTTPException, OSError, ssl.SSLError, ValueError) as error:
                self._drop_locked()
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
