"""Single-model, sequential batch-one loopback HTTP detector service.

GET /health and POST /infer require Bearer authentication and
X-Detector-Identity. Infer additionally requires X-Model-Id, X-Frame-Width,
X-Frame-Height and X-Timestamp-Ms; the body is raw contiguous uint8 HWC BGR.
There is no pickle transport. No listening readiness is published before a
real warmup prediction completes on the explicitly selected device.
"""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import os
from pathlib import Path
import signal
import socket

from inference.experts.protocol import (MAX_RESPONSE, TIMEOUT_SECONDS, VERSION, read_token,
    settings_identity, validate_dimensions, validate_result, validate_service)

LOG = logging.getLogger(__name__)


class DetectorHTTPServer(HTTPServer):
    request_queue_size = 64

    def server_bind(self):
        # Windows SO_REUSEADDR permits a second listener on an occupied port.
        # Reserve exclusively there; Unix allows restart but never two listeners.
        if os.name == 'nt':
            self.allow_reuse_address = False
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(TIMEOUT_SECONDS)
        return connection, address


class DetectorHandler(BaseHTTPRequestHandler):
    # Deliberately HTTP/1.0: a single worker cannot monopolize a persistent socket.
    def log_message(self, fmt, *args):
        pass

    def _reply(self, status, payload):
        raw = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()
        if len(raw) > MAX_RESPONSE:
            status, raw = 503, b'{"error":"response exceeds size limit"}'
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def _authorized(self):
        values = self.headers.get_all('Authorization', [])
        if (len(values) != 1 or not values[0].isascii()
                or not hmac.compare_digest(values[0], 'Bearer ' + self.server.token)):
            self._reply(401, {'error': 'authentication required'})
            return False
        if self.headers.get_all('X-Detector-Identity', []) != [self.server.request_identity]:
            self._reply(409, {'error': 'detector identity mismatch'})
            return False
        return True

    def do_GET(self):
        if not self._authorized():
            return
        if self.path != '/health':
            self._reply(404, {'error': 'unknown endpoint'})
            return
        self._reply(200, self.server.health)

    def do_POST(self):
        if not self._authorized():
            return
        if self.path != '/infer':
            self._reply(404, {'error': 'unknown endpoint'})
            return
        if self.server.health['status'] != 'ready':
            self._reply(503, {'error': 'detector is not ready'})
            return
        if self.headers.get_all('X-Model-Id', []) != [self.server.detector.model_id]:
            self._reply(409, {'error': 'model identity changed'})
            return
        try:
            if self.headers.get('Transfer-Encoding') is not None:
                raise ValueError('Transfer-Encoding is not supported')
            if self.headers.get_all('Content-Type', []) != ['application/octet-stream']:
                raise ValueError('expected application/octet-stream')
            numbers = []
            for name in ('X-Frame-Height', 'X-Frame-Width', 'X-Timestamp-Ms', 'Content-Length'):
                values = self.headers.get_all(name, [])
                if len(values) != 1 or not values[0].isascii() or not values[0].isdigit() or len(values[0]) > 20:
                    raise ValueError('invalid numeric header ' + name)
                numbers.append(int(values[0]))
            height, width, timestamp, length = numbers
            expected = validate_dimensions(height, width, timestamp)
            if length != expected:
                raise ValueError('Content-Length does not match dimensions')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('truncated frame body')
        except (ValueError, OSError) as exc:
            self._reply(400, {'error': str(exc)})
            return
        try:
            import numpy as np
            frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
            result = self.server.detector.predict(frame, timestamp)
            validate_result(result, self.server.detector.model_id, timestamp, width, height,
                            self.server.max_predictions)
        except Exception:
            LOG.exception('Detector inference failed; no backend fallback')
            self._reply(503, {'error': 'detector inference failed; no backend fallback'})
            return
        self._reply(200, result)


def create_server(settings, service, detector=None, *, port=None):
    """Reserve the port before optional heavyweight detector construction."""
    validate_service(service)
    token = read_token(service)
    identity = settings_identity(**settings)
    server = DetectorHTTPServer((service['host'], service['port'] if port is None else port), DetectorHandler)
    server.detector, server.token, server.request_identity = None, token, identity
    server.ready_file = None
    server.ready_stat = None
    server.settings = dict(settings)
    server.max_predictions = sum((settings['input_size'] // stride) ** 2 for stride in (8, 16, 32))
    server.health = dict(status='warming', protocol=VERSION, request_identity=identity,
                         model_id=None, identity=None,
                         host=service['host'], port=server.server_address[1], pid=os.getpid())
    if detector is not None:
        attach_detector(server, detector)
    return server


def attach_detector(server, detector):
    server.detector = detector
    server.health.update(model_id=detector.model_id, identity=detector.identity)


def remove_owned_ready(server):
    """Only remove the exact readiness file created by this server instance."""
    ready = server.ready_file
    if ready is not None:
        try:
            state = ready.lstat()
            if (state.st_dev, state.st_ino) == server.ready_stat:
                ready.unlink()
        except FileNotFoundError:
            pass
        server.ready_file = server.ready_stat = None


def warmup_and_write_ready(server, ready_file):
    import numpy as np
    ready = Path(ready_file)
    remove_owned_ready(server)
    server.health['status'] = 'warming'
    if os.path.lexists(ready):
        raise FileExistsError(f'ready file already exists: {ready}')
    size = server.settings['input_size']
    result = server.detector.predict(np.zeros((size, size, 3), np.uint8), 0)
    validate_result(result, server.detector.model_id, 0, size, size, server.max_predictions)
    server.health['status'] = 'ready'
    ready.parent.mkdir(parents=True, exist_ok=True)
    temporary = ready.with_name(ready.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(server.health, sort_keys=True, allow_nan=False), encoding='utf-8')
        # Atomic publication without replacing an existing service's readiness.
        os.link(temporary, ready)
        state = ready.lstat()
        server.ready_file, server.ready_stat = ready, (state.st_dev, state.st_ino)
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info('Detector ready: %s', json.dumps(server.health, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ready-file', required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from inference.perception.runtime import load_runtime_config
    from inference.experts.yolo11_adapter import YOLO11Adapter
    ready = Path(args.ready_file)
    if os.path.lexists(ready):
        raise FileExistsError(f'ready file already exists: {ready}; choose a fresh session path')
    detector, server = None, None
    previous = {}

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
        settings = dict(load_runtime_config(args.config)['objects'])
        service = settings.pop('service')
        validate_service(service)
        read_token(service)
        server = create_server(settings, service)
        detector = YOLO11Adapter(**settings)
        attach_detector(server, detector)
        warmup_and_write_ready(server, ready)
        server.serve_forever(poll_interval=.25)
    finally:
        if server is not None:
            server.server_close()
            remove_owned_ready(server)
        if detector is not None:
            detector.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    main()
