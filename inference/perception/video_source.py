"""Local PTS-based decode and bounded-memory causal resampling, no FPS guessing."""
from dataclasses import dataclass
from fractions import Fraction
import importlib
import math
from pathlib import Path

import numpy as np

DECODE_OPTIONS = {'format_whitelist': 'mov,matroska,webm,avi', 'protocol_whitelist': 'file',
                  'enable_drefs': '0', 'use_absolute_path': '0'}


@dataclass(frozen=True)
class VideoFrame:
    timestamp_ms: int
    bgr: np.ndarray


def decode_video(path, metadata=None):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError('Input must be a local video file')
    try:
        av = importlib.import_module('av')
    except ImportError as error:
        raise RuntimeError('PyAV is required for video PTS decoding; no automatic install') from error
    # Reject playlists/concat before opening their secondary inputs. MOV external tracks stay disabled.
    with av.open(str(path), mode='r', options=dict(DECODE_OPTIONS)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Expected exactly one video stream; select/remux multi-stream sources explicitly')
        origin, previous = None, None
        for frame in container.decode(container.streams.video[0]):
            if frame.is_corrupt or frame.pts is None or frame.time_base is None or frame.time_base <= 0:
                raise ValueError('Corrupt frame or missing PTS/time_base; no guessed-FPS fallback')
            instant = Fraction(frame.pts) * frame.time_base
            if origin is None:
                origin = instant
                if metadata is not None:
                    metadata.update(first_pts=frame.pts, first_time_base=str(frame.time_base),
                                    origin_seconds=str(origin), timestamp_quantization='ceil(relative_PTS_ms)',
                                    demuxer_policy=dict(DECODE_OPTIONS))
            timestamp = math.ceil((instant-origin)*1000)
            if timestamp < 0 or timestamp > 2**63-1 or (previous is not None and timestamp <= previous):
                raise ValueError('Decoded PTS must be strictly increasing after millisecond quantization')
            previous = timestamp
            yield VideoFrame(timestamp, frame.to_ndarray(format='bgr24'))


def sample_frames(frames):
    """Yield (25Hz slot, latest past frame); never extend past the last source PTS."""
    previous, shape, slot = None, None, 0
    for frame in frames:
        time, bgr = frame.timestamp_ms, frame.bgr
        if type(time) is not int or not 0 <= time <= 2**63-1 or (previous is None and time != 0) or (
                previous is not None and time <= previous.timestamp_ms):
            raise ValueError('Source times must start at zero and strictly increase')
        if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 1:
            raise ValueError('Expected nonempty uint8 HWC BGR source frames')
        if max(bgr.shape[:2]) > 16384 or bgr.shape[0]*bgr.shape[1] > 64*1024*1024:
            raise ValueError('Source frame exceeds supported dimensions')
        if shape is not None and bgr.shape != shape:
            raise ValueError('Video dimensions must remain fixed')
        shape = bgr.shape
        if previous is not None:
            while slot < time:
                yield slot, previous
                slot += 40
        previous = frame
    if previous is None:
        raise ValueError('Video contains no decoded frames')
    if slot <= previous.timestamp_ms:
        yield slot, previous
