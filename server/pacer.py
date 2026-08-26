"""녹음 wav를 실시간 속도로 흘려보내는 페이서."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import numpy as np
import soundfile as sf

from server.config import get as cfg_get

OnFrame = Callable[[float, np.ndarray, np.ndarray], Any]


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    # numpy 선형 보간만으로 리샘플한다 (요구사항 밖 라이브러리를 안 쓴다)
    if orig_sr == target_sr:
        return data
    duration = len(data) / orig_sr
    n_target = max(1, round(duration * target_sr))
    x_old = np.linspace(0.0, duration, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)


async def play(wav_path: str, on_frame: OnFrame, speed: float = 1.0) -> None:
    # 왼쪽=고객, 오른쪽=상담사인 2채널 wav를 frame_ms씩 잘라 실시간 속도로 on_frame(t_sec, left, right)를 부른다
    target_sr = cfg_get("audio", "sample_rate")
    frame_ms = cfg_get("audio", "frame_ms")

    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    if data.shape[1] != 2:
        raise ValueError(f"{wav_path} 는 2채널이 아닙니다 (채널 수: {data.shape[1]})")

    left = _resample(data[:, 0], sr, target_sr)
    right = _resample(data[:, 1], sr, target_sr)

    frame_len = int(target_sr * frame_ms / 1000)
    n_frames = -(-len(left) // frame_len)  # 올림 나눗셈
    sleep_s = (frame_ms / 1000.0) / speed

    for i in range(n_frames):
        start = i * frame_len
        end = min(start + frame_len, len(left))
        t_sec = start / target_sr
        result = on_frame(t_sec, left[start:end], right[start:end])
        if asyncio.iscoroutine(result):
            await result
        await asyncio.sleep(sleep_s)
