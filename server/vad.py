"""채널 하나의 100ms 프레임을 받아 silero-vad로 발화 구간을 잘라내는 VAD."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from silero_vad import VADIterator, load_silero_vad

from server.config import get as cfg_get

OnStart = Callable[[float], None]
OnUtterance = Callable[[float, float, np.ndarray], None]


class ChannelVAD:
    def __init__(self, on_start: OnStart, on_utterance: OnUtterance) -> None:
        self.on_start = on_start
        self.on_utterance = on_utterance

        self._sr = cfg_get("audio", "sample_rate")
        self._chunk_samples = cfg_get("vad", "chunk_samples")
        self._max_utt_s = cfg_get("vad", "max_utt_s")
        self._split_overlap_s = cfg_get("vad", "split_overlap_s")

        model = load_silero_vad()
        self._vad = VADIterator(
            model,
            threshold=cfg_get("vad", "threshold"),
            sampling_rate=self._sr,
            min_silence_duration_ms=cfg_get("vad", "min_silence_duration_ms"),
            speech_pad_ms=cfg_get("vad", "speech_pad_ms"),
        )

        self._carry = np.array([], dtype=np.float32)  # 512로 나누고 남는 샘플 — 다음 프레임 앞에 이어붙인다
        self._prev_chunk: Optional[np.ndarray] = None  # 발화 시작 직전 32ms — 패딩용으로 앞에 붙인다
        self._vad_origin_t: Optional[float] = None  # VADIterator 내부 샘플카운터의 0번째 샘플이 가리키는 절대 시각

        self._triggered = False
        self._utt_t0 = 0.0
        self._utt_buffer: list[np.ndarray] = []

        self._segments: list[tuple[float, float]] = []
        self._stream_end_t = 0.0

    def push_frame(self, t_sec: float, frame: np.ndarray) -> None:
        # 100ms 프레임을 받아 512샘플(32ms) 조각으로 다시 잘라 VADIterator에 먹인다
        frame = np.asarray(frame, dtype=np.float32)
        if len(self._carry):
            combined = np.concatenate([self._carry, frame])
            combined_start_t = t_sec - len(self._carry) / self._sr
        else:
            combined = frame
            combined_start_t = t_sec

        n_chunks = len(combined) // self._chunk_samples
        for i in range(n_chunks):
            chunk = combined[i * self._chunk_samples : (i + 1) * self._chunk_samples]
            chunk_t = combined_start_t + i * self._chunk_samples / self._sr
            self._feed_chunk(chunk, chunk_t)

        usable = n_chunks * self._chunk_samples
        self._carry = combined[usable:].copy()  # 남는 샘플은 버리지 않고 보관
        self._stream_end_t = max(self._stream_end_t, t_sec + len(frame) / self._sr)

    def _feed_chunk(self, chunk: np.ndarray, chunk_t: float) -> None:
        if self._vad_origin_t is None:
            self._vad_origin_t = chunk_t

        result = self._vad(chunk, return_seconds=False)

        if not self._triggered:
            if result and "start" in result:
                self._triggered = True
                self._utt_t0 = self._vad_origin_t + result["start"] / self._sr
                self._utt_buffer = [self._prev_chunk] if self._prev_chunk is not None else []
                self._utt_buffer.append(chunk)
                self.on_start(self._utt_t0)
        else:
            self._utt_buffer.append(chunk)
            if result and "end" in result:
                t1 = self._vad_origin_t + result["end"] / self._sr
                self._finish_utterance(t1)
            elif self._utt_duration_s() >= self._max_utt_s:
                self._force_split()

        self._prev_chunk = chunk

    def _utt_duration_s(self) -> float:
        return sum(len(c) for c in self._utt_buffer) / self._sr

    def _finish_utterance(self, t1: float) -> None:
        audio = np.concatenate(self._utt_buffer)
        self._triggered = False
        self._utt_buffer = []
        self._emit_utterance(self._utt_t0, t1, audio)

    def _force_split(self) -> None:
        # max_utt_s를 넘으면 에너지가 가장 작은 조각 경계에서 자르고 0.5초 겹쳐 이어간다
        lengths = [len(c) for c in self._utt_buffer]
        cumulative = np.cumsum(lengths)
        overlap_samples = int(self._split_overlap_s * self._sr)

        # cut 지점이 겹침 길이보다 앞쪽이면 overlap이 0으로 눌려 이미 내보낸 구간이
        # 다음 조각에 통째로 다시 들어간다 — 겹침 확보가 가능한 지점에서만 고른다
        valid_idx = [i for i, c in enumerate(cumulative) if c >= overlap_samples]
        if not valid_idx:
            valid_idx = list(range(len(self._utt_buffer)))

        energies = [
            np.sqrt(np.mean(self._utt_buffer[i].astype(np.float64) ** 2)) for i in valid_idx
        ]
        cut_idx = valid_idx[int(np.argmin(energies))]
        cut_sample = int(cumulative[cut_idx])

        concat = np.concatenate(self._utt_buffer)
        cut_t = self._utt_t0 + cut_sample / self._sr
        self._emit_utterance(self._utt_t0, cut_t, concat[:cut_sample])

        overlap_start = max(0, cut_sample - overlap_samples)
        remainder = concat[overlap_start:]
        self._utt_t0 = cut_t - (cut_sample - overlap_start) / self._sr
        self._utt_buffer = [remainder] if len(remainder) else []

    def _emit_utterance(self, t0: float, t1: float, audio: np.ndarray) -> None:
        self._segments.append((t0, t1))
        self.on_utterance(t0, t1, audio)

    def flush(self, t_end: float) -> None:
        # 파일 끝의 마지막 발화를 강제로 확정해 on_utterance로 내보낸다 — EOF 직전에 양 채널 모두 불러야 한다
        if self._triggered and self._utt_buffer:
            audio = np.concatenate(self._utt_buffer)
            self._triggered = False
            self._utt_buffer = []
            self._emit_utterance(self._utt_t0, t_end, audio)
        self._carry = np.array([], dtype=np.float32)
        self._stream_end_t = max(self._stream_end_t, t_end)

    def get_stats(self) -> dict:
        # 지금까지의 발화 구간 목록과 총 발화/침묵 시간을 반환한다
        speech_s = sum(t1 - t0 for t0, t1 in self._segments)
        silence_s = max(0.0, self._stream_end_t - speech_s)
        return {
            "segments": list(self._segments),
            "speech_s": speech_s,
            "silence_s": silence_s,
        }
