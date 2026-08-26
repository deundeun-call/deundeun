"""pacer -> vad -> stt 파이프라인을 call1.wav로 손으로 확인하는 임시 스크립트.

실행:
    .venv/bin/python scripts/test_pipeline_call1.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from server import pacer, stt
from server.config import get as cfg_get
from server.vad import ChannelVAD

WAV_PATH = Path(__file__).resolve().parent.parent / "data" / "audio" / "merged" / "call1.wav"


async def main() -> None:
    sr = cfg_get("audio", "sample_rate")
    utterances: list[tuple[float, float, object]] = []
    t_end = 0.0

    def on_start(t: float) -> None:
        print(f"[VAD] 발화 시작 t={t:.2f}s")

    def on_utterance(t0: float, t1: float, audio) -> None:
        print(f"[VAD] 발화 종료 t0={t0:.2f}s t1={t1:.2f}s ({len(audio)}샘플)")
        utterances.append((t0, t1, audio))

    vad = ChannelVAD(on_start=on_start, on_utterance=on_utterance)

    async def on_frame(t: float, left, right) -> None:
        nonlocal t_end
        vad.push_frame(t, left)  # 고객 채널(왼쪽)만 VAD에 먹인다
        t_end = max(t_end, t + len(left) / sr)

    await pacer.play(str(WAV_PATH), on_frame, speed=20.0)
    vad.flush(t_end)

    print(f"\n총 발화 {len(utterances)}건 — STT 시작\n")
    for t0, t1, audio in utterances:
        text = await stt.transcribe(audio, sr)
        print(f"[STT] {t0:.2f}~{t1:.2f}s : {text}")

    stats = vad.get_stats()
    print(
        f"\n[통계] 발화 {stats['speech_s']:.1f}s / 침묵 {stats['silence_s']:.1f}s "
        f"/ 구간 {len(stats['segments'])}개"
    )


if __name__ == "__main__":
    asyncio.run(main())
