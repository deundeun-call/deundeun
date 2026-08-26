"""고객·상담사 모노 wav 두 개를 좌우 채널 스테레오 wav로 합친다."""
import sys

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_mono_16k(path: str) -> np.ndarray:
    # 모노 wav를 읽어 16kHz가 아니면 오류를 낸다 (리샘플링은 하지 않는다)
    data, sr = sf.read(path, always_2d=False)
    if data.ndim != 1:
        raise ValueError(f"{path} 는 모노가 아닙니다 (채널 수: {data.ndim})")
    if sr != TARGET_SR:
        raise ValueError(f"{path} 의 샘플레이트가 {sr}Hz 입니다 (16000Hz 필요)")
    return data


def make_stereo(customer_path: str, agent_path: str, out_path: str) -> None:
    # 두 모노 신호를 읽고 짧은 쪽을 무음으로 채워 길이를 맞춘다
    customer = load_mono_16k(customer_path)
    agent = load_mono_16k(agent_path)

    length = max(len(customer), len(agent))
    left = np.zeros(length, dtype=customer.dtype)
    right = np.zeros(length, dtype=agent.dtype)
    left[: len(customer)] = customer
    right[: len(agent)] = agent

    stereo = np.stack([left, right], axis=1)
    sf.write(out_path, stereo, TARGET_SR, subtype="PCM_16")


def main() -> None:
    if len(sys.argv) != 4:
        print("사용법: python scripts/make_stereo.py customer.wav agent.wav out.wav")
        sys.exit(1)
    make_stereo(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"저장 완료: {sys.argv[3]}")


if __name__ == "__main__":
    main()
