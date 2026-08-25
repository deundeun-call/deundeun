"""
requirements.txt에 명시된 패키지들이 정상적으로 import되는지 하나씩 확인하는 스크립트.

사용법:
    (.venv 활성화 후)
    python scripts/check_env.py
"""

import importlib

# (표시용 패키지명, 실제 import 모듈명)
PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn[standard]", "uvicorn"),
    ("websockets", "websockets"),
    ("pydantic", "pydantic"),
    ("python-dotenv", "dotenv"),
    ("pyyaml", "yaml"),
    ("sqlalchemy", "sqlalchemy"),
    ("numpy", "numpy"),
    ("soundfile", "soundfile"),
    ("faster-whisper", "faster_whisper"),
    ("silero-vad", "silero_vad"),
    ("praat-parselmouth", "parselmouth"),
    ("chromadb", "chromadb"),
    ("sentence-transformers", "sentence_transformers"),
    ("pdfplumber", "pdfplumber"),
    ("rank_bm25", "rank_bm25"),
    ("kiwipiepy", "kiwipiepy"),
    ("httpx", "httpx"),
    ("cryptography", "cryptography"),
    ("matplotlib", "matplotlib"),
]


def main():
    name_width = max(len(name) for name, _ in PACKAGES) + 2
    ok_count = 0
    fail_count = 0

    for display_name, module_name in PACKAGES:
        try:
            importlib.import_module(module_name)
        except Exception as e:  # noqa: BLE001
            print(f"{display_name:<{name_width}} FAIL  ({type(e).__name__}: {e})")
            fail_count += 1
        else:
            print(f"{display_name:<{name_width}} OK")
            ok_count += 1

    print()
    print(f"총 {len(PACKAGES)}개 중 OK {ok_count}개, 실패 {fail_count}개")


if __name__ == "__main__":
    main()
