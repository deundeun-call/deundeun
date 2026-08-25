"""규정집 적재 실행기.

사용법:
    (.venv 활성화 후, 프로젝트 루트에서)
    python scripts/run_ingest.py            # data/corpus/*.pdf 를 이어서 적재
    python scripts/run_ingest.py --reset    # 기존 컬렉션을 지우고 새로 적재
"""

import argparse
import sys
from pathlib import Path

# server 패키지를 어디서 실행해도 찾을 수 있게 프로젝트 루트를 sys.path에 넣는다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.rag.ingest import ingest_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="data/corpus/*.pdf를 chromadb에 적재한다")
    parser.add_argument(
        "--reset", action="store_true", help="기존 컬렉션을 지우고 새로 넣는다"
    )
    args = parser.parse_args()
    ingest_corpus(reset=args.reset)


if __name__ == "__main__":
    main()
