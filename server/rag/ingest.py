"""data/corpus/*.pdf를 읽어 chromadb에 조각(청크)째로 넣는다.

🔴 컬렉션은 항상 metadata={"hnsw:space": "cosine"}로 만든다 — chroma 기본 거리는
   코사인이 아니라서 기본값으로 두면 rag.tau=0.568 문턱이 다른 척도가 된다.
🔴 임베딩 모델 이름/경로는 config.yaml의 rag.embed_model·rag.embed_model_dir 하나뿐이다.
   server/rag/search.py도 반드시 이 두 값을 config에서 읽어써서 적재·조회가 같은 모델을 쓰게 한다.
"""

import argparse
import re
from pathlib import Path
from typing import Iterable

import chromadb
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from server.config import get as cfg_get

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CLAUSE_RE = re.compile(r"제\s*\d+\s*조(?:의\s*\d+)?")


def _resolve(path_str: str) -> Path:
    # config.yaml에 적힌 상대경로를 프로젝트 루트 기준 절대경로로 바꾼다
    path = Path(path_str)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def chunk_text_with_offsets(
    text: str, size: int, tolerance: int, overlap: int
) -> list[tuple[str, int, int]]:
    # 문장 경계(.!?) 근처에서 size±tolerance로 자르고, 다음 조각은 overlap만큼 겹쳐 시작한다
    text = text.strip()
    n = len(text)
    if not text:
        return []
    boundary_re = re.compile(r"[.!?]")
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < n:
        target_end = min(start + size, n)
        if target_end >= n:
            piece = text[start:n].strip()
            if piece:
                chunks.append((piece, start, n))
            break
        lo = max(start + 1, target_end - tolerance)
        hi = min(n, target_end + tolerance)
        best_end = None
        for m in boundary_re.finditer(text, lo, hi):
            pos = m.end()
            if best_end is None or abs(pos - target_end) < abs(best_end - target_end):
                best_end = pos
        end = best_end if best_end is not None else target_end
        piece = text[start:end].strip()
        if piece:
            chunks.append((piece, start, end))
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def extract_pdf_chunks(
    pdf_path: Path, size: int, tolerance: int, overlap: int
) -> Iterable[tuple[str, dict]]:
    # PDF 한 개를 페이지 순서대로 훑으며 (조각 텍스트, {doc,clause,page}) 를 만든다
    doc_name = pdf_path.stem
    current_clause = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            clause_matches = list(_CLAUSE_RE.finditer(text))
            for piece, start, _end in chunk_text_with_offsets(text, size, tolerance, overlap):
                for m in clause_matches:
                    if m.start() <= start:
                        current_clause = m.group().replace(" ", "")
                    else:
                        break
                yield piece, {"doc": doc_name, "clause": current_clause, "page": page_no}


def load_embedder(model_name: str, local_dir: Path) -> SentenceTransformer:
    # 로컬 경로에 이미 받아둔 모델이 있으면 그걸 쓰고, 없으면 한 번 받아서 저장해둔다
    import os

    if local_dir.exists() and any(local_dir.iterdir()):
        return SentenceTransformer(str(local_dir))
    if os.environ.get("OFFLINE") == "1":
        raise RuntimeError(
            f"OFFLINE=1인데 로컬 임베딩 모델이 없습니다: {local_dir}\n"
            "인터넷 되는 곳에서 OFFLINE=0으로 한 번 실행해 먼저 받아두세요."
        )
    model = SentenceTransformer(model_name)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(local_dir))
    return model


def get_collection(persist_dir: Path, name: str, reset: bool):
    # 🔴 hnsw:space를 cosine으로 고정해서 만든다 — 컬렉션 생성 시에만 적용되는 값이라 매번 이 함수를 거쳐야 한다
    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
        return client.create_collection(name, metadata={"hnsw:space": "cosine"})
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name, metadata={"hnsw:space": "cosine"})


def ingest_corpus(corpus_dir: Path | None = None, reset: bool = False) -> int:
    # data/corpus/*.pdf 전부를 조각내 임베딩하고 chromadb에 넣는다. 총 적재 청크 수를 반환한다
    size = cfg_get("rag", "chunk_size")
    tolerance = cfg_get("rag", "chunk_tolerance")
    overlap = cfg_get("rag", "chunk_overlap")
    model_name = cfg_get("rag", "embed_model")
    model_dir = _resolve(cfg_get("rag", "embed_model_dir"))
    persist_dir = _resolve(cfg_get("rag", "persist_dir"))
    collection_name = cfg_get("rag", "collection")

    corpus_dir = corpus_dir or (_PROJECT_ROOT / "data" / "corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(corpus_dir.glob("*.pdf"))

    texts: list[str] = []
    metadatas: list[dict] = []
    for pdf_path in pdf_paths:
        for piece, meta in extract_pdf_chunks(pdf_path, size, tolerance, overlap):
            texts.append(piece)
            metadatas.append(meta)

    total = len(texts)
    if total == 0:
        get_collection(persist_dir, collection_name, reset)
        print(f"총 {total}개 청크 적재")
        return total

    embedder = load_embedder(model_name, model_dir)
    embeddings = embedder.encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    collection = get_collection(persist_dir, collection_name, reset)
    ids = [f"{m['doc']}::{m['page']}::{i}" for i, m in enumerate(metadatas)]
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    print(f"총 {total}개 청크 적재")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="data/corpus/*.pdf를 chromadb에 적재한다")
    parser.add_argument(
        "--reset", action="store_true", help="기존 컬렉션을 지우고 새로 넣는다"
    )
    args = parser.parse_args()
    ingest_corpus(reset=args.reset)


if __name__ == "__main__":
    main()
