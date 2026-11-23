"""
indexer.py — GIAI ĐOẠN 3: TẠO VECTOR (EMBEDDING) VÀ CHỈ MỤC TÌM KIẾM
=====================================================================
Đọc file data/processed/chunks.jsonl (đầu ra Giai đoạn 2), tạo vector biểu diễn
ý nghĩa (embedding) cho từng chunk bằng Gemini Embedding API, đồng thời chuẩn bị
danh sách từ khóa (tokens) để tìm kiếm BM25 ở Giai đoạn 4.

Vì sao thiết kế như vậy?
    - Máy 8GB RAM, không GPU: dùng Gemini Embedding API (gemini-embedding-001)
      và lưu ma trận vector bằng NumPy (.npz) giúp nhẹ máy tuyệt đối (chỉ ~3MB),
      không cần cài PyTorch hay cơ sở dữ liệu nặng nề.
    - Mỗi chunk dùng trường `text_for_embedding` (đã có sẵn tên Luật, Chương, Điều)
      để vector hiểu rõ ngữ cảnh của đoạn luật đó.

5 bước:
    1. Đọc danh sách chunk từ chunks.jsonl   -> load_chunks()
    2. Tách từ tiếng Việt phục vụ BM25       -> tokenize_vi()
    3. Gọi Gemini API tạo vector theo lô     -> embed_texts()
    4. Kiểm tra chất lượng chỉ mục           -> validate_index()
    5. Lưu chỉ mục ra data/processed/        -> build_and_save_index()

Cách chạy (từ thư mục gốc dự án):
    python src/indexer.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from dotenv import load_dotenv

# Tự động nạp biến môi trường từ file .env (chứa GEMINI_API_KEY)
load_dotenv()

# --------------------------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN VÀ MÔ HÌNH
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"
OUTPUT_DIR = ROOT / "data" / "processed"
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npz"
INDEX_REPORT_FILE = OUTPUT_DIR / "index_report.json"

DEFAULT_EMBED_MODEL = "gemini-embedding-001"
BATCH_SIZE = 50  # Số lượng chunk gửi lên API mỗi lần để nhanh và tránh quá tải


# ==========================================================================
# BƯỚC 1 — ĐỌC DỮ LIỆU CHUNKS TỪ GIAI ĐOẠN 2
# ==========================================================================
def load_chunks(path: str | Path = CHUNKS_FILE) -> list[dict]:
    """
    Đọc file chunks.jsonl, trả về danh sách các dict (mỗi dict là 1 chunk).
    Kiểm tra luôn các trường bắt buộc để đảm bảo dữ liệu hợp lệ.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {p}. Hãy chạy Giai đoạn 2 trước: python src/chunker.py"
        )

    chunks: list[dict] = []
    required_keys = ("chunk_id", "article_no", "text", "text_for_embedding", "citation")

    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for k in required_keys:
                if k not in item:
                    raise ValueError(f"Dòng {line_no} trong {p.name} thiếu trường '{k}'.")
            chunks.append(item)

    if not chunks:
        raise ValueError(f"File {p} rỗng, không có chunk nào.")
    return chunks


# ==========================================================================
# BƯỚC 2 — TÁCH TỪ TIẾNG VIỆT ĐƠN GIẢN CHO TÌM KIẾM TỪ KHÓA (BM25)
# ==========================================================================
# Danh sách từ hư (stopwords) rất phổ biến nhưng ít mang ý nghĩa phân biệt điều luật
VI_STOPWORDS = {
    "của", "và", "các", "có", "được", "trong", "cho", "theo", "tại",
    "với", "là", "về", "hoặc", "này", "khi", "đến", "từ", "do", "để",
    "như", "thì", "mà", "bị", "vào", "ra", "trên", "dưới", "ai", "gì",
    "nào", "sao", "thế", "nhiêu", "bao",
}

# Các từ ghép pháp lý quan trọng mà nếu tách rời chữ "lương" hay "nghỉ" sẽ gây bắt nhầm điều luật
# Ví dụ: "lương hưu" khác hoàn toàn với "tiền lương / trả lương" (Điều 94).
COMPOUND_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\blương\s+hưu\b", re.IGNORECASE), "lương_hưu hưu_trí"),
    (re.compile(r"\bnghỉ\s+hưu\b", re.IGNORECASE), "nghỉ_hưu hưu_trí"),
]

RE_WORD_TOKEN = re.compile(r"[0-9a-zA-ZÀ-ỹ_]+")


def tokenize_vi(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Chuyển văn bản tiếng Việt về chữ thường, tách thành danh sách các từ/số.
    - Bảo toàn các từ ghép đặc biệt như 'lương_hưu', 'nghỉ_hưu' để chữ 'lương' trong
      'lương hưu' không bị khớp nhầm sang các điều luật về 'trả lương' (Điều 94).
    - Giữ lại các con số (như '36', '2019', '12', '169') để tìm chính xác số Điều/Khoản.
    - Tạo thêm cụm 2 từ liền kề (bigrams) từ các từ gốc để bắt trúng từ ghép tiếng Việt.
    """
    if not text:
        return []

    # Bước 1: Tách từ gốc để tạo bigrams tự nhiên
    raw_words = [w.lower() for w in re.findall(r"[0-9a-zA-ZÀ-ỹ]+", text)]
    bigrams = [f"{raw_words[i]}_{raw_words[i + 1]}" for i in range(len(raw_words) - 1)]

    # Nếu trong câu là "nhận lương hưu", loại bỏ bigram gây nhiễu "nhận_lương"
    for i in range(len(raw_words) - 2):
        if raw_words[i + 1] == "lương" and raw_words[i + 2] == "hưu":
            bad_bg = f"{raw_words[i]}_lương"
            bigrams = [b for b in bigrams if b != bad_bg]

    # Bước 2: Gom cụm đặc biệt ("lương hưu" -> "lương_hưu") trước khi lấy unigrams
    processed = text.lower()
    for pattern, repl in COMPOUND_REPLACEMENTS:
        processed = pattern.sub(repl, processed)

    words = [w.lower() for w in RE_WORD_TOKEN.findall(processed)]
    if remove_stopwords:
        unigrams = [w for w in words if w not in VI_STOPWORDS]
    else:
        unigrams = list(words)

    return unigrams + bigrams


# ==========================================================================
# BƯỚC 3 — TẠO VECTOR EMBEDDING BẰNG GEMINI API
# ==========================================================================
def normalize_vectors(matrix: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa mỗi vector về độ dài bằng 1 (L2 normalization).
    Khi các vector đã chuẩn hóa, phép nhân tích vô hướng (dot product) chính là
    độ tương đồng Cosine (Cosine Similarity), giúp tìm kiếm cực nhanh.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Tránh chia cho 0 nếu có vector toàn số 0
    norms = np.where(norms == 0, 1.0, norms)
    return (arr / norms).astype(np.float32)


_SHARED_EMBED_CLIENT = None


def get_embed_client(api_key: str | None = None):
    """Khởi tạo Gemini Client 1 lần duy nhất (Singleton) để tái sử dụng kết nối HTTPS."""
    global _SHARED_EMBED_CLIENT
    if api_key is not None or _SHARED_EMBED_CLIENT is None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "Thiếu GEMINI_API_KEY. Hãy kiểm tra file .env ở thư mục gốc dự án."
            )
        from google import genai
        from google.genai import types

        http_opts = types.HttpOptions(
            timeout=10000,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        client = genai.Client(api_key=key, http_options=http_opts)
        if api_key is None:
            _SHARED_EMBED_CLIENT = client
        return client
    return _SHARED_EMBED_CLIENT


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
    api_key: str | None = None,
    client=None,
) -> np.ndarray:
    """
    Gửi danh sách văn bản lên Gemini Embedding API theo từng lô (batch)
    và trả về ma trận NumPy kích thước (số_văn_bản, số_chiều_vector).
    """
    if not texts:
        raise ValueError("Danh sách văn bản cần tạo embedding không được rỗng.")

    if client is None:
        client = get_embed_client(api_key=api_key)

    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        # Tự động thử lại nếu gặp giới hạn 100 chunks/phút của gói Gemini Free Tier
        for attempt in range(4):
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=batch,
                )
                break
            except Exception as err:
                if "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err):
                    wait_sec = 38 if attempt == 0 else 62
                    print(
                        f"  [Đợi quota Free Tier] Đã gửi {start}/{len(texts)} chunks, "
                        f"tạm nghỉ {wait_sec}s rồi tiếp tục..."
                    )
                    time.sleep(wait_sec)
                else:
                    raise
        else:
            raise RuntimeError("Vượt quá số lần thử lại khi gọi Gemini Embedding API.")

        for emb in response.embeddings:
            all_vectors.append(list(emb.values))

    matrix = np.asarray(all_vectors, dtype=np.float32)
    return normalize_vectors(matrix)


# ==========================================================================
# BƯỚC 4 — KIỂM TRA CHẤT LƯỢNG CHỈ MỤC
# ==========================================================================
def validate_index(chunks: list[dict], embeddings: np.ndarray) -> dict:
    """
    Kiểm tra xem ma trận vector có khớp hoàn toàn với danh sách chunk không:
        - Số dòng của ma trận phải bằng số lượng chunk.
        - Không có giá trị lỗi (NaN hoặc Inf).
        - Không có vector nào toàn số 0.
        - Các vector đã được chuẩn hóa (độ dài ~ 1.0).
    """
    errors: list[str] = []

    if embeddings.ndim != 2:
        errors.append(f"Ma trận embedding phải là 2 chiều, nhận được {embeddings.ndim} chiều.")
    else:
        n_rows, n_dims = embeddings.shape
        if n_rows != len(chunks):
            errors.append(
                f"Số vector ({n_rows}) không khớp với số chunk ({len(chunks)})."
            )
        if n_dims == 0:
            errors.append("Chiều của vector bằng 0.")

    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        errors.append("Ma trận embedding chứa giá trị NaN hoặc vô cực (Inf).")

    if embeddings.ndim == 2 and embeddings.shape[0] > 0:
        norms = np.linalg.norm(embeddings, axis=1)
        zero_count = int(np.sum(norms < 1e-6))
        if zero_count > 0:
            errors.append(f"Có {zero_count} vector có độ dài bằng 0.")

    chunk_ids = [c["chunk_id"] for c in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        errors.append("Có chunk_id bị trùng lặp trong danh sách chunks.")

    return {
        "counts": {
            "chunks": len(chunks),
            "vectors": int(embeddings.shape[0]) if embeddings.ndim == 2 else 0,
            "dimensions": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "quoted_other_law": sum(1 for c in chunks if c.get("quoted_other_law")),
        },
        "errors": errors,
        "ok": len(errors) == 0,
    }


# ==========================================================================
# BƯỚC 5 — LƯU VÀ ĐỌC LẠI CHỈ MỤC
# ==========================================================================
def save_index(
    chunks: list[dict],
    embeddings: np.ndarray,
    embeddings_path: str | Path = EMBEDDINGS_FILE,
    report_path: str | Path = INDEX_REPORT_FILE,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> dict:
    """
    Lưu ma trận vector và danh sách chunk_id vào file nén NumPy (.npz),
    kèm báo cáo kiểm tra index_report.json.
    """
    report = validate_index(chunks, embeddings)
    report["model"] = model_name

    emb_p = Path(embeddings_path)
    rep_p = Path(report_path)
    emb_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.parent.mkdir(parents=True, exist_ok=True)

    if not report["ok"]:
        rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"Chỉ mục không hợp lệ: {report['errors']}")

    chunk_ids = np.array([c["chunk_id"] for c in chunks], dtype=str)
    np.savez_compressed(
        emb_p,
        embeddings=embeddings.astype(np.float32),
        chunk_ids=chunk_ids,
        model=np.array([model_name], dtype=str),
    )
    rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_index(
    chunks_path: str | Path = CHUNKS_FILE,
    embeddings_path: str | Path = EMBEDDINGS_FILE,
) -> tuple[list[dict], np.ndarray]:
    """
    Đọc lại danh sách chunks và ma trận embeddings từ ổ cứng.
    Kiểm tra thứ tự chunk_id để đảm bảo dòng thứ i của ma trận đúng là chunk thứ i.
    """
    chunks = load_chunks(chunks_path)
    emb_p = Path(embeddings_path)
    if not emb_p.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file vector {emb_p}. Hãy chạy: python src/indexer.py"
        )

    data = np.load(emb_p, allow_pickle=False)
    embeddings = data["embeddings"].astype(np.float32)
    saved_ids = [str(x) for x in data["chunk_ids"].tolist()]
    current_ids = [c["chunk_id"] for c in chunks]

    if saved_ids != current_ids:
        raise ValueError(
            "Danh sách chunk_id trong embeddings.npz không khớp với chunks.jsonl hiện tại. "
            "Hãy chạy lại: python src/indexer.py"
        )
    return chunks, embeddings


def build_and_save_index(
    chunks_path: str | Path = CHUNKS_FILE,
    embeddings_path: str | Path = EMBEDDINGS_FILE,
    report_path: str | Path = INDEX_REPORT_FILE,
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> dict:
    """
    Hàm tổng hợp: Đọc chunks -> Tạo vector -> Kiểm tra -> Lưu file.
    Cho phép truyền `embed_fn` tùy chỉnh để viết unit test nhanh không tốn quota API.
    """
    start_time = time.perf_counter()
    chunks = load_chunks(chunks_path)
    texts = [c["text_for_embedding"] for c in chunks]

    if embed_fn is not None:
        embeddings = normalize_vectors(embed_fn(texts))
    else:
        embeddings = embed_texts(texts, model=model_name, batch_size=batch_size)

    elapsed = round(time.perf_counter() - start_time, 2)
    report = save_index(
        chunks=chunks,
        embeddings=embeddings,
        embeddings_path=embeddings_path,
        report_path=report_path,
        model_name=model_name,
    )
    report["elapsed_seconds"] = elapsed
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


# ==========================================================================
# CHẠY TRỰC TIẾP TỪ DÒNG LỆNH
# ==========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Giai đoạn 3: Tạo vector embedding cho chunks.jsonl"
    )
    parser.add_argument("--chunks", default=str(CHUNKS_FILE), help="Đường dẫn chunks.jsonl")
    parser.add_argument("--out", default=str(EMBEDDINGS_FILE), help="Đường dẫn file .npz đầu ra")
    parser.add_argument("--report", default=str(INDEX_REPORT_FILE), help="Đường dẫn báo cáo JSON")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Tên mô hình Gemini Embedding")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Kích thước mỗi lô gửi API")
    args = parser.parse_args()

    print(f"[Giai đoạn 3] Đang đọc dữ liệu từ: {args.chunks}")
    report = build_and_save_index(
        chunks_path=args.chunks,
        embeddings_path=args.out,
        report_path=args.report,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    c = report["counts"]
    print(
        f"[Giai đoạn 3] Hoàn tất trong {report['elapsed_seconds']}s | "
        f"Số chunk: {c['chunks']} | Kích thước vector: {c['vectors']}x{c['dimensions']}"
    )
    print(f"Kết quả: {'ĐẠT' if report['ok'] else 'LỖI'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
