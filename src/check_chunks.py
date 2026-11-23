"""
check_chunks.py — KIỂM TRA CHẤT LƯỢNG DỮ LIỆU CHUNKS (GIAI ĐOẠN 1 - YÊU CẦU 1)
===============================================================================
Quét toàn bộ các chunk trong data/processed/chunks.jsonl để kiểm tra:
    1. Có chunk nào rỗng hoặc nội dung quá ngắn (< 50 ký tự) không.
    2. Mỗi chunk có đầy đủ metadata bắt buộc (tên văn bản, chương, điều, khoản,
       trích dẫn) không.
    3. Kiểm tra riêng Điều 46 (Trợ cấp thôi việc) và Điều 169 (Tuổi nghỉ hưu)
       của Bộ luật Lao động 2019 xem có đầy đủ toàn bộ các khoản và nội dung không.

Cách chạy:
    python src/check_chunks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"
MIN_CHARS = 50

REQUIRED_METADATA = (
    "chunk_id",
    "doc_id",
    "law_name",
    "chapter_no",
    "chapter_title",
    "article_no",
    "article_title",
    "clause_nos",
    "citation",
    "text",
    "text_for_embedding",
)


def inspect_chunks(path: str | Path = CHUNKS_FILE, min_chars: int = MIN_CHARS) -> dict:
    """
    Kiểm tra từng dòng trong chunks.jsonl.
    Trả về báo cáo chi tiết các chunk quá ngắn, thiếu metadata, và nội dung Điều 46, 169.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file {p}")

    chunks = [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    short_chunks: list[dict] = []
    missing_meta: list[dict] = []
    by_article: dict[tuple[str, int], list[dict]] = {}

    for c in chunks:
        text_clean = (c.get("text") or "").strip()
        if len(text_clean) < min_chars:
            short_chunks.append(
                {
                    "chunk_id": c.get("chunk_id"),
                    "article_no": c.get("article_no"),
                    "char_len": len(text_clean),
                    "text": text_clean,
                }
            )

        missing = [k for k in REQUIRED_METADATA if k not in c or c[k] is None or c[k] == ""]
        if missing:
            missing_meta.append(
                {
                    "chunk_id": c.get("chunk_id"),
                    "missing_keys": missing,
                }
            )

        key = (str(c.get("doc_id", "")), int(c.get("article_no", 0)))
        by_article.setdefault(key, []).append(c)

    # Kiểm tra riêng Điều 46 và Điều 169 Bộ luật Lao động 2019 (45/2019/QH14)
    art_46_chunks = by_article.get(("45/2019/QH14", 46), [])
    art_169_chunks = by_article.get(("45/2019/QH14", 169), [])

    art_46_text = "\n".join(c["text"] for c in art_46_chunks)
    art_169_text = "\n".join(c["text"] for c in art_169_chunks)

    art_46_ok = (
        len(art_46_chunks) >= 1
        and len(art_46_text) > 500
        and all(f"{i}." in art_46_text for i in (1, 2, 3, 4))
    )
    art_169_ok = (
        len(art_169_chunks) >= 1
        and len(art_169_text) > 500
        and all(f"{i}." in art_169_text for i in (1, 2, 3, 4, 5))
    )

    return {
        "total_chunks": len(chunks),
        "total_articles": len(by_article),
        "min_chars_threshold": min_chars,
        "short_chunks_count": len(short_chunks),
        "short_chunks": short_chunks,
        "missing_metadata_count": len(missing_meta),
        "missing_metadata": missing_meta,
        "special_checks": {
            "dieu_46": {
                "ok": art_46_ok,
                "chunks_count": len(art_46_chunks),
                "char_length": len(art_46_text),
                "clause_nos": art_46_chunks[0].get("clause_nos", []) if art_46_chunks else [],
                "preview": art_46_text[:250] + "..." if art_46_text else "",
            },
            "dieu_169": {
                "ok": art_169_ok,
                "chunks_count": len(art_169_chunks),
                "char_length": len(art_169_text),
                "clause_nos": art_169_chunks[0].get("clause_nos", []) if art_169_chunks else [],
                "preview": art_169_text[:250] + "..." if art_169_text else "",
            },
        },
        "ok": len(short_chunks) == 0 and len(missing_meta) == 0 and art_46_ok and art_169_ok,
    }


def main() -> int:
    report = inspect_chunks()
    print("=== KẾT QUẢ KIỂM TRA CHẤT LƯỢNG CHUNKS ===")
    print(f"Tổng số chunks   : {report['total_chunks']}")
    print(f"Tổng số điều luật: {report['total_articles']}")
    print(f"Số chunk < {report['min_chars_threshold']} ký tự: {report['short_chunks_count']}")
    if report["short_chunks"]:
        for sc in report["short_chunks"]:
            print(f"  - Cảnh báo chunk ngắn: {sc['chunk_id']} ({sc['char_len']} ký tự): {sc['text']}")
    print(f"Số chunk thiếu metadata: {report['missing_metadata_count']}")
    if report["missing_metadata"]:
        for mm in report["missing_metadata"]:
            print(f"  - Thiếu metadata: {mm['chunk_id']} -> {mm['missing_keys']}")

    d46 = report["special_checks"]["dieu_46"]
    d169 = report["special_checks"]["dieu_169"]
    print(
        f"Kiểm tra Điều 46 BLLĐ 2019 : {'ĐẠT' if d46['ok'] else 'LỖI'} "
        f"({d46['char_length']} ký tự, các khoản: {d46['clause_nos']})"
    )
    print(
        f"Kiểm tra Điều 169 BLLĐ 2019: {'ĐẠT' if d169['ok'] else 'LỖI'} "
        f"({d169['char_length']} ký tự, các khoản: {d169['clause_nos']})"
    )
    print(f"Kết luận chung: {'ĐẠT' if report['ok'] else 'CẦN XỬ LÝ'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
