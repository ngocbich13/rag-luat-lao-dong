"""
ingest.py — GIAI ĐOẠN 1 (YÊU CẦU 1 & 2): NẠP ĐA VĂN BẢN LUẬT & KIỂM TRA CHẤT LƯỢNG CHUNK
=========================================================================================
Quản lý việc nạp nhiều văn bản pháp luật từ thư mục `data/raw/` (Bộ luật Lao động 2019,
Luật Bảo hiểm xã hội 2024, Nghị định 145/2020/NĐ-CP và các văn bản hướng dẫn sau này)
vào kho dữ liệu `data/processed/chunks.jsonl`.

Cam kết nguyên tắc:
    1. KHÔNG tự bịa nội dung điều luật: Chỉ nạp những file `.docx` có thật trong `data/raw/`.
       Với các văn bản trong danh mục mục tiêu nhưng chưa có file `.docx` trong `data/raw/`,
       script sẽ báo rõ tên file bạn cần cung cấp.
    2. Mỗi chunk được kiểm tra nghiêm ngặt: đủ nội dung (>= 50 ký tự) và đủ metadata
       (`doc_id`, `law_name`, `chapter_no`, `article_no`, `clause_nos`, `citation`).
    3. Riêng Điều 46 và Điều 169 Bộ luật Lao động 2019 luôn được kiểm tra tự động
       để đảm bảo đủ 100% các khoản.

Cách chạy:
    python src/ingest.py
    python src/ingest.py --rebuild-index   # Nạp lại văn bản và cập nhật luôn embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.check_chunks import MIN_CHARS, inspect_chunks
from src.chunker import MAX_WORDS, OVERLAP_WORDS, chunk_document
from src.parser import (
    attach_metadata,
    build_tree,
    check_input_file,
    clean_document,
    extract_document_info,
    read_docx,
)

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
INGEST_REPORT_FILE = PROCESSED_DIR / "ingest_report.json"

# --------------------------------------------------------------------------
# DANH MỤC CÁC VĂN BẢN LUẬT HỖ TRỢ TRONG HỆ THỐNG
# Khi muốn thêm văn bản mới, chỉ cần bỏ file .docx tương ứng vào data/raw/
# rồi chạy `python src/ingest.py --rebuild-index`
# --------------------------------------------------------------------------
LAW_CATALOG: list[dict] = [
    {
        "filename": "luatLaoDong2019.docx",
        "doc_id": "45/2019/QH14",
        "title": "Bộ luật Lao động",
        "issued_date": "2019-11-20",
        "effective_date": "2021-01-01",
        "description": "Bộ luật Lao động 2019 (Văn bản hợp nhất 18/VBHN-VPQH)",
        "required": True,
    },
    {
        "filename": "luatBaoHiemXaHoi2024.docx",
        "doc_id": "41/2024/QH15",
        "title": "Luật Bảo hiểm xã hội",
        "issued_date": "2024-06-29",
        "effective_date": "2025-07-01",
        "description": "Luật Bảo hiểm xã hội số 41/2024/QH15 (quy định chi tiết điều kiện hưởng lương hưu, chế độ thai sản, ốm đau, BHXH một lần)",
        "required": False,
    },
    {
        "filename": "nghiDinh145_2020.docx",
        "doc_id": "145/2020/NĐ-CP",
        "title": "Nghị định 145/2020/NĐ-CP",
        "issued_date": "2020-12-14",
        "effective_date": "2021-02-01",
        "description": "Nghị định 145/2020/NĐ-CP quy định chi tiết và hướng dẫn thi hành Bộ luật Lao động về điều kiện lao động và quan hệ lao động",
        "required": False,
    },
]


def _extract_generic_doc_info(preamble: list[dict], tables: list[dict], fallback_meta: dict | None = None) -> dict:
    """
    Trích xuất metadata từ phần mở đầu của file Word.
    Nếu là Luật Bảo hiểm xã hội hoặc Nghị định không khớp mẫu 'Bộ luật ...' mặc định,
    dùng regex mở rộng hoặc lấy từ danh mục `fallback_meta`.
    """
    info = extract_document_info(preamble, tables)

    if not info.get("doc_id") or not info.get("title"):
        text = " ".join(p["text"] for p in preamble)
        m = re.search(
            r"((?:Bộ luật|Luật|Nghị định)\s+[^\d,;]+?)\s+số\s+(\d+/\d{4}/[A-ZĐ0-9\-]+)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            info["title"] = info["title"] or m.group(1).strip()
            info["doc_id"] = info["doc_id"] or m.group(2).strip()

    if fallback_meta:
        info["doc_id"] = info.get("doc_id") or fallback_meta.get("doc_id")
        info["title"] = info.get("title") or fallback_meta.get("title")
        info["issued_date"] = info.get("issued_date") or fallback_meta.get("issued_date")
        info["effective_date"] = info.get("effective_date") or fallback_meta.get("effective_date")

    return info


def parse_single_docx(
    docx_path: Path,
    fallback_meta: dict | None = None,
    max_words: int = MAX_WORDS,
    overlap: int = OVERLAP_WORDS,
) -> tuple[dict, list[dict]]:
    """
    Đọc 1 file .docx văn bản luật -> trả về (structured_doc, danh_sách_chunks).
    """
    valid_path = check_input_file(docx_path)
    raw_lines, tables = read_docx(valid_path)
    body, footnotes, _stats = clean_document(raw_lines)
    tree, _warnings = build_tree(body)
    info = _extract_generic_doc_info(tree["preamble"], tables, fallback_meta=fallback_meta)
    doc = attach_metadata(tree, info, valid_path.name)
    doc["footnotes"] = footnotes
    chunks = chunk_document(doc, max_words=max_words, overlap=overlap)
    return doc, chunks


def ingest_all_laws(
    raw_dir: str | Path = RAW_DIR,
    out_dir: str | Path = PROCESSED_DIR,
    rebuild_index: bool = False,
) -> dict:
    """
    Quét thư mục `data/raw/`, nạp tất cả các file `.docx` hiện có, kiểm tra chất lượng
    chunk (>= 50 ký tự, đủ metadata, Điều 46 & 169 đủ nội dung) và ghi ra `chunks.jsonl`.
    """
    raw_p = Path(raw_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    catalog_by_name = {item["filename"]: item for item in LAW_CATALOG}

    ingested_docs: list[dict] = []
    missing_optional_docs: list[dict] = []
    all_chunks: list[dict] = []

    # 1. Duyệt các văn bản trong danh mục chuẩn LAW_CATALOG theo đúng thứ tự
    processed_filenames: set[str] = set()
    for item in LAW_CATALOG:
        fname = item["filename"]
        fpath = raw_p / fname
        if fpath.exists():
            doc, chunks = parse_single_docx(fpath, fallback_meta=item)
            all_chunks.extend(chunks)
            processed_filenames.add(fname)
            ingested_docs.append(
                {
                    "filename": fname,
                    "doc_id": doc["doc_id"],
                    "law_name": doc["law_name"],
                    "effective_date": doc.get("effective_date"),
                    "chapters_count": len(doc["chapters"]),
                    "articles_count": sum(len(ch["articles"]) for ch in doc["chapters"]),
                    "chunks_count": len(chunks),
                }
            )
        else:
            if item.get("required"):
                raise FileNotFoundError(
                    f"Thiếu file văn bản luật bắt buộc: {fpath}"
                )
            missing_optional_docs.append(
                {
                    "filename": fname,
                    "expected_path": str(fpath),
                    "doc_id": item["doc_id"],
                    "law_name": f"{item['title']} {(item['issued_date'] or '')[:4]}".strip(),
                    "description": item["description"],
                }
            )

    # 2. Tự động phát hiện bất kỳ file .docx nào khác mà người dùng mới bỏ thêm vào data/raw/
    if raw_p.exists():
        for extra_path in sorted(raw_p.glob("*.docx")):
            if extra_path.name.startswith("~$") or extra_path.name in processed_filenames:
                continue
            fallback = catalog_by_name.get(extra_path.name, {
                "doc_id": extra_path.stem.upper(),
                "title": extra_path.stem,
                "issued_date": "2024-01-01",
                "effective_date": "2024-01-01",
            })
            doc, chunks = parse_single_docx(extra_path, fallback_meta=fallback)
            all_chunks.extend(chunks)
            processed_filenames.add(extra_path.name)
            ingested_docs.append(
                {
                    "filename": extra_path.name,
                    "doc_id": doc["doc_id"],
                    "law_name": doc["law_name"],
                    "effective_date": doc.get("effective_date"),
                    "chapters_count": len(doc["chapters"]),
                    "articles_count": sum(len(ch["articles"]) for ch in doc["chapters"]),
                    "chunks_count": len(chunks),
                }
            )

    # 3. Ghi toàn bộ chunks ra data/processed/chunks.jsonl
    chunks_path = out_p / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 4. Chạy kiểm tra chất lượng toàn bộ chunks (< 50 ký tự, metadata, Điều 46 & 169)
    quality_report = inspect_chunks(chunks_path, min_chars=MIN_CHARS)

    # 5. Nếu người dùng yêu cầu --rebuild-index thì tạo lại embeddings.npz
    if rebuild_index:
        from src.indexer import build_and_save_index
        build_and_save_index(chunks_path=chunks_path)

    report = {
        "ingested_laws": ingested_docs,
        "pending_laws_to_provide": missing_optional_docs,
        "quality_check": quality_report,
        "ok": quality_report["ok"],
    }
    report_path = out_p / "ingest_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Nạp đa văn bản luật từ data/raw/ và kiểm tra chất lượng chunk"
    )
    ap.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Tự động gọi indexer.py để cập nhật embeddings.npz sau khi nạp",
    )
    args = ap.parse_args()

    report = ingest_all_laws(rebuild_index=args.rebuild_index)
    print("=== BÁO CÁO NẠP DỮ LIỆU LUẬT (src/ingest.py) ===")
    print("1. Các văn bản luật ĐÃ NẠP thành công:")
    for doc in report["ingested_laws"]:
        print(
            f"   + {doc['law_name']} ({doc['doc_id']}) từ file `data/raw/{doc['filename']}`: "
            f"{doc['articles_count']} Điều -> {doc['chunks_count']} chunks"
        )

    print("\n2. Các văn bản luật CHỜ BẠN CUNG CẤP FILE vào thư mục `data/raw/`:")
    for miss in report["pending_laws_to_provide"]:
        print(
            f"   - Cần file: `data/raw/{miss['filename']}` ({miss['law_name']} - {miss['doc_id']})\n"
            f"     Vai trò : {miss['description']}"
        )

    qc = report["quality_check"]
    print("\n3. Kiểm tra chất lượng dữ liệu:")
    print(f"   - Tổng số chunks          : {qc['total_chunks']}")
    print(f"   - Số chunk < 50 ký tự     : {qc['short_chunks_count']}")
    print(f"   - Số chunk thiếu metadata : {qc['missing_metadata_count']}")
    print(f"   - Điều 46 BLLĐ 2019 đủ ý  : {'ĐẠT' if qc['special_checks']['dieu_46']['ok'] else 'LỖI'}")
    print(f"   - Điều 169 BLLĐ 2019 đủ ý : {'ĐẠT' if qc['special_checks']['dieu_169']['ok'] else 'LỖI'}")
    print(f"Kết quả: {'ĐẠT' if report['ok'] else 'LỖI'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
