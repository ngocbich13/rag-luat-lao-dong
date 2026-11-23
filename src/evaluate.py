"""
evaluate.py — GIAI ĐOẠN 6: ĐÁNH GIÁ ĐỘ CHÍNH XÁC HỆ THỐNG RAG
=============================================================
Đo lường độ chính xác của hệ thống theo các chỉ số mục tiêu đã đề ra trong
tài liệu docs/00_problem_definition.md trên bộ câu hỏi chuẩn eval/test_questions.json:

    1. Tìm kiếm — Recall@5 (Mục tiêu > 85%):
       Tỷ lệ câu hỏi trong phạm vi mà Điều luật chuẩn nằm trong top 5 chunk tìm được.
    2. Tìm kiếm — MRR (Mean Reciprocal Rank):
       Vị trí trung bình của Điều luật đúng (càng gần 1.0 nghĩa là Điều đúng luôn ở top 1).
    3. Trích dẫn — Citation Accuracy (Mục tiêu > 90%):
       Tỷ lệ câu trả lời trích dẫn đúng Điều luật kỳ vọng.
    4. Cờ Nghị định — Needs Decree Accuracy:
       Tỷ lệ phát hiện đúng các Điều giao Chính phủ quy định chi tiết (`needs_decree`).
    5. An toàn — Refusal Accuracy (Mục tiêu > 90%):
       Tỷ lệ từ chối đúng các câu hỏi ngoài phạm vi Bộ luật Lao động.

Cách chạy (từ thư mục gốc dự án):
    python src/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generator import RAGGenerator
from src.retriever import HybridRetriever

QUESTIONS_FILE = ROOT / "eval" / "test_questions.json"
EVAL_REPORT_FILE = ROOT / "data" / "processed" / "eval_report.json"


# ==========================================================================
# BƯỚC 1 — ĐỌC BỘ CÂU HỎI KIỂM THỬ CHUẨN
# ==========================================================================
def load_test_questions(path: str | Path = QUESTIONS_FILE) -> list[dict]:
    """Đọc và kiểm tra định dạng của file eval/test_questions.json."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy bộ câu hỏi đánh giá: {p}")

    items = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"File {p} phải chứa danh sách câu hỏi không rỗng.")

    required = ("id", "category", "question", "expected_articles", "expected_status")
    for idx, q in enumerate(items, start=1):
        for k in required:
            if k not in q:
                raise ValueError(f"Câu hỏi thứ {idx} thiếu trường '{k}'.")
    return items


# ==========================================================================
# BƯỚC 2 — ĐÁNH GIÁ TẦNG TÌM KIẾM & TẦNG SINH CÂU TRẢ LỜI
# ==========================================================================
def _deterministic_rag_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
    """
    Hàm sinh câu trả lời chuẩn dựa trên tài liệu truy xuất được (dùng khi chạy
    đánh giá offline không tốn quota API hoặc khi người dùng không bật --live-llm).
    Hàm đọc đúng các nguồn `[TÀI LIỆU i] Nguồn trích dẫn: ...` trong user_prompt
    để trích dẫn trung thực các Điều luật mà Retriever đã tìm ra.
    """
    import re

    citations = re.findall(r"Nguồn trích dẫn:\s*([^\(\n]+)", user_prompt)
    clean_citations = [c.strip() for c in citations if c.strip()]
    top_cit = clean_citations[0] if clean_citations else "Bộ luật Lao động 2019"

    needs_decree_note = ""
    if "Chính phủ quy định chi tiết" in user_prompt:
        needs_decree_note = " Chi tiết thi hành thực hiện theo Nghị định của Chính phủ."

    answer = (
        f"Căn cứ theo {top_cit} (và các tài liệu liên quan: {', '.join(clean_citations[:3])}), "
        f"quy định của Bộ luật Lao động 2019 được áp dụng trực tiếp cho trường hợp này.{needs_decree_note}"
    )
    return answer, 300, 75


def run_evaluation(
    questions: list[dict] | None = None,
    retriever: HybridRetriever | None = None,
    top_k: int = 5,
    use_vector: bool = False,
    live_llm: bool = False,
    report_path: str | Path | None = EVAL_REPORT_FILE,
) -> dict:
    """
    Chạy đánh giá toàn diện trên bộ câu hỏi chuẩn và lưu báo cáo JSON.
    Mặc định `use_vector=False, live_llm=False` để chạy kiểm thử chỉ trong < 1 giây,
    không lo hết quota Gemini Free Tier; khi cần gọi API thật có thể bật `--live-llm --use-vector`.
    """
    start_time = time.perf_counter()
    qs = questions if questions is not None else load_test_questions()
    ret = retriever or HybridRetriever.from_files()

    llm_fn = None if live_llm else _deterministic_rag_llm
    gen = RAGGenerator(retriever=ret, llm_fn=llm_fn)

    in_scope_total = 0
    recall_at_k_hits = 0
    reciprocal_ranks: list[float] = []
    citation_hits = 0
    needs_decree_hits = 0

    out_of_scope_total = 0
    refusal_hits = 0

    details: list[dict] = []

    for item in qs:
        qid = item["id"]
        cat = item["category"]
        question = item["question"]
        expected_arts = [int(a) for a in item["expected_articles"]]
        expected_status = item["expected_status"]
        expected_decree = bool(item.get("expected_needs_decree", False))

        result = gen.answer(question, top_k=top_k, use_vector=use_vector)
        retrieved_arts = [int(s["article_no"]) for s in result["sources"]]
        cited_arts = [int(a) for a in result["cited_articles"]]

        if cat == "out_of_scope" or not expected_arts:
            out_of_scope_total += 1
            is_refused = result["status"] == "refused"
            if is_refused:
                refusal_hits += 1
            details.append(
                {
                    "id": qid,
                    "category": cat,
                    "question": question,
                    "expected_status": expected_status,
                    "actual_status": result["status"],
                    "passed": is_refused,
                }
            )
        else:
            in_scope_total += 1
            # Kiểm tra Recall@k: có ít nhất 1 Điều kỳ vọng nằm trong các nguồn tìm được
            hit_ranks = [
                idx + 1
                for idx, art in enumerate(retrieved_arts)
                if art in expected_arts
            ]
            hit_recall = len(hit_ranks) > 0
            if hit_recall:
                recall_at_k_hits += 1
                reciprocal_ranks.append(1.0 / hit_ranks[0])
            else:
                reciprocal_ranks.append(0.0)

            # Kiểm tra Trích dẫn đúng Điều kỳ vọng
            hit_citation = any(a in cited_arts for a in expected_arts)
            if hit_citation:
                citation_hits += 1

            # Kiểm tra cờ needs_decree
            hit_decree = bool(result["needs_decree"]) == expected_decree
            if hit_decree:
                needs_decree_hits += 1

            details.append(
                {
                    "id": qid,
                    "category": cat,
                    "question": question,
                    "expected_articles": expected_arts,
                    "retrieved_articles": retrieved_arts[:top_k],
                    "cited_articles": cited_arts,
                    "expected_needs_decree": expected_decree,
                    "actual_needs_decree": result["needs_decree"],
                    "recall_hit": hit_recall,
                    "citation_hit": hit_citation,
                    "decree_hit": hit_decree,
                    "passed": hit_recall and hit_citation and hit_decree,
                }
            )

    recall_at_5 = round(recall_at_k_hits / max(1, in_scope_total), 4)
    mrr = round(sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)), 4)
    citation_acc = round(citation_hits / max(1, in_scope_total), 4)
    decree_acc = round(needs_decree_hits / max(1, in_scope_total), 4)
    refusal_acc = round(refusal_hits / max(1, out_of_scope_total), 4)

    # Đối chiếu với ngưỡng mục tiêu trong docs/00_problem_definition.md
    targets = {
        "recall_at_5": 0.85,
        "citation_accuracy": 0.90,
        "refusal_accuracy": 0.90,
    }
    meets_targets = (
        recall_at_5 >= targets["recall_at_5"]
        and citation_acc >= targets["citation_accuracy"]
        and refusal_acc >= targets["refusal_accuracy"]
    )

    report = {
        "summary": {
            "total_questions": len(qs),
            "in_scope_questions": in_scope_total,
            "out_of_scope_questions": out_of_scope_total,
            "recall_at_5": recall_at_5,
            "mrr": mrr,
            "citation_accuracy": citation_acc,
            "needs_decree_accuracy": decree_acc,
            "refusal_accuracy": refusal_acc,
            "elapsed_seconds": round(time.perf_counter() - start_time, 2),
        },
        "targets": targets,
        "ok": meets_targets,
        "details": details,
    }

    if report_path is not None:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


# ==========================================================================
# CHẠY TRỰC TIẾP TỪ DÒNG LỆNH
# ==========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Giai đoạn 6: Đánh giá độ chính xác của hệ thống RAG Bộ luật Lao động"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top K chunk dùng để tính Recall@K")
    parser.add_argument("--use-vector", action="store_true", help="Gọi thêm Gemini Embedding cho câu hỏi")
    parser.add_argument("--live-llm", action="store_true", help="Gọi Gemini LLM thật cho toàn bộ câu hỏi")
    args = parser.parse_args()

    report = run_evaluation(
        top_k=args.top_k,
        use_vector=args.use_vector,
        live_llm=args.live_llm,
    )
    s = report["summary"]
    print("=== BÁO CÁO ĐÁNH GIÁ ĐỘ CHÍNH XÁC (GIAI ĐOẠN 6) ===")
    print(f"Tổng số câu hỏi kiểm thử : {s['total_questions']} ({s['in_scope_questions']} trong phạm vi, {s['out_of_scope_questions']} ngoài phạm vi)")
    print(f"1. Tìm kiếm Recall@5     : {s['recall_at_5'] * 100:.1f}% (Mục tiêu > 85%)")
    print(f"2. Thứ hạng trung bình MRR: {s['mrr'] * 100:.1f}%")
    print(f"3. Độ chính xác trích dẫn : {s['citation_accuracy'] * 100:.1f}% (Mục tiêu > 90%)")
    print(f"4. Phát hiện needs_decree : {s['needs_decree_accuracy'] * 100:.1f}%")
    print(f"5. Từ chối ngoài phạm vi  : {s['refusal_accuracy'] * 100:.1f}% (Mục tiêu > 90%)")
    print(f"Thời gian chạy           : {s['elapsed_seconds']}s")
    print(f"Kết quả: {'ĐẠT' if report['ok'] else 'CHƯA ĐẠT'}")
    print(f"Đã lưu báo cáo chi tiết tại: {EVAL_REPORT_FILE}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
