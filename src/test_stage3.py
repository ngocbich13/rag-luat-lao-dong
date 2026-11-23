"""
test_stage3.py — GIAI ĐOẠN 3: BỘ KIỂM THỬ TỰ ĐỘNG 22 CÂU HỎI MẪU (CHẠY BẰNG 1 LỆNH)
====================================================================================
Kiểm thử toàn diện chất lượng truy xuất (Top-3 Accuracy), chất lượng câu trả lời
và tốc độ phản hồi (Time To First Token - TTFT < ~2 giây) trên 22 câu hỏi mẫu:
    - 18 câu hỏi TRONG phạm vi (bao gồm đầy đủ 6 câu bắt buộc của đề bài)
    - 4 câu hỏi NGOÀI phạm vi (bao gồm "thủ tục ly hôn") để kiểm tra bot trả lời
      "Tôi chưa tìm thấy căn cứ pháp lý phù hợp trong dữ liệu hiện có."

Với mỗi câu hỏi, script in ra:
    1. Các điều luật được truy xuất + điểm liên quan (`score`)
    2. Câu trả lời của bot
    3. Thời gian phản hồi chi tiết (TTFT và tổng thời gian)

Cách chạy bằng 1 lệnh duy nhất:
    python src/test_stage3.py
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

from src.generator import NOT_FOUND_MESSAGE, RAGGenerator
from src.retriever import HybridRetriever

STAGE3_REPORT_FILE = ROOT / "data" / "processed" / "stage3_eval_report.json"

# Danh sách 22 câu hỏi kiểm thử (18 câu trong phạm vi + 4 câu ngoài phạm vi)
STAGE3_QUESTIONS: list[dict] = [
    # --- 6 CÂU HỎI BẮT BUỘC TRONG ĐỀ BÀI ---
    {
        "id": 1,
        "scope": "in_scope",
        "query": "ai được quyền nhận lương hưu",
        "expected_top3": [169, 168],
    },
    {
        "id": 2,
        "scope": "in_scope",
        "query": "tuổi nghỉ hưu năm 2026 là bao nhiêu",
        "expected_top3": [169],
    },
    {
        "id": 3,
        "scope": "in_scope",
        "query": "báo trước bao nhiêu ngày khi nghỉ việc",
        "expected_top3": [35],
    },
    {
        "id": 4,
        "scope": "in_scope",
        "query": "trợ cấp thôi việc tính thế nào",
        "expected_top3": [46],
    },
    {
        "id": 5,
        "scope": "in_scope",
        "query": "thời gian thử việc tối đa",
        "expected_top3": [25],
    },
    {
        "id": 6,
        "scope": "in_scope",
        "query": "làm thêm giờ được trả bao nhiêu",
        "expected_top3": [98],
    },
    # --- 12 CÂU HỎI PHỔ BIẾN KHÁC TRONG PHẠM VI BỘ LUẬT LAO ĐỘNG ---
    {
        "id": 7,
        "scope": "in_scope",
        "query": "tiền lương trong thời gian thử việc ít nhất bằng bao nhiêu phần trăm",
        "expected_top3": [26],
    },
    {
        "id": 8,
        "scope": "in_scope",
        "query": "lao động nữ sinh con được nghỉ thai sản trước và sau khi sinh mấy tháng",
        "expected_top3": [139],
    },
    {
        "id": 9,
        "scope": "in_scope",
        "query": "công ty có được giữ bản chính bằng đại học hay căn cước công dân của tôi không",
        "expected_top3": [17],
    },
    {
        "id": 10,
        "scope": "in_scope",
        "query": "trợ cấp mất việc làm tính mỗi năm bao nhiêu tháng lương",
        "expected_top3": [47],
    },
    {
        "id": 11,
        "scope": "in_scope",
        "query": "làm đủ 12 tháng thì người lao động có bao nhiêu ngày nghỉ phép hằng năm",
        "expected_top3": [113],
    },
    {
        "id": 12,
        "scope": "in_scope",
        "query": "bản thân kết hôn hoặc bố mẹ mất được nghỉ việc riêng hưởng nguyên lương mấy ngày",
        "expected_top3": [115],
    },
    {
        "id": 13,
        "scope": "in_scope",
        "query": "người lao động tự ý bỏ việc bao nhiêu ngày thì bị xử lý kỷ luật sa thải",
        "expected_top3": [125],
    },
    {
        "id": 14,
        "scope": "in_scope",
        "query": "công ty được khấu trừ tiền lương hằng tháng tối đa bao nhiêu phần trăm",
        "expected_top3": [102],
    },
    {
        "id": 15,
        "scope": "in_scope",
        "query": "thời giờ làm việc bình thường tối đa mấy giờ trong 01 ngày và 01 tuần",
        "expected_top3": [105],
    },
    {
        "id": 16,
        "scope": "in_scope",
        "query": "người lao động được nghỉ lễ tết hưởng nguyên lương những ngày nào",
        "expected_top3": [112],
    },
    {
        "id": 17,
        "scope": "in_scope",
        "query": "hợp đồng lao động được giao kết theo những loại nào",
        "expected_top3": [20],
    },
    {
        "id": 18,
        "scope": "in_scope",
        "query": "công ty trả lương chậm từ 15 ngày trở lên thì phải đền bù cho người lao động thế nào",
        "expected_top3": [97],
    },
    # --- 4 CÂU HỎI NGOÀI PHẠM VI ---
    {
        "id": 19,
        "scope": "out_of_scope",
        "query": "thủ tục ly hôn",
        "expected_top3": [],
    },
    {
        "id": 20,
        "scope": "out_of_scope",
        "query": "thuế thu nhập cá nhân bậc 3 phải đóng bao nhiêu phần trăm",
        "expected_top3": [],
    },
    {
        "id": 21,
        "scope": "out_of_scope",
        "query": "thủ tục làm sổ đỏ và sang tên quyền sử dụng đất",
        "expected_top3": [],
    },
    {
        "id": 22,
        "scope": "out_of_scope",
        "query": "vượt đèn đỏ và vi phạm nồng độ cồn bị phạt bao nhiêu tiền",
        "expected_top3": [],
    },
]


def run_stage3_suite(
    top_k: int = 3,
    use_vector: bool = True,
) -> dict:
    """
    Chạy tự động toàn bộ 22 câu hỏi mẫu của Giai đoạn 3, in chi tiết từng câu
    và tổng hợp các chỉ số đạt được.
    """
    print("=" * 80)
    print("BẮT ĐẦU KIỂM THỬ TỰ ĐỘNG GIAI ĐOẠN 3 (22 CÂU HỎI MẪU)")
    print("=" * 80)

    # Khởi tạo 1 lần duy nhất (Singleton) giống như khi khởi động Web Server
    t_init0 = time.perf_counter()
    retriever = HybridRetriever.from_files()
    generator = RAGGenerator(retriever=retriever)
    generator.warmup()
    init_ms = round((time.perf_counter() - t_init0) * 1000.0, 2)
    print(f"[Khởi tạo chỉ mục 258x3072 & kết nối Gemini một lần: {init_ms} ms]\n")

    in_scope_count = 0
    top3_hits = 0
    out_of_scope_count = 0
    refusal_hits = 0
    ttft_list_ms: list[float] = []
    total_list_ms: list[float] = []
    records: list[dict] = []

    for item in STAGE3_QUESTIONS:
        qid = item["id"]
        scope = item["scope"]
        query = item["query"]
        expected = item["expected_top3"]

        res = generator.answer(
            query,
            top_k=top_k,
            use_vector=use_vector,
            use_cache=False,
        )
        sources = res.get("sources", [])
        retrieved_articles = [int(s["article_no"]) for s in sources[:3]]
        timings = res.get("metrics", {}).get("timings_ms", {})
        ttft_ms = float(timings.get("ttft_ms", 0.0))
        total_ms = float(timings.get("total_ms", 0.0))

        ttft_list_ms.append(ttft_ms)
        total_list_ms.append(total_ms)

        if scope == "in_scope":
            in_scope_count += 1
            passed = any(art in retrieved_articles for art in expected)
            if passed:
                top3_hits += 1
        else:
            out_of_scope_count += 1
            passed = (
                res.get("status") == "refused"
                and NOT_FOUND_MESSAGE in res.get("answer", "")
                and len(sources) == 0
            )
            if passed:
                refusal_hits += 1

        status_icon = "ĐẠT" if passed else "CHƯA ĐẠT"
        print(f"Câu {qid:02d} [{scope}] -> [{status_icon}]")
        print(f"  Câu hỏi          : {query}")
        if sources:
            src_strs = [
                f"{s['citation']} ({s['article_title']} | điểm={s['score']:.2f})"
                for s in sources
            ]
            print(f"  Điều truy xuất   : {'; '.join(src_strs)}")
        else:
            print("  Điều truy xuất   : (Không có điều luật nào vượt ngưỡng min_score)")

        print(
            f"  Thời gian (ms)   : TTFT = {ttft_ms:.1f} ms ({ttft_ms/1000:.2f}s) | "
            f"Tổng = {total_ms:.1f} ms ({total_ms/1000:.2f}s) "
            f"[rewrite={timings.get('rewrite_ms', 0)}ms, embed={timings.get('embedding_ms', 0)}ms, "
            f"search={timings.get('search_ms', 0)}ms, rerank={timings.get('rerank_ms', 0)}ms, "
            f"llm_ttft={timings.get('llm_ttft_ms', 0)}ms]"
        )
        ans_preview = res["answer"].replace("\n", "\n    ")
        print(f"  Câu trả lời      :\n    {ans_preview}")
        print("-" * 80)

        records.append(
            {
                "id": qid,
                "scope": scope,
                "query": query,
                "expected_top3": expected,
                "retrieved_top3": retrieved_articles,
                "sources": [
                    {
                        "citation": s["citation"],
                        "article_no": s["article_no"],
                        "article_title": s["article_title"],
                        "score": s["score"],
                    }
                    for s in sources
                ],
                "status": res["status"],
                "passed": passed,
                "ttft_ms": ttft_ms,
                "total_ms": total_ms,
                "timings_ms": timings,
                "answer": res["answer"],
            }
        )

    # Kiểm tra thêm tốc độ Cache Hit cho câu hỏi lặp lại
    cache_probe = generator.answer("ai được quyền nhận lương hưu", top_k=top_k, use_cache=True)
    _ = generator.answer("ai được quyền nhận lương hưu", top_k=top_k, use_cache=True)
    cached_res = generator.answer("ai được quyền nhận lương hưu", top_k=top_k, use_cache=True)
    cache_hit_ms = cached_res["metrics"]["timings_ms"]["total_ms"]

    top3_accuracy = round(top3_hits / max(1, in_scope_count), 4)
    refusal_accuracy = round(refusal_hits / max(1, out_of_scope_count), 4)
    avg_ttft_ms = round(sum(ttft_list_ms) / max(1, len(ttft_list_ms)), 1)
    median_ttft_ms = round(sorted(ttft_list_ms)[len(ttft_list_ms) // 2], 1)
    avg_total_ms = round(sum(total_list_ms) / max(1, len(total_list_ms)), 1)

    summary = {
        "total_questions": len(STAGE3_QUESTIONS),
        "in_scope_questions": in_scope_count,
        "out_of_scope_questions": out_of_scope_count,
        "top3_hits": top3_hits,
        "top3_accuracy": top3_accuracy,
        "refusal_hits": refusal_hits,
        "refusal_accuracy": refusal_accuracy,
        "avg_ttft_ms": avg_ttft_ms,
        "median_ttft_ms": median_ttft_ms,
        "avg_total_ms": avg_total_ms,
        "repeat_query_cache_hit_ms": cache_hit_ms,
    }

    report = {
        "summary": summary,
        "records": records,
    }
    STAGE3_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    STAGE3_REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("TỔNG KẾT KẾT QUẢ KIỂM THỬ GIAI ĐOẠN 3")
    print("=" * 80)
    print(f"- Tổng số câu hỏi kiểm thử          : {summary['total_questions']} câu ({in_scope_count} trong phạm vi, {out_of_scope_count} ngoài phạm vi)")
    print(f"- Độ chính xác Điều luật đúng Top 3 : {top3_hits}/{in_scope_count} ({top3_accuracy * 100:.1f}%)")
    print(f"- Độ chính xác Từ chối ngoài phạm vi: {refusal_hits}/{out_of_scope_count} ({refusal_accuracy * 100:.1f}%)")
    print(f"- Thời gian tới token đầu (TTFT TB) : {avg_ttft_ms} ms ({avg_ttft_ms / 1000:.2f}s) | Trung vị: {median_ttft_ms} ms ({median_ttft_ms / 1000:.2f}s)")
    print(f"- Tổng thời gian phản hồi trung bình: {avg_total_ms} ms ({avg_total_ms / 1000:.2f}s)")
    print(f"- Thời gian phản hồi câu hỏi lặp lại: {cache_hit_ms} ms (Cache Hit)")
    print(f"- Đã lưu báo cáo chi tiết tại       : {STAGE3_REPORT_FILE}")
    print("=" * 80)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Chạy bộ kiểm thử tự động 22 câu hỏi Giai đoạn 3")
    parser.add_argument("--top-k", type=int, default=3, help="Số điều luật tối đa đưa vào prompt")
    parser.add_argument("--no-vector", action="store_true", help="Chỉ dùng BM25 + Rerank")
    args = parser.parse_args()

    report = run_stage3_suite(top_k=args.top_k, use_vector=not args.no_vector)
    s = report["summary"]
    ok = s["top3_accuracy"] >= 0.85 and s["refusal_accuracy"] >= 0.90
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
