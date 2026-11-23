"""
test_generator.py — KIỂM THỬ GIAI ĐOẠN 1 & 5 (GEMINI RAG GENERATOR, NGƯỠNG ĐIỂM & FALLBACK)
===========================================================================================
Kiểm tra toàn bộ cơ chế kiểm soát chất lượng câu trả lời trong src/generator.py:
    1. Chặn câu hỏi ngoài phạm vi hoặc điểm < min_score -> trả về đúng câu:
       "Tôi chưa tìm thấy căn cứ pháp lý phù hợp trong dữ liệu hiện có." và sources=[].
    2. Cho phép câu hỏi có từ khóa lao động đi qua tới bộ tìm kiếm.
    3. Phát hiện chính xác cờ `needs_decree` khi điều luật giao Chính phủ quy định chi tiết.
    4. Trả về đầy đủ trích dẫn (`cited_articles`), nguyên văn nguồn (`sources`) không trùng lặp,
       và kết thúc câu trả lời bằng dòng lưu ý không phải tư vấn pháp lý chính thức.
    5. Khi LLM lỗi/timeout, hiển thị thông báo trung thực:
       "Hệ thống AI đang gián đoạn, dưới đây là các điều luật có thể liên quan (chưa được kiểm chứng)"
       và chỉ hiển thị khi điểm liên quan vượt ngưỡng, không trùng lặp điều luật.
"""

from __future__ import annotations

from src.generator import (
    DISCLAIMER_TEXT,
    FALLBACK_HEADER_MESSAGE,
    NOT_FOUND_MESSAGE,
    RAGGenerator,
    check_needs_decree,
    check_out_of_scope,
    format_context,
)
from src.retriever import HybridRetriever


# --------------------------------------------------------------------------
# 1. KIỂM THỬ BỘ LỌC NGOÀI PHẠM VI & CỜ CẦN NGHỊ ĐỊNH
# --------------------------------------------------------------------------
def test_check_out_of_scope_chan_cau_hoi_ngoai_luat_lao_dong():
    """Các câu hỏi về thuế TNCN, sổ đỏ, ly hôn, hình sự phải bị phát hiện ngoài phạm vi."""
    assert check_out_of_scope("Thuế thu nhập cá nhân của tôi tính như thế nào?") is not None
    assert check_out_of_scope("Thủ tục làm sổ đỏ mất bao nhiêu ngày?") is not None
    assert check_out_of_scope("Thủ tục ly hôn và chia tài sản vợ chồng?") is not None
    assert check_out_of_scope("Vượt đèn đỏ và phạt nồng độ cồn bao nhiêu tiền?") is not None

    # Câu hỏi về lao động phải trả về None (tức là nằm trong phạm vi)
    assert check_out_of_scope("Làm việc 3 năm nghỉ việc thì trợ cấp thôi việc tính sao?") is None
    assert check_out_of_scope("Thử việc tối đa bao nhiêu ngày?") is None
    assert check_out_of_scope("ai được quyền nhận lương hưu") is None


def test_check_needs_decree_phat_hien_dieu_giao_chinh_phu():
    """Phát hiện đúng khi trong văn bản điều luật có câu 'Chính phủ quy định chi tiết Điều này'."""
    chunks_with_decree = [
        {"chunk_id": "D046", "article_no": 46, "text": "1. Trợ cấp thôi việc...\n4. Chính phủ quy định chi tiết Điều này."}
    ]
    chunks_without_decree = [
        {"chunk_id": "D013", "article_no": 13, "text": "Hợp đồng lao động là sự thỏa thuận giữa người lao động..."}
    ]
    assert check_needs_decree(chunks_with_decree) is True
    assert check_needs_decree(chunks_without_decree) is False


# --------------------------------------------------------------------------
# 2. KIỂM THỬ RAG GENERATOR VỚI HÀM LLM GIẢ LẬP (MOCK) ĐỂ CHẠY NHANH
# --------------------------------------------------------------------------
def test_generator_tra_loi_co_trich_dan_va_needs_decree():
    """
    Khi hỏi về Điều 46 (Trợ cấp thôi việc), Generator phải:
        - Truy xuất được Điều 46 với đầy đủ nội dung (998 ký tự)
        - Bật cờ needs_decree = True, status = 'needs_decree'
        - Trích xuất được cited_articles = [46]
        - Kết thúc câu trả lời bằng dòng lưu ý không phải tư vấn pháp lý chính thức
    """
    retriever = HybridRetriever.from_files()

    def fake_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        assert "Điều 46" in user_prompt
        return (
            "Theo khoản 1 Điều 46 Bộ luật Lao động 2019, người lao động làm việc từ đủ 12 tháng "
            "trở lên khi chấm dứt hợp đồng hợp pháp được hưởng trợ cấp thôi việc mỗi năm nửa tháng lương. "
            "Chi tiết thực hiện theo Nghị định của Chính phủ.",
            350,
            80,
        )

    gen = RAGGenerator(retriever=retriever, llm_fn=fake_llm)
    res = gen.answer("Điều 46 quy định trợ cấp thôi việc thế nào?", top_k=3, use_vector=False)

    assert res["status"] == "needs_decree"
    assert res["needs_decree"] is True
    assert 46 in res["cited_articles"]
    assert "không phải là tư vấn pháp lý chính thức" in res["answer"].lower()
    assert res["disclaimer"] == DISCLAIMER_TEXT
    assert len(res["sources"]) >= 1
    assert res["sources"][0]["article_no"] == 46
    assert len(res["sources"][0]["text"]) >= 900
    assert res["metrics"]["input_tokens"] == 350


def test_generator_tu_choi_cau_hoi_ngoai_pham_vi_khong_goi_llm():
    """
    Câu hỏi ngoài phạm vi hoặc điểm dưới ngưỡng phải trả lời rõ:
    'Tôi chưa tìm thấy căn cứ pháp lý phù hợp trong dữ liệu hiện có' và sources=[].
    """
    retriever = HybridRetriever.from_files()
    called = {"count": 0}

    def fake_llm(sys_p: str, usr_p: str) -> tuple[str, int, int]:
        called["count"] += 1
        return ("...", 10, 10)

    gen = RAGGenerator(retriever=retriever, llm_fn=fake_llm)
    res = gen.answer("Thuế thu nhập cá nhân bậc 3 là bao nhiêu phần trăm?", use_vector=False)

    assert res["status"] == "refused"
    assert NOT_FOUND_MESSAGE in res["answer"]
    assert res["sources"] == []
    assert called["count"] == 0


def test_generator_fallback_khi_llm_loi_khong_trung_lap_dieu_luat():
    """
    Khi LLM gặp lỗi/timeout, Generator phải trả về thông báo trung thực:
    'Hệ thống AI đang gián đoạn, dưới đây là các điều luật có thể liên quan (chưa được kiểm chứng)'
    và không bao giờ hiển thị trùng lặp cùng 1 điều luật 2 lần.
    """
    retriever = HybridRetriever.from_files()

    def failing_llm(sys_p: str, usr_p: str) -> tuple[str, int, int]:
        raise RuntimeError("503 UNAVAILABLE")

    gen = RAGGenerator(retriever=retriever, llm_fn=failing_llm)
    res = gen.answer("báo trước bao nhiêu ngày khi nghỉ việc", top_k=4, use_vector=False)

    assert res["status"] == "fallback"
    assert FALLBACK_HEADER_MESSAGE in res["answer"]
    art_nos = [s["article_no"] for s in res["sources"]]
    assert len(art_nos) == len(set(art_nos)), f"Trùng lặp điều luật trong fallback: {art_nos}"
    assert 35 in art_nos


def test_generator_xu_ly_cau_hoi_rong():
    """Câu hỏi rỗng hoặc chỉ có khoảng trắng phải trả về trạng thái refused an toàn."""
    retriever = HybridRetriever.from_files()
    gen = RAGGenerator(retriever=retriever, llm_fn=lambda s, u: ("", 0, 0))
    res = gen.answer("   ")
    assert res["status"] == "refused"
    assert NOT_FOUND_MESSAGE in res["answer"]
    assert res["sources"] == []


def test_format_context_co_nhan_nguon_ro_rang():
    """Hàm format_context phải đưa citation và text vào chuỗi ngữ cảnh."""
    sample = [
        {
            "chunk_id": "D001",
            "citation": "Điều 1 Bộ luật Lao động 2019",
            "full_path": "Chương I > Điều 1",
            "article_title": "Phạm vi điều chỉnh",
            "text": "Bộ luật Lao động quy định tiêu chuẩn lao động...",
        }
    ]
    ctx = format_context(sample)
    assert "Điều 1 Bộ luật Lao động 2019" in ctx
    assert "Phạm vi điều chỉnh" in ctx
    assert "Bộ luật Lao động quy định tiêu chuẩn lao động..." in ctx


def test_generator_cache_va_timings_ms():
    """
    Giai đoạn 2: Kiểm tra đo thời gian từng bước (timings_ms) và cache câu hỏi lặp lại.
    Lần gọi thứ 2 với cùng câu hỏi phải trả về cache_hit=True mà không gọi lại LLM.
    """
    retriever = HybridRetriever.from_files()
    calls = {"count": 0}

    def fake_llm(sys_p: str, usr_p: str) -> tuple[str, int, int]:
        calls["count"] += 1
        return ("Theo Điều 25 Bộ luật Lao động 2019, thời gian thử việc tối đa 60 ngày.", 200, 40)

    gen = RAGGenerator(retriever=retriever, llm_fn=fake_llm)
    res1 = gen.answer("thời gian thử việc tối đa", top_k=3, use_vector=False)
    assert calls["count"] == 1
    assert res1["metrics"]["cache_hit"] is False
    t1 = res1["metrics"]["timings_ms"]
    for key in ("rewrite_ms", "embedding_ms", "search_ms", "rerank_ms", "llm_ttft_ms", "llm_total_ms", "ttft_ms", "total_ms"):
        assert key in t1

    res2 = gen.answer("thời gian thử việc tối đa", top_k=3, use_vector=False)
    assert calls["count"] == 1
    assert res2["metrics"]["cache_hit"] is True
    assert res2["metrics"]["timings_ms"]["total_ms"] < 10.0
