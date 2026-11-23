"""
test_app.py — KIỂM THỬ GIAI ĐOẠN 1 & 7 (WEB CHATBOT FASTAPI)
============================================================
Kiểm tra toàn bộ các đường dẫn (endpoints) của ứng dụng Web src/app.py:
    1. GET /           -> Trả về giao diện HTML tiếng Việt, đã bỏ nhãn "Đã đối chiếu Bộ luật Lao động 2019".
    2. GET /api/health -> Trả về trạng thái 'ok' và đủ 258 chunks.
    3. POST /api/retrieve -> Trả về đúng các điều luật đã khử trùng lặp và gộp đủ nội dung.
    4. POST /api/chat  -> Trả về câu trả lời, danh sách nguyên văn `sources`,
                          cờ `needs_decree` và từ chối đúng câu hỏi ngoài phạm vi.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import src.app as app_module
from src.generator import NOT_FOUND_MESSAGE, RAGGenerator
from src.retriever import HybridRetriever


def _setup_mock_generator(monkeypatch):
    """Gắn RAGGenerator dùng hàm LLM giả lập để test Web API nhanh và ổn định."""
    retriever = HybridRetriever.from_files()

    def fake_llm(sys_p: str, usr_p: str) -> tuple[str, int, int]:
        return (
            "Theo khoản 1 Điều 46 Bộ luật Lao động 2019, người lao động làm việc từ đủ 12 tháng "
            "trở lên được hưởng trợ cấp thôi việc. Chi tiết theo Nghị định Chính phủ.",
            320,
            65,
        )

    mock_gen = RAGGenerator(retriever=retriever, llm_fn=fake_llm)
    monkeypatch.setattr(app_module, "_generator", mock_gen)
    return TestClient(app_module.app)


def test_web_home_page_hien_thi_giao_dien(monkeypatch):
    """
    Trang chủ GET / phải trả về mã 200, hiển thị đúng giao diện theo mockup mới:
        - Đã bỏ badge trạng thái ("Căn cứ một phần - cần đối chiếu thêm") và khung vàng ("Chưa đủ căn cứ để trả lời trọn vẹn")
        - Có nút "Xóa toàn bộ" và "deleteHistoryItem" trong Lịch sử
        - Có Panel "NGUYÊN VĂN ĐIỀU LUẬT", thu gọn khoản không trích ("nhấn để mở"), Mục lục luật và nút Feedback Hữu ích/Báo sai.
    """
    client = _setup_mock_generator(monkeypatch)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Tra cứu Bộ luật Lao động 2019" in resp.text
    assert "Căn cứ một phần - cần đối chiếu thêm" not in resp.text
    assert "Chưa đủ căn cứ để trả lời trọn vẹn" not in resp.text
    assert "Rê chuột vào thẻ" not in resp.text
    assert "Đã đối chiếu Bộ luật Lao động 2019" not in resp.text
    assert "Xóa toàn bộ" in resp.text
    assert "deleteHistoryItem" in resp.text
    assert "NGUYÊN VĂN ĐIỀU LUẬT" in resp.text
    assert "nhấn để mở" in resp.text
    assert "Mục lục luật" in resp.text
    assert "/api/chat" in resp.text


def test_api_toc_va_feedback(monkeypatch, tmp_path):
    """GET /api/toc phải trả về 17 Chương và POST /api/feedback phải lưu log thành công."""
    client = _setup_mock_generator(monkeypatch)
    monkeypatch.setattr(app_module, "FEEDBACK_LOG_FILE", tmp_path / "feedback_log.jsonl")

    toc_resp = client.get("/api/toc")
    assert toc_resp.status_code == 200
    chapters = toc_resp.json()["chapters"]
    assert len(chapters) == 17
    total_arts = sum(len(ch["articles"]) for ch in chapters)
    assert total_arts == 220

    fb_resp = client.post(
        "/api/feedback",
        json={
            "query": "ai được quyền nhận lương hưu",
            "answer": "Người lao động đủ tuổi nghỉ hưu...",
            "rating": "helpful",
            "cited_articles": [169],
        },
    )
    assert fb_resp.status_code == 200
    assert fb_resp.json()["status"] == "recorded"
    assert (tmp_path / "feedback_log.jsonl").exists()


def test_api_get_article_tra_ve_nguyen_van_day_du(monkeypatch):
    """GET /api/article/{article_no} phải trả về nguyên văn đầy đủ các khoản của Điều 169 và Điều 35."""
    client = _setup_mock_generator(monkeypatch)
    resp169 = client.get("/api/article/169")
    assert resp169.status_code == 200
    d169 = resp169.json()
    assert d169["article_no"] == 169
    assert len(d169["text"]) >= 1200

    resp35 = client.get("/api/article/35")
    assert resp35.status_code == 200
    d35 = resp35.json()
    assert d35["article_no"] == 35
    assert "1. Người lao động" in d35["text"]
    assert "2. Người lao động" in d35["text"]


def test_api_health_tra_ve_du_258_chunks(monkeypatch):
    """GET /api/health phải báo status='ok' và chunks_count=258."""
    client = _setup_mock_generator(monkeypatch)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["chunks_count"] == 258


def test_api_retrieve_tim_dung_dieu_luat_khong_trung_lap(monkeypatch):
    """POST /api/retrieve phải trả về danh sách điều luật đã khử trùng lặp."""
    client = _setup_mock_generator(monkeypatch)
    resp = client.post(
        "/api/retrieve",
        json={"query": "Trợ cấp thôi việc theo Điều 46", "top_k": 3, "use_vector": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    assert data["results"][0]["article_no"] == 46
    art_nos = [r["article_no"] for r in data["results"]]
    assert len(art_nos) == len(set(art_nos))


def test_api_chat_tra_loi_va_kem_nguyen_van_dieu_luat(monkeypatch):
    """POST /api/chat với câu hỏi trong phạm vi phải trả lời kèm sources đầy đủ nội dung."""
    client = _setup_mock_generator(monkeypatch)
    resp = client.post(
        "/api/chat",
        json={"query": "Làm 3 năm nghỉ việc thì tính trợ cấp thôi việc theo Điều 46 thế nào?", "top_k": 3, "use_vector": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "needs_decree"
    assert data["needs_decree"] is True
    assert 46 in data["cited_articles"]
    assert len(data["sources"]) >= 1
    assert "disclaimer" in data


def test_api_chat_tu_choi_cau_ngoai_pham_vi(monkeypatch):
    """POST /api/chat với câu hỏi ngoài phạm vi phải trả về status='refused' và NOT_FOUND_MESSAGE."""
    client = _setup_mock_generator(monkeypatch)
    resp = client.post(
        "/api/chat",
        json={"query": "Thuế thu nhập cá nhân bậc 2 đóng bao nhiêu tiền?", "top_k": 3, "use_vector": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "refused"
    assert NOT_FOUND_MESSAGE in data["answer"]
    assert data["sources"] == []


def test_api_chat_stream_tra_ve_sse_events(monkeypatch):
    """POST /api/chat/stream phải trả về luồng sự kiện meta, token và done (Giai đoạn 2)."""
    client = _setup_mock_generator(monkeypatch)
    resp = client.post(
        "/api/chat/stream",
        json={"query": "Trợ cấp thôi việc theo Điều 46 tính thế nào?", "top_k": 3, "use_vector": False},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    lines = [line for line in resp.text.splitlines() if line.startswith("data: ")]
    assert len(lines) >= 3
    assert '"type": "meta"' in lines[0]
    assert '"type": "done"' in lines[-1]
